use memmap2::Mmap;
use tempfile::TempPath;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::File,
    io::{BufWriter, Write},
    marker::PhantomData,
    path::Path,
};

use crate::error::{Result, RingDbError};
use crate::persist::{move_file, read_u64_file, write_u64_file};
use super::traits::{PayloadBuilderOps, open_mmap};
use super::OwnedPayloadStore;

// ─── SerdeStoreBuilder ────────────────────────────────────────────────────────

pub struct SerdeStoreBuilder<T> {
    writer: BufWriter<File>,
    temp_path: TempPath,
    offsets: Vec<u64>,
    cursor: u64,
    _marker: PhantomData<T>,
}

impl<T: Serialize> SerdeStoreBuilder<T> {
    pub fn new() -> Result<Self> {
        let named = tempfile::NamedTempFile::new()?;
        let (file, temp_path) = named.into_parts();
        Ok(Self {
            writer: BufWriter::new(file),
            temp_path,
            offsets: vec![0u64],
            cursor: 0,
            _marker: PhantomData,
        })
    }

    fn push_inner(&mut self, payload: T) -> Result<()> {
        let bytes = bincode::serialize(&payload)
            .map_err(|e| RingDbError::Payload(e.to_string()))?;
        self.writer.write_all(&bytes)?;
        self.cursor += bytes.len() as u64;
        self.offsets.push(self.cursor);
        Ok(())
    }

    fn finish_inner(self) -> Result<SerdeStore<T>> {
        let Self { writer, temp_path, offsets, cursor, _marker } = self;
        writer.into_inner().map_err(|e| e.into_error())?;
        let mmap = open_mmap(temp_path.as_ref(), cursor)?;
        Ok(SerdeStore {
            mmap,
            offsets,
            _marker: PhantomData,
        })
    }

    fn finish_persisted_inner(
        self,
        payloads_path: &Path,
        offsets_path: &Path,
    ) -> Result<SerdeStore<T>> {
        let Self { writer, temp_path, offsets, cursor, _marker } = self;
        writer.into_inner().map_err(|e| e.into_error())?;
        write_u64_file(offsets_path, &offsets)?;
        // move_file handles cross-filesystem moves with a copy fallback.
        // If it fails, temp_path is still alive and its Drop will clean up.
        move_file(temp_path.as_ref(), payloads_path)?;
        let mmap = open_mmap(payloads_path, cursor)?;
        Ok(SerdeStore {
            mmap,
            offsets,
            _marker: PhantomData,
        })
    }
}

impl<T: Serialize> PayloadBuilderOps<T> for SerdeStoreBuilder<T> {
    type Store = SerdeStore<T>;

    fn push(&mut self, payload: T) -> Result<()> {
        self.push_inner(payload)
    }

    fn finish(self) -> Result<SerdeStore<T>> {
        self.finish_inner()
    }

    fn finish_persisted(self, payloads_path: &Path, offsets_path: &Path) -> Result<SerdeStore<T>> {
        self.finish_persisted_inner(payloads_path, offsets_path)
    }
}

// ─── SerdeStore ───────────────────────────────────────────────────────────────

pub struct SerdeStore<T> {
    mmap: Option<Mmap>,
    offsets: Vec<u64>,
    _marker: PhantomData<T>,
}

impl<T: DeserializeOwned> SerdeStore<T> {
    pub fn load(payloads_path: &Path, offsets_path: &Path) -> Result<Self> {
        let offsets = read_u64_file(offsets_path)?;
        let total_bytes = offsets.last().copied().unwrap_or(0);
        let mmap = open_mmap(payloads_path, total_bytes)?;
        Ok(SerdeStore {
            mmap,
            offsets,
            _marker: PhantomData,
        })
    }

    fn fetch_inner(&self, id: u32) -> Result<T> {
        let idx = id as usize;
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        let bytes = match &self.mmap {
            Some(mmap) => &mmap[start..end],
            None => &[],
        };
        bincode::deserialize(bytes).map_err(|e| RingDbError::Payload(e.to_string()))
    }
}

impl<T: DeserializeOwned> OwnedPayloadStore<T> for SerdeStore<T> {
    fn fetch_owned(&self, id: u32) -> Result<T> {
        self.fetch_inner(id)
    }
}
