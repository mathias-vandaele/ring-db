use memmap2::Mmap;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::File,
    io::{BufWriter, Write},
    marker::PhantomData,
    mem,
    path::{Path, PathBuf},
};

use crate::error::{Result, RingDbError};
use crate::persist::{move_file, read_u64_file, write_u64_file};
use super::traits::{PayloadBuilderOps, flush_writer, new_temp_path, open_mmap};
use super::{OwnedPayloadStore};

// ─── SerdeStoreBuilder ────────────────────────────────────────────────────────

pub struct SerdeStoreBuilder<T> {
    writer: Option<BufWriter<File>>,
    temp_path: PathBuf,
    offsets: Vec<u64>,
    cursor: u64,
    _marker: PhantomData<T>,
}

impl<T: Serialize + 'static> SerdeStoreBuilder<T> {
    pub fn new() -> Result<Self> {
        let temp_path = new_temp_path();
        let file = File::create(&temp_path)?;
        Ok(Self {
            writer: Some(BufWriter::new(file)),
            temp_path,
            offsets: vec![0u64],
            cursor: 0,
            _marker: PhantomData,
        })
    }

    fn push_inner(&mut self, payload: T) -> Result<()> {
        let bytes = bincode::serialize(&payload)
            .map_err(|e| RingDbError::Payload(e.to_string()))?;
        self.writer
            .as_mut()
            .expect("push called after finish")
            .write_all(&bytes)?;
        self.cursor += bytes.len() as u64;
        self.offsets.push(self.cursor);
        Ok(())
    }

    fn finish_inner(mut self) -> Result<SerdeStore<T>> {
        flush_writer(&mut self.writer)?;
        let mmap = open_mmap(&self.temp_path, self.cursor)?;
        Ok(SerdeStore {
            mmap,
            offsets: mem::take(&mut self.offsets),
            path: mem::take(&mut self.temp_path),
            delete_on_drop: true,
            _marker: PhantomData,
        })
    }

    fn finish_persisted_inner(
        mut self,
        payloads_path: &Path,
        offsets_path: &Path,
    ) -> Result<SerdeStore<T>> {
        flush_writer(&mut self.writer)?;
        write_u64_file(offsets_path, &self.offsets)?;
        move_file(&self.temp_path, payloads_path)?;
        self.temp_path = PathBuf::new();
        let mmap = open_mmap(payloads_path, self.cursor)?;
        Ok(SerdeStore {
            mmap,
            offsets: mem::take(&mut self.offsets),
            path: payloads_path.to_path_buf(),
            delete_on_drop: false,
            _marker: PhantomData,
        })
    }
}

impl<T: Serialize + 'static> PayloadBuilderOps<T> for SerdeStoreBuilder<T> {
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

impl<T> Drop for SerdeStoreBuilder<T> {
    fn drop(&mut self) {
        drop(self.writer.take());
        if !self.temp_path.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}

// ─── SerdeStore ───────────────────────────────────────────────────────────────

pub struct SerdeStore<T> {
    mmap: Option<Mmap>,
    offsets: Vec<u64>,
    path: PathBuf,
    delete_on_drop: bool,
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
            path: payloads_path.to_path_buf(),
            delete_on_drop: false,
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

impl<T> Drop for SerdeStore<T> {
    fn drop(&mut self) {
        drop(self.mmap.take());
        if self.delete_on_drop && !self.path.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
