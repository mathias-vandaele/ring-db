use memmap2::Mmap;
use std::{
    fs::File,
    io::{BufWriter, Write},
    marker::PhantomData,
    path::Path,
};
use tempfile::TempPath;

use super::traits::{PayloadBuilderOps, open_mmap};
use super::{OwnedPayloadStore, RefPayloadStore};
use crate::error::Result;
use crate::persist::move_file;

// ─── PodStoreBuilder ──────────────────────────────────────────────────────────
//
// Payloads are stored as raw bytes (`bytemuck::bytes_of`), back-to-back,
// with no offset table. `fetch_ref(id)` returns a zero-copy `&T` in O(1):
//
//   offset = id * size_of::<T>()
//   &T     = bytemuck::from_bytes(&mmap[offset..offset + size_of::<T>()])

pub struct PodStoreBuilder<T> {
    writer: BufWriter<File>,
    temp_path: TempPath,
    n_pushed: usize,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> PodStoreBuilder<T> {
    pub fn new() -> Result<Self> {
        let named = tempfile::NamedTempFile::new()?;
        let (file, temp_path) = named.into_parts();
        Ok(Self {
            writer: BufWriter::new(file),
            temp_path,
            n_pushed: 0,
            _marker: PhantomData,
        })
    }

    fn push_inner(&mut self, payload: T) -> Result<()> {
        self.writer.write_all(bytemuck::bytes_of(&payload))?;
        self.n_pushed += 1;
        Ok(())
    }

    fn finish_inner(self) -> Result<PodStore<T>> {
        let Self {
            writer,
            temp_path,
            n_pushed,
            _marker,
        } = self;
        let total = (n_pushed * size_of::<T>()) as u64;
        writer.into_inner().map_err(|e| e.into_error())?;
        let mmap = open_mmap(temp_path.as_ref(), total)?;
        Ok(PodStore {
            mmap,
            _marker: PhantomData,
        })
    }

    fn finish_persisted_inner(self, payloads_path: &Path) -> Result<PodStore<T>> {
        let Self {
            writer,
            temp_path,
            n_pushed,
            _marker,
        } = self;
        let total = (n_pushed * size_of::<T>()) as u64;
        writer.into_inner().map_err(|e| e.into_error())?;
        // move_file handles cross-filesystem moves with a copy fallback.
        // If it fails, temp_path is still alive and its Drop will clean up.
        move_file(temp_path.as_ref(), payloads_path)?;
        let mmap = open_mmap(payloads_path, total)?;
        Ok(PodStore {
            mmap,
            _marker: PhantomData,
        })
    }
}

impl<T: bytemuck::Pod> PayloadBuilderOps<T> for PodStoreBuilder<T> {
    type Store = PodStore<T>;

    fn push(&mut self, payload: T) -> Result<()> {
        self.push_inner(payload)
    }

    fn finish(self) -> Result<PodStore<T>> {
        self.finish_inner()
    }

    /// Pod storage has no offset table; `offsets_path` is ignored.
    fn finish_persisted(self, payloads_path: &Path, _offsets_path: &Path) -> Result<PodStore<T>> {
        self.finish_persisted_inner(payloads_path)
    }
}

// ─── PodStore ─────────────────────────────────────────────────────────────────

pub struct PodStore<T> {
    mmap: Option<Mmap>,
    // Declared after `mmap` so it drops after the mmap is released.
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> PodStore<T> {
    pub fn load(payloads_path: &Path) -> Result<Self> {
        let total_bytes = std::fs::metadata(payloads_path)?.len();
        let mmap = open_mmap(payloads_path, total_bytes)?;
        Ok(PodStore {
            mmap,
            _marker: PhantomData,
        })
    }
}

impl<T: bytemuck::Pod> OwnedPayloadStore<T> for PodStore<T> {
    /// Deserializes by copying `size_of::<T>()` bytes — no bincode, no heap alloc.
    fn fetch_owned(&self, id: u32) -> crate::error::Result<T> {
        Ok(*self.fetch_ref(id))
    }
}

impl<T: bytemuck::Pod> RefPayloadStore<T> for PodStore<T> {
    /// Zero-copy reference into the mmap — O(1), no allocation.
    fn fetch_ref(&self, id: u32) -> &T {
        let size = std::mem::size_of::<T>();
        let offset = id as usize * size;
        bytemuck::from_bytes(
            &self.mmap.as_ref().expect("fetch_ref on empty store")[offset..offset + size],
        )
    }
}
