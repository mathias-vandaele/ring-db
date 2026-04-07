use memmap2::Mmap;
use std::{
    fs::File,
    io::{BufWriter, Write},
    marker::PhantomData,
    mem,
    path::{Path, PathBuf},
};

use crate::error::Result;
use crate::persist::move_file;
use super::traits::{PayloadBuilderOps, flush_writer, new_temp_path, open_mmap};
use super::{OwnedPayloadStore, RefPayloadStore};

// ─── PodStoreBuilder ──────────────────────────────────────────────────────────
//
// Payloads are stored as raw bytes (`bytemuck::bytes_of`), back-to-back,
// with no offset table. `fetch_ref(id)` returns a zero-copy `&T` in O(1):
//
//   offset = id * size_of::<T>()
//   &T     = bytemuck::from_bytes(&mmap[offset..offset + size_of::<T>()])

pub struct PodStoreBuilder<T> {
    writer: Option<BufWriter<File>>,
    temp_path: PathBuf,
    n_pushed: usize,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> PodStoreBuilder<T> {
    pub fn new() -> Result<Self> {
        let temp_path = new_temp_path();
        let file = File::create(&temp_path)?;
        Ok(Self {
            writer: Some(BufWriter::new(file)),
            temp_path,
            n_pushed: 0,
            _marker: PhantomData,
        })
    }

    fn push_inner(&mut self, payload: T) -> Result<()> {
        self.writer
            .as_mut()
            .expect("push called after finish")
            .write_all(bytemuck::bytes_of(&payload))?;
        self.n_pushed += 1;
        Ok(())
    }

    fn total_bytes(&self) -> u64 {
        (self.n_pushed * std::mem::size_of::<T>()) as u64
    }

    fn finish_inner(mut self) -> Result<PodStore<T>> {
        flush_writer(&mut self.writer)?;
        let total = self.total_bytes();
        let mmap = open_mmap(&self.temp_path, total)?;
        Ok(PodStore {
            mmap,
            path: mem::take(&mut self.temp_path),
            delete_on_drop: true,
            _marker: PhantomData,
        })
    }

    fn finish_persisted_inner(mut self, payloads_path: &Path) -> Result<PodStore<T>> {
        flush_writer(&mut self.writer)?;
        move_file(&self.temp_path, payloads_path)?;
        self.temp_path = PathBuf::new();
        let total = self.total_bytes();
        let mmap = open_mmap(payloads_path, total)?;
        Ok(PodStore {
            mmap,
            path: payloads_path.to_path_buf(),
            delete_on_drop: false,
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

impl<T> Drop for PodStoreBuilder<T> {
    fn drop(&mut self) {
        drop(self.writer.take());
        if !self.temp_path.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}

// ─── PodStore ─────────────────────────────────────────────────────────────────

pub struct PodStore<T> {
    mmap: Option<Mmap>,
    path: PathBuf,
    delete_on_drop: bool,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> PodStore<T> {
    pub fn load(payloads_path: &Path) -> Result<Self> {
        let total_bytes = std::fs::metadata(payloads_path)?.len();
        let mmap = open_mmap(payloads_path, total_bytes)?;
        Ok(PodStore {
            mmap,
            path: payloads_path.to_path_buf(),
            delete_on_drop: false,
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

impl<T> Drop for PodStore<T> {
    fn drop(&mut self) {
        drop(self.mmap.take());
        if self.delete_on_drop && !self.path.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
