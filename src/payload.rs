use memmap2::Mmap;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::File,
    io::{BufWriter, Write},
    marker::PhantomData,
    mem,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::error::{Result, RingDbError};
use crate::persist::{move_file, write_u64_file};

fn new_temp_path() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("ringdb-payloads-{}-{}.bin", std::process::id(), id))
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Flush and close a `BufWriter<File>` wrapped in an `Option`.
///
/// Takes the writer out of `slot` so callers don't see a partial flush on drop.
fn flush_writer(slot: &mut Option<BufWriter<File>>) -> Result<()> {
    if let Some(writer) = slot.take() {
        writer.into_inner().map_err(|e| e.into_error())?;
    }
    Ok(())
}

/// Mmap `path` read-only, or return `None` if `total_bytes == 0`.
///
/// # Safety
/// Callers must ensure the file is not modified after this call.
fn open_mmap(path: &Path, total_bytes: u64) -> Result<Option<Mmap>> {
    if total_bytes == 0 {
        return Ok(None);
    }
    let file = File::open(path)?;
    // SAFETY: caller guarantees the file is read-only from this point on.
    Ok(Some(unsafe { Mmap::map(&file) }?))
}

/// Write-side of cold payload storage.
///
/// Each payload is serialized immediately on [`push`] and streamed to a temp
/// file — no `Vec<T>` ever accumulates in RAM. Peak memory per call is just
/// `sizeof(T) + serialized_size(T)`, both dropped before returning.
///
/// Call [`finish`] once all vectors have been added to close the file and
/// hand off a read-only mmapped [`PayloadStore`].
pub struct PayloadStoreBuilder<T> {
    /// Buffered writer into the temp file. Wrapped in `Option` so we can
    /// take ownership in `finish` and `drop`.
    writer: Option<BufWriter<File>>,
    temp_path: PathBuf,
    /// `offsets[i]` is the byte start of payload `i`; `offsets[n]` is sentinel.
    offsets: Vec<u64>,
    cursor: u64,
    _marker: PhantomData<T>,
}

impl<T: Serialize> PayloadStoreBuilder<T> {
    pub(crate) fn new() -> Result<Self> {
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

    /// Serialize `payload` and stream it directly to the temp file.
    ///
    /// `payload` is consumed and dropped immediately after serialization;
    /// only the transient `Vec<u8>` from `bincode` briefly exists in RAM.
    pub(crate) fn push(&mut self, payload: T) -> Result<()> {
        let bytes =
            bincode::serialize(&payload).map_err(|e| RingDbError::Payload(e.to_string()))?;
        self.writer
            .as_mut()
            .expect("push called after finish")
            .write_all(&bytes)?;
        self.cursor += bytes.len() as u64;
        self.offsets.push(self.cursor);
        Ok(())
    }

    /// Flush the temp file and mmap it read-only.
    ///
    /// After this call, `self` is consumed. The returned [`PayloadStore`]
    /// owns the mmap and is responsible for deleting the temp file on drop.
    pub(crate) fn finish(mut self) -> Result<PayloadStore<T>> {
        flush_writer(&mut self.writer)?;
        let mmap = open_mmap(&self.temp_path, self.cursor)?;

        // Transfer ownership to PayloadStore and clear our temp_path so
        // Drop does not double-delete the file.
        let store = PayloadStore {
            mmap,
            offsets: mem::take(&mut self.offsets),
            path: mem::take(&mut self.temp_path),
            delete_on_drop: true,
            _marker: PhantomData,
        };

        Ok(store)
        // `self` is dropped here with an empty temp_path → Drop is a no-op.
    }

    /// Flush the staging data to persistent files, then mmap the payload file
    /// read-only.
    ///
    /// - `payloads_path` — destination for the raw payload bytes (content of
    ///   the current temp file).
    /// - `offsets_path` — destination for the `offsets` table.
    ///
    /// The returned [`PayloadStore`] points at the persisted files and will
    /// **not** delete them on drop.
    pub(crate) fn finish_persisted(
        mut self,
        payloads_path: &Path,
        offsets_path: &Path,
    ) -> Result<PayloadStore<T>> {
        flush_writer(&mut self.writer)?;

        // Persist the offset table.
        write_u64_file(offsets_path, &self.offsets)?;

        // Move the temp payload file to its final location.
        move_file(&self.temp_path, payloads_path)?;
        // Clear temp_path so Drop doesn't try to remove the now-moved file.
        self.temp_path = PathBuf::new();

        let mmap = open_mmap(payloads_path, self.cursor)?;

        Ok(PayloadStore {
            mmap,
            offsets: mem::take(&mut self.offsets),
            path: payloads_path.to_path_buf(),
            delete_on_drop: false,
            _marker: PhantomData,
        })
    }
}

impl<T> Drop for PayloadStoreBuilder<T> {
    fn drop(&mut self) {
        // Close the write handle first (Windows: file must be closed before delete).
        drop(self.writer.take());
        if !self.temp_path.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}

/// Read-only cold storage for per-vector payloads.
///
/// Payload bytes live in a file-backed mmap; the OS can page them out under
/// memory pressure. The only hot data is the offset table (`Vec<u64>`,
/// 8 bytes per vector).
pub struct PayloadStore<T> {
    mmap: Option<Mmap>,
    offsets: Vec<u64>,
    /// File backing the mmap (temp or persisted).
    path: PathBuf,
    /// Whether to delete `path` on drop.  `true` for anonymous temp files,
    /// `false` for user-managed persisted files.
    delete_on_drop: bool,
    _marker: PhantomData<T>,
}

impl<T: DeserializeOwned> PayloadStore<T> {
    /// Reconstruct a `PayloadStore` from files written by
    /// [`PayloadStoreBuilder::finish_persisted`].
    ///
    /// The files are mmapped read-only; they are **not** deleted on drop.
    pub(crate) fn load(
        payloads_path: &std::path::Path,
        offsets_path: &std::path::Path,
    ) -> Result<Self> {
        let offsets = crate::persist::read_u64_file(offsets_path)?;

        // The last entry in `offsets` is the sentinel (total byte size).
        let total_bytes = offsets.last().copied().unwrap_or(0);
        let mmap = open_mmap(payloads_path, total_bytes)?;

        Ok(PayloadStore {
            mmap,
            offsets,
            path: payloads_path.to_path_buf(),
            delete_on_drop: false,
            _marker: PhantomData,
        })
    }

    /// Deserialize the payload for a single vector ID.
    pub fn fetch(&self, id: u32) -> Result<T> {
        let idx = id as usize;
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        let bytes = match &self.mmap {
            Some(mmap) => &mmap[start..end],
            None => &[],
        };
        bincode::deserialize(bytes).map_err(|e| RingDbError::Payload(e.to_string()))
    }

    /// Deserialize payloads for a slice of vector IDs, in order.
    pub fn fetch_many(&self, ids: &[u32]) -> Result<Vec<T>> {
        ids.iter().map(|&id| self.fetch(id)).collect()
    }
}

impl<T> Drop for PayloadStore<T> {
    fn drop(&mut self) {
        // Drop the mmap first to release all OS-level file handles.
        // This is required on Windows before the file can be deleted.
        drop(self.mmap.take());
        if self.delete_on_drop && !self.path.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
