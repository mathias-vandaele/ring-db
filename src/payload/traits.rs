use memmap2::Mmap;
use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::error::Result;

// ─── Temp-file helpers ────────────────────────────────────────────────────────

pub(super) fn new_temp_path() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ringdb-payloads-{}-{}.bin",
        std::process::id(),
        id
    ))
}

pub(super) fn flush_writer(slot: &mut Option<BufWriter<File>>) -> Result<()> {
    if let Some(writer) = slot.take() {
        writer.into_inner().map_err(|e| e.into_error())?;
    }
    Ok(())
}

/// Mmap `path` read-only, or return `None` if `total_bytes == 0`.
///
/// # Safety
/// Callers must ensure the file is not modified after this call.
pub(super) fn open_mmap(path: &Path, total_bytes: u64) -> Result<Option<Mmap>> {
    if total_bytes == 0 {
        return Ok(None);
    }
    let file = File::open(path)?;
    // SAFETY: caller guarantees the file is read-only from this point on.
    Ok(Some(unsafe { Mmap::map(&file) }?))
}

// ─── Builder trait ────────────────────────────────────────────────────────────
//
// No longer object-safe (finish consumes self by value) — that's intentional.
// RingDb<T: Payload> holds T::Builder as a concrete field, so no Box<dyn> needed.

/// Internal trait implemented by `SerdeStoreBuilder<T>` and `PodStoreBuilder<T>`.
///
/// The associated type `Store` ties the builder to the store it produces, which
/// lets `RingDb::build()` return a `SealedRingDb<T>` without knowing the
/// concrete storage type at the call site.
#[doc(hidden)]
pub trait PayloadBuilderOps<T>: Sized + 'static {
    type Store;

    fn push(&mut self, payload: T) -> Result<()>;

    /// Flush and mmap the temp file for in-memory-only use.
    fn finish(self) -> Result<Self::Store>;

    /// Flush, persist the payload file to `payloads_path`, and — for Serde
    /// storage — write the offset table to `offsets_path`.
    /// Pod storage ignores `offsets_path`.
    fn finish_persisted(self, payloads_path: &Path, offsets_path: &Path) -> Result<Self::Store>;
}
