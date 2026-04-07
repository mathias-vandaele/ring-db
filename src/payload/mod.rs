pub mod pod;
pub mod serde;
pub(crate) mod traits;

pub use pod::{PodStore, PodStoreBuilder};
pub use serde::{SerdeStore, SerdeStoreBuilder};
pub use traits::PayloadBuilderOps;

use crate::error::Result;

// ─── Store traits ─────────────────────────────────────────────────────────────

/// Implemented by all payload stores. Provides owned deserialization.
///
/// `SerdeStore<T>` deserializes via bincode.
/// `PodStore<T>` copies `size_of::<T>()` bytes from the mmap.
#[doc(hidden)]
pub trait OwnedPayloadStore<T> {
    fn fetch_owned(&self, id: u32) -> Result<T>;

    fn fetch_many_owned(&self, ids: &[u32]) -> Result<Vec<T>> {
        ids.iter().map(|&id| self.fetch_owned(id)).collect()
    }
}

/// Implemented only by `PodStore<T>`. Provides zero-copy `&T` references.
///
/// This trait is the static gate for [`SealedRingDb::fetch_pod`]: the method
/// only exists when `T::Store: RefPayloadStore<T>`, which is only true for
/// types whose `#[derive(Payload)]` chose `storage = "pod"`.
#[doc(hidden)]
pub trait RefPayloadStore<T> {
    fn fetch_ref(&self, id: u32) -> &T;

    fn fetch_many_ref(&self, ids: &[u32]) -> Vec<&T> {
        ids.iter().map(|&id| self.fetch_ref(id)).collect()
    }
}

// ─── Payload trait ────────────────────────────────────────────────────────────

/// Marker trait that ties a user type to its payload storage strategy.
///
/// Implement this trait via `#[derive(Payload)]` from the `ring-db-derive`
/// crate (re-exported as `ringdb::Payload`).
///
/// Two storage strategies are available:
///
/// | Strategy | Attribute | Fetch return |
/// |----------|-----------|--------------|
/// | Serde (default) | *(none)* | `T` (owned, bincode) |
/// | Pod | `#[payload(storage = "pod")]` | `&T` (zero-copy mmap) |
pub trait Payload: Sized {
    /// The read-only store produced by `Builder` and held inside
    /// [`SealedRingDb`](crate::SealedRingDb).
    type Store: OwnedPayloadStore<Self>;

    /// The write-side builder held inside [`RingDb`](crate::RingDb) during
    /// insertion. Consumed by `build()` to produce `Store`.
    type Builder: PayloadBuilderOps<Self, Store = Self::Store>;

    /// Create a fresh builder (called by `RingDb::new`).
    fn make_builder() -> Result<Self::Builder>;

    /// Load the store from a persisted directory (called by `RingDb::load`).
    fn load_store(dir: &std::path::Path) -> Result<Self::Store>;
}

// ─── impl Payload for () ──────────────────────────────────────────────────────
//
// Allows `RingDb::new(config)` without specifying a payload type when none
// is needed. Serde serializes `()` to 0 bytes, so the mmap stays empty.

impl Payload for () {
    type Store = SerdeStore<()>;
    type Builder = SerdeStoreBuilder<()>;

    fn make_builder() -> Result<Self::Builder> {
        SerdeStoreBuilder::new()
    }

    fn load_store(dir: &std::path::Path) -> Result<Self::Store> {
        SerdeStore::load(&dir.join("payloads.bin"), &dir.join("offsets.bin"))
    }
}
