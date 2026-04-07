//! # ringDB
//!
//! A vector database specialised for **ring queries** in high-dimensional spaces.
//!
//! Instead of nearest-neighbour search, ringDB retrieves all vectors whose
//! Euclidean distance to a query lies within a specified interval `[d-λ, d+λ]`.
//!
//! ## Quick start
//!
//! ```
//! use ringdb::{RingDb, RingDbConfig, RingQuery};
//!
//! let mut db = RingDb::new(RingDbConfig::new(4)).unwrap();
//! db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], ()).unwrap();
//! db.add_vector(&[0.0, 5.0, 0.0, 0.0], ()).unwrap();
//!
//! let db = db.build().unwrap();
//! let result = db.query(&RingQuery { query: &[0.0f32; 4], d: 1.0, lambda: 0.1 }).unwrap();
//! // result.ids contains IDs of all vectors at distance ≈ 1.0 from origin
//! ```
mod backend;
mod config;
mod engine;
mod error;
pub mod payload;
mod persist;
mod query;

pub use config::{BackendPreference, RingDbConfig};
pub use engine::{RingDb, SealedRingDb};
pub use error::RingDbError;
pub use payload::{OwnedPayloadStore, Payload, RefPayloadStore};
pub use query::{DiskQuery, QueryResult, RangeQuery, RingQuery};

// Re-export the derive macro so users write `use ringdb::Payload` for both
// the trait and the derive, mirroring the serde pattern.
pub use ring_db_derive::Payload;

/// Private implementation details referenced by the `#[derive(Payload)]`
/// generated code. Not part of the public API; subject to change.
#[doc(hidden)]
pub mod __private {
    pub use crate::error::Result;
    pub use crate::payload::pod::{PodStore, PodStoreBuilder};
    pub use crate::payload::serde::{SerdeStore, SerdeStoreBuilder};
}
