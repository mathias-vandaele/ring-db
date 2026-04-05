//! # ringDB
//!
//! A high-performance vector database specialised for **ring queries** in
//! high-dimensional spaces.
//!
//! Instead of nearest-neighbour search, ringDB retrieves all vectors whose
//! Euclidean distance to a query lies within a specified interval `[d-λ, d+λ]`.
//!
//! The engine uses **brute-force GPU computation** as a first-class algorithm,
//! leveraging massive parallel dot-product throughput.
//!
//! ## Execution modes
//!
//! | Mode | Vectors | Results | Speed |
//! |------|---------|---------|-------|
//! | Exact float32 | f32 | Exact | Baseline |
//! | Approximate Q8 | int8 | Approximate | Faster |
//!
//! ## Backend selection
//!
//! `BackendPreference::Auto` (default) selects the best available backend at
//! runtime: CUDA → WGPU (Metal/Vulkan/DX12) → CPU.
//!
//! ## Quick start
//!
//! ```
//! use ringdb::{RingDb, RingDbConfig, RingQuery};
//!
//! let mut db = RingDb::new(RingDbConfig::new(4)).unwrap();
//! db.add_vectors(&[1.0f32, 0.0, 0.0, 0.0,
//!                  0.0, 5.0, 0.0, 0.0]).unwrap();
//!
//! let q = [0.0f32; 4];
//! let result = db.query(&RingQuery { query: &q, d: 1.0, lambda: 0.1 }).unwrap();
//! // result.ids contains IDs of all vectors at distance ≈ 1.0 from origin
//! ```

mod backend;
mod config;
mod engine;
mod error;
mod query;
pub mod quant;

pub use config::{BackendPreference, QuantizationMode, RingDbConfig};
pub use engine::RingDb;
pub use error::RingDbError;
pub use query::{QueryResult, RingQuery};
