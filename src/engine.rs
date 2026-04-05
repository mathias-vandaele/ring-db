use std::time::Instant;

use crate::backend::{CpuBackend, RingComputeBackend};
use crate::config::RingDbConfig;
use crate::error::Result;
use crate::query::{QueryResult, RingQuery};

/// Builder for a ring-query vector database.
///
/// Insert vectors with [`add_vector()`](Self::add_vector), then call
/// [`build()`](Self::build) to transfer ownership to the compute backend
/// and obtain a [`SealedRingDb`] that can be queried.
///
/// # Example
///
/// ```
/// use ringdb::{RingDb, RingDbConfig, RingQuery};
///
/// let config = RingDbConfig::new(4);
/// let mut db = RingDb::new(config).unwrap();
///
/// db.add_vector(&[1.0, 0.0, 0.0, 0.0]).unwrap();
/// db.add_vector(&[0.0, 1.0, 0.0, 0.0]).unwrap();
/// db.add_vector(&[3.0, 4.0, 0.0, 0.0]).unwrap();
///
/// let db = db.build().unwrap();
///
/// let q = [1.0f32, 0.0, 0.0, 0.0];
/// let result = db.query(&RingQuery { query: &q, d: 5.0, lambda: 1.0 }).unwrap();
/// println!("hits: {:?}", result.ids);
/// ```
pub struct RingDb {
    config: RingDbConfig,
    backend: Box<dyn RingComputeBackend>,
    n_vectors: usize,

    /// Staging buffer: f32 vectors, row-major, `n_vectors × dims`.
    vectors: Vec<f32>,

    /// Staging buffer: per-vector squared L2 norm.
    norms_sq: Vec<f32>,
}

impl RingDb {
    /// Create a new empty `RingDb` with the given configuration.
    pub fn new(config: RingDbConfig) -> Result<Self> {
        Ok(Self {
            config,
            backend: Box::new(CpuBackend::new()),
            n_vectors: 0,
            vectors: Vec::new(),
            norms_sq: Vec::new(),
        })
    }

    /// Insert a single vector into the database.
    ///
    /// Vectors are assigned sequential IDs starting from 0.
    /// The slice length must equal `dims`.
    pub fn add_vector(&mut self, vector: &[f32]) -> Result<()> {
        let dims = self.config.dims;
        if vector.len() != dims {
            return Err(crate::error::RingDbError::DimensionMismatch {
                expected: dims,
                got: vector.len(),
            });
        }

        let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
        self.norms_sq.push(norm_sq);
        self.vectors.extend_from_slice(vector);
        self.n_vectors += 1;
        Ok(())
    }

    /// Transfer ownership of the accumulated vectors to the compute backend
    /// and seal the database.
    ///
    /// Consumes `self` and returns a [`SealedRingDb`] that can only be
    /// queried. The staging buffers are moved into the backend (zero-cost
    /// for the CPU backend) and then dropped.
    pub fn build(mut self) -> Result<SealedRingDb> {
        let dims = self.config.dims;
        let n_vectors = self.n_vectors;
        self.backend
            .upload_f32_dataset(dims, self.vectors, self.norms_sq)?;
        Ok(SealedRingDb {
            config: self.config,
            backend: self.backend,
            n_vectors,
        })
    }

    /// Number of vectors currently staged.
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Returns `true` if no vectors have been inserted.
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Number of dimensions per vector.
    pub fn dims(&self) -> usize {
        self.config.dims
    }

    /// Name of the backend currently in use.
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}

/// Sealed (immutable) ring-query database.
///
/// Obtained by calling [`RingDb::build()`]. Vectors can no longer be
/// inserted — only queries are allowed.
///
/// All data is owned by the compute backend; the `SealedRingDb` itself
/// holds no copy of the vector data.
pub struct SealedRingDb {
    config: RingDbConfig,
    backend: Box<dyn RingComputeBackend>,
    n_vectors: usize,
}

impl SealedRingDb {
    /// Execute a ring query and return matching vector IDs.
    pub fn query(&self, q: &RingQuery<'_>) -> Result<QueryResult> {
        let dims = self.config.dims;
        if q.query.len() != dims {
            return Err(crate::error::RingDbError::DimensionMismatch {
                expected: dims,
                got: q.query.len(),
            });
        }

        let t = Instant::now();
        let ids = self.backend.ring_query_f32(dims, q.query, q.d, q.lambda)?;
        let elapsed = t.elapsed();

        Ok(QueryResult {
            ids,
            backend_used: self.backend.name(),
            elapsed,
        })
    }

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Returns `true` if the database contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Number of dimensions per vector.
    pub fn dims(&self) -> usize {
        self.config.dims
    }

    /// Name of the backend currently in use.
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}
