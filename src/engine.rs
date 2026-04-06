use std::path::Path;
use std::time::Instant;

use serde::{Serialize, de::DeserializeOwned};

use crate::backend::{CpuBackend, RingComputeBackend};
use crate::config::RingDbConfig;
use crate::error::{Result, RingDbError};
use crate::payload::{PayloadStore, PayloadStoreBuilder};
use crate::persist::{read_f32_file, read_meta, write_f32_file, write_meta};
use crate::query::{DiskQuery, QueryResult, RangeQuery, RingQuery};

/// Builder for a ring-query vector database.
///
/// Insert vectors (and their associated payloads) with
/// [`add_vector()`](Self::add_vector), then call [`build()`](Self::build) to
/// transfer ownership to the compute backend and obtain a [`SealedRingDb`]
/// that can be queried.
///
/// `T` is the payload type stored alongside each vector. Use `T = ()` when
/// no payload is needed.
///
/// # Example — no payload
///
/// ```
/// use ringdb::{RingDb, RingDbConfig, RingQuery};
///
/// let config = RingDbConfig::new(4);
/// let mut db = RingDb::new(config).unwrap();
///
/// db.add_vector(&[1.0, 0.0, 0.0, 0.0], ()).unwrap();
/// db.add_vector(&[0.0, 1.0, 0.0, 0.0], ()).unwrap();
///
/// let db = db.build().unwrap();
/// let result = db.query(&RingQuery { query: &[1.0f32, 0.0, 0.0, 0.0], d: 1.0, lambda: 0.1 }).unwrap();
/// println!("hits: {:?}", result.ids);
/// ```
///
/// # Example — with payload
///
/// ```
/// use ringdb::{RingDb, RingDbConfig, RingQuery};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct Meta { label: String }
///
/// let mut db: RingDb<Meta> = RingDb::new(RingDbConfig::new(2)).unwrap();
/// db.add_vector(&[1.0, 0.0], Meta { label: "dog".into() }).unwrap();
/// db.add_vector(&[0.0, 1.0], Meta { label: "cat".into() }).unwrap();
///
/// let db = db.build().unwrap();
/// let result = db.query(&RingQuery { query: &[1.0f32, 0.0], d: 1.0, lambda: 0.1 }).unwrap();
/// let payloads = db.fetch_payloads(&result.ids).unwrap();
/// ```
pub struct RingDb<T = ()> {
    config: RingDbConfig,
    backend: Box<dyn RingComputeBackend>,
    n_vectors: usize,

    /// Staging buffer: f32 vectors, row-major, `n_vectors × dims`.
    vectors: Vec<f32>,

    /// Staging buffer: per-vector squared L2 norm.
    norms_sq: Vec<f32>,

    /// Streams payloads to a temp file as they arrive; never accumulates in RAM.
    payload_builder: PayloadStoreBuilder<T>,
}

impl<T: Serialize + DeserializeOwned> RingDb<T> {
    /// Create a new empty `RingDb` with the given configuration.
    pub fn new(config: RingDbConfig) -> Result<Self> {
        Ok(Self {
            config,
            backend: Box::new(CpuBackend::new()),
            n_vectors: 0,
            vectors: Vec::new(),
            norms_sq: Vec::new(),
            payload_builder: PayloadStoreBuilder::new()?,
        })
    }

    /// Insert a single vector and its associated payload.
    ///
    /// Vectors are assigned sequential IDs starting from 0.
    /// The slice length must equal `dims`.
    pub fn add_vector(&mut self, vector: &[f32], payload: T) -> Result<()> {
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
        self.payload_builder.push(payload)?;
        self.n_vectors += 1;
        Ok(())
    }

    /// Transfer ownership of the accumulated data to the compute backend and
    /// seal the database.
    ///
    /// Vector data is moved into the backend (zero-cost for the CPU backend).
    /// Payloads are serialized and moved into a cold anonymous mmap — the
    /// staging `Vec<T>` is dropped immediately after.
    ///
    /// If [`RingDbConfig::persist_dir`] is set the following files are written
    /// to that directory before sealing:
    ///
    /// | File | Content |
    /// |------|---------|
    /// | `meta.bin` | `dims` + `n_vectors` as little-endian u64 |
    /// | `vectors.bin` | raw f32 vectors (row-major) |
    /// | `norms_sq.bin` | raw f32 squared norms |
    /// | `payloads.bin` | concatenated bincode payload bytes |
    /// | `offsets.bin` | byte offsets (u64) into `payloads.bin` |
    ///
    /// The database can be reloaded later with [`RingDb::load()`].
    pub fn build(mut self) -> Result<SealedRingDb<T>> {
        let dims = self.config.dims;
        let n_vectors = self.n_vectors;

        if let Some(dir) = self.config.persist_dir.clone() {
            std::fs::create_dir_all(&dir)?;

            write_meta(&dir.join("meta.bin"), dims, n_vectors)?;
            write_f32_file(&dir.join("vectors.bin"), &self.vectors)?;
            write_f32_file(&dir.join("norms_sq.bin"), &self.norms_sq)?;

            let payload_store = self
                .payload_builder
                .finish_persisted(&dir.join("payloads.bin"), &dir.join("offsets.bin"))?;

            self.backend
                .upload_f32_dataset(dims, self.vectors, self.norms_sq)?;

            return Ok(SealedRingDb {
                config: self.config,
                backend: self.backend,
                n_vectors,
                payload_store,
            });
        }

        self.backend
            .upload_f32_dataset(dims, self.vectors, self.norms_sq)?;
        let payload_store = self.payload_builder.finish()?;
        Ok(SealedRingDb {
            config: self.config,
            backend: self.backend,
            n_vectors,
            payload_store,
        })
    }

    /// Reconstruct a [`SealedRingDb`] from a directory previously written by
    /// [`RingDb::build()`] when [`RingDbConfig::persist_dir`] was set.
    ///
    /// Reads `meta.bin`, `vectors.bin`, `norms_sq.bin`, `payloads.bin`, and
    /// `offsets.bin` from `dir`, re-uploads the vectors and norms to the
    /// requested backend, and memory-maps the payload file.
    ///
    /// Pass [`BackendPreference::Cpu`] for the CPU backend (the only option
    /// today; CUDA and others will be added later).
    ///
    /// # Errors
    ///
    /// Returns [`RingDbError::Corrupt`] if any file is missing, has an
    /// unexpected size, or the dimension/count metadata is inconsistent.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ringdb::{RingDb, RingDbConfig, SealedRingDb};
    /// use ringdb::BackendPreference;
    /// use std::path::Path;
    ///
    /// // --- save ---
    /// let mut db = RingDb::<()>::new(RingDbConfig::new(4).with_persist_dir("/tmp/mydb")).unwrap();
    /// db.add_vector(&[1.0, 0.0, 0.0, 0.0], ()).unwrap();
    /// let _sealed = db.build().unwrap(); // writes files to /tmp/mydb
    ///
    /// // --- load ---
    /// let loaded = RingDb::<()>::load(Path::new("/tmp/mydb"), BackendPreference::Cpu).unwrap();
    /// ```
    pub fn load(
        dir: &Path,
        backend_preference: crate::config::BackendPreference,
    ) -> Result<SealedRingDb<T>> {
        let (dims, n_vectors) = read_meta(&dir.join("meta.bin"))?;

        let vectors = read_f32_file(&dir.join("vectors.bin"))?;
        let norms_sq = read_f32_file(&dir.join("norms_sq.bin"))?;

        let expected_vec_len = n_vectors * dims;
        if vectors.len() != expected_vec_len {
            return Err(RingDbError::Corrupt(format!(
                "vectors.bin has {} f32 values, expected {} (n_vectors={} × dims={})",
                vectors.len(),
                expected_vec_len,
                n_vectors,
                dims,
            )));
        }
        if norms_sq.len() != n_vectors {
            return Err(RingDbError::Corrupt(format!(
                "norms_sq.bin has {} f32 values, expected {}",
                norms_sq.len(),
                n_vectors,
            )));
        }

        let mut backend: Box<dyn RingComputeBackend> = match backend_preference {
            crate::config::BackendPreference::Cpu => Box::new(CpuBackend::new()),
        };
        backend.upload_f32_dataset(dims, vectors, norms_sq)?;

        let payload_store =
            PayloadStore::load(&dir.join("payloads.bin"), &dir.join("offsets.bin"))?;

        Ok(SealedRingDb {
            config: RingDbConfig::new(dims)
                .with_persist_dir(dir)
                .with_backend_preference(backend_preference),
            backend,
            n_vectors,
            payload_store,
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
/// Obtained by calling [`RingDb::build()`] or [`RingDb::load()`]. Vectors can
/// no longer be inserted — only queries and payload fetches are allowed.
///
/// The hot side (vectors + norms) is owned by the compute backend.
/// The cold side (payloads) lives in an anonymous mmap managed by
/// [`PayloadStore`].
pub struct SealedRingDb<T = ()> {
    config: RingDbConfig,
    backend: Box<dyn RingComputeBackend>,
    n_vectors: usize,
    payload_store: PayloadStore<T>,
}

impl<T: Serialize + DeserializeOwned> SealedRingDb<T> {
    /// Execute a ring query and return matching vector IDs.
    ///
    /// The ring `[d - lambda, d + lambda]` is converted to `[d_min, d_max]`
    /// internally; negative lower bounds are clamped to 0.
    pub fn query(&self, q: &RingQuery<'_>) -> Result<QueryResult> {
        let dims = self.config.dims;
        if q.query.len() != dims {
            return Err(crate::error::RingDbError::DimensionMismatch {
                expected: dims,
                got: q.query.len(),
            });
        }

        let d_min = (q.d - q.lambda).max(0.0);
        let d_max = q.d + q.lambda;

        let t = Instant::now();
        let ids = self.backend.ring_query_f32(dims, q.query, d_min, d_max)?;
        let elapsed = t.elapsed();

        Ok(QueryResult {
            ids,
            backend_used: self.backend.name(),
            elapsed,
        })
    }

    /// Execute a range query and return matching vector IDs.
    ///
    /// Returns all vectors whose Euclidean distance to the query lies in
    /// `[d_min, d_max]`.
    pub fn query_range(&self, q: &RangeQuery<'_>) -> Result<QueryResult> {
        let dims = self.config.dims;
        if q.query.len() != dims {
            return Err(crate::error::RingDbError::DimensionMismatch {
                expected: dims,
                got: q.query.len(),
            });
        }

        let t = Instant::now();
        let ids = self
            .backend
            .ring_query_f32(dims, q.query, q.d_min, q.d_max)?;
        let elapsed = t.elapsed();

        Ok(QueryResult {
            ids,
            backend_used: self.backend.name(),
            elapsed,
        })
    }

    /// Execute a disk query and return matching vector IDs.
    ///
    /// Returns all vectors within Euclidean distance `d_max` of the query
    /// (i.e. the full ball of radius `d_max`, equivalent to `d_min = 0`).
    pub fn query_disk(&self, q: &DiskQuery<'_>) -> Result<QueryResult> {
        let dims = self.config.dims;
        if q.query.len() != dims {
            return Err(crate::error::RingDbError::DimensionMismatch {
                expected: dims,
                got: q.query.len(),
            });
        }

        let t = Instant::now();
        let ids = self.backend.ring_query_f32(dims, q.query, 0.0, q.d_max)?;
        let elapsed = t.elapsed();

        Ok(QueryResult {
            ids,
            backend_used: self.backend.name(),
            elapsed,
        })
    }

    /// Fetch the payload for a single vector ID.
    ///
    /// Reads and deserializes from the cold mmap. Call this after
    /// [`query`](Self::query) to retrieve metadata for the matching vectors.
    pub fn fetch_payload(&self, id: u32) -> Result<T> {
        self.payload_store.fetch(id)
    }

    /// Fetch payloads for a slice of vector IDs, in order.
    pub fn fetch_payloads(&self, ids: &[u32]) -> Result<Vec<T>> {
        self.payload_store.fetch_many(ids)
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
