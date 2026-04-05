use std::time::Instant;

use crate::backend::{CpuBackend, RingComputeBackend, WgpuBackend};
use crate::config::{BackendPreference, QuantizationMode, RingDbConfig};
use crate::error::{Result, RingDbError};
use crate::query::{QueryResult, RingQuery};
use crate::quant::{pad_dataset_i8, padded_dims, quantize_dataset};

/// High-performance vector database for ring queries.
///
/// A ring query returns all vector IDs whose Euclidean distance to a
/// query vector lies within the interval `[d - lambda, d + lambda]`.
///
/// # Execution modes
///
/// ## Exact mode (default)
///
/// Vectors are stored as float32. All backends produce results that are
/// mathematically identical to the CPU reference implementation.
///
/// ## Approximate Q8 mode
///
/// Vectors are quantized to int8 at insertion time. Queries run on the
/// quantized data and results are **approximate by design** — false
/// positives and false negatives near the ring boundary are expected.
/// Q8 mode uses less memory bandwidth and runs faster than exact mode.
///
/// # Example
///
/// ```
/// use ringdb::{RingDb, RingDbConfig, RingQuery};
///
/// let config = RingDbConfig::new(4);
/// let mut db = RingDb::new(config).unwrap();
///
/// db.add_vectors(&[
///     1.0, 0.0, 0.0, 0.0,
///     0.0, 1.0, 0.0, 0.0,
///     3.0, 4.0, 0.0, 0.0,
/// ]).unwrap();
///
/// let q = [1.0f32, 0.0, 0.0, 0.0];
/// let result = db.query(&RingQuery { query: &q, d: 5.0, lambda: 1.0 }).unwrap();
/// println!("hits: {:?}, backend: {}", result.ids, result.backend_used);
/// ```
pub struct RingDb {
    config: RingDbConfig,
    backend: Box<dyn RingComputeBackend>,
    /// Number of vectors currently stored (tracked separately because the
    /// engine-side f32 buffer is freed after upload to save memory).
    n_vectors: usize,
}

impl RingDb {
    /// Create a new empty `RingDb` with the given configuration.
    ///
    /// Backend selection for `BackendPreference::Auto`:
    /// CUDA (if feature enabled) → WGPU → CPU (always available).
    pub fn new(config: RingDbConfig) -> Result<Self> {
        let backend = select_backend(&config)?;
        Ok(Self {
            config,
            backend,
            n_vectors: 0,
        })
    }

    /// Create a `RingDb` pre-populated with `vectors`.
    ///
    /// Equivalent to `RingDb::new(config)` followed by `add_vectors(vectors)`.
    pub fn with_vectors(config: RingDbConfig, vectors: &[f32]) -> Result<Self> {
        let mut db = Self::new(config)?;
        db.add_vectors(vectors)?;
        Ok(db)
    }

    /// Insert vectors into the database.
    ///
    /// Vectors are appended to any previously inserted data. IDs are
    /// assigned in insertion order starting from 0.
    ///
    /// The slice length must be a multiple of `dims`.
    pub fn add_vectors(&mut self, vectors: &[f32]) -> Result<()> {
        let dims = self.config.dims;
        if vectors.is_empty() {
            return Err(RingDbError::EmptyDataset);
        }
        if vectors.len() % dims != 0 {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: vectors.len() % dims,
            });
        }

        let n_new = vectors.len() / dims;

        // Build the full accumulated dataset.  If this is the first call
        // we can avoid a copy by moving directly into the backend; on
        // subsequent calls we reclaim the data the backend held, append
        // the new vectors, and hand ownership back.
        let all = if self.n_vectors == 0 {
            // First call — just adopt the caller's data.
            vectors.to_vec()
        } else {
            // Incremental add — get old data back from the backend.
            let mut prev = self.backend.take_f32_vectors().unwrap_or_default();
            prev.extend_from_slice(vectors);
            prev
        };

        self.n_vectors += n_new;
        let total = self.n_vectors;

        // Compute per-vector squared norms over the full dataset.
        let norms_sq: Vec<f32> = (0..total)
            .map(|i| {
                let base = i * dims;
                all[base..base + dims]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
            })
            .collect();

        // Upload to the backend — ownership is transferred so the engine
        // holds no long-lived copy of the vectors.
        match self.config.quantization {
            QuantizationMode::None => {
                self.backend
                    .upload_f32_dataset(dims, all, norms_sq)?;
            }
            QuantizationMode::Q8 => {
                let (q8_flat, scales) = quantize_dataset(&all, dims);
                // `all` is no longer needed after quantization — drop it
                // before allocating the padded buffer to reduce peak RAM.
                drop(all);
                let pdims = padded_dims(dims);
                let q8_padded = if pdims != dims {
                    pad_dataset_i8(&q8_flat, dims)
                } else {
                    q8_flat
                };
                self.backend
                    .upload_q8_dataset(dims, q8_padded, norms_sq, scales)?;
            }
        }

        Ok(())
    }

    /// Execute a ring query and return matching vector IDs.
    pub fn query(&self, q: &RingQuery<'_>) -> Result<QueryResult> {
        let dims = self.config.dims;
        if q.query.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: q.query.len(),
            });
        }

        let t = Instant::now();
        let ids = match self.config.quantization {
            QuantizationMode::None => {
                self.backend.ring_query_f32(dims, q.query, q.d, q.lambda)?
            }
            QuantizationMode::Q8 => {
                self.backend.ring_query_q8(dims, q.query, q.d, q.lambda)?
            }
        };
        let elapsed = t.elapsed();

        Ok(QueryResult {
            ids,
            backend_used: self.backend.name().to_string(),
            elapsed,
        })
    }

    /// Number of vectors currently stored in the database.
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

/// Select and initialise the best available backend for the given config.
fn select_backend(config: &RingDbConfig) -> Result<Box<dyn RingComputeBackend>> {
    match config.backend_preference {
        BackendPreference::Cpu => Ok(Box::new(CpuBackend::new())),

        BackendPreference::Wgpu => {
            WgpuBackend::try_new().map(|b| Box::new(b) as Box<dyn RingComputeBackend>)
        }

        #[cfg(feature = "cuda")]
        BackendPreference::Cuda => {
            use crate::backend::CudaBackend;
            CudaBackend::try_new().map(|b| Box::new(b) as Box<dyn RingComputeBackend>)
        }
        #[cfg(not(feature = "cuda"))]
        BackendPreference::Cuda => Err(RingDbError::BackendUnavailable(
            "CUDA support not compiled in (enable the `cuda` feature)".to_string(),
        )),

        BackendPreference::Auto => {
            #[cfg(feature = "cuda")]
            {
                use crate::backend::CudaBackend;
                if let Ok(b) = CudaBackend::try_new() {
                    return Ok(Box::new(b));
                }
            }

            if let Ok(b) = WgpuBackend::try_new() {
                return Ok(Box::new(b));
            }

            Ok(Box::new(CpuBackend::new()))
        }
    }
}
