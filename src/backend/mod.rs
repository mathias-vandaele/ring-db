use crate::error::Result;

pub mod cpu;
pub mod wgpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda;

pub use cpu::CpuBackend;
pub use wgpu_backend::WgpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

/// Trait implemented by all compute backends (CPU, WGPU, CUDA).
///
/// The separation between upload and query methods reflects the core
/// performance design: the dataset is uploaded once to device memory,
/// then many queries are executed against the resident data without
/// re-uploading.
pub trait RingComputeBackend: Send + Sync {
    /// Human-readable backend name (e.g. `"cpu"`, `"wgpu"`, `"cuda"`).
    fn name(&self) -> &'static str;

    /// Return the previously uploaded f32 vectors, removing them from the
    /// backend.  Used by the engine to support incremental `add_vectors`
    /// without keeping a permanent engine-side copy.
    ///
    /// Returns `None` if no f32 dataset has been uploaded or if the backend
    /// does not retain CPU-side data (e.g. GPU-only backends).
    fn take_f32_vectors(&mut self) -> Option<Vec<f32>> {
        None
    }

    /// Upload a float32 dataset to the backend.
    ///
    /// - `dims`: number of dimensions per vector.
    /// - `vectors`: flat row-major buffer, length `n * dims`.  Ownership is
    ///   transferred so CPU backends can store the data without copying.
    /// - `norms_sq`: precomputed squared L2 norm per vector, length `n`.
    fn upload_f32_dataset(
        &mut self,
        dims: usize,
        vectors: Vec<f32>,
        norms_sq: Vec<f32>,
    ) -> Result<()>;

    /// Upload a quantized (Q8) dataset to the backend.
    ///
    /// - `dims`: logical number of dimensions per vector.
    /// - `vectors_q8`: flat row-major int8 buffer (padded to multiple of 4
    ///   for GPU backends), length `n * dims_padded`.  Ownership transferred.
    /// - `norms_sq`: squared L2 norms of the *original* float32 vectors.
    /// - `scales`: per-vector quantization scale (float32 → int8).
    fn upload_q8_dataset(
        &mut self,
        dims: usize,
        vectors_q8: Vec<i8>,
        norms_sq: Vec<f32>,
        scales: Vec<f32>,
    ) -> Result<()>;

    /// Execute an exact float32 ring search.
    ///
    /// Returns IDs of all vectors with squared distance to `query` in
    /// `[lower_sq, upper_sq]` where `lower_sq = max(0, d-lambda)²`
    /// and `upper_sq = (d+lambda)²`.
    fn ring_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        d: f32,
        lambda: f32,
    ) -> Result<Vec<u32>>;

    /// Execute an approximate Q8 ring search.
    ///
    /// The `query` is provided in float32 and quantized internally.
    /// Results are approximate: false positives and false negatives near
    /// the ring boundary are expected due to quantization error.
    fn ring_query_q8(
        &self,
        dims: usize,
        query: &[f32],
        d: f32,
        lambda: f32,
    ) -> Result<Vec<u32>>;
}
