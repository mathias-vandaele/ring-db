/// All errors that ringdb can produce.
#[derive(Debug, thiserror::Error)]
pub enum RingDbError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input: no vectors provided")]
    EmptyDataset,

    #[error("WGPU init error: {0}")]
    WgpuInit(String),

    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(String),
}

pub type Result<T> = std::result::Result<T, RingDbError>;
