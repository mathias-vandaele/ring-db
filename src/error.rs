/// All errors that ringdb can produce.
#[derive(Debug, thiserror::Error)]
pub enum RingDbError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, RingDbError>;
