/// All errors that ringdb can produce.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum RingDbError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("payload serialization error: {0}")]
    Payload(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("corrupt database: {0}")]
    Corrupt(String),
}

pub type Result<T> = std::result::Result<T, RingDbError>;
