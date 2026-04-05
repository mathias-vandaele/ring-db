/// Which compute backend to use for ring queries.
///
/// Only CPU is supported for now. This enum exists so the config is ready
/// to be extended with additional backends later.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum BackendPreference {
    /// CPU brute-force (always available).
    #[default]
    Cpu,
}

/// Configuration for a `RingDb` instance.
#[derive(Debug, Clone)]
pub struct RingDbConfig {
    /// Number of dimensions per vector. Must be > 0.
    pub dims: usize,
    /// Backend selection strategy.
    pub backend_preference: BackendPreference,
}

impl RingDbConfig {
    /// Create a config with default settings (CPU backend).
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            backend_preference: BackendPreference::Cpu,
        }
    }
}
