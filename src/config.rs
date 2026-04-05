/// Which compute backend to use for ring queries.
///
/// `Auto` selects the best available backend at runtime:
/// CUDA (if feature enabled and device present) → WGPU → CPU.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum BackendPreference {
    /// Automatically select the best available backend.
    #[default]
    Auto,
    /// Force CPU brute-force (always available).
    Cpu,
    /// Force WGPU (Metal on macOS, Vulkan/DX12 elsewhere).
    Wgpu,
    /// Force CUDA (requires `cuda` feature and NVIDIA GPU).
    Cuda,
}

/// Vector quantization mode for stored dataset and queries.
///
/// ## Exact mode (`None`)
/// Vectors are stored as float32. Ring search is exact.
/// All backends produce identical results to the CPU reference.
///
/// ## Approximate mode (`Q8`)
/// Vectors are quantized to int8 at insertion time. Ring search is
/// approximate — results may include false positives or miss hits near
/// the ring boundary. Q8 mode is faster and uses less GPU memory bandwidth
/// than exact mode. The user explicitly accepts approximation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum QuantizationMode {
    /// Exact float32 brute-force ring search.
    #[default]
    None,
    /// Approximate int8 brute-force ring search.
    ///
    /// **Results are inexact by design.** False positives and false negatives
    /// near ring boundaries are expected due to quantization error.
    Q8,
}

/// Configuration for a `RingDb` instance.
///
/// # Example
/// ```
/// use ringdb::{RingDbConfig, BackendPreference, QuantizationMode};
///
/// // Exact GPU ring search on 128-dimensional vectors
/// let config = RingDbConfig {
///     dims: 128,
///     backend_preference: BackendPreference::Auto,
///     quantization: QuantizationMode::None,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RingDbConfig {
    /// Number of dimensions per vector. Must be > 0.
    pub dims: usize,
    /// Backend selection strategy.
    pub backend_preference: BackendPreference,
    /// Whether to use exact float32 or approximate Q8 mode.
    pub quantization: QuantizationMode,
}

impl RingDbConfig {
    /// Create a config with default settings (Auto backend, exact mode).
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            backend_preference: BackendPreference::Auto,
            quantization: QuantizationMode::None,
        }
    }
}
