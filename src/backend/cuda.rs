/// CUDA backend (feature-gated: `--features cuda`).
///
/// **Stage 1 stub**: This implementation falls back to CPU execution until
/// proper PTX/cudarc kernels are implemented.
///
/// To add GPU execution:
/// 1. Write a `.ptx` or inline PTX kernel for float32 ring search.
/// 2. Write an equivalent kernel for Q8 ring search.
/// 3. Use `cudarc` to upload data to device memory and launch kernels.
/// 4. Replace the CPU fallback calls below.
use crate::backend::{cpu::CpuBackend, RingComputeBackend};
use crate::error::Result;

/// CUDA compute backend (stub — delegates to CPU until PTX kernels are added).
pub struct CudaBackend {
    cpu: CpuBackend,
}

impl CudaBackend {
    /// Try to initialise the CUDA backend.
    ///
    /// Currently always succeeds (stub) and creates a CPU fallback.
    /// In a full implementation this would call `cudarc::driver::CudaDevice::new(0)`.
    pub fn try_new() -> Result<Self> {
        Ok(Self {
            cpu: CpuBackend::new(),
        })
    }
}

impl RingComputeBackend for CudaBackend {
    fn name(&self) -> &'static str {
        // Distinguish from pure CPU so callers know CUDA was selected.
        "cuda-stub"
    }

    fn take_f32_vectors(&mut self) -> Option<Vec<f32>> {
        self.cpu.take_f32_vectors()
    }

    fn upload_f32_dataset(
        &mut self,
        dims: usize,
        vectors: Vec<f32>,
        norms_sq: Vec<f32>,
    ) -> Result<()> {
        self.cpu.upload_f32_dataset(dims, vectors, norms_sq)
    }

    fn upload_q8_dataset(
        &mut self,
        dims: usize,
        vectors_q8: Vec<i8>,
        norms_sq: Vec<f32>,
        scales: Vec<f32>,
    ) -> Result<()> {
        self.cpu.upload_q8_dataset(dims, vectors_q8, norms_sq, scales)
    }

    fn ring_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        d: f32,
        lambda: f32,
    ) -> Result<Vec<u32>> {
        self.cpu.ring_query_f32(dims, query, d, lambda)
    }

    fn ring_query_q8(
        &self,
        dims: usize,
        query: &[f32],
        d: f32,
        lambda: f32,
    ) -> Result<Vec<u32>> {
        self.cpu.ring_query_q8(dims, query, d, lambda)
    }
}
