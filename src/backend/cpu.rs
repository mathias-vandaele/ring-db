use crate::backend::RingComputeBackend;
use crate::error::{Result, RingDbError};
use crate::quant::quantize_vec;

/// CPU brute-force backend.
///
/// This backend serves two roles:
///
/// 1. **Reference implementation** for correctness testing: the f32 path
///    produces exact results that all other backends must match.
///
/// 2. **Fallback** when no GPU is available.
///
/// Both the f32 and Q8 paths are single-threaded and not optimised for
/// throughput. For production workloads, prefer the WGPU or CUDA backends.
pub struct CpuBackend {
    dims: usize,
    n_vectors: usize,

    // --- exact float32 path ---
    vectors_f32: Vec<f32>,  // row-major, n_vectors × dims
    norms_sq_f32: Vec<f32>, // per-vector squared L2 norm

    // --- approximate Q8 path ---
    vectors_q8: Vec<i8>,  // row-major, n_vectors × dims
    norms_sq_q8: Vec<f32>, // squared L2 norms of the original f32 vectors
    scales_q8: Vec<f32>,   // per-vector quantization scale
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            dims: 0,
            n_vectors: 0,
            vectors_f32: Vec::new(),
            norms_sq_f32: Vec::new(),
            vectors_q8: Vec::new(),
            norms_sq_q8: Vec::new(),
            scales_q8: Vec::new(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RingComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn take_f32_vectors(&mut self) -> Option<Vec<f32>> {
        if self.vectors_f32.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.vectors_f32))
        }
    }

    fn upload_f32_dataset(
        &mut self,
        dims: usize,
        vectors: Vec<f32>,
        norms_sq: Vec<f32>,
    ) -> Result<()> {
        self.dims = dims;
        self.n_vectors = norms_sq.len();
        self.vectors_f32 = vectors;
        self.norms_sq_f32 = norms_sq;
        Ok(())
    }

    fn upload_q8_dataset(
        &mut self,
        dims: usize,
        vectors_q8: Vec<i8>,
        norms_sq: Vec<f32>,
        scales: Vec<f32>,
    ) -> Result<()> {
        self.dims = dims;
        self.n_vectors = norms_sq.len();
        self.vectors_q8 = vectors_q8;
        self.norms_sq_q8 = norms_sq;
        self.scales_q8 = scales;
        Ok(())
    }

    fn ring_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        d: f32,
        lambda: f32,
    ) -> Result<Vec<u32>> {
        if self.n_vectors == 0 {
            return Ok(Vec::new());
        }
        if query.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: query.len(),
            });
        }

        let norm_sq_q: f32 = query.iter().map(|x| x * x).sum();
        let lower = (d - lambda).max(0.0);
        let lower_sq = lower * lower;
        let upper_sq = (d + lambda) * (d + lambda);

        let mut ids = Vec::new();
        for i in 0..self.n_vectors {
            let base = i * dims;
            let row = &self.vectors_f32[base..base + dims];
            let dot: f32 = row.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            let dist_sq = self.norms_sq_f32[i] + norm_sq_q - 2.0 * dot;
            if dist_sq >= lower_sq && dist_sq <= upper_sq {
                ids.push(i as u32);
            }
        }
        Ok(ids)
    }

    fn ring_query_q8(
        &self,
        dims: usize,
        query: &[f32],
        d: f32,
        lambda: f32,
    ) -> Result<Vec<u32>> {
        if self.n_vectors == 0 {
            return Ok(Vec::new());
        }
        if query.len() != dims {
            return Err(RingDbError::DimensionMismatch {
                expected: dims,
                got: query.len(),
            });
        }

        let (query_q8, scale_q) = quantize_vec(query);
        let norm_sq_q: f32 = query.iter().map(|x| x * x).sum();
        let lower = (d - lambda).max(0.0);
        let lower_sq = lower * lower;
        let upper_sq = (d + lambda) * (d + lambda);

        let mut ids = Vec::new();
        for i in 0..self.n_vectors {
            let base = i * dims;
            let row = &self.vectors_q8[base..base + dims];

            // Integer dot product then scale back to float
            let dot_i32: i32 = row
                .iter()
                .zip(query_q8.iter())
                .map(|(&a, &b)| a as i32 * b as i32)
                .sum();
            let dot_f32 = dot_i32 as f32 * self.scales_q8[i] * scale_q;
            let dist_sq = self.norms_sq_q8[i] + norm_sq_q - 2.0 * dot_f32;

            if dist_sq >= lower_sq && dist_sq <= upper_sq {
                ids.push(i as u32);
            }
        }
        Ok(ids)
    }
}
