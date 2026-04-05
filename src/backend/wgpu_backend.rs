use crate::backend::RingComputeBackend;
use crate::error::{Result, RingDbError};
use crate::quant::{pack_i8_to_i32, padded_dims, quantize_vec};
use bytemuck::cast_slice;

/// WGPU compute backend.
///
/// Uses WGSL compute shaders for brute-force ring search. Works on any
/// platform supported by WGPU: Metal (macOS), Vulkan (Linux/Windows),
/// DX12 (Windows).
///
/// The dataset buffers are uploaded once and reside permanently on the GPU.
/// Each query creates small per-query buffers (params uniform + query vector
/// + output bitmask) and reads back the bitmask after dispatch.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,

    dims: usize,
    n_vectors: usize,

    // --- f32 path ---
    buf_vectors_f32: Option<wgpu::Buffer>,
    buf_norms_sq_f32: Option<wgpu::Buffer>,

    // --- Q8 path ---
    buf_vectors_q8: Option<wgpu::Buffer>,  // packed i32 (4× i8)
    buf_norms_sq_q8: Option<wgpu::Buffer>,
    buf_scales_q8: Option<wgpu::Buffer>,

    // Pipelines compiled once at init.
    pipeline_f32: wgpu::ComputePipeline,
    pipeline_q8: wgpu::ComputePipeline,

    bgl_f32: wgpu::BindGroupLayout,
    bgl_q8: wgpu::BindGroupLayout,
}

// -------------------------------------------------------------------------------------------------
// Params uniform structs (must be bytemuck-safe and match WGSL struct layouts exactly)
// -------------------------------------------------------------------------------------------------

/// Params uniform for the f32 shader (32 bytes, 8 × u32).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsF32 {
    n_vectors: u32,
    dims: u32,
    lower_sq: f32,
    upper_sq: f32,
    norm_sq_q: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Params uniform for the Q8 shader (32 bytes, 8 × u32).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsQ8 {
    n_vectors: u32,
    dims_div4: u32,
    lower_sq: f32,
    upper_sq: f32,
    norm_sq_q: f32,
    scale_q: f32,
    _pad0: u32,
    _pad1: u32,
}

// -------------------------------------------------------------------------------------------------
// Construction
// -------------------------------------------------------------------------------------------------

impl WgpuBackend {
    /// Try to initialise a WGPU backend.
    ///
    /// Returns `Err(RingDbError::WgpuInit(...))` if no suitable GPU adapter
    /// is found (e.g. running on a headless server without GPU).
    pub fn try_new() -> Result<Self> {
        pollster::block_on(Self::init())
    }

    async fn init() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| RingDbError::WgpuInit("no GPU adapter found".to_string()))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| RingDbError::WgpuInit(e.to_string()))?;

        // Compile both shaders up front.
        let shader_f32 =
            device.create_shader_module(wgpu::include_wgsl!("../gpu/ring_search_f32.wgsl"));
        let shader_q8 =
            device.create_shader_module(wgpu::include_wgsl!("../gpu/ring_search_q8.wgsl"));

        // ---- f32 bind group layout (5 bindings) ----
        let bgl_f32 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_f32"),
            entries: &[
                bgl_uniform(0),     // params
                bgl_storage_ro(1),  // vectors f32
                bgl_storage_ro(2),  // norms_sq
                bgl_storage_ro(3),  // query
                bgl_storage_rw(4),  // output bitmask
            ],
        });

        // ---- Q8 bind group layout (6 bindings) ----
        let bgl_q8 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_q8"),
            entries: &[
                bgl_uniform(0),     // params
                bgl_storage_ro(1),  // vectors_q8 (packed i32)
                bgl_storage_ro(2),  // norms_sq
                bgl_storage_ro(3),  // scales
                bgl_storage_ro(4),  // query_q8
                bgl_storage_rw(5),  // output bitmask
            ],
        });

        let pipeline_f32 = make_pipeline(&device, "ring_f32", &bgl_f32, &shader_f32, "main");
        let pipeline_q8 = make_pipeline(&device, "ring_q8", &bgl_q8, &shader_q8, "main");

        Ok(Self {
            device,
            queue,
            dims: 0,
            n_vectors: 0,
            buf_vectors_f32: None,
            buf_norms_sq_f32: None,
            buf_vectors_q8: None,
            buf_norms_sq_q8: None,
            buf_scales_q8: None,
            pipeline_f32,
            pipeline_q8,
            bgl_f32,
            bgl_q8,
        })
    }
}

// -------------------------------------------------------------------------------------------------
// Helpers for bind group layout entries
// -------------------------------------------------------------------------------------------------

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    label: &str,
    bgl: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

// -------------------------------------------------------------------------------------------------
// Buffer creation helpers
// -------------------------------------------------------------------------------------------------

fn create_storage_buf(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_uniform_buf(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Allocate a zeroed output bitmask buffer (STORAGE + COPY_SRC).
fn create_bitmask_buf(device: &wgpu::Device, n_vectors: usize) -> wgpu::Buffer {
    let words = (n_vectors + 31) / 32;
    let size = (words * 4).max(4) as u64; // at least 4 bytes
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bitmask"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Allocate a staging (readback) buffer (MAP_READ + COPY_DST).
fn create_staging_buf(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

// -------------------------------------------------------------------------------------------------
// Bitmask → ID compaction (CPU side)
// -------------------------------------------------------------------------------------------------

fn bitmask_to_ids(words: &[u32], n_vectors: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    for (w, &word) in words.iter().enumerate() {
        if word == 0 {
            continue;
        }
        for b in 0..32u32 {
            let id = w as u32 * 32 + b;
            if id >= n_vectors as u32 {
                break;
            }
            if (word >> b) & 1 != 0 {
                ids.push(id);
            }
        }
    }
    ids
}

// -------------------------------------------------------------------------------------------------
// RingComputeBackend impl
// -------------------------------------------------------------------------------------------------

impl RingComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn upload_f32_dataset(
        &mut self,
        dims: usize,
        vectors: Vec<f32>,
        norms_sq: Vec<f32>,
    ) -> Result<()> {
        self.dims = dims;
        self.n_vectors = norms_sq.len();

        self.buf_vectors_f32 = Some(create_storage_buf(
            &self.device,
            "vectors_f32",
            cast_slice(&vectors),
        ));
        self.buf_norms_sq_f32 = Some(create_storage_buf(
            &self.device,
            "norms_sq_f32",
            cast_slice(&norms_sq),
        ));
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

        let pdims = padded_dims(dims);
        let n = norms_sq.len();

        // Pack i8 → i32 for WGSL (which has no i8 array type).
        // vectors_q8 is already padded to pdims per vector by the engine.
        let packed_len = n * pdims; // total i8 values (padded)
        let padded_q8: Vec<i8> = if vectors_q8.len() == packed_len {
            vectors_q8
        } else {
            // Shouldn't happen if engine padded correctly, but be safe.
            let mut v = vectors_q8;
            v.resize(packed_len, 0);
            v
        };
        let packed_i32 = pack_i8_to_i32(&padded_q8);

        self.buf_vectors_q8 = Some(create_storage_buf(
            &self.device,
            "vectors_q8",
            cast_slice(&packed_i32),
        ));
        self.buf_norms_sq_q8 = Some(create_storage_buf(
            &self.device,
            "norms_sq_q8",
            cast_slice(&norms_sq),
        ));
        self.buf_scales_q8 = Some(create_storage_buf(
            &self.device,
            "scales_q8",
            cast_slice(&scales),
        ));
        Ok(())
    }

    fn ring_query_f32(
        &self,
        dims: usize,
        query: &[f32],
        d: f32,
        lambda: f32,
    ) -> Result<Vec<u32>> {
        let n = self.n_vectors;
        if n == 0 {
            return Ok(Vec::new());
        }

        let buf_vectors = self
            .buf_vectors_f32
            .as_ref()
            .ok_or_else(|| RingDbError::WgpuInit("f32 dataset not uploaded".to_string()))?;
        let buf_norms = self
            .buf_norms_sq_f32
            .as_ref()
            .ok_or_else(|| RingDbError::WgpuInit("f32 norms not uploaded".to_string()))?;

        let norm_sq_q: f32 = query.iter().map(|x| x * x).sum();
        let lower = (d - lambda).max(0.0);
        let params = ParamsF32 {
            n_vectors: n as u32,
            dims: dims as u32,
            lower_sq: lower * lower,
            upper_sq: (d + lambda) * (d + lambda),
            norm_sq_q,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let buf_params =
            create_uniform_buf(&self.device, "params_f32", cast_slice(&[params]));
        let buf_query =
            create_storage_buf(&self.device, "query_f32", cast_slice(query));
        let buf_bitmask = create_bitmask_buf(&self.device, n);

        let bitmask_size = buf_bitmask.size();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_f32"),
            layout: &self.bgl_f32,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_norms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_query.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_bitmask.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ring_f32"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_f32);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy bitmask to staging before submission.
        let staging = create_staging_buf(&self.device, bitmask_size);
        encoder.copy_buffer_to_buffer(&buf_bitmask, 0, &staging, 0, bitmask_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read back.
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let words: &[u32] = cast_slice(&data);
        let ids = bitmask_to_ids(words, n);
        drop(data);
        staging.unmap();

        Ok(ids)
    }

    fn ring_query_q8(
        &self,
        dims: usize,
        query: &[f32],
        d: f32,
        lambda: f32,
    ) -> Result<Vec<u32>> {
        let n = self.n_vectors;
        if n == 0 {
            return Ok(Vec::new());
        }

        let buf_vectors = self
            .buf_vectors_q8
            .as_ref()
            .ok_or_else(|| RingDbError::WgpuInit("Q8 dataset not uploaded".to_string()))?;
        let buf_norms = self
            .buf_norms_sq_q8
            .as_ref()
            .ok_or_else(|| RingDbError::WgpuInit("Q8 norms not uploaded".to_string()))?;
        let buf_scales = self
            .buf_scales_q8
            .as_ref()
            .ok_or_else(|| RingDbError::WgpuInit("Q8 scales not uploaded".to_string()))?;

        let (query_q8, scale_q) = quantize_vec(query);
        let norm_sq_q: f32 = query.iter().map(|x| x * x).sum();

        let pdims = padded_dims(dims);
        // Pad the quantized query if necessary.
        let mut query_q8_padded = query_q8;
        query_q8_padded.resize(pdims, 0i8);
        let query_packed = pack_i8_to_i32(&query_q8_padded);

        let lower = (d - lambda).max(0.0);
        let params = ParamsQ8 {
            n_vectors: n as u32,
            dims_div4: (pdims / 4) as u32,
            lower_sq: lower * lower,
            upper_sq: (d + lambda) * (d + lambda),
            norm_sq_q,
            scale_q,
            _pad0: 0,
            _pad1: 0,
        };

        let buf_params =
            create_uniform_buf(&self.device, "params_q8", cast_slice(&[params]));
        let buf_query =
            create_storage_buf(&self.device, "query_q8", cast_slice(&query_packed));
        let buf_bitmask = create_bitmask_buf(&self.device, n);

        let bitmask_size = buf_bitmask.size();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_q8"),
            layout: &self.bgl_q8,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_norms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_scales.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_query.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_bitmask.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ring_q8"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_q8);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging = create_staging_buf(&self.device, bitmask_size);
        encoder.copy_buffer_to_buffer(&buf_bitmask, 0, &staging, 0, bitmask_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let words: &[u32] = cast_slice(&data);
        let ids = bitmask_to_ids(words, n);
        drop(data);
        staging.unmap();

        Ok(ids)
    }
}
