// ringdb — approximate Q8 (int8) ring search shader
//
// WGSL has no i8 array type, so 4 consecutive int8 values are packed into
// one i32 word (little-endian byte order). The `dims` field in Params is the
// *padded* dimension count (always a multiple of 4), so dims/4 packed words
// cover the full vector.
//
// Each thread:
//   1. Unpacks 4 bytes per word from the quantized vector and query.
//   2. Accumulates a 32-bit integer dot product.
//   3. Scales back to float using per-vector and per-query scales.
//   4. Sets a bitmask bit if the approximate distance is in [lower_sq, upper_sq].
//
// Results are APPROXIMATE. False positives and false negatives near the ring
// boundary are expected due to quantization error.

struct Params {
    n_vectors : u32,
    dims_div4 : u32,  // padded_dims / 4 — number of packed i32 words per vector
    lower_sq  : f32,
    upper_sq  : f32,
    norm_sq_q : f32,  // squared L2 norm of the original (float32) query
    scale_q   : f32,  // quantization scale for the query
    _pad0     : u32,
    _pad1     : u32,
}

@group(0) @binding(0) var<uniform>            params    : Params;
@group(0) @binding(1) var<storage, read>      vectors   : array<i32>;  // packed i8
@group(0) @binding(2) var<storage, read>      norms_sq  : array<f32>;  // per-vector f32 norm
@group(0) @binding(3) var<storage, read>      scales    : array<f32>;  // per-vector quant scale
@group(0) @binding(4) var<storage, read>      query_q8  : array<i32>;  // packed quantized query
@group(0) @binding(5) var<storage, read_write> output   : array<atomic<u32>>;

// Sign-extend individual bytes from a packed i32.
fn unpack0(p: i32) -> i32 { return (p << 24) >> 24; }
fn unpack1(p: i32) -> i32 { return (p << 16) >> 24; }
fn unpack2(p: i32) -> i32 { return (p << 8)  >> 24; }
fn unpack3(p: i32) -> i32 { return  p         >> 24; }

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vid: u32 = gid.x;
    if vid >= params.n_vectors {
        return;
    }

    var dot_i32: i32 = 0;
    let base: u32 = vid * params.dims_div4;
    for (var k: u32 = 0u; k < params.dims_div4; k = k + 1u) {
        let xp: i32 = vectors[base + k];
        let qp: i32 = query_q8[k];
        dot_i32 = dot_i32 + unpack0(xp) * unpack0(qp);
        dot_i32 = dot_i32 + unpack1(xp) * unpack1(qp);
        dot_i32 = dot_i32 + unpack2(xp) * unpack2(qp);
        dot_i32 = dot_i32 + unpack3(xp) * unpack3(qp);
    }

    let dot_f32: f32 = f32(dot_i32) * scales[vid] * params.scale_q;
    let dist_sq: f32 = norms_sq[vid] + params.norm_sq_q - 2.0 * dot_f32;

    if dist_sq >= params.lower_sq && dist_sq <= params.upper_sq {
        let word: u32 = vid / 32u;
        let bit:  u32 = vid % 32u;
        atomicOr(&output[word], 1u << bit);
    }
}
