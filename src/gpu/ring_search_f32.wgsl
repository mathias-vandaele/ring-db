// ringdb — exact float32 ring search shader
//
// Each thread handles one vector.
// Computes: dist_sq = norm_sq[i] + norm_sq_q - 2 * dot(vectors[i], query)
// Sets a bit in the output bitmask if lower_sq <= dist_sq <= upper_sq.
//
// Output is a packed bitmask: output[vid/32] bit (vid%32) = 1 means hit.

struct Params {
    n_vectors : u32,
    dims      : u32,
    lower_sq  : f32,
    upper_sq  : f32,
    norm_sq_q : f32,
    _pad0     : u32,
    _pad1     : u32,
    _pad2     : u32,
}

@group(0) @binding(0) var<uniform>            params   : Params;
@group(0) @binding(1) var<storage, read>      vectors  : array<f32>;
@group(0) @binding(2) var<storage, read>      norms_sq : array<f32>;
@group(0) @binding(3) var<storage, read>      query    : array<f32>;
@group(0) @binding(4) var<storage, read_write> output  : array<atomic<u32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vid: u32 = gid.x;
    if vid >= params.n_vectors {
        return;
    }

    var dot: f32 = 0.0;
    let base: u32 = vid * params.dims;
    for (var j: u32 = 0u; j < params.dims; j = j + 1u) {
        dot = dot + vectors[base + j] * query[j];
    }

    let dist_sq: f32 = norms_sq[vid] + params.norm_sq_q - 2.0 * dot;

    if dist_sq >= params.lower_sq && dist_sq <= params.upper_sq {
        let word: u32 = vid / 32u;
        let bit:  u32 = vid % 32u;
        atomicOr(&output[word], 1u << bit);
    }
}
