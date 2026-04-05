/// Per-vector symmetric int8 quantization.
///
/// Scale is chosen as `max(|v[i]|) / 127.0`. This maps the largest
/// magnitude component to ±127, using the full int8 range symmetrically.
///
/// **Approximation warning**: quantization introduces error proportional to
/// the scale. For Q8 mode results will differ from float32 exact mode.
///
/// Returns `(quantized_values, scale)`.
pub fn quantize_vec(v: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = v
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max)
        .max(1e-9); // avoid divide-by-zero for zero vectors

    let scale = max_abs / 127.0;
    let inv_scale = 1.0 / scale;

    let q: Vec<i8> = v
        .iter()
        .map(|&x| {
            let r = (x * inv_scale).round();
            r.clamp(-127.0, 127.0) as i8
        })
        .collect();

    (q, scale)
}

/// Quantize an entire dataset (flat row-major buffer) per-vector.
///
/// Returns `(quantized_flat, scales)` where `scales[i]` is the scale for
/// vector `i`. The quantized buffer has the same length as the input.
pub fn quantize_dataset(vectors: &[f32], dims: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(dims > 0);
    let n = vectors.len() / dims;
    let mut q_all = Vec::with_capacity(vectors.len());
    let mut scales = Vec::with_capacity(n);

    for i in 0..n {
        let start = i * dims;
        let (qv, scale) = quantize_vec(&vectors[start..start + dims]);
        q_all.extend_from_slice(&qv);
        scales.push(scale);
    }

    (q_all, scales)
}

/// Pack a flat i8 slice into a packed i32 slice (4 bytes per i32, little-endian).
///
/// The input length must already be a multiple of 4 (pad with zeros if needed).
/// This is required for the Q8 WGSL shaders which use `array<i32>` since WGSL
/// has no native i8 array type.
pub fn pack_i8_to_i32(values: &[i8]) -> Vec<i32> {
    assert!(
        values.len() % 4 == 0,
        "values length must be a multiple of 4 for packing"
    );
    values
        .chunks_exact(4)
        .map(|c| {
            (c[0] as i32 & 0xFF)
                | ((c[1] as i32 & 0xFF) << 8)
                | ((c[2] as i32 & 0xFF) << 16)
                | ((c[3] as i32 & 0xFF) << 24)
        })
        .collect()
}

/// Pad `dims` up to the next multiple of 4 (required for i8→i32 packing).
pub fn padded_dims(dims: usize) -> usize {
    (dims + 3) / 4 * 4
}

/// Pad a single vector to `padded_dims(dims)` by appending zeros.
pub fn pad_vec(v: &[f32], target_len: usize) -> Vec<f32> {
    let mut out = v.to_vec();
    out.resize(target_len, 0.0);
    out
}

/// Pad a flat dataset (row-major) so each vector has `padded_dims(dims)` entries.
pub fn pad_dataset_f32(vectors: &[f32], dims: usize) -> Vec<f32> {
    let n = vectors.len() / dims;
    let padded = padded_dims(dims);
    if padded == dims {
        return vectors.to_vec();
    }
    let mut out = Vec::with_capacity(n * padded);
    for i in 0..n {
        out.extend_from_slice(&vectors[i * dims..(i + 1) * dims]);
        out.extend(std::iter::repeat(0.0f32).take(padded - dims));
    }
    out
}

/// Pad a flat i8 dataset (row-major) so each vector has `padded_dims(dims)` entries.
pub fn pad_dataset_i8(vectors: &[i8], dims: usize) -> Vec<i8> {
    let n = vectors.len() / dims;
    let padded = padded_dims(dims);
    if padded == dims {
        return vectors.to_vec();
    }
    let mut out = Vec::with_capacity(n * padded);
    for i in 0..n {
        out.extend_from_slice(&vectors[i * dims..(i + 1) * dims]);
        out.extend(std::iter::repeat(0i8).take(padded - dims));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_roundtrip_small_error() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let (q, scale) = quantize_vec(&v);
        for (orig, &qi) in v.iter().zip(&q) {
            let decoded = qi as f32 * scale;
            let err = (decoded - orig).abs();
            assert!(
                err < scale * 1.5,
                "roundtrip error {err} too large for scale {scale}"
            );
        }
    }

    #[test]
    fn pack_unpack_identity() {
        let vals: Vec<i8> = vec![1, -1, 127, -127, 0, 64, -64, 32];
        let packed = pack_i8_to_i32(&vals);
        // unpack manually
        let unpacked: Vec<i8> = packed
            .iter()
            .flat_map(|&p| {
                [
                    ((p << 24) >> 24) as i8,
                    ((p << 16) >> 24) as i8,
                    ((p << 8) >> 24) as i8,
                    (p >> 24) as i8,
                ]
            })
            .collect();
        assert_eq!(unpacked, vals);
    }
}
