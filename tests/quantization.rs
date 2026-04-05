/// Tests for the Q8 approximate mode.
///
/// Q8 mode is intentionally approximate. These tests verify:
/// - quantization encode/decode roundtrip has bounded error
/// - Q8 queries run without panics
/// - returned IDs are valid (in range, no duplicates)
/// - Q8 CPU results are a reasonable approximation of f32 CPU results
///   (high overlap, not required to be identical)
use rand::{Rng, SeedableRng};
use ringdb::{
    quant::{pack_i8_to_i32, quantize_vec},
    BackendPreference, QuantizationMode, RingDb, RingDbConfig, RingQuery,
};

// ---- quantization unit tests ----

#[test]
fn quantize_zero_vector() {
    let v = vec![0.0f32; 8];
    let (q, scale) = quantize_vec(&v);
    assert!(scale > 0.0, "scale must be positive even for zero vector");
    assert!(q.iter().all(|&x| x == 0), "zero vector should quantize to zeros");
}

#[test]
fn quantize_roundtrip_error_bounded() {
    let v: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
    let (q, scale) = quantize_vec(&v);

    let max_abs = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let expected_max_error = scale; // at most 0.5 * scale (rounding), but allow scale for safety

    for (orig, &qi) in v.iter().zip(&q) {
        let decoded = qi as f32 * scale;
        let err = (decoded - orig).abs();
        assert!(
            err <= expected_max_error + 1e-6,
            "quantization error {err} > allowed {expected_max_error} (scale={scale}, max_abs={max_abs})"
        );
    }
}

#[test]
fn quantize_extremes_fit_in_i8() {
    let v = vec![100.0f32, -100.0, 50.0, -50.0, 0.0];
    let (q, _scale) = quantize_vec(&v);
    // All values should fit in [-127, 127]
    for &qi in &q {
        assert!(qi >= -127, "quantized value {qi} below -127");
    }
}

#[test]
fn pack_i8_i32_roundtrip() {
    let vals: Vec<i8> = vec![0, 1, -1, 127, -127, 64, -64, 32, -32, 0, 0, 0];
    let packed = pack_i8_to_i32(&vals);
    // Manually unpack and verify.
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

// ---- Q8 database tests ----

fn q8_db(dims: usize) -> RingDb {
    RingDb::new(RingDbConfig {
        dims,
        backend_preference: BackendPreference::Cpu,
        quantization: QuantizationMode::Q8,
    })
    .unwrap()
}

fn f32_db(dims: usize) -> RingDb {
    RingDb::new(RingDbConfig {
        dims,
        backend_preference: BackendPreference::Cpu,
        quantization: QuantizationMode::None,
    })
    .unwrap()
}

fn assert_valid_ids(ids: &[u32], n: usize) {
    for &id in ids {
        assert!((id as usize) < n, "ID {id} out of range n={n}");
    }
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "duplicate IDs");
}

#[test]
fn q8_query_no_panic_small() {
    let mut db = q8_db(8);
    let vecs: Vec<f32> = (0..10 * 8).map(|i| (i as f32) / 80.0 - 0.5).collect();
    db.add_vectors(&vecs).unwrap();

    let q = vec![0.0f32; 8];
    let r = db
        .query(&RingQuery {
            query: &q,
            d: 0.5,
            lambda: 0.3,
        })
        .unwrap();
    assert_valid_ids(&r.ids, 10);
}

#[test]
fn q8_query_returns_valid_ids() {
    let n = 1_000;
    let dims = 64;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(17);
    let vecs: Vec<f32> = (0..n * dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let mut db = q8_db(dims);
    db.add_vectors(&vecs).unwrap();

    let query: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let r = db
        .query(&RingQuery {
            query: &query,
            d: 4.0,
            lambda: 0.5,
        })
        .unwrap();

    assert_valid_ids(&r.ids, n);
}

#[test]
fn q8_results_overlap_significantly_with_f32() {
    // Q8 results are approximate but should overlap substantially with f32
    // results, especially with a generous lambda.
    let n = 2_000;
    let dims = 32;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(31);
    let vecs: Vec<f32> = (0..n * dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let mut f32db = f32_db(dims);
    f32db.add_vectors(&vecs).unwrap();

    let mut q8db = q8_db(dims);
    q8db.add_vectors(&vecs).unwrap();

    let query: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let d = 3.5f32;
    let lambda = 1.0f32; // wide ring → should have good overlap

    let f32_ids: std::collections::HashSet<u32> = f32db
        .query(&RingQuery {
            query: &query,
            d,
            lambda,
        })
        .unwrap()
        .ids
        .into_iter()
        .collect();

    let q8_ids: std::collections::HashSet<u32> = q8db
        .query(&RingQuery {
            query: &query,
            d,
            lambda,
        })
        .unwrap()
        .ids
        .into_iter()
        .collect();

    if f32_ids.is_empty() {
        // No hits in f32 mode → Q8 can't be expected to match anything
        return;
    }

    // Compute Jaccard similarity
    let intersection = f32_ids.intersection(&q8_ids).count();
    let union = f32_ids.union(&q8_ids).count();
    let jaccard = intersection as f32 / union as f32;

    // For a wide ring (lambda=1.0), we expect at least 40% overlap.
    // This threshold is intentionally lenient — Q8 is approximate.
    assert!(
        jaccard >= 0.4,
        "Q8 Jaccard similarity {jaccard:.2} too low (intersection={intersection}, union={union})"
    );
}

#[test]
fn q8_backend_name() {
    let db = q8_db(4);
    assert_eq!(db.backend_name(), "cpu");
}
