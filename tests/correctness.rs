/// Correctness tests for the CPU exact (float32) backend.
///
/// These tests use hand-crafted vectors at known distances from a query so
/// the expected result sets are computed analytically, not by another code path.
use ringdb::{BackendPreference, QuantizationMode, RingDb, RingDbConfig, RingQuery};

fn cpu_exact(dims: usize) -> RingDb {
    RingDb::new(RingDbConfig {
        dims,
        backend_preference: BackendPreference::Cpu,
        quantization: QuantizationMode::None,
    })
    .unwrap()
}

// ---- 2-D helpers ----

/// Distance between two 2-D points.
fn dist2(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

#[test]
fn test_basic_hits() {
    // Query at origin, ring [4.5, 5.5].
    // Points at distances ≈ 3.0, 5.0, 5.2, 10.0.
    // Expected hits: IDs 1 and 2.
    let q = (0.0f32, 0.0f32);
    let pts: &[(f32, f32)] = &[
        (3.0, 0.0),  // dist = 3.0  → miss
        (5.0, 0.0),  // dist = 5.0  → hit
        (3.6, 3.6),  // dist ≈ 5.09 → hit
        (10.0, 0.0), // dist = 10.0 → miss
    ];
    let vecs: Vec<f32> = pts.iter().flat_map(|&(x, y)| [x, y]).collect();

    let mut db = cpu_exact(2);
    db.add_vectors(&vecs).unwrap();

    let result = db
        .query(&RingQuery {
            query: &[q.0, q.1],
            d: 5.0,
            lambda: 0.5,
        })
        .unwrap();

    let mut ids = result.ids;
    ids.sort_unstable();

    // Verify analytically
    for (i, &pt) in pts.iter().enumerate() {
        let d = dist2(q, pt);
        let in_ring = d >= 4.5 && d <= 5.5;
        let found = ids.contains(&(i as u32));
        assert_eq!(
            in_ring, found,
            "vector {i} at dist={d:.3} should {} in ring [4.5, 5.5]",
            if in_ring { "be" } else { "NOT be" }
        );
    }
}

#[test]
fn test_lambda_zero() {
    // With lambda=0 only vectors exactly at distance d match.
    // Use a vector at exactly distance 3 from origin: (3, 0).
    let mut db = cpu_exact(2);
    db.add_vectors(&[3.0f32, 0.0, 4.0, 0.0, 0.0, 3.0]).unwrap();

    // d=3, lambda=0 → ring is [3,3] (squared: [9,9])
    let result = db
        .query(&RingQuery {
            query: &[0.0, 0.0],
            d: 3.0,
            lambda: 0.0,
        })
        .unwrap();

    let mut ids = result.ids;
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 2], "only vectors at dist=3 should match");
}

#[test]
fn test_no_results() {
    let mut db = cpu_exact(2);
    db.add_vectors(&[1.0f32, 0.0, 0.0, 1.0]).unwrap();

    // Ring far away from both vectors.
    let result = db
        .query(&RingQuery {
            query: &[0.0, 0.0],
            d: 100.0,
            lambda: 0.1,
        })
        .unwrap();

    assert!(result.ids.is_empty());
}

#[test]
fn test_all_results() {
    // All vectors within the ring.
    let vecs: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
    let mut db = cpu_exact(2);
    db.add_vectors(&vecs).unwrap();

    // Ring [0.9, 1.1] → all four unit vectors match.
    let result = db
        .query(&RingQuery {
            query: &[0.0, 0.0],
            d: 1.0,
            lambda: 0.1,
        })
        .unwrap();

    assert_eq!(result.ids.len(), 4);
}

#[test]
fn test_identical_vectors() {
    // Multiple identical vectors all get the same distance → all should match.
    let v = [3.0f32, 4.0]; // dist = 5.0
    let vecs: Vec<f32> = v.iter().cloned().cycle().take(2 * 5).collect();

    let mut db = cpu_exact(2);
    db.add_vectors(&vecs).unwrap();

    let result = db
        .query(&RingQuery {
            query: &[0.0, 0.0],
            d: 5.0,
            lambda: 0.1,
        })
        .unwrap();

    assert_eq!(result.ids.len(), 5);
}

#[test]
fn test_d_less_than_lambda() {
    // When d < lambda, d - lambda < 0. The lower bound should clamp to 0.
    // A vector at the origin (dist=0) should match ring [0, 1.0+lambda].
    let mut db = cpu_exact(2);
    // Vector at origin: dist = 0.0
    // Vector at (2, 0): dist = 2.0
    db.add_vectors(&[0.0f32, 0.0, 2.0, 0.0]).unwrap();

    let result = db
        .query(&RingQuery {
            query: &[0.0, 0.0],
            d: 0.5,
            lambda: 1.0, // ring would be [-0.5, 1.5], clamped to [0, 1.5]
        })
        .unwrap();

    let mut ids = result.ids;
    ids.sort_unstable();
    assert!(ids.contains(&0), "vector at origin should match");
    assert!(!ids.contains(&1), "vector at dist=2 should not match");
}

#[test]
fn test_backend_name() {
    let db = cpu_exact(4);
    assert_eq!(db.backend_name(), "cpu");
}

#[test]
fn test_dims_and_len() {
    let mut db = cpu_exact(8);
    assert_eq!(db.dims(), 8);
    assert_eq!(db.len(), 0);
    assert!(db.is_empty());

    db.add_vectors(&vec![0.0f32; 8 * 10]).unwrap();
    assert_eq!(db.len(), 10);
    assert!(!db.is_empty());
}

#[test]
fn test_higher_dims() {
    let dims = 64usize;
    let mut db = cpu_exact(dims);

    // 100 random-looking vectors (deterministic).
    let vecs: Vec<f32> = (0..100 * dims)
        .map(|i| ((i as f32 * 1.6180339) % 2.0) - 1.0)
        .collect();
    db.add_vectors(&vecs).unwrap();

    // Brute-force reference on CPU.
    let query: Vec<f32> = (0..dims).map(|i| ((i as f32 * 2.7182818) % 2.0) - 1.0).collect();
    let d = 5.0f32;
    let lambda = 0.5f32;

    let result = db
        .query(&RingQuery {
            query: &query,
            d,
            lambda,
        })
        .unwrap();

    // Verify each returned ID is actually in the ring.
    let norm_sq_q: f32 = query.iter().map(|x| x * x).sum();
    for &id in &result.ids {
        let base = id as usize * dims;
        let row = &vecs[base..base + dims];
        let dot: f32 = row.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
        let norm_sq_x: f32 = row.iter().map(|x| x * x).sum();
        let dist_sq = norm_sq_x + norm_sq_q - 2.0 * dot;
        let dist = dist_sq.sqrt();
        assert!(
            dist >= d - lambda - 1e-4 && dist <= d + lambda + 1e-4,
            "returned ID {id} has dist {dist:.4} outside [{}, {}]",
            d - lambda,
            d + lambda
        );
    }
}
