/// Correctness tests for the CPU exact (float32) backend.
///
/// These tests use hand-crafted vectors at known distances from a query so
/// the expected result sets are computed analytically, not by another code path.
use ringdb::{DiskQuery, RangeQuery, RingDb, RingDbConfig, RingQuery};

fn cpu_db(dims: usize) -> RingDb {
    RingDb::new(RingDbConfig::new(dims)).unwrap()
}

// ---- 2-D helpers ----

/// Distance between two 2-D points.
fn dist2(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

#[test]
fn test_basic_hits() {
    let q = (0.0f32, 0.0f32);
    let pts: &[(f32, f32)] = &[
        (3.0, 0.0),  // dist = 3.0  → miss
        (5.0, 0.0),  // dist = 5.0  → hit
        (3.6, 3.6),  // dist ≈ 5.09 → hit
        (10.0, 0.0), // dist = 10.0 → miss
    ];

    let mut db = cpu_db(2);
    for &(x, y) in pts {
        db.add_vector(&[x, y], ()).unwrap();
    }
    let db = db.build().unwrap();

    let result = db
        .query(&RingQuery {
            query: &[q.0, q.1],
            d: 5.0,
            lambda: 0.5,
        })
        .unwrap();

    let mut ids = result.ids;
    ids.sort_unstable();

    for (i, &pt) in pts.iter().enumerate() {
        let d = dist2(q, pt);
        let in_ring = d >= 4.5 && d <= 5.5;
        let found = ids.contains(&(i as u32));
        assert_eq!(
            in_ring,
            found,
            "vector {i} at dist={d:.3} should {} in ring [4.5, 5.5]",
            if in_ring { "be" } else { "NOT be" }
        );
    }
}

#[test]
fn test_lambda_zero() {
    let mut db = cpu_db(2);
    db.add_vector(&[3.0f32, 0.0], ()).unwrap();
    db.add_vector(&[4.0, 0.0], ()).unwrap();
    db.add_vector(&[0.0, 3.0], ()).unwrap();
    let db = db.build().unwrap();

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
    let mut db = cpu_db(2);
    db.add_vector(&[1.0f32, 0.0], ()).unwrap();
    db.add_vector(&[0.0, 1.0], ()).unwrap();
    let db = db.build().unwrap();

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
    let mut db = cpu_db(2);
    db.add_vector(&[1.0, 0.0], ()).unwrap();
    db.add_vector(&[0.0, 1.0], ()).unwrap();
    db.add_vector(&[-1.0, 0.0], ()).unwrap();
    db.add_vector(&[0.0, -1.0], ()).unwrap();
    let db = db.build().unwrap();

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
    let mut db = cpu_db(2);
    for _ in 0..5 {
        db.add_vector(&[3.0f32, 4.0], ()).unwrap();
    }
    let db = db.build().unwrap();

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
    let mut db = cpu_db(2);
    db.add_vector(&[0.0f32, 0.0], ()).unwrap();
    db.add_vector(&[2.0, 0.0], ()).unwrap();
    let db = db.build().unwrap();

    let result = db
        .query(&RingQuery {
            query: &[0.0, 0.0],
            d: 0.5,
            lambda: 1.0,
        })
        .unwrap();

    let mut ids = result.ids;
    ids.sort_unstable();
    assert!(ids.contains(&0), "vector at origin should match");
    assert!(!ids.contains(&1), "vector at dist=2 should not match");
}

#[test]
fn test_backend_name() {
    let db = cpu_db(4);
    assert_eq!(db.backend_name(), "cpu");
}

#[test]
fn test_dims_and_len() {
    let mut db = cpu_db(8);
    assert_eq!(db.dims(), 8);
    assert_eq!(db.len(), 0);
    assert!(db.is_empty());

    for _ in 0..10 {
        db.add_vector(&[0.0f32; 8], ()).unwrap();
    }
    assert_eq!(db.len(), 10);
    assert!(!db.is_empty());
}

#[test]
fn test_higher_dims() {
    let dims = 64usize;
    let mut db = cpu_db(dims);

    let vecs: Vec<f32> = (0..100 * dims)
        .map(|i| ((i as f32 * 1.6180339) % 2.0) - 1.0)
        .collect();
    for chunk in vecs.chunks(dims) {
        db.add_vector(chunk, ()).unwrap();
    }
    let db = db.build().unwrap();

    let query: Vec<f32> = (0..dims)
        .map(|i| ((i as f32 * 2.7182818) % 2.0) - 1.0)
        .collect();
    let d = 5.0f32;
    let lambda = 0.5f32;

    let result = db
        .query(&RingQuery {
            query: &query,
            d,
            lambda,
        })
        .unwrap();

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

#[test]
fn test_payload_roundtrip() {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Meta {
        label: String,
        score: f64,
    }

    let mut db: RingDb<Meta> = RingDb::new(RingDbConfig::new(2)).unwrap();
    db.add_vector(
        &[1.0, 0.0],
        Meta {
            label: "dog".into(),
            score: 0.9,
        },
    )
    .unwrap();
    db.add_vector(
        &[0.0, 1.0],
        Meta {
            label: "cat".into(),
            score: 0.7,
        },
    )
    .unwrap();
    db.add_vector(
        &[5.0, 0.0],
        Meta {
            label: "bird".into(),
            score: 0.5,
        },
    )
    .unwrap();

    let db = db.build().unwrap();

    let result = db
        .query(&RingQuery {
            query: &[0.0, 0.0],
            d: 1.0,
            lambda: 0.1,
        })
        .unwrap();

    let mut ids = result.ids.clone();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1]);

    let payloads = db.fetch_payloads(&result.ids).unwrap();
    assert_eq!(payloads.len(), 2);

    let p0 = db.fetch_payload(0).unwrap();
    assert_eq!(p0.label, "dog");
    assert_eq!(p0.score, 0.9);

    let p1 = db.fetch_payload(1).unwrap();
    assert_eq!(p1.label, "cat");
}

// ---- RangeQuery tests ----

#[test]
fn test_range_query_basic() {
    // Vectors at distances 1, 3, 5, 7 from origin.
    let mut db = cpu_db(2);
    db.add_vector(&[1.0f32, 0.0], ()).unwrap(); // dist 1 → miss (< 2.5)
    db.add_vector(&[3.0, 0.0], ()).unwrap(); // dist 3 → hit
    db.add_vector(&[5.0, 0.0], ()).unwrap(); // dist 5 → hit
    db.add_vector(&[7.0, 0.0], ()).unwrap(); // dist 7 → miss (> 6)
    let db = db.build().unwrap();

    let result = db
        .query_range(&RangeQuery {
            query: &[0.0, 0.0],
            d_min: 2.5,
            d_max: 6.0,
        })
        .unwrap();

    let mut ids = result.ids;
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2], "only vectors at dist 3 and 5 should match");
}

#[test]
fn test_range_query_inclusive_bounds() {
    // Vectors are placed exactly on the boundaries.
    let mut db = cpu_db(2);
    db.add_vector(&[3.0f32, 0.0], ()).unwrap(); // dist = 3.0 (on d_min)
    db.add_vector(&[5.0, 0.0], ()).unwrap(); // dist = 5.0 (on d_max)
    db.add_vector(&[4.0, 0.0], ()).unwrap(); // dist = 4.0 (inside)
    let db = db.build().unwrap();

    let result = db
        .query_range(&RangeQuery {
            query: &[0.0, 0.0],
            d_min: 3.0,
            d_max: 5.0,
        })
        .unwrap();

    let mut ids = result.ids;
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![0, 1, 2],
        "boundary and interior vectors must all match"
    );
}

#[test]
fn test_range_query_empty_interval() {
    // d_min == d_max: only vectors at exactly that distance should match.
    let mut db = cpu_db(2);
    db.add_vector(&[3.0f32, 0.0], ()).unwrap(); // dist = 3
    db.add_vector(&[4.0, 0.0], ()).unwrap(); // dist = 4
    let db = db.build().unwrap();

    let result = db
        .query_range(&RangeQuery {
            query: &[0.0, 0.0],
            d_min: 3.0,
            d_max: 3.0,
        })
        .unwrap();

    let mut ids = result.ids;
    ids.sort_unstable();
    assert_eq!(ids, vec![0], "only the vector at dist=3 should match");
}

#[test]
fn test_range_query_no_results() {
    let mut db = cpu_db(2);
    db.add_vector(&[1.0f32, 0.0], ()).unwrap();
    db.add_vector(&[2.0, 0.0], ()).unwrap();
    let db = db.build().unwrap();

    let result = db
        .query_range(&RangeQuery {
            query: &[0.0, 0.0],
            d_min: 10.0,
            d_max: 20.0,
        })
        .unwrap();

    assert!(result.ids.is_empty());
}

/// A RangeQuery with d_min=(d-lambda).max(0) and d_max=d+lambda must return
/// the same IDs as the equivalent RingQuery.
#[test]
fn test_range_equivalent_to_ring() {
    let mut db = cpu_db(2);
    for x in [1.0f32, 3.0, 5.0, 7.0, 9.0] {
        db.add_vector(&[x, 0.0], ()).unwrap();
    }
    let db = db.build().unwrap();

    let (d, lambda) = (5.0f32, 1.5f32);

    let ring_result = db
        .query(&RingQuery {
            query: &[0.0, 0.0],
            d,
            lambda,
        })
        .unwrap();

    let range_result = db
        .query_range(&RangeQuery {
            query: &[0.0, 0.0],
            d_min: (d - lambda).max(0.0),
            d_max: d + lambda,
        })
        .unwrap();

    let mut ring_ids = ring_result.ids;
    let mut range_ids = range_result.ids;
    ring_ids.sort_unstable();
    range_ids.sort_unstable();
    assert_eq!(
        ring_ids, range_ids,
        "RingQuery and RangeQuery must return identical IDs"
    );
}

// ---- DiskQuery tests ----

#[test]
fn test_disk_query_basic() {
    // Vectors at distances 1, 3, 5, 7 from origin.
    let mut db = cpu_db(2);
    db.add_vector(&[1.0f32, 0.0], ()).unwrap(); // dist 1 → hit
    db.add_vector(&[3.0, 0.0], ()).unwrap(); // dist 3 → hit
    db.add_vector(&[5.0, 0.0], ()).unwrap(); // dist 5 → hit (on boundary)
    db.add_vector(&[7.0, 0.0], ()).unwrap(); // dist 7 → miss
    let db = db.build().unwrap();

    let result = db
        .query_disk(&DiskQuery {
            query: &[0.0, 0.0],
            d_max: 5.0,
        })
        .unwrap();

    let mut ids = result.ids;
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![0, 1, 2],
        "all vectors within radius 5 should match"
    );
}

#[test]
fn test_disk_query_includes_origin() {
    // A vector at the origin is at distance 0 — must always be inside any disk.
    let mut db = cpu_db(2);
    db.add_vector(&[0.0f32, 0.0], ()).unwrap(); // dist 0
    db.add_vector(&[10.0, 0.0], ()).unwrap(); // dist 10
    let db = db.build().unwrap();

    let result = db
        .query_disk(&DiskQuery {
            query: &[0.0, 0.0],
            d_max: 1.0,
        })
        .unwrap();

    assert!(
        result.ids.contains(&0),
        "vector at origin must be inside disk"
    );
    assert!(!result.ids.contains(&1), "far vector must be outside disk");
}

#[test]
fn test_disk_query_d_max_zero() {
    // Disk of radius 0: only vectors coincident with the query match.
    let mut db = cpu_db(2);
    db.add_vector(&[0.0f32, 0.0], ()).unwrap(); // coincident  → hit
    db.add_vector(&[1.0, 0.0], ()).unwrap(); // dist 1 → miss
    let db = db.build().unwrap();

    let result = db
        .query_disk(&DiskQuery {
            query: &[0.0, 0.0],
            d_max: 0.0,
        })
        .unwrap();

    let mut ids = result.ids;
    ids.sort_unstable();
    assert_eq!(ids, vec![0]);
}

/// A DiskQuery must return a superset of a RingQuery with the same outer bound,
/// because the disk covers d_min=0 while the ring excludes the interior.
#[test]
fn test_disk_is_superset_of_ring() {
    let mut db = cpu_db(2);
    for x in [1.0f32, 2.0, 3.0, 4.0, 5.0, 8.0] {
        db.add_vector(&[x, 0.0], ()).unwrap();
    }
    let db = db.build().unwrap();

    let disk_result = db
        .query_disk(&DiskQuery {
            query: &[0.0, 0.0],
            d_max: 5.0,
        })
        .unwrap();

    let ring_result = db
        .query(&RingQuery {
            query: &[0.0, 0.0],
            d: 4.0,
            lambda: 1.5, // covers [2.5, 5.5]
        })
        .unwrap();

    // Every ring hit must also appear in the disk.
    for &id in &ring_result.ids {
        assert!(
            disk_result.ids.contains(&id),
            "ring hit ID {id} not found in disk result"
        );
    }
    // The disk must have at least as many hits.
    assert!(disk_result.ids.len() >= ring_result.ids.len());
}

/// DiskQuery with d_max equals RangeQuery with d_min=0 and same d_max.
#[test]
fn test_disk_equivalent_to_range_d_min_zero() {
    let mut db = cpu_db(2);
    for x in [0.0f32, 1.0, 2.5, 4.9, 5.0, 6.0] {
        db.add_vector(&[x, 0.0], ()).unwrap();
    }
    let db = db.build().unwrap();

    let disk_result = db
        .query_disk(&DiskQuery {
            query: &[0.0, 0.0],
            d_max: 5.0,
        })
        .unwrap();

    let range_result = db
        .query_range(&RangeQuery {
            query: &[0.0, 0.0],
            d_min: 0.0,
            d_max: 5.0,
        })
        .unwrap();

    let mut disk_ids = disk_result.ids;
    let mut range_ids = range_result.ids;
    disk_ids.sort_unstable();
    range_ids.sort_unstable();
    assert_eq!(
        disk_ids, range_ids,
        "DiskQuery must equal RangeQuery(d_min=0)"
    );
}
