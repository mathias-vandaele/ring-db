/// Randomised sanity tests: no panics, valid IDs, no duplicates.
use rand::{Rng, SeedableRng};
use ringdb::{DiskQuery, RangeQuery, RingDb, RingDbConfig, RingQuery, SealedRingDb};

fn random_db(dims: usize, n: usize, seed: u64) -> SealedRingDb {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut db = RingDb::new(RingDbConfig::new(dims)).unwrap();
    let mut buf = vec![0.0f32; dims];
    for _ in 0..n {
        for x in buf.iter_mut() {
            *x = rng.gen_range(-1.0f32..1.0);
        }
        db.add_vector(&buf, ()).unwrap();
    }
    db.build().unwrap()
}

fn assert_valid_result(ids: &[u32], n_vectors: usize) {
    for &id in ids {
        assert!(
            (id as usize) < n_vectors,
            "ID {id} out of range (n={n_vectors})"
        );
    }
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "duplicate IDs in result");
}

#[test]
fn random_no_panic_small() {
    let db = random_db(16, 50, 1);
    let q: Vec<f32> = vec![0.1f32; 16];
    let r = db
        .query(&RingQuery {
            query: &q,
            d: 1.5,
            lambda: 0.3,
        })
        .unwrap();
    assert_valid_result(&r.ids, 50);
}

#[test]
fn random_no_panic_medium() {
    let db = random_db(128, 5_000, 2);
    let q: Vec<f32> = vec![0.0f32; 128];
    let r = db
        .query(&RingQuery {
            query: &q,
            d: 4.0,
            lambda: 0.5,
        })
        .unwrap();
    assert_valid_result(&r.ids, 5_000);
}

#[test]
fn random_various_dims() {
    for &dims in &[1usize, 2, 3, 7, 16, 64, 128, 256] {
        let db = random_db(dims, 200, dims as u64 * 7);
        let q: Vec<f32> = vec![0.0f32; dims];
        let r = db
            .query(&RingQuery {
                query: &q,
                d: 3.0,
                lambda: 1.0,
            })
            .unwrap();
        assert_valid_result(&r.ids, 200);
    }
}

#[test]
fn elapsed_is_non_zero_with_data() {
    let db = random_db(64, 1_000, 77);
    let q = vec![0.0f32; 64];
    let r = db
        .query(&RingQuery {
            query: &q,
            d: 5.0,
            lambda: 1.0,
        })
        .unwrap();
    assert!(
        r.elapsed.as_nanos() > 0,
        "elapsed should be > 0 with 1000 vectors"
    );
}

#[test]
fn empty_db_returns_empty() {
    let db: SealedRingDb = RingDb::new(RingDbConfig::new(8)).unwrap().build().unwrap();
    let q = vec![0.0f32; 8];
    let r = db
        .query(&RingQuery {
            query: &q,
            d: 1.0,
            lambda: 0.5,
        })
        .unwrap();
    assert!(r.ids.is_empty());
}

#[test]
fn add_vectors_multiple_calls() {
    let dims = 4usize;
    let mut db = RingDb::new(RingDbConfig::new(dims)).unwrap();

    db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], ()).unwrap(); // ID 0
    db.add_vector(&[0.0f32, 1.0, 0.0, 0.0], ()).unwrap(); // ID 1
    db.add_vector(&[0.0f32, 0.0, 1.0, 0.0], ()).unwrap(); // ID 2

    assert_eq!(db.len(), 3);
    let db = db.build().unwrap();

    let q = [0.0f32; 4];
    let r = db
        .query(&RingQuery {
            query: &q,
            d: 1.0,
            lambda: 0.1,
        })
        .unwrap();

    assert_eq!(r.ids.len(), 3);
}

// ---- RangeQuery randomised tests ----

#[test]
fn range_query_no_panic_small() {
    let db = random_db(16, 50, 10);
    let q = vec![0.1f32; 16];
    let r = db
        .query_range(&RangeQuery {
            query: &q,
            d_min: 1.0,
            d_max: 2.0,
        })
        .unwrap();
    assert_valid_result(&r.ids, 50);
}

#[test]
fn range_query_no_panic_medium() {
    let db = random_db(128, 5_000, 20);
    let q = vec![0.0f32; 128];
    let r = db
        .query_range(&RangeQuery {
            query: &q,
            d_min: 3.0,
            d_max: 5.0,
        })
        .unwrap();
    assert_valid_result(&r.ids, 5_000);
}

#[test]
fn range_query_various_dims() {
    for &dims in &[1usize, 2, 3, 7, 16, 64, 128, 256] {
        let db = random_db(dims, 200, dims as u64 * 13);
        let q = vec![0.0f32; dims];
        let r = db
            .query_range(&RangeQuery {
                query: &q,
                d_min: 2.0,
                d_max: 4.0,
            })
            .unwrap();
        assert_valid_result(&r.ids, 200);
    }
}

/// For the same distance interval, RingQuery and RangeQuery must agree.
#[test]
fn range_matches_ring_random() {
    let dims = 32usize;
    let n = 1_000usize;
    let db = random_db(dims, n, 99);
    let q = vec![0.0f32; dims];
    let (d, lambda) = (4.0f32, 1.0f32);

    let ring_r = db
        .query(&RingQuery {
            query: &q,
            d,
            lambda,
        })
        .unwrap();

    let range_r = db
        .query_range(&RangeQuery {
            query: &q,
            d_min: (d - lambda).max(0.0),
            d_max: d + lambda,
        })
        .unwrap();

    let mut ring_ids = ring_r.ids;
    let mut range_ids = range_r.ids;
    ring_ids.sort_unstable();
    range_ids.sort_unstable();
    assert_eq!(ring_ids, range_ids, "RingQuery and RangeQuery must agree");
}

// ---- DiskQuery randomised tests ----

#[test]
fn disk_query_no_panic_small() {
    let db = random_db(16, 50, 30);
    let q = vec![0.0f32; 16];
    let r = db
        .query_disk(&DiskQuery {
            query: &q,
            d_max: 3.0,
        })
        .unwrap();
    assert_valid_result(&r.ids, 50);
}

#[test]
fn disk_query_no_panic_medium() {
    let db = random_db(128, 5_000, 40);
    let q = vec![0.0f32; 128];
    let r = db
        .query_disk(&DiskQuery {
            query: &q,
            d_max: 6.0,
        })
        .unwrap();
    assert_valid_result(&r.ids, 5_000);
}

#[test]
fn disk_query_various_dims() {
    for &dims in &[1usize, 2, 3, 7, 16, 64, 128, 256] {
        let db = random_db(dims, 200, dims as u64 * 17);
        let q = vec![0.0f32; dims];
        let r = db
            .query_disk(&DiskQuery {
                query: &q,
                d_max: 5.0,
            })
            .unwrap();
        assert_valid_result(&r.ids, 200);
    }
}

/// Disk with d_max must return at least as many results as a ring contained
/// within it (same d_max, any positive d_min).
#[test]
fn disk_is_superset_of_contained_range() {
    let dims = 32usize;
    let n = 1_000usize;
    let db = random_db(dims, n, 55);
    let q = vec![0.0f32; dims];

    let disk_r = db
        .query_disk(&DiskQuery {
            query: &q,
            d_max: 5.0,
        })
        .unwrap();

    let range_r = db
        .query_range(&RangeQuery {
            query: &q,
            d_min: 2.0,
            d_max: 5.0,
        })
        .unwrap();

    assert!(
        disk_r.ids.len() >= range_r.ids.len(),
        "disk (d_max=5) must have >= hits than range [2,5]"
    );

    for &id in &range_r.ids {
        assert!(
            disk_r.ids.contains(&id),
            "range hit {id} not found in disk result"
        );
    }
}

/// DiskQuery must equal RangeQuery(d_min=0, same d_max).
#[test]
fn disk_equals_range_d_min_zero_random() {
    let dims = 32usize;
    let n = 1_000usize;
    let db = random_db(dims, n, 66);
    let q = vec![0.0f32; dims];

    let disk_r = db
        .query_disk(&DiskQuery {
            query: &q,
            d_max: 4.5,
        })
        .unwrap();

    let range_r = db
        .query_range(&RangeQuery {
            query: &q,
            d_min: 0.0,
            d_max: 4.5,
        })
        .unwrap();

    let mut disk_ids = disk_r.ids;
    let mut range_ids = range_r.ids;
    disk_ids.sort_unstable();
    range_ids.sort_unstable();
    assert_eq!(disk_ids, range_ids, "DiskQuery must equal RangeQuery(d_min=0)");
}

