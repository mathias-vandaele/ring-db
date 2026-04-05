/// Randomised sanity tests: no panics, valid IDs, no duplicates.
use rand::{Rng, SeedableRng};
use ringdb::{BackendPreference, QuantizationMode, RingDb, RingDbConfig, RingQuery};

fn random_db(dims: usize, n: usize, seed: u64) -> RingDb {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let vecs: Vec<f32> = (0..n * dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let mut db = RingDb::new(RingDbConfig {
        dims,
        backend_preference: BackendPreference::Cpu,
        quantization: QuantizationMode::None,
    })
    .unwrap();
    db.add_vectors(&vecs).unwrap();
    db
}

fn assert_valid_result(ids: &[u32], n_vectors: usize) {
    // All IDs in range.
    for &id in ids {
        assert!(
            (id as usize) < n_vectors,
            "ID {id} out of range (n={n_vectors})"
        );
    }
    // No duplicates.
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
    // elapsed should be at least 1 ns for a non-trivial dataset
    assert!(
        r.elapsed.as_nanos() > 0,
        "elapsed should be > 0 with 1000 vectors"
    );
}

#[test]
fn empty_db_returns_empty() {
    let db = RingDb::new(RingDbConfig {
        dims: 8,
        backend_preference: BackendPreference::Cpu,
        quantization: QuantizationMode::None,
    })
    .unwrap();
    // Query on empty DB should succeed and return no IDs.
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
fn add_vectors_multiple_batches() {
    let dims = 4usize;
    let mut db = RingDb::new(RingDbConfig {
        dims,
        backend_preference: BackendPreference::Cpu,
        quantization: QuantizationMode::None,
    })
    .unwrap();

    // Add two batches; IDs should be cumulative.
    db.add_vectors(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        .unwrap(); // IDs 0, 1
    db.add_vectors(&[0.0f32, 0.0, 1.0, 0.0]).unwrap(); // ID 2

    assert_eq!(db.len(), 3);

    let q = [0.0f32; 4];
    let r = db
        .query(&RingQuery {
            query: &q,
            d: 1.0,
            lambda: 0.1,
        })
        .unwrap();

    // All three unit vectors are at dist=1 from origin.
    assert_eq!(r.ids.len(), 3);
}
