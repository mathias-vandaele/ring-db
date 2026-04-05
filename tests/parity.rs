/// Backend parity tests: CPU f32 exact vs WGPU f32 exact.
///
/// In exact mode, all backends must return identical result sets.
/// These tests use a fixed random seed for reproducibility.
use rand::{Rng, SeedableRng};
use ringdb::{BackendPreference, QuantizationMode, RingDb, RingDbConfig, RingQuery};

fn make_db(backend: BackendPreference, dims: usize) -> RingDb {
    RingDb::new(RingDbConfig {
        dims,
        backend_preference: backend,
        quantization: QuantizationMode::None,
    })
    .unwrap()
}

fn random_vecs(n: usize, dims: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    (0..n * dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn sorted_ids(mut ids: Vec<u32>) -> Vec<u32> {
    ids.sort_unstable();
    ids
}

/// Compare CPU and WGPU f32 results on the same dataset.
#[test]
fn cpu_wgpu_f32_parity() {
    let dims = 64;
    let n = 1_000;
    let seed = 42u64;
    let vecs = random_vecs(n, dims, seed);

    let mut cpu_db = make_db(BackendPreference::Cpu, dims);
    cpu_db.add_vectors(&vecs).unwrap();

    let wgpu_result = RingDb::new(RingDbConfig {
        dims,
        backend_preference: BackendPreference::Wgpu,
        quantization: QuantizationMode::None,
    });

    let Ok(mut wgpu_db) = wgpu_result else {
        eprintln!("WGPU not available, skipping parity test");
        return;
    };
    wgpu_db.add_vectors(&vecs).unwrap();

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed + 1);
    let query: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let d = 3.0f32;
    let lambda = 0.5f32;

    let cpu_ids = sorted_ids(
        cpu_db
            .query(&RingQuery {
                query: &query,
                d,
                lambda,
            })
            .unwrap()
            .ids,
    );
    let wgpu_ids = sorted_ids(
        wgpu_db
            .query(&RingQuery {
                query: &query,
                d,
                lambda,
            })
            .unwrap()
            .ids,
    );

    assert_eq!(
        cpu_ids, wgpu_ids,
        "CPU and WGPU f32 results must be identical"
    );
}

/// Parity across multiple queries.
#[test]
fn cpu_wgpu_f32_parity_multi_query() {
    let dims = 32;
    let n = 500;
    let seed = 99u64;
    let vecs = random_vecs(n, dims, seed);

    let mut cpu_db = make_db(BackendPreference::Cpu, dims);
    cpu_db.add_vectors(&vecs).unwrap();

    let wgpu_result = RingDb::new(RingDbConfig {
        dims,
        backend_preference: BackendPreference::Wgpu,
        quantization: QuantizationMode::None,
    });

    let Ok(mut wgpu_db) = wgpu_result else {
        eprintln!("WGPU not available, skipping parity test");
        return;
    };
    wgpu_db.add_vectors(&vecs).unwrap();

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed + 100);

    for q_idx in 0..10 {
        let query: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let d = 2.0f32 + (q_idx as f32) * 0.3;
        let lambda = 0.3f32;

        let cpu_ids = sorted_ids(
            cpu_db
                .query(&RingQuery {
                    query: &query,
                    d,
                    lambda,
                })
                .unwrap()
                .ids,
        );
        let wgpu_ids = sorted_ids(
            wgpu_db
                .query(&RingQuery {
                    query: &query,
                    d,
                    lambda,
                })
                .unwrap()
                .ids,
        );

        assert_eq!(
            cpu_ids, wgpu_ids,
            "query {q_idx}: CPU and WGPU must agree"
        );
    }
}
