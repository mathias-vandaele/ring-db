use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{Rng, SeedableRng};
use ringdb::{RingDb, RingDbConfig, RingQuery};
use std::time::Duration;

const N: usize = 30_000_000;
const SEED: u64 = 42;

fn populate_db(db: &mut RingDb, n: usize, dims: usize) {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
    let mut buf = vec![0.0f32; dims];
    for _ in 0..n {
        for x in buf.iter_mut() {
            *x = rng.gen_range(-1.0f32..1.0);
        }
        db.add_vector(&buf).unwrap();
    }
}

fn make_query(dims: usize) -> Vec<f32> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
    (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn bench_cpu_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_f32");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &dims in &[64, 128] {
        let query = make_query(dims);
        let mut db = RingDb::new(RingDbConfig::new(dims)).unwrap();
        populate_db(&mut db, N, dims);
        let db = db.build().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| {
                db.query(&RingQuery {
                    query: &query,
                    d: 5.0,
                    lambda: 0.5,
                })
                .unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cpu_f32);
criterion_main!(benches);
