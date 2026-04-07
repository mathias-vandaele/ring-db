use bytemuck::{Pod, Zeroable};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng};
use ringdb::{Payload, RingDb, RingDbConfig, RingQuery};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const N: usize = 30_000_000;
const SEED: u64 = 42;

#[derive(Serialize, Deserialize, Payload)]
struct DynamicPayload {
    label: String,
}

#[derive(Clone, Copy, Pod, Zeroable, Payload)]
#[repr(C)]
#[payload(storage = "pod")]
struct StaticPayload {
    score: f32,
}

fn bench_cpu_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_f32");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &dims in &[64usize, 128usize] {
        // Expected L2 distance between random vectors in [-1,1]^d is sqrt(2d/3).
        // 64 dims → ~6.5, 128 dims → ~9.2
        let expected_dist = ((2.0 * dims as f32) / 3.0).sqrt();
        let d = expected_dist;
        let lambda = expected_dist * 0.0005; // .05% ring width → guaranteed hits

        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
        let mut buf = vec![0.0f32; dims];

        let mut db = RingDb::new(RingDbConfig::new(dims)).unwrap();
        for _ in 0..N {
            for x in buf.iter_mut() {
                *x = rng.gen_range(-1.0f32..1.0);
            }
            db.add_vector(&buf, ()).unwrap();
        }
        let db = db.build().unwrap();

        let query: Vec<f32> = {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
            (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
        };

        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, _| {
            b.iter(|| {
                db.query(&RingQuery {
                    query: &query,
                    d,
                    lambda,
                })
                .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_payload_fetch_dynamic(c: &mut Criterion) {
    let mut group = c.benchmark_group("payload_fetch_dynamic");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let label: String = "x".repeat(92);

    for &dims in &[64usize, 128usize] {
        let expected_dist = ((2.0 * dims as f32) / 3.0).sqrt();
        let d = expected_dist;
        let lambda = expected_dist * 0.0005;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
        let mut buf = vec![0.0f32; dims];

        let mut db: RingDb<DynamicPayload> = RingDb::new(RingDbConfig::new(dims)).unwrap();
        for _ in 0..N {
            for x in buf.iter_mut() {
                *x = rng.gen_range(-1.0f32..1.0);
            }
            db.add_vector(
                &buf,
                DynamicPayload {
                    label: label.clone(),
                },
            )
            .unwrap();
        }
        let db = db.build().unwrap();

        let query: Vec<f32> = {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
            (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
        };

        let result = db
            .query(&RingQuery {
                query: &query,
                d,
                lambda,
            })
            .unwrap();
        let n_hits = result.ids.len();

        println!(
            "dims={dims} ring=[{:.3}, {:.3}] hits={n_hits} ({:.2}%)",
            d - lambda,
            d + lambda,
            100.0 * n_hits as f32 / N as f32
        );

        group.bench_with_input(
            BenchmarkId::new("100B_string_payload", dims),
            &n_hits,
            |b, _| b.iter(|| db.fetch_payloads(&result.ids).unwrap()),
        );
    }

    group.finish();
}

fn bench_payload_fetch_static(c: &mut Criterion) {
    let mut group = c.benchmark_group("payload_fetch_static");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let score = 42f32;
    for &dims in &[64usize, 128usize] {
        let expected_dist = ((2.0 * dims as f32) / 3.0).sqrt();
        let d = expected_dist;
        let lambda = expected_dist * 0.0005;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
        let mut buf = vec![0.0f32; dims];

        let mut db: RingDb<StaticPayload> = RingDb::new(RingDbConfig::new(dims)).unwrap();
        for _ in 0..N {
            for x in buf.iter_mut() {
                *x = rng.gen_range(-1.0f32..1.0);
            }
            db.add_vector(&buf, StaticPayload { score }).unwrap();
        }
        let db = db.build().unwrap();

        let query: Vec<f32> = {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
            (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
        };

        let result = db
            .query(&RingQuery {
                query: &query,
                d,
                lambda,
            })
            .unwrap();
        let n_hits = result.ids.len();

        println!(
            "dims={dims} ring=[{:.3}, {:.3}] hits={n_hits} ({:.2}%)",
            d - lambda,
            d + lambda,
            100.0 * n_hits as f32 / N as f32
        );

        group.bench_with_input(
            BenchmarkId::new("100B_string_payload", dims),
            &n_hits,
            |b, _| b.iter(|| db.fetch_pods(&result.ids)),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cpu_f32,
    bench_payload_fetch_dynamic,
    bench_payload_fetch_static
);
criterion_main!(benches);
