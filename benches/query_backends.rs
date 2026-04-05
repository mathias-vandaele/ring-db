use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{Rng, SeedableRng};
use ringdb::{BackendPreference, QuantizationMode, RingDb, RingDbConfig, RingQuery};
use std::time::Duration;

const N: usize = 30_000_000;
const SEED: u64 = 42;

fn make_vecs(n: usize, dims: usize) -> Vec<f32> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED);
    (0..n * dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn make_query(dims: usize) -> Vec<f32> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(SEED + 1);
    (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn bench_cpu_f32(c: &mut Criterion) {
    println!("Running CPU f32 benchmark...");
    let mut group = c.benchmark_group("cpu_f32");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    // CPU queries are O(N*dims) — reduce samples for large dims to avoid hour-long runs.
    group.sample_size(20);

    for &dims in &[64, 128] {
        let query = make_query(dims);
        let mut db = RingDb::new(RingDbConfig {
            dims,
            backend_preference: BackendPreference::Cpu,
            quantization: QuantizationMode::None,
        })
        .unwrap();
        // Build vecs in a block so the source buffer is freed before
        // benchmarking starts — avoids holding two full copies in RAM.
        {
            let vecs = make_vecs(N, dims);
            db.add_vectors(&vecs).unwrap();
        }

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

fn bench_cpu_q8(c: &mut Criterion) {
    println!("Running CPU q8 benchmark...");
    let mut group = c.benchmark_group("cpu_q8");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &dims in &[64, 128] {
        let query = make_query(dims);
        let mut db = RingDb::new(RingDbConfig {
            dims,
            backend_preference: BackendPreference::Cpu,
            quantization: QuantizationMode::Q8,
        })
        .unwrap();
        {
            let vecs = make_vecs(N, dims);
            db.add_vectors(&vecs).unwrap();
        }

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

fn bench_wgpu_f32(c: &mut Criterion) {
    println!("Checking WGPU availability for f32 benchmark...");
    // Single availability check — no throw-away device.
    if RingDb::new(RingDbConfig {
        dims: 4,
        backend_preference: BackendPreference::Wgpu,
        quantization: QuantizationMode::None,
    })
    .is_err()
    {
        eprintln!("WGPU not available, skipping wgpu_f32 benchmark");
        return;
    }
    let mut group = c.benchmark_group("wgpu_f32");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for &dims in &[64, 128] {
        let query = make_query(dims);
        let mut db = RingDb::new(RingDbConfig {
            dims,
            backend_preference: BackendPreference::Wgpu,
            quantization: QuantizationMode::None,
        })
        .unwrap();
        {
            let vecs = make_vecs(N, dims);
            db.add_vectors(&vecs).unwrap();
        }

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

fn bench_wgpu_q8(c: &mut Criterion) {
    println!("Checking WGPU availability for Q8 benchmark...");
    if RingDb::new(RingDbConfig {
        dims: 4,
        backend_preference: BackendPreference::Wgpu,
        quantization: QuantizationMode::Q8,
    })
    .is_err()
    {
        eprintln!("WGPU not available, skipping wgpu_q8 benchmark");
        return;
    }

    let mut group = c.benchmark_group("wgpu_q8");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for &dims in &[64, 128] {
        let query = make_query(dims);
        let mut db = RingDb::new(RingDbConfig {
            dims,
            backend_preference: BackendPreference::Wgpu,
            quantization: QuantizationMode::Q8,
        })
        .unwrap();
        {
            let vecs = make_vecs(N, dims);
            db.add_vectors(&vecs).unwrap();
        }

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

criterion_group!(benches, bench_cpu_f32, bench_cpu_q8, bench_wgpu_f32, bench_wgpu_q8);
criterion_main!(benches);
