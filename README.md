# ring-db

A Rust library for **ring queries** in high-dimensional vector spaces.

Instead of nearest-neighbour search, ring-db retrieves every vector whose
Euclidean distance to a query falls inside a specified interval
**[d − λ, d + λ]** — a hollow sphere (ring) rather than a ball.

```
         ┌─────── λ ──────┐
 ─────── d ────────────────── query
         └────────────────┘
   vectors in this shell are returned
```

---

## Table of contents

- [Why ring queries?](#why-ring-queries)
- [Use cases](#use-cases)
- [Quick start](#quick-start)
- [API overview](#api-overview)
- [Payload storage](#payload-storage)
- [Architecture](#architecture)
- [Performance](#performance)
- [Running the benchmarks](#running-the-benchmarks)
- [Running the tests](#running-the-tests)
- [CLI example](#cli-example)
- [Roadmap](#roadmap)

---

## Why ring queries?

Most vector databases answer "what is *closest* to X?". Ring queries answer a
different question: "what is *at distance d* from X, give or take λ?"

This matters when the *magnitude* of similarity has semantic meaning and you
want to filter by it rather than rank by it. Nearest-neighbour returns a ranked
list; a ring query returns a boolean membership set — every result satisfies the
same geometric constraint.

Internally, the library avoids computing square roots by working with squared
L2 distances:

```
lower_sq = max(0, d − λ)²
upper_sq = (d + λ)²
dist_sq  = ‖x‖² + ‖q‖² − 2·(x · q)
```

The dot product and norms are computed with four independent accumulators so
the compiler can emit fused multiply-add instructions (`fmla`/`vfmadd`).

---

## Use cases

| Domain | Query | Ring semantics |
|---|---|---|
| **Anomaly detection** | Is this embedding at an unusual distance from the cluster centroid? | Query at `d = expected_radius`, flag hits at `d ± small_λ` |
| **Content deduplication** | Find near-duplicates, exclude exact copies and fully different items | Set `d` to a "suspicious similarity" threshold, narrow `λ` |
| **Recommendation diversity** | Return items *similar to but not the same as* the seed | Exclude the `d ≈ 0` ball with a positive lower bound |
| **Spatial shells** | Find all POIs within a GPS distance band (e.g. 500 m–1 km) | Direct geometric interpretation |
| **Music / audio fingerprinting** | Match recordings at a specific perceptual distance | Encodes "sounds like but is not identical" |
| **Security / threat intel** | Find embeddings of known-bad patterns at a specific mutation distance | Ring encodes "one edit away from known threat" |

---

## Quick start

Add to `Cargo.toml`:

```toml
[dependencies]
ring-db = { path = "." }         # or version once published
serde = { version = "1", features = ["derive"] }
```

### Minimal example — no payload

```rust
use ringdb::{RingDb, RingDbConfig, RingQuery};

// 1. Build
let mut db = RingDb::new(RingDbConfig::new(4)).unwrap();
db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], ()).unwrap();
db.add_vector(&[0.0, 5.0, 0.0, 0.0], ()).unwrap();
db.add_vector(&[3.0, 4.0, 0.0, 0.0], ()).unwrap(); // dist = 5.0 from origin

// 2. Seal
let db = db.build().unwrap();

// 3. Query — find all vectors at distance ≈ 5 from the origin (±0.5)
let result = db.query(&RingQuery {
    query: &[0.0f32; 4],
    d: 5.0,
    lambda: 0.5,
}).unwrap();

println!("hits: {:?}", result.ids);          // [1, 2]
println!("backend: {}", result.backend_used); // "cpu"
println!("elapsed: {:?}", result.elapsed);
```

### With a typed payload

```rust
use ringdb::{RingDb, RingDbConfig, RingQuery};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Document {
    title: String,
    score: f64,
}

let mut db: RingDb<Document> = RingDb::new(RingDbConfig::new(128)).unwrap();

db.add_vector(&embedding_of("Rust programming"), Document {
    title: "The Rust Book".into(),
    score: 0.92,
}).unwrap();

// … insert more vectors …

let db = db.build().unwrap();
let result = db.query(&RingQuery { query: &my_query, d: 3.5, lambda: 0.2 }).unwrap();

// Fetch payloads for matching IDs
let docs = db.fetch_payloads(&result.ids).unwrap();
for doc in &docs {
    println!("{} (score={})", doc.title, doc.score);
}
```

---

## API overview

### `RingDbConfig`

```rust
let config = RingDbConfig::new(dims);          // CPU backend (default)
```

| Field | Type | Description |
|---|---|---|
| `dims` | `usize` | Number of dimensions per vector. Must be > 0. |
| `backend_preference` | `BackendPreference` | Backend selection. Only `Cpu` is available today. |

### `RingDb<T>` — builder

| Method | Description |
|---|---|
| `RingDb::new(config)` | Create an empty database. |
| `add_vector(v, payload)` | Insert one vector and its associated payload. |
| `build()` | Seal the database; returns `SealedRingDb<T>`. |
| `len()` / `is_empty()` | Number of staged vectors. |
| `dims()` / `backend_name()` | Configuration introspection. |

Vectors are assigned sequential IDs starting from **0** in insertion order.

### `SealedRingDb<T>` — query

| Method | Description |
|---|---|
| `query(q)` | Execute a ring query; returns `QueryResult`. |
| `fetch_payload(id)` | Deserialize the payload for a single ID. |
| `fetch_payloads(ids)` | Deserialize payloads for a slice of IDs, in order. |
| `len()` / `is_empty()` / `dims()` | Introspection. |

### `RingQuery<'a>`

```rust
RingQuery {
    query: &[f32],   // query vector, length == dims
    d: f32,          // centre of the ring (target distance)
    lambda: f32,     // half-width of the ring
}
```

### `QueryResult`

```rust
result.ids           // Vec<u32> — matching vector IDs
result.backend_used  // &'static str — e.g. "cpu"
result.elapsed       // Duration — wall-clock query time
```

### Error handling

All fallible operations return `Result<_, RingDbError>`:

```rust
pub enum RingDbError {
    DimensionMismatch { expected: usize, got: usize },
    Payload(String),   // serialization / deserialization error
    Io(std::io::Error),
}
```

---

## Payload storage

Payloads are designed to consume **zero hot memory** after the build phase:

1. **Build phase** — each call to `add_vector` serializes the payload
   immediately (via `bincode`) and streams it to a temporary file.
   No `Vec<T>` accumulates in RAM.

2. **Query phase** — after `build()`, the temp file is mmap-ed read-only.
   The OS can page payload bytes out under memory pressure. The only
   hot-path data is the offset table (8 bytes × number of vectors).

3. **Cleanup** — the temp file is deleted automatically when `SealedRingDb`
   is dropped, on all platforms (Windows: mmap is dropped before deletion).

The payload type `T` must implement `serde::Serialize + serde::DeserializeOwned`.
Use `T = ()` when no payload is needed — the file is never written in that case.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  RingDb<T>  (builder, mutable)                       │
│                                                      │
│  ┌─────────────────────┐   ┌──────────────────────┐  │
│  │  vectors: Vec<f32>  │   │  PayloadStoreBuilder │  │
│  │  norms_sq: Vec<f32> │   │  (BufWriter → tmpfile│  │
│  └──────────┬──────────┘   └──────────┬─��─────────┘  │
│             │  build()                │              │
└─────────────┼─────────────────────────┼──────────────┘
              ▼                         ▼
┌──────────────────────────────────────────────────────┐
│  SealedRingDb<T>  (immutable, Send + Sync)           │
│                                                      │
│  ┌─────────────────────────┐  ┌────────────────────┐ │
│  │  RingComputeBackend     │  │  PayloadStore      │ │
│  │  (trait object)         │  │  mmap + offsets    │ │
│  │                         │  │  (cold, page-able) │ │
│  │  ┌─────────────────┐    │  └────────────────────┘ │
│  │  │  CpuBackend     │    │                         │
│  │  │  Rayon parallel │    │                         │
│  │  │  4-acc dot prod │    │                         │
│  │  └─────────────────┘    │                         │
│  └─────────────────────────┘                         │
└──────────────────────────────────────────────────────┘
```

### Backend trait

The `RingComputeBackend` trait separates the upload step from the query step,
reflecting the intended usage: upload once, then run many queries against
resident data without re-uploading.

```rust
pub trait RingComputeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn upload_f32_dataset(&mut self, dims: usize, vectors: Vec<f32>, norms_sq: Vec<f32>) -> Result<()>;
    fn ring_query_f32(&self, dims: usize, query: &[f32], d: f32, lambda: f32) -> Result<Vec<u32>>;
}
```

New backends (WGPU, CUDA, AVX-512 hand-rolled) can be added without touching
the public API.

### CPU backend implementation

- **Parallelism** — Rayon `par_chunks_exact` splits the vector matrix across
  all logical cores.
- **SIMD-friendly dot product** — four independent float accumulators break the
  dependency chain so the compiler emits `fmla.4s` (NEON) or `vfmadd231ps`
  (AVX), yielding ~4× FP throughput on the reduction.
- **Precomputed norms** — `‖x‖²` is stored at insertion time; only a dot
  product is computed per vector at query time (no square root).
- **Squared bounds** — both bounds are squared once before the loop to avoid
  `sqrt` entirely.

---

## Performance

Benchmarks run on 30 000 000 randomly generated float32 vectors in `[-1, 1]^d`,
with a ring width of 0.05 % of the expected inter-vector distance.
Measured with [Criterion](https://github.com/bheisler/criterion.rs).

### Ring query throughput (no payload)

| Dimensions | Vectors | Query time | Hit rate |
|---:|---:|---:|---:|
| 64 | 30 M | **~68 ms** | ~0.55 % |
| 128 | 30 M | **~138 ms** | ~0.61 % |

At 64 dims this is **~440 M vectors/second** scanned on a single machine.

### Payload fetch (100-byte string payload, mmap)

Fetching all payloads for matching vectors after a query (ring=[6.529, 6.535],
~164 K hits out of 30 M):

| Dimensions | Hits | Fetch time |
|---:|---:|---:|
| 64 | ~164 K | **~24 ms** |
| 128 | ~184 K | **~26 ms** |

Both are O(hits), not O(dataset), because payloads are addressed by offset
table and read directly from the mmap.

> Benchmarks show a **95 % improvement** over the previous streaming-decode
> baseline after switching to mmap-backed cold storage.

---

## Running the benchmarks

```bash
# Full benchmark suite (HTML report in target/criterion/)
cargo bench

# Single group
cargo bench --bench query_backends cpu_f32
cargo bench --bench query_backends payload_fetch

# With native CPU optimisations (recommended)
RUSTFLAGS="-C target-cpu=native" cargo bench
```

Results are written to `target/criterion/` as an interactive HTML report.

---

## Running the tests

```bash
# All tests
cargo test

# Correctness tests only (hand-crafted, analytically verified)
cargo test --test correctness

# Randomised sanity tests
cargo test --test random

# With output (useful to see ring stats)
cargo test -- --nocapture
```

---

## CLI example

A small CLI that builds a random dataset and runs a ring query:

```bash
cargo run --example cli -- --help

# 100 000 vectors, 64 dims, ring centred at d=5.0 with λ=0.2
cargo run --release --example cli -- \
    --n 100000 --dims 64 --d 5.0 --lambda 0.2
```

Sample output:

```
ringdb demo
  dataset : 100000 vectors × 64 dims
  ring    : d=5, λ=0.2

Building database … done in 45.12ms
  backend : cpu

Running ring query … done

Results:
  backend      : cpu
  hits         : 312
  query time   : 2.34ms
  hit IDs      : [142, 891, … 312 total]
```

---

## Roadmap

- [ ] GPU backend via `wgpu` (compute shaders, cross-platform)
- [ ] CUDA backend for datacenter workloads
- [ ] AVX-512 hand-rolled kernel (skipping Rayon for very small datasets)
- [ ] Approximate ring search (LSH / quantisation) for datasets > 1 B vectors
- [ ] Persistent on-disk format (memory-mapped, no load time)
- [ ] Python bindings (PyO3)

---

## License

MIT OR Apache-2.0
