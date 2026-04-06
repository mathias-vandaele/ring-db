# ring-db

A Rust library for **ring queries** in high-dimensional vector spaces.

Instead of nearest-neighbour search, ring-db retrieves every vector whose
Euclidean distance to a query falls inside a specified interval.  
Three query shapes are supported out of the box:

| Query type | Interval | Typical use |
|---|---|---|
| **`RingQuery`** | [d − λ, d + λ] | Hollow shell at distance d |
| **`RangeQuery`** | [d_min, d_max] | Arbitrary distance band |
| **`DiskQuery`** | [0, d_max] | Full ball / nearest-within-radius |

```
  RingQuery          RangeQuery          DiskQuery
  ┌──── λ ───┐       ┌──────────┐        ┌──────────────┐
  d ─────────── q    d_min  d_max  q     0      d_max   q
  └──────────┘
```

---

## Table of contents

- [Why ring queries?](#why-ring-queries)
- [Use cases](#use-cases)
- [Quick start](#quick-start)
- [API overview](#api-overview)
- [Payload storage](#payload-storage)
- [Persistence](#persistence)
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

The three query types let you express the constraint directly:

- **`RingQuery`** — symmetric band defined by a centre distance `d` and
  half-width `λ`: interval `[d-λ, d+λ]`.
- **`RangeQuery`** — arbitrary interval `[d_min, d_max]`, no conversion needed.
- **`DiskQuery`** — full ball of radius `d_max`, equivalent to
  `RangeQuery { d_min: 0, d_max }`.

Internally, the library avoids computing square roots by working with squared
L2 distances:

```
lower_sq = d_min²
upper_sq = d_max²
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

### Ring query — hollow shell at distance d ± λ

```rust
use ringdb::{RingDb, RingDbConfig, RingQuery};

let mut db = RingDb::new(RingDbConfig::new(4)).unwrap();
db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], ()).unwrap();
db.add_vector(&[0.0, 5.0, 0.0, 0.0], ()).unwrap();
db.add_vector(&[3.0, 4.0, 0.0, 0.0], ()).unwrap(); // dist = 5.0 from origin

let db = db.build().unwrap();

// Find all vectors at distance ≈ 5 from the origin (band: [4.5, 5.5])
let result = db.query(&RingQuery {
    query: &[0.0f32; 4],
    d: 5.0,
    lambda: 0.5,
}).unwrap();

println!("hits: {:?}", result.ids);           // [1, 2]
println!("backend: {}", result.backend_used); // "cpu"
println!("elapsed: {:?}", result.elapsed);
```

### Range query — explicit [d_min, d_max] band

```rust
use ringdb::{RingDb, RingDbConfig, RangeQuery};

let mut db = RingDb::new(RingDbConfig::new(4)).unwrap();
db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], ()).unwrap();
db.add_vector(&[3.0, 4.0, 0.0, 0.0], ()).unwrap();
let db = db.build().unwrap();

// Find all vectors between distance 3.0 and 6.0 from the query
let result = db.query_range(&RangeQuery {
    query: &[0.0f32; 4],
    d_min: 3.0,
    d_max: 6.0,
}).unwrap();
```

### Disk query — everything within radius d_max

```rust
use ringdb::{RingDb, RingDbConfig, DiskQuery};

let mut db = RingDb::new(RingDbConfig::new(4)).unwrap();
db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], ()).unwrap();
db.add_vector(&[3.0, 4.0, 0.0, 0.0], ()).unwrap();
let db = db.build().unwrap();

// Find all vectors within distance 5.0 from the query (full ball)
let result = db.query_disk(&DiskQuery {
    query: &[0.0f32; 4],
    d_max: 5.0,
}).unwrap();
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

let mut db: RingDb<Document> = RingDb::new(RingDbConfig::new(4)).unwrap();

db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], Document {
    title: "The Rust Book".into(),
    score: 0.92,
}).unwrap();
db.add_vector(&[0.0, 1.0, 0.0, 0.0], Document {
    title: "Rustonomicon".into(),
    score: 0.85,
}).unwrap();

let query = [0.8f32, 0.0, 0.0, 0.0];
let db = db.build().unwrap();
let result = db.query(&RingQuery { query: &query, d: 0.2, lambda: 0.1 }).unwrap();

// Fetch payloads for matching IDs
let docs = db.fetch_payloads(&result.ids).unwrap();
for doc in &docs {
    println!("{} (score={})", doc.title, doc.score);
}
```

---

## API overview

### `RingDbConfig`

```
let config = RingDbConfig::new(dims);                             // in-memory, CPU backend (defaults)
let config = RingDbConfig::new(dims).with_persist_dir("/my/db"); // persisted to disk
```

| Field | Type | Description |
|---|---|---|
| `dims` | `usize` | Number of dimensions per vector. Must be > 0. |
| `backend_preference` | `BackendPreference` | Backend selection. Only `Cpu` is available today. |
| `persist_dir` | `Option<PathBuf>` | Directory to persist the database. `None` = in-memory only. |

| Builder method | Description |
|---|---|
| `with_persist_dir(path)` | Enable persistence to the given directory (created automatically). |
| `with_backend_preference(p)` | Override the backend (default: `BackendPreference::Cpu`). |

### `RingDb<T>` — builder

| Method | Description |
|---|---|
| `RingDb::new(config)` | Create an empty in-memory (or persist-configured) database. |
| `RingDb::load(dir, backend)` | Rehydrate a previously persisted database from `dir`. |
| `add_vector(v, payload)` | Insert one vector and its associated payload. |
| `build()` | Seal the database; returns `SealedRingDb<T>`. Writes files to disk if `persist_dir` is set. |
| `len()` / `is_empty()` | Number of staged vectors. |
| `dims()` / `backend_name()` | Configuration introspection. |

Vectors are assigned sequential IDs starting from **0** in insertion order.

### `SealedRingDb<T>` — query

| Method | Description |
|---|---|
| `query(q: &RingQuery)` | Ring query: interval `[d-λ, d+λ]`. |
| `query_range(q: &RangeQuery)` | Range query: explicit `[d_min, d_max]`. |
| `query_disk(q: &DiskQuery)` | Disk query: full ball `[0, d_max]`. |
| `fetch_payload(id)` | Deserialize the payload for a single ID. |
| `fetch_payloads(ids)` | Deserialize payloads for a slice of IDs, in order. |
| `len()` / `is_empty()` / `dims()` | Introspection. |

### `RingQuery<'a>`

```
RingQuery {
    query: &[f32],   // query vector, length == dims
    d: f32,          // centre of the ring (target distance)
    lambda: f32,     // half-width; interval = [max(0, d-λ), d+λ]
}
```

### `RangeQuery<'a>`

```
RangeQuery {
    query: &[f32],   // query vector, length == dims
    d_min: f32,      // lower bound of the distance interval (inclusive, ≥ 0)
    d_max: f32,      // upper bound of the distance interval (inclusive, ≥ d_min)
}
```

### `DiskQuery<'a>`

```
DiskQuery {
    query: &[f32],   // query vector, length == dims
    d_max: f32,      // radius of the ball (inclusive, ≥ 0)
}
// Equivalent to RangeQuery { d_min: 0.0, d_max }
```

### `QueryResult`

```
result.ids           // Vec<u32> — matching vector IDs
result.backend_used  // &'static str — e.g. "cpu"
result.elapsed       // Duration — wall-clock query time
```

### Error handling

All fallible operations return `Result<_, RingDbError>`:

```
pub enum RingDbError {
    DimensionMismatch { expected: usize, got: usize },
    Payload(String),   // serialization / deserialization error
    Io(std::io::Error),
    Corrupt(String),   // missing / inconsistent persistence files
}
```

---

## Payload storage

Payloads are designed to consume **zero hot memory** after the build phase:

1. **Build phase** — each call to `add_vector` serializes the payload
   immediately (via `bincode`) and streams it to a temporary file.

2. **Query phase** — after `build()`, the temp file is mmap-ed read-only.
   The OS can page payload bytes out under memory pressure. The only
   hot-path data is the offset table (8 bytes × number of vectors).

3. **Cleanup** — when **no** `persist_dir` is set, the temp file is deleted
   automatically when `SealedRingDb` is dropped. When `persist_dir` is set,
   the files are kept on disk and can be reloaded later with `RingDb::load()`.

---

## Persistence

ring-db can save the full database to disk and reload it in a later process.
No re-insertion is required — the original vectors, norms, and payloads are
reconstructed exactly.

### Saving

Set `persist_dir` on the config before calling `build()`:

```rust
use ringdb::{RingDb, RingDbConfig};

let mut db = RingDb::<()>::new(
    RingDbConfig::new(4).with_persist_dir("/tmp/mydb")
).unwrap();

db.add_vector(&[1.0, 0.0, 0.0, 0.0], ()).unwrap();
db.add_vector(&[0.0, 1.0, 0.0, 0.0], ()).unwrap();

let _sealed = db.build().unwrap(); // writes files to /tmp/mydb
```

`build()` creates the directory if it does not exist and writes:

| File | Content |
|---|---|
| `meta.bin` | `dims` + `n_vectors` as little-endian `u64` |
| `vectors.bin` | Raw f32 vectors (row-major, `n_vectors × dims` floats) |
| `norms_sq.bin` | Raw f32 squared L2 norms (`n_vectors` floats) |
| `payloads.bin` | Concatenated bincode-serialized payload bytes |
| `offsets.bin` | Byte offsets (`u64`) into `payloads.bin`, one per vector |

### Loading

Pass the directory and your chosen backend to `RingDb::load()`:

```rust
use ringdb::{RingDb, BackendPreference, RingQuery};
use std::path::Path;

let db = RingDb::<()>::load(
    Path::new("/tmp/mydb"),
    BackendPreference::Cpu,
).unwrap();

// db is a fully usable SealedRingDb — query immediately
let result = db.query(&RingQuery {
    query: &[1.0f32, 0.0, 0.0, 0.0],
    d: 1.0,
    lambda: 0.1,
}).unwrap();
```

`load()` validates the file sizes against the stored metadata and returns
`RingDbError::Corrupt` if anything is inconsistent.

### With a typed payload

```rust
use ringdb::{RingDb, RingDbConfig, BackendPreference, RingQuery};
use serde::{Serialize, Deserialize};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
struct Document { title: String }

// --- save ---
let mut db: RingDb<Document> = RingDb::new(
    RingDbConfig::new(4).with_persist_dir("/tmp/docs")
).unwrap();
db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], Document { title: "Hello".into() }).unwrap();
let _sealed = db.build().unwrap();

// --- load in a new process ---
let db = RingDb::<Document>::load(Path::new("/tmp/docs"), BackendPreference::Cpu).unwrap();
let result = db.query(&RingQuery { query: &[1.0f32, 0.0, 0.0, 0.0], d: 1.0, lambda: 0.5 }).unwrap();
let docs = db.fetch_payloads(&result.ids).unwrap();
```

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
│  └──────────┬──────────┘   └──────────┬───────────┘  │
│             │  build()                │              │
└─────────────┼─────────────────────────┼──────────────┘
              │                         │
              │  (if persist_dir set)   │
              ▼                         ▼
        ┌─────────────────────────────────────┐
        │  Disk  (/persist_dir/)              │
        │  meta.bin  vectors.bin  norms_sq.bin│
        │  payloads.bin  offsets.bin          │
        └──────────────┬──────────────────────┘
                       │  RingDb::load(dir, backend)
                       ▼
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

```
pub trait RingComputeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn upload_f32_dataset(&mut self, dims: usize, vectors: Vec<f32>, norms_sq: Vec<f32>) -> Result<()>;
    fn ring_query_f32(&self, dims: usize, query: &[f32], d_min: f32, d_max: f32) -> Result<Vec<u32>>;
}
```

All three public query methods (`query`, `query_range`, `query_disk`) translate
their respective inputs into a `(d_min, d_max)` pair and delegate to this single
backend method.

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
- [x] Persistent on-disk format (`build()` writes, `RingDb::load()` rehydrates)
- [ ] Python bindings (PyO3)

---

## License

MIT OR Apache-2.0
