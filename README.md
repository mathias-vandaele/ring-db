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
- [Payload strategies](#payload-strategies)
- [API overview](#api-overview)
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
ring-db = "0.4"
ring-db-derive = "0.4"
serde = { version = "1", features = ["derive"] }   # only needed for serde payloads
bytemuck = { version = "1", features = ["derive"] } # only needed for pod payloads
```

### Ring query — no payload

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

// result.hits is Vec<Hit> — each Hit carries the id and squared distance
for hit in &result.hits {
    println!("id={} dist={:.3}", hit.id, hit.dist_sq.sqrt());
}
println!("backend: {}", result.backend_used); // "cpu"
println!("elapsed: {:?}", result.elapsed);

// ids() is a convenience method when you need a plain Vec<u32>
let ids = result.ids(); // → vec![1, 2]
```

### Range query — explicit [d_min, d_max] band

```rust
use ringdb::{RingDb, RingDbConfig, RangeQuery};

let mut db = RingDb::new(RingDbConfig::new(4)).unwrap();
db.add_vector(&[1.0f32, 0.0, 0.0, 0.0], ()).unwrap();
db.add_vector(&[3.0, 4.0, 0.0, 0.0], ()).unwrap();
let db = db.build().unwrap();

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

let result = db.query_disk(&DiskQuery {
    query: &[0.0f32; 4],
    d_max: 5.0,
}).unwrap();
```

---

## Payload strategies

ring-db ships a `#[derive(Payload)]` macro (from the `ring-db-derive` crate,
re-exported as `ringdb::Payload`) that wires a user type to its storage
strategy at compile time — **no runtime dispatch, no trait objects**.

Two strategies are available:

| Strategy | Attribute | Storage | Fetch | Best for |
|---|---|---|---|---|
| **Serde** (default) | *(none)* | bincode, variable-size | `T` (owned) | Strings, enums, dynamic data |
| **Pod** | `#[payload(storage = "pod")]` | raw bytes, fixed-size | `&T` zero-copy | `repr(C)` structs, numeric records |

### Serde payload — flexible, variable-size

```rust
use ringdb::{RingDb, RingDbConfig, RingQuery, Payload};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Payload)]
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

let db = db.build().unwrap();
let result = db.query(&RingQuery { query: &[0.8f32, 0.0, 0.0, 0.0], d: 0.2, lambda: 0.1 }).unwrap();

// Fetches payloads by deserializing from mmap via bincode
let docs = db.fetch_payloads(&result.ids()).unwrap();
for doc in &docs {
    println!("{} (score={})", doc.title, doc.score);
}
```

### Pod payload — zero-copy, maximum retrieval performance

Use `#[payload(storage = "pod")]` when your payload is a **fixed-size
plain-old-data struct** (`#[repr(C)]` + `bytemuck::Pod`). The library stores
raw bytes back-to-back — no offset table — and returns a `&T` pointing
directly into the mmap. **No allocation, no deserialization, O(1).**

> **When to choose Pod:** if retrieval latency is critical (real-time serving,
> large hit counts, tight SLA), the Pod strategy is the right choice for
> numeric or fixed-size payloads. On 30 M vectors with ~164 K hits, Pod
> payload fetch is **~288× faster** than Serde (92 µs vs 26 ms — see
> [Performance](#performance)).

```rust
use ringdb::{RingDb, RingDbConfig, RingQuery, Payload};
use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Pod, Zeroable, Payload)]
#[repr(C)]
#[payload(storage = "pod")]
struct GeoPoint {
    lat: f32,
    lon: f32,
    altitude: f32,
}

let mut db: RingDb<GeoPoint> = RingDb::new(RingDbConfig::new(3)).unwrap();

db.add_vector(&[48.8566f32, 2.3522, 35.0], GeoPoint { lat: 48.8566, lon: 2.3522, altitude: 35.0 }).unwrap();
db.add_vector(&[51.5074f32, -0.1278, 11.0], GeoPoint { lat: 51.5074, lon: -0.1278, altitude: 11.0 }).unwrap();

let db = db.build().unwrap();
let result = db.query(&RingQuery { query: &[48.0f32, 2.0, 30.0], d: 1.5, lambda: 0.5 }).unwrap();

// Zero-copy: &GeoPoint points directly into the mmap — no heap allocation
let points: Vec<&GeoPoint> = db.fetch_pods(&result.ids());
for pt in points {
    println!("lat={} lon={} alt={}", pt.lat, pt.lon, pt.altitude);
}
```

**Requirements for Pod storage:**

- The type must be `#[repr(C)]`
- Must derive `bytemuck::Pod` + `bytemuck::Zeroable`
- Must be `Copy` (fixed-size by definition)
- No heap-allocated fields (no `String`, `Vec`, etc.)

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
| `fetch_payload(id)` | Deserialize the payload for a single ID (serde or pod). |
| `fetch_payloads(ids)` | Deserialize payloads for a slice of IDs, in order. |
| `fetch_pod(id)` | **Pod only** — zero-copy `&T` reference into the mmap for a single ID. |
| `fetch_pods(ids)` | **Pod only** — zero-copy `Vec<&T>` references for a slice of IDs. |
| `len()` / `is_empty()` / `dims()` | Introspection. |

> `fetch_pod` and `fetch_pods` are statically gated: they only exist when
> `T::Store: RefPayloadStore<T>`, which is only true for
> `#[payload(storage = "pod")]` types. Calling them on a Serde type is a
> **compile error**, not a runtime panic.

### `#[derive(Payload)]`

```rust
use ringdb::Payload;

// Default: serde storage — requires Serialize + DeserializeOwned
#[derive(serde::Serialize, serde::Deserialize, Payload)]
struct MyDoc { title: String }

// Pod storage — requires repr(C) + bytemuck::Pod
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Payload)]
#[repr(C)]
#[payload(storage = "pod")]
struct MyRecord { x: f32, y: f32 }
```

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
result.hits          // Vec<Hit>        — all matches with their squared distances
result.backend_used  // &'static str    — e.g. "cpu"
result.elapsed       // Duration        — wall-clock query time

result.ids()         // Vec<u32>        — convenience: just the IDs, for fetch_payloads / fetch_pods
```

### `Hit`

Each entry in `result.hits` is a `Hit`:

```
hit.id               // u32  — insertion-order ID of the matching vector
hit.dist_sq          // f32  — squared Euclidean distance to the query
                     //        call hit.dist_sq.sqrt() for the actual distance
```

`Hit` is 8 bytes (two `f32`-sized fields, no padding) and derives `Debug`, `Clone`, `Copy`, and `PartialEq`.

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

## Payload storage internals

Payloads are designed to consume **zero hot memory** after the build phase.

### Serde storage

1. **Build phase** — each call to `add_vector` serializes the payload
   immediately (via `bincode`) and streams it to a temporary file.
2. **Query phase** — after `build()`, the temp file is mmap-ed read-only.
   The OS can page payload bytes out under memory pressure. The only
   hot-path data is the offset table (8 bytes × number of vectors).
3. **Fetch** — `fetch_payloads(ids)` slices the mmap by offset and runs
   `bincode::deserialize` per hit. O(hits), not O(dataset).

### Pod storage

1. **Build phase** — each payload is written as raw bytes
   (`bytemuck::bytes_of`) back-to-back, with no offset table.
2. **Query phase** — the file is mmap-ed read-only. No offset table is kept
   in memory at all: position = `id × size_of::<T>()`.
3. **Fetch** — `fetch_pods(ids)` computes the offset arithmetically and calls
   `bytemuck::from_bytes` — **zero copies, zero allocations**. The returned
   `&T` borrows directly from the mmap.

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
| `payloads.bin` | Serde: bincode bytes · Pod: raw `T` bytes |
| `offsets.bin` | Serde only: byte offsets (`u64`) into `payloads.bin` |

### Loading

Pass the directory and your chosen backend to `RingDb::load()`. The correct
storage variant (`SerdeStore` or `PodStore`) is selected automatically from `T`:

```rust
use ringdb::{RingDb, BackendPreference, RingQuery};
use std::path::Path;

let db = RingDb::<()>::load(
    Path::new("/tmp/mydb"),
    BackendPreference::Cpu,
).unwrap();

let result = db.query(&RingQuery {
    query: &[1.0f32, 0.0, 0.0, 0.0],
    d: 1.0,
    lambda: 0.1,
}).unwrap();
```

`load()` validates the file sizes against the stored metadata and returns
`RingDbError::Corrupt` if anything is inconsistent.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  RingDb<T: Payload>  (builder, mutable)              │
│                                                      │
│  ┌─────────────────────┐   ┌──────────────────────┐  │
│  │  vectors: Vec<f32>  │   │  T::Builder          │  │
│  │  norms_sq: Vec<f32> │   │  SerdeStoreBuilder   │  │
│  └──────────┬──────────┘   │  or PodStoreBuilder  │  │
│             │  build()     └──────────┬───────────┘  │
└─────────────┼─────────────────────────┼──────────────┘
              │                         │
              │  (if persist_dir set)   │
              ▼                         ▼
        ┌─────────────────────────────────────┐
        │  Disk  (/persist_dir/)              │
        │  meta.bin  vectors.bin  norms_sq.bin│
        │  payloads.bin  [offsets.bin]        │
        └──────────────┬──────────────────────┘
                       │  RingDb::load(dir, backend)
                       ▼
┌──────────────────────────────────────────────────────┐
│  SealedRingDb<T>  (immutable, Send + Sync)           │
│                                                      │
│  ┌─────────────────────────┐  ┌────────────────────┐ │
│  │  RingComputeBackend     │  │  T::Store          │ │
│  │  (trait object)         │  │  SerdeStore<T>     │ │
│  │                         │  │  or PodStore<T>    │ │
│  │  ┌─────────────────┐    │  │  (mmap, page-able) │ │
│  │  │  CpuBackend     │    │  └────────────────────┘ │
│  │  │  Rayon parallel │    │                         │
│  │  │  4-acc dot prod │    │                         │
│  │  └─────────────────┘    │                         │
│  └─────────────────────────┘                         │
└──────────────────────────────────────────────────────┘
```

### Payload trait & derive

The `Payload` trait ties a user type to its concrete storage strategy at
compile time. It is implemented via `#[derive(Payload)]`:

```rust
pub trait Payload: Sized {
    type Store: OwnedPayloadStore<Self>;
    type Builder: PayloadBuilderOps<Self, Store = Self::Store>;

    fn make_builder() -> Result<Self::Builder>;
    fn load_store(dir: &Path) -> Result<Self::Store>;
}
```

`RingDb<T>` holds `T::Builder` and `SealedRingDb<T>` holds `T::Store` as
**concrete fields** — no `Box<dyn>`, no heap indirection on the payload path.
The compiler resolves the entire storage path at monomorphization time.

### Backend trait

The `RingComputeBackend` trait separates the upload step from the query step,
reflecting the intended usage: upload once, then run many queries against
resident data without re-uploading.

```
pub trait RingComputeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn upload_f32_dataset(&mut self, dims: usize, vectors: Vec<f32>, norms_sq: Vec<f32>) -> Result<()>;

    // Ring / range search: returns all vectors with dist in [d_min, d_max]
    fn ring_query_f32(&self, dims: usize, query: &[f32], d_min: f32, d_max: f32) -> Result<Vec<Hit>>;

    // Disk (ball) search: returns all vectors with dist in [0, d_max]
    // Default impl delegates to ring_query_f32; backends can override for a
    // tighter loop that skips the lower-bound comparison entirely.
    fn disk_query_f32(&self, dims: usize, query: &[f32], d_max: f32) -> Result<Vec<Hit>> { … }
}
```

`query` and `query_range` delegate to `ring_query_f32`. `query_disk` delegates
to `disk_query_f32`, which the CPU backend overrides to remove the
`dist_sq >= lower_sq` branch — a small but measurable saving when scanning
hundreds of millions of vectors.

New backends (WGPU, CUDA, AVX-512 hand-rolled) can be added without touching
the public API. Backends that do not override `disk_query_f32` get the
ring-delegating default for free.

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
- **Dedicated disk path** — `disk_query_f32` removes the lower-bound
  comparison (`dist_sq >= 0`) that is always true, giving the branch predictor
  one fewer condition to evaluate per vector.

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

### Payload fetch — Serde vs Pod (100-byte string payload, ~164 K hits)

Ring: `[6.529, 6.535]` · 164 298 hits out of 30 M vectors · 64 dimensions.

| Strategy | Fetch time | Notes |
|---|---:|---|
| **Serde** (`bincode`) | **~26.6 ms** | Variable-size, heap-allocates per hit |
| **Pod** (zero-copy mmap) | **~92 µs** | Fixed-size, `&T` from mmap, no alloc |

**Pod is ~288× faster** for payload retrieval when the payload is a
fixed-size `repr(C)` struct. Use `#[payload(storage = "pod")]` whenever
maximum retrieval throughput is required.

```
payload_fetch_dynamic/100B_string_payload/64
                        time:   [25.959 ms 26.630 ms 27.608 ms]

payload_fetch_static/100B_string_payload/64
                        time:   [92.162 µs 92.298 µs 92.443 µs]
```

Both are O(hits), not O(dataset), because payloads are addressed directly
by position in the mmap.

---

## Running the benchmarks

```bash
# Full benchmark suite (HTML report in target/criterion/)
cargo bench

# Single group
cargo bench --bench query_backends cpu_f32
cargo bench --bench query_backends cpu_disk_f32
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
- [x] `#[derive(Payload)]` macro — serde and pod storage strategies
- [x] Zero-copy Pod payload fetch (`fetch_pod` / `fetch_pods`)
- [ ] Python bindings (PyO3)

---

## License

MIT OR Apache-2.0
