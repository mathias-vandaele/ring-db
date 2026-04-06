# AGENT.md — ring-db

> Quick-start context for AI agents. Read this before touching any code.

## What is ring-db?

**ring-db** is a Rust library (published as `ringdb` on crates.io) for **ring queries** in high-dimensional vector spaces. Unlike nearest-neighbor databases, it retrieves all vectors whose Euclidean distance to a query falls within a specified distance interval `[d_min, d_max]`.

Use cases: anomaly detection, content deduplication, recommendation diversity, spatial shells (GPS bands), audio fingerprinting, security threat detection.

- Version: 0.2.0
- Rust Edition: 2024
- License: MIT OR Apache-2.0
- Repo: https://github.com/mathias-vandaele/ring-db

---

## Directory Structure

```
ring-db/
├── src/
│   ├── lib.rs             # Public API root, re-exports
│   ├── engine.rs          # RingDb (builder) and SealedRingDb (immutable DB)
│   ├── query.rs           # RingQuery, RangeQuery, DiskQuery, QueryResult
│   ├── config.rs          # RingDbConfig and BackendPreference
│   ├── error.rs           # RingDbError enum and Result alias
│   ├── payload.rs         # PayloadStoreBuilder + PayloadStore (mmap cold storage)
│   └── backend/
│       ├── mod.rs         # RingComputeBackend trait
│       └── cpu.rs         # CpuBackend: brute-force, Rayon-parallel, SIMD-friendly
├── examples/
│   └── cli.rs             # Standalone CLI: random dataset + ring query demo
├── benches/
│   └── query_backends.rs  # Criterion.rs benchmarks (30M vector scale)
├── tests/
│   ├── correctness.rs     # Hand-crafted vectors, analytically verified
│   └── random.rs          # Randomized sanity tests
├── .github/workflows/
│   ├── rust.yml           # CI: build + test on push/PR to master
│   └── publish.yml        # Publish to crates.io on v*.*.* tags
├── release_tag.sh         # Release automation (bump, fmt, clippy, tag, push)
└── Cargo.toml
```

---

## Key Source Files

### [src/engine.rs](src/engine.rs)

Two main public types:

**`RingDb<T>`** — mutable builder (generic over payload type `T`, default `()`)
- Holds staging buffers: `vectors` (f32 flat row-major), `norms_sq` (precomputed)
- Streams payloads to a temp file immediately (no RAM accumulation)
- Methods: `new()`, `add_vector()`, `build()`, `len()`, `is_empty()`, `dims()`, `backend_name()`

**`SealedRingDb<T>`** — immutable, `Send + Sync`
- Owns the backend's hot data (vectors + norms)
- Owns the cold payload store (memory-mapped temp file)
- Methods: `query()`, `query_range()`, `query_disk()`, `fetch_payload()`, `fetch_payloads()`

### [src/query.rs](src/query.rs)

All query types take a borrowed `query: &[f32]`:

| Type | Semantics |
|------|-----------|
| `RingQuery` | Symmetric band `[d − λ, d + λ]` |
| `RangeQuery` | Explicit `[d_min, d_max]` interval |
| `DiskQuery` | Full ball `[0, d_max]` |

All three normalize to `(d_min, d_max)` and delegate to `backend.ring_query_f32()`.

`QueryResult` — `ids: Vec<u32>`, `backend_used: &'static str`, `elapsed: Duration`

### [src/backend/cpu.rs](src/backend/cpu.rs)

The only available backend. Key optimizations:
1. **Four-accumulator dot product** — breaks dependency chains so LLVM emits `fmla.4s` / `vfmadd231ps`; ~4× FP throughput
2. **Precomputed `‖x‖²`** — inserted at index time, only dot product computed at query time
3. **Squared bounds** — `d_min²` and `d_max²` computed once before the loop (no `sqrt`)
4. **Rayon `par_chunks_exact(dims)`** — parallelizes the scan across all logical cores

Distance formula: `dist_sq = norm_sq_i + norm_sq_q − 2·dot(row, query)`

### [src/payload.rs](src/payload.rs)

**`PayloadStoreBuilder<T>`** (write phase):
- Creates `ringdb-payloads-{pid}-{counter}.bin` in `temp_dir()`
- `push(payload)` serializes via bincode and streams to `BufWriter` immediately
- Maintains `offsets: Vec<u64>` (8 bytes per vector, always hot)

**`PayloadStore<T>`** (read phase):
- Memory-maps the temp file read-only after build
- Payloads paged by OS; only the offset table stays resident
- Auto-cleanup on drop (closes mmap, deletes temp file)

### [src/error.rs](src/error.rs)

```rust
pub enum RingDbError {
    DimensionMismatch { expected: usize, got: usize },
    Payload(String),        // bincode error
    Io(std::io::Error),     // temp file I/O
}
```

---

## Architecture Decisions

| Decision | Reason |
|----------|--------|
| Two-phase builder (`RingDb` → `SealedRingDb`) | Upload once, query many; type-safe separation of concerns |
| Cold payload storage via mmap | Scales beyond available RAM; OS pages out under pressure |
| No `sqrt` in distance computation | Preserves ordering; avoids expensive operation; better SIMD scheduling |
| Backend trait abstraction | Add GPU/CUDA/AVX-512 backends without changing public API |
| Payload streamed to disk immediately | O(1) peak memory during build regardless of `T` size |

---

## Build and Test Commands

```bash
# Build
cargo build
cargo build --release
RUSTFLAGS="-C target-cpu=native" cargo build --release  # native SIMD

# Test
cargo test
cargo test --test correctness
cargo test --test random
cargo test -- --nocapture

# Benchmark (recommended with native CPU features)
RUSTFLAGS="-C target-cpu=native" cargo bench
cargo bench --bench query_backends cpu_f32

# Example CLI
cargo run --release --example cli -- --n 100000 --dims 64 --d 5.0 --lambda 0.2
cargo run --example cli -- --help

# Docs
cargo doc --open

# Release (automated)
./release_tag.sh <version>          # formats, clippy, bumps, tags, pushes
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `rayon` | Data parallelism (work-stealing) |
| `thiserror` | Error derive macro |
| `memmap2` | Memory-mapped file I/O |
| `bincode` | Binary serialization for payloads |
| `serde` | Serialization framework |
| `criterion` (dev) | Statistical benchmarking |
| `rand` (dev) | Deterministic random generation |
| `clap` (dev) | CLI argument parsing (example only) |

---

## Data Flow Cheatsheet

### Insert
```
add_vector(&[f32], T)
  → [dim check]
  → [compute norm_sq]
  → [extend vectors staging buffer]
  → [serialize T → temp file]
```

### Build
```
db.build()
  → backend.upload_f32_dataset(dims, vectors, norms_sq)
  → payload_builder.finish() → mmap
  → SealedRingDb { backend, payload_store }
```

### Query
```
db.query(&RingQuery { query, d, lambda })
  → normalize to (d_min, d_max)
  → backend.ring_query_f32(dims, query, d_min, d_max)
      → compute norm_sq_q
      → square bounds: lower_sq, upper_sq
      → par_chunks_exact(dims):
          dist_sq = norm_sq_i + norm_sq_q - 2·dot(row_i, query)
          if lower_sq ≤ dist_sq ≤ upper_sq → include id i
  → QueryResult { ids, backend_used, elapsed }
```

### Payload fetch
```
db.fetch_payloads(&[id])
  → for each id: offsets[id]..offsets[id+1] → mmap slice → bincode::deserialize
  → Vec<T>
```

---

## Public API Quick Reference

```rust
use ringdb::{RingDb, RingDbConfig, RingQuery, RangeQuery, DiskQuery};

// Build
let mut db: RingDb<MyPayload> = RingDb::new(RingDbConfig::new(dims))?;
db.add_vector(&vec_f32, my_payload)?;
let db = db.build()?;  // → SealedRingDb<MyPayload>

// Query variants
let r = db.query(&RingQuery { query: &q, d: 5.0, lambda: 0.5 })?;
let r = db.query_range(&RangeQuery { query: &q, d_min: 4.5, d_max: 5.5 })?;
let r = db.query_disk(&DiskQuery { query: &q, d_max: 5.5 })?;

// Fetch payloads
let payloads: Vec<MyPayload> = db.fetch_payloads(&r.ids)?;
```

`T` defaults to `()` (no payload). Any `T: Serialize + DeserializeOwned` works.

---

## CI/CD

- **Push/PR to `master`** → `rust.yml`: `cargo build` + `cargo test`
- **Tag `v*.*.*`** → `publish.yml`: full test, dry-run publish, publish to crates.io, create GitHub release

---