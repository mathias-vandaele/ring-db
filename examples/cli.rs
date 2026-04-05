use clap::Parser;
use rand::{Rng, SeedableRng};
use ringdb::{BackendPreference, QueryResult, QuantizationMode, RingDb, RingDbConfig, RingQuery};

/// ringdb CLI — generate a random dataset and run a ring query.
///
/// Examples:
///
///   cargo run --example cli -- --n 100000 --dims 64 --backend auto --quant none --d 5.0 --lambda 0.2
///
///   cargo run --example cli -- --n 100000 --dims 64 --backend auto --quant q8 --d 5.0 --lambda 0.2
#[derive(Parser)]
#[command(name = "ringdb-cli", about = "ringdb demo: random dataset ring query")]
struct Cli {
    /// Number of random vectors to generate.
    #[arg(short, long, default_value_t = 10_000)]
    n: usize,

    /// Number of dimensions per vector.
    #[arg(long, default_value_t = 128)]
    dims: usize,

    /// Target distance (centre of ring).
    #[arg(short = 'd', long, default_value_t = 5.0)]
    d: f32,

    /// Half-width of the ring (λ).
    #[arg(short = 'l', long, default_value_t = 0.5)]
    lambda: f32,

    /// Backend: auto | cpu | wgpu | cuda
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Quantization mode: none | q8
    #[arg(long, default_value = "none")]
    quant: String,

    /// Random seed for reproducibility.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let cli = Cli::parse();

    // ---- parse backend preference ----
    let backend_preference = match cli.backend.to_lowercase().as_str() {
        "cpu" => BackendPreference::Cpu,
        "wgpu" => BackendPreference::Wgpu,
        "cuda" => BackendPreference::Cuda,
        _ => BackendPreference::Auto,
    };

    // ---- parse quantization mode ----
    let quantization = match cli.quant.to_lowercase().as_str() {
        "q8" => QuantizationMode::Q8,
        _ => QuantizationMode::None,
    };

    let config = RingDbConfig {
        dims: cli.dims,
        backend_preference,
        quantization,
    };

    println!("ringdb demo");
    println!("  dataset : {} vectors × {} dims", cli.n, cli.dims);
    println!("  quant   : {}", cli.quant);
    println!("  ring    : d={}, λ={}", cli.d, cli.lambda);
    println!();

    // ---- generate random vectors ----
    let mut rng = rand::rngs::SmallRng::seed_from_u64(cli.seed);
    let total_floats = cli.n * cli.dims;
    let vectors: Vec<f32> = (0..total_floats).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    // ---- build the database ----
    print!("Building database … ");
    let build_start = std::time::Instant::now();
    let mut db = RingDb::new(config).expect("failed to create RingDb");
    db.add_vectors(&vectors).expect("failed to add vectors");
    let build_elapsed = build_start.elapsed();
    println!("done in {:.2?}", build_elapsed);
    println!("  backend : {}", db.backend_name());
    println!();

    // ---- run a ring query (use the first vector as query) ----
    let query_vec = &vectors[..cli.dims];
    let rq = RingQuery {
        query: query_vec,
        d: cli.d,
        lambda: cli.lambda,
    };

    print!("Running ring query … ");
    let QueryResult {
        ids,
        backend_used,
        elapsed,
    } = db.query(&rq).expect("query failed");

    println!("done");
    println!();
    println!("Results:");
    println!("  backend      : {backend_used}");
    println!("  hits         : {}", ids.len());
    println!("  query time   : {elapsed:.2?}");

    if ids.len() <= 20 {
        println!("  hit IDs      : {ids:?}");
    } else {
        println!("  hit IDs      : [{}, {}, … {} total]", ids[0], ids[1], ids.len());
    }
}
