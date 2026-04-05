use std::time::Duration;

/// A ring query: find all vectors whose Euclidean distance to `query`
/// lies within `[d - lambda, d + lambda]`.
///
/// Internally, ringdb uses squared L2 distances to avoid computing square
/// roots. The ring bounds become:
///
/// ```text
/// lower_sq = max(0, d - lambda)²
/// upper_sq = (d + lambda)²
/// ```
pub struct RingQuery<'a> {
    /// The query vector. Must have length equal to `RingDb::dims()`.
    pub query: &'a [f32],
    /// Target distance (centre of the ring).
    pub d: f32,
    /// Half-width of the ring.
    pub lambda: f32,
}

/// Result of a ring query.
pub struct QueryResult {
    /// IDs of all vectors whose distance to the query falls within
    /// `[d - lambda, d + lambda]`. IDs correspond to insertion order
    /// (first inserted vector has ID 0).
    pub ids: Vec<u32>,
    /// Name of the backend that executed the query (e.g. `"cpu"`, `"wgpu"`, `"cuda"`).
    pub backend_used: &'static str,
    /// Wall-clock time for the query (excluding dataset upload).
    pub elapsed: Duration,
}
