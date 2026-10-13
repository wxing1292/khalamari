// Empirical operator criteria. These coefficients do not claim compiler register limits or residency.
#[derive(Clone, Copy, Deserialize, Serialize)]
struct Policy {
    #[serde(rename = "short_rows")]
    short_q_token_heads: u32,
    #[serde(rename = "row_elements")]
    q_token_head_values: u32,
    loader_tokens: u32,
    max_head_base_groups: u32,
    max_floor_base_groups: u32,
    target_split_groups: u32,
    upper_slice_shift: u32,
    #[serde(default)]
    decode: DecodePolicy,
}

// Profiles: D128/G<=8, D128/G>8, D256/G<=8, D256/G>8.
// Columns: rounded query capacities 1,2,4,8,16,32.
// Values: 0=small-query D128, 1=resource Q-token/head positions, 2=legacy QTokensPerHead, 3=single-query KV reuse.
const STRATEGIES: [[u8; 6]; 4] = [
    [3, 0, 1, 1, 1, 1],
    [3, 1, 1, 1, 1, 1],
    [3, 1, 1, 2, 2, 2],
    [3, 1, 1, 1, 1, 1],
];

// Fixed uses the earlier large-head partition calibration. New fits can select the shared grid criterion.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DecodePolicy {
    #[default]
    Fixed,
    Adaptive {
        min_grid_groups: u32,
        target_groups: u32,
    },
}

fn decode_slice(end: u32, kv_work: u128, target_groups: u32, max_slice: u32) -> u32 {
    let work = (kv_work / target_groups as u128).max(1);
    // The slice has a bounded power-of-two domain. Clamp before narrowing.
    let slice = 1_u128 << work.ilog2();
    slice.clamp(256, max_slice as u128).min(end as u128) as u32
}

// Prior M3 Max calibration at history 8192. The slice does not claim an optimum at other contexts.
const DECODE_REUSE_GRID: u128 = 8;
const DECODE_SLICE: u32 = 512;

// Counts describe positive causal requests after input validation.
// num_visible_kv_tokens sums request endpoints. It counts shared prefixes once per request.
#[derive(Clone, Copy)]
struct BatchShape {
    num_requests: u32,
    max_q_tokens: u32,
    max_visible_kv_tokens: u32,
    num_visible_kv_tokens: u128,
}

fn proposed_candidate(w: Workload, policy: Policy) -> Option<Candidate> {
    let end = w.history + w.queries;
    proposed_batch_candidate(
        w.dim,
        w.ratio,
        w.kv_heads,
        BatchShape {
            num_requests: w.requests,
            max_q_tokens: w.queries,
            max_visible_kv_tokens: end,
            num_visible_kv_tokens: end as u128 * w.requests as u128,
        },
        |tile| w.requests as u128 * w.queries.div_ceil(tile) as u128,
        |slice| w.requests as u128 * end.div_ceil(slice) as u128,
        policy,
    )
}

fn proposed_batch_candidate(
    dim: u32,
    ratio: u32,
    kv_heads: u32,
    batch: BatchShape,
    q_token_range_count: impl Fn(u32) -> u128,
    kv_split_count: impl Fn(u32) -> u128,
    policy: Policy,
) -> Option<Candidate> {
    let end = batch.max_visible_kv_tokens;
    let query_tile = batch.max_q_tokens.min(32).next_power_of_two();
    let stage: u32 = if dim == 128 { 64 } else { 16 };
    let index = usize::from(dim == 256) * 2 + usize::from(ratio > 8);
    let strategy = STRATEGIES[index][query_tile.ilog2() as usize];
    let (layout, tile, stage) = if end <= 32 || (query_tile > 1 && end <= 512) {
        let tile = query_tile.min(1 << (policy.short_q_token_heads / ratio).max(1).ilog2());
        (Layout::QTokensAndHeads, tile, 16)
    } else {
        match strategy {
            0 => {
                if end > 4096 {
                    (Layout::SlicedKv, 1, dim)
                } else {
                    (Layout::QHeadsPerToken, query_tile, 16)
                }
            },
            1 => {
                let q_token_heads = (policy.q_token_head_values / dim).max(8);
                let tile = query_tile.min(1 << (q_token_heads / ratio).max(1).ilog2());
                let joint = (tile * ratio).div_ceil(8);
                let head = tile * ratio.div_ceil(8);
                let layout = if head > joint && stage.div_ceil(joint) > policy.loader_tokens {
                    Layout::QHeadsPerToken
                } else {
                    Layout::QTokensAndHeads
                };
                (layout, tile, stage)
            },
            2 => (Layout::QTokensPerHead, 8, 16),
            3 => {
                // SlicedKv holds at most eight heads. QTokensAndHeads avoids repeated D256 KV reads
                // when a complete logical group needs two head tiles and the base grid is sufficient.
                let matrix_base_tasks = batch.num_requests as u128 * kv_heads as u128;
                let matrix_groups = ratio.div_ceil(8) as u128;
                let matrix_slice = match policy.decode {
                    DecodePolicy::Fixed => DECODE_SLICE.min(end),
                    DecodePolicy::Adaptive { target_groups, .. } => {
                        decode_slice(
                            end,
                            batch.num_visible_kv_tokens * kv_heads as u128 * matrix_groups,
                            target_groups,
                            512,
                        )
                    },
                };
                let enough_grid = match policy.decode {
                    DecodePolicy::Fixed => matrix_base_tasks >= DECODE_REUSE_GRID,
                    DecodePolicy::Adaptive { min_grid_groups, .. } => {
                        kv_split_count(matrix_slice) * kv_heads as u128 * matrix_groups >= min_grid_groups as u128
                    },
                };
                if dim == 256 && ratio > 8 && enough_grid {
                    (Layout::QTokensAndHeads, 1, 16)
                } else {
                    (Layout::SlicedKv, 1, dim)
                }
            },
            _ => unreachable!("invalid policy strategy"),
        }
    };
    let heads = if layout == Layout::SlicedKv { ratio.min(8) } else { ratio };
    let base_tasks = q_token_range_count(tile) * kv_heads as u128 * ratio.div_ceil(heads) as u128;
    let mut candidate = Candidate {
        layout,
        tile,
        heads,
        stage,
        slice: end,
        direct: false,
    };
    let joint = (tile * heads).div_ceil(8) as u128;
    let head = (tile * heads.div_ceil(8)) as u128;
    // Extra padded Q heads can improve the base grid before partitioning. This is a calibrated proxy.
    if end > 512 && layout == Layout::QTokensAndHeads && head > joint && base_tasks * head <= policy.max_head_base_groups as u128
    {
        candidate.layout = Layout::QHeadsPerToken;
    }
    let threads = candidate.threads();
    if threads > 1024 {
        return None;
    }
    if strategy == 3 && (end > 512 || query_tile == 1) {
        candidate.slice = match policy.decode {
            DecodePolicy::Fixed => DECODE_SLICE.min(end),
            DecodePolicy::Adaptive { target_groups, .. } => {
                let groups = if layout == Layout::SlicedKv { 4 } else { joint };
                let max_slice = if layout == Layout::SlicedKv { 1024 } else { 512 };
                decode_slice(
                    end,
                    batch.num_visible_kv_tokens * kv_heads as u128 * ratio.div_ceil(heads) as u128 * groups,
                    target_groups,
                    max_slice,
                )
            },
        };
    } else if end > 512 {
        // The input boundary validates token/head domains. The features fit u128.
        let groups = (threads / 32) as u128;
        let work = (end as u128).pow(2) * batch.num_requests as u128 * groups.pow(2);
        let upper = 1_u128
            << (34
                + if layout == Layout::SlicedKv {
                    0
                } else {
                    policy.upper_slice_shift
                });
        let slice = [(1_u128 << 26, 64), (1 << 30, 128), (upper, 256), (u128::MAX, 512)]
            .into_iter()
            .find(|&(limit, _)| work <= limit)
            .unwrap()
            .1;
        candidate.slice = slice;
        // A disabled floor has max_floor_base_groups=0. Positive validated base work cannot enter it.
        if layout != Layout::SlicedKv && base_tasks * groups <= policy.max_floor_base_groups as u128 {
            let work = end as u128 * base_tasks * groups;
            let target = policy.target_split_groups as u128;
            let floor = [
                (target * 64, 64),
                (target * 128, 128),
                (target * 256, 256),
                (u128::MAX, 512),
            ]
            .into_iter()
            .find(|&(limit, _)| work <= limit)
            .unwrap()
            .1;
            candidate.slice = candidate.slice.max(floor);
        }
        candidate.slice = candidate.slice.min(end);
    }
    candidate.direct = candidate.layout != Layout::SlicedKv && candidate.slice == end;
    Some(candidate)
}
