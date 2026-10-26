//! Token and wide-hidden routes for a native Qwen4x proposal wave.
//!
//! For module m, continuation length V, Main start p, and reusable prefix r:
//!   r = min(V,m), or zero when a known token replaces a cached draft.
//!   start = p - (m-r)
//!   tokens = continuation[r..] + sampled + drafts[..m]
//!   end = p + V + 1, for every module.
//! Hidden routes read old boundary rows before replacing the retained tail.
use inference_executor_core::model::qwen::v3_5::Qwen35DecodeDecision;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::sampling::SamplerConfig;

#[derive(Clone, Default)]
pub struct History {
    pub pending_token_index: u32,
    pub token_ids: Vec<i32>,
    pub has_hidden: bool,
}

pub struct Request {
    pub req_slot: u32,
    pub block_index: usize,
    pub main_hidden_flat_start: usize,
    pub pending_token_index: u32,
    pub next_pending_token_index: u32,
    pub sampler_config: SamplerConfig,
    repair_tail: bool,
    pub kind: Kind,
}
pub enum Kind {
    Prefill(Vec<Vec<i32>>),
    Decode {
        continuation_token_ids: Vec<i32>,
        sampled_token_id: i32,
        decision_index: usize,
        proposal_sample_position: u32,
    },
}
pub struct Batch {
    pub microbatch: Qwen35Microbatch,
    pub hidden_routes: Vec<u32>,
    pub write_routes: Vec<u32>,
    pub distribution_indices: Vec<u32>,
    pub sample_slots: Vec<u32>,
    pub sample_positions: Vec<u32>,
    pub sampler_configs: Vec<SamplerConfig>,
}

pub fn requests(
    batch: &Qwen35Microbatch,
    decisions: &[Qwen35DecodeDecision],
    history: &[History],
    depth: usize,
) -> Vec<Request> {
    let mut decision_index = 0;
    (0..batch.num_reqs())
        .map(|i| {
            let slot = batch.req_slots()[i];
            let position = batch.token_indices()[i];
            let start = batch.cu_tokens()[i] as usize;
            let end = batch.cu_tokens()[i + 1] as usize;
            let input = &batch.flat_token_ids()[start..end];
            let old = &history[slot as usize];
            let known: Vec<_> = if batch.is_decode_req(i) {
                input.iter().copied().take(depth).collect()
            } else {
                input
                    .iter()
                    .copied()
                    .chain((1..=depth).map(|lane| *batch.token_ids_for_lane(i, lane).last().unwrap()))
                    .take(depth)
                    .collect()
            };
            if !old.token_ids.is_empty() {
                debug_assert_eq!(old.pending_token_index, position);
                debug_assert_eq!(old.token_ids[0], known[0]);
            }
            let changed = old.token_ids.iter().zip(&known).any(|(a, b)| a != b);
            debug_assert!(old.has_hidden || !changed, "canonical MTP lookahead must remain stable");
            let repair = old.has_hidden && changed;
            let (kind, next_position) = if batch.is_decode_req(i) {
                let decision = &decisions[decision_index];
                let fixed = input.len() - batch.num_spec_tokens(i) as usize;
                debug_assert!(fixed > 0);
                debug_assert!(old.has_hidden || fixed >= depth);
                let continuation: Vec<_> = input[1..fixed]
                    .iter()
                    .copied()
                    .chain(decision.validated_tokens.iter().map(|&t| t as i32))
                    .collect();
                let next_position = position + fixed as u32 + decision.validated_tokens.len() as u32;
                let kind = Kind::Decode {
                    continuation_token_ids: continuation,
                    sampled_token_id: decision.sampled_token as i32,
                    decision_index,
                    proposal_sample_position: next_position + 1,
                };
                decision_index += 1;
                (kind, next_position)
            } else {
                let lanes = (1..=depth)
                    .map(|lane| {
                        let mut tokens = batch.token_ids_for_lane(i, lane).to_vec();
                        if repair {
                            tokens.splice(0..0, known[1..lane].iter().copied());
                        }
                        tokens
                    })
                    .collect();
                (Kind::Prefill(lanes), position + input.len() as u32)
            };
            Request {
                req_slot: slot,
                block_index: batch.block_indices()[i],
                main_hidden_flat_start: start,
                pending_token_index: position,
                next_pending_token_index: next_position,
                sampler_config: batch.sampler_configs()[i],
                repair_tail: repair,
                kind,
            }
        })
        .collect()
}

fn reused(request: &Request, module: usize) -> usize {
    if request.repair_tail {
        return 0;
    }
    match &request.kind {
        Kind::Prefill(_) => module,
        Kind::Decode {
            continuation_token_ids: continuation,
            ..
        } => continuation.len().min(module),
    }
}
fn count(request: &Request, module: usize) -> usize {
    match &request.kind {
        Kind::Prefill(lanes) => lanes[module].len(),
        Kind::Decode {
            continuation_token_ids: continuation,
            ..
        } => continuation.len() + 1 + module - reused(request, module),
    }
}
fn cache_base(max_requests: usize, slot: u32, module: usize) -> usize {
    max_requests * module * (module + 1) / 2 + slot as usize * (module + 1)
}

pub fn batch(requests: &[Request], module: usize, depth: usize, max_requests: usize, drafts: &[Vec<i32>]) -> Batch {
    let mut tokens = Vec::new();
    let mut cu = vec![0];
    let mut positions = Vec::new();
    let mut masks = Vec::new();
    let mut hidden_routes = Vec::new();
    let mut write_routes = Vec::new();
    let mut distribution_indices = Vec::new();
    let mut sample_slots = Vec::new();
    let mut sample_positions = Vec::new();
    let mut sample_configs = Vec::new();
    let mut previous_start = 0;
    for request in requests {
        let r = reused(request, module);
        let position = request.pending_token_index - (module - r) as u32;
        positions.push(position);
        let begin = tokens.len();
        match &request.kind {
            Kind::Prefill(lanes) => tokens.extend_from_slice(&lanes[module]),
            Kind::Decode {
                continuation_token_ids: continuation,
                sampled_token_id: sampled,
                decision_index: decision,
                proposal_sample_position: sample_position,
            } => {
                tokens.extend_from_slice(&continuation[r..]);
                tokens.push(*sampled);
                tokens.extend(drafts.iter().take(module).map(|step| step[*decision]));
                distribution_indices.push((request.req_slot as usize * depth + module) as u32);
                sample_slots.push(request.req_slot);
                sample_positions.push(*sample_position + module as u32);
                sample_configs.push(request.sampler_config);
            },
        }
        let n = tokens.len() - begin;
        debug_assert_eq!(n, count(request, module));
        for j in 0..n {
            let (source, row) = if module == 0 {
                (0, request.main_hidden_flat_start + j)
            } else if module > r && j == 0 {
                (2, cache_base(max_requests, request.req_slot, module - 1) + r)
            } else {
                (1, previous_start + j - usize::from(module > r))
            };
            hidden_routes.extend_from_slice(&[source, row as u32]);
        }
        if module > 0 {
            let previous_count = count(request, module - 1);
            if matches!(request.kind, Kind::Decode { .. }) {
                let destination = cache_base(max_requests, request.req_slot, module - 1);
                for j in 0..module {
                    write_routes.extend_from_slice(&[
                        (previous_start + previous_count - module + j) as u32,
                        (destination + j) as u32,
                    ]);
                }
            }
            previous_start += previous_count;
        }
        masks.extend(std::iter::repeat_n(false, n));
        if matches!(request.kind, Kind::Decode { .. }) {
            *masks.last_mut().unwrap() = true;
        }
        cu.push(tokens.len() as u32);
    }
    Batch {
        microbatch: Qwen35Microbatch::new(
            requests.iter().map(|r| r.req_slot).collect(),
            requests.iter().map(|r| r.block_index).collect(),
            positions,
            tokens,
            cu,
            vec![Vec::new(); requests.len()],
            requests.iter().map(|r| r.sampler_config).collect(),
            masks,
        ),
        hidden_routes,
        write_routes,
        distribution_indices,
        sample_slots,
        sample_positions,
        sampler_configs: sample_configs,
    }
}
