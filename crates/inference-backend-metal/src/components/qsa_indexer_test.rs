use half::bf16;
use inference_executor_core::attn::qsa_indexer::core::QSAIndexerCore;
use inference_executor_core::attn::qsa_indexer::reference::Precision;
use inference_executor_core::attn::qsa_indexer::reference::{self};

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;
use crate::test_support::ReplayTestCache;

const ACTIVE: ReplayParameterKey = ReplayParameterKey::new("test.qsa_indexer.num_active_tokens");

#[test]
fn test_prepare_select_bucketing() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    for (heads, dim, ratio, top_k, groups, dtype, zero_queries) in [
        (2, 7, 3, 3, 769, Dtype::Float32, false),
        (4, 128, 4, 512, 65537, Dtype::Bfloat16, false),
        (4, 128, 4, 512, 2051, Dtype::Bfloat16, true),
    ] {
        let config = Config {
            num_q_heads: heads,
            head_dim: dim,
            compress_ratio: ratio,
            top_k,
            rope_dim: if dim == 7 { 4 } else { 64 },
            rope_theta: 10_000_000.0,
            norm_eps: 1e-6,
            dtype,
        };
        let fixture = Fixture::new(&device, config, groups, zero_queries);
        let shape = Shape {
            num_total_tokens: 4,
            num_total_groups: groups,
        };
        let mut cache = ReplayTestCache::new();
        let (_, hit) = cache.record(shape, || {
            let mut builder = stream.create_replay_program();
            builder.record(fixture.compute.prepare(
                shape.num_total_tokens,
                fixture.layout,
                ReplayU32::Parameter(ACTIVE),
                fixture.prepare_buffers(),
            ));
            builder.record_with_barrier_before(fixture.compute.select(
                shape,
                fixture.layout,
                ReplayU32::Parameter(ACTIVE),
                fixture.select_buffers(),
                fixture.scratch(),
            ));
            builder.build()
        });
        assert!(!hit);
        for active in [1, 4, 3, 2] {
            let before_pages = values(&fixture.pages, dtype);
            let before_states = values(&fixture.states, dtype);
            let (replay, hit) = cache.record(shape, || unreachable!());
            assert!(hit);
            stream
                .submit_replay_with_arguments(replay, &ReplayArguments::new().with_u32(ACTIVE, active))
                .wait();
            fixture.check(active as usize, &before_pages, &before_states);
        }
        let smaller = Shape {
            num_total_tokens: 2,
            ..shape
        };
        assert!(
            !cache
                .record(smaller, || {
                    let mut builder = stream.create_replay_program();
                    builder.record(fixture.compute.prepare(
                        smaller.num_total_tokens,
                        fixture.layout,
                        ReplayU32::Fixed(2),
                        fixture.prepare_buffers(),
                    ));
                    builder.record_with_barrier_before(fixture.compute.select(
                        smaller,
                        fixture.layout,
                        ReplayU32::Fixed(2),
                        fixture.select_buffers(),
                        fixture.scratch(),
                    ));
                    builder.build()
                })
                .1
        );
    }
}

struct Fixture {
    config: Config,
    compute: Compute,
    layout: CacheLayout,
    metadata: Vec<Token>,
    page_ids_values: Vec<u32>,
    references: Vec<reference::Output>,
    qk: Buffer,
    tokens: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    pages: Buffer,
    page_ids: Buffer,
    states: Buffer,
    ids: Buffer,
    q: Buffer,
    scores: Buffer,
    candidates: [Buffer; 2],
}

impl Fixture {
    fn new(device: &Device, config: Config, groups: u32, zero_queries: bool) -> Self {
        let core = QSAIndexerCore {
            hidden_dim: 32,
            num_q_heads: config.num_q_heads as usize,
            head_dim: config.head_dim as usize,
            compress_ratio: config.compress_ratio as usize,
            top_k: config.top_k as usize,
            rope_dim: config.rope_dim as usize,
            rope_theta: config.rope_theta,
            norm_eps: config.norm_eps,
        };
        let precision = match config.dtype {
            Dtype::Float32 => Precision::F32,
            _ => Precision::Bf16,
        };
        let groups_per_page = if config.head_dim == 128 { 128 } else { 5 };
        let layout = CacheLayout {
            num_req_slots: 4,
            num_pages_per_request: groups.div_ceil(groups_per_page),
            num_state_slots: 8,
            page_bytes: groups_per_page * config.head_dim * config.dtype.item_size() as u32,
        };
        let num_pages = layout.num_req_slots * layout.num_pages_per_request;
        let page_ids_values: Vec<_> = (0..num_pages).rev().collect();
        let mut page_values = vec![0.125; num_pages as usize * groups_per_page as usize * core.head_dim];
        let mut state_values = vec![0.25; 8 * core.state_dim()];
        let mut qk_values = fixture_values(4 * core.qk_dim(), 17, config.dtype);
        if zero_queries {
            for row in qk_values.chunks_exact_mut(core.qk_dim()) {
                row[..core.num_q_heads * core.head_dim].fill(0.0);
            }
        }
        let q_norm_values: Vec<_> = fixture_values(core.head_dim, 19, config.dtype)
            .into_iter()
            .map(|x| x * 0.125)
            .collect();
        let k_norm_values: Vec<_> = fixture_values(core.head_dim, 23, config.dtype)
            .into_iter()
            .map(|x| x * 0.125)
            .collect();
        let positions = [
            1,
            config.compress_ratio * (config.top_k + 1) - 2,
            config.compress_ratio * (config.top_k + 1) - 1,
            config.compress_ratio * groups - 2,
        ];
        let mut metadata = Vec::new();
        let mut references = Vec::new();
        for (request, &position) in positions.iter().enumerate() {
            let raw_history = fixture_values(position as usize * core.head_dim, 101 + request as u64, config.dtype);
            let initial = reference::projected_indexer(reference::Input {
                core,
                precision,
                qk: &[],
                raw_history: &raw_history,
                group_start: 0,
                q_norm: &q_norm_values,
                k_norm: &k_norm_values,
            });
            for (group, row) in initial.keys.chunks_exact(core.head_dim).enumerate() {
                let page =
                    page_ids_values[request * layout.num_pages_per_request as usize + group / groups_per_page as usize];
                let offset =
                    (page as usize * groups_per_page as usize + group % groups_per_page as usize) * core.head_dim;
                page_values[offset..offset + core.head_dim].copy_from_slice(row);
            }
            let carry_start = position as usize / core.compress_ratio * core.compress_ratio * core.head_dim;
            let carry = &raw_history[carry_start..];
            state_values[request * core.state_dim()..request * core.state_dim() + carry.len()].copy_from_slice(carry);
            let group_start = if request == 3 { 7 } else { 0 };
            references.push(reference::projected_indexer(reference::Input {
                core,
                precision,
                qk: &qk_values[request * core.qk_dim()..(request + 1) * core.qk_dim()],
                raw_history: &raw_history,
                group_start,
                q_norm: &q_norm_values,
                k_norm: &k_norm_values,
            }));
            metadata.push(Token {
                batch_start: request as u32,
                token_index: position,
                req_slot: request as u32,
                src_state_slot: request as u32,
                dst_state_slot: request as u32 + 4,
                group_start: group_start as u32,
            });
        }
        let compute = Compute::new(device, config);
        let shape = Shape {
            num_total_tokens: 4,
            num_total_groups: groups,
        };
        Self {
            config,
            layout,
            metadata: metadata.clone(),
            page_ids_values,
            references,
            qk: buffer(device, &qk_values, config.dtype),
            tokens: Buffer::from_slice(
                device,
                &metadata.iter().flat_map(|token| token.to_array()).collect::<Vec<_>>(),
            ),
            q_norm: buffer(device, &q_norm_values, config.dtype),
            k_norm: buffer(device, &k_norm_values, config.dtype),
            pages: buffer(device, &page_values, config.dtype),
            page_ids: Buffer::from_slice(device, &(0..num_pages).rev().collect::<Vec<_>>()),
            states: buffer(device, &state_values, config.dtype),
            ids: Buffer::new_zeroed_elements(device, 4 * core.top_k, Dtype::Uint32),
            q: Buffer::new_zeroed_elements(device, 4 * core.num_q_heads * core.head_dim, config.dtype),
            scores: Buffer::new_zeroed_elements(device, 4 * groups as usize, Dtype::Float32),
            candidates: std::array::from_fn(|_| Buffer::new_zeroed(device, compute.candidate_bytes(shape))),
            compute,
        }
    }

    fn prepare_buffers(&self) -> PrepareBuffers<'_> {
        PrepareBuffers {
            qk: &self.qk,
            tokens: &self.tokens,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            pages: &self.pages,
            page_ids: &self.page_ids,
            raw_states: &self.states,
            q: &self.q,
        }
    }

    fn select_buffers(&self) -> SelectBuffers<'_> {
        SelectBuffers {
            q: &self.q,
            tokens: &self.tokens,
            pages: &self.pages,
            page_ids: &self.page_ids,
            group_ids: &self.ids,
        }
    }

    fn scratch(&self) -> SelectScratch<'_> {
        SelectScratch {
            scores: &self.scores,
            candidates: [&self.candidates[0], &self.candidates[1]],
        }
    }

    fn check(&self, active: usize, before_pages: &[f32], before_states: &[f32]) {
        let c = self.config;
        let dim = c.head_dim as usize;
        let tolerance = if c.dtype == Dtype::Float32 { 0.0001 } else { 0.03125 };
        let q = values(&self.q, c.dtype);
        let scores = self.scores.read_typed::<f32>(0, self.scores.len_bytes() / 4);
        let groups = scores.len() / 4;
        let ids = self.ids.read_typed::<u32>(0, 4 * c.top_k as usize);
        let mut expected_pages = before_pages.to_vec();
        let mut expected_states = before_states.to_vec();
        let mut changed_pages = vec![false; before_pages.len()];
        for token in 0..active {
            let reference = &self.references[token];
            assert_close(
                &q[token * c.num_q_heads as usize * dim..(token + 1) * c.num_q_heads as usize * dim],
                &reference.q,
                tolerance,
            );
            let selected = &reference.group_ids[0];
            let actual = &ids[token * c.top_k as usize..(token + 1) * c.top_k as usize];
            let metadata = self.metadata[token];
            let group_end = (metadata.token_index + 1) / c.compress_ratio;
            if group_end - metadata.group_start > c.top_k {
                let gpu_scores = &scores[token * groups..token * groups + reference.scores[0].len()];
                assert_close(gpu_scores, &reference.scores[0], tolerance * c.num_q_heads as f32);
                // Exact CPU sorting of GPU scores isolates selection correctness from dot-product rounding.
                let mut expected: Vec<_> = (metadata.group_start..group_end).collect();
                expected.sort_by(|&a, &b| {
                    gpu_scores[(b - metadata.group_start) as usize]
                        .total_cmp(&gpu_scores[(a - metadata.group_start) as usize])
                        .then(a.cmp(&b))
                });
                assert_eq!(actual, &expected[..c.top_k as usize]);
                let threshold =
                    reference.scores[0][selected[selected.len() - 1] as usize - metadata.group_start as usize];
                for &id in actual {
                    assert!(
                        reference.scores[0][(id - metadata.group_start) as usize] + tolerance * c.num_q_heads as f32
                            >= threshold
                    );
                }
            } else {
                assert!(
                    actual.iter().all(|&id| id == 0),
                    "dense queries must not write selection"
                );
            }
            let end = metadata.token_index as usize + 1;
            let carry = end % c.compress_ratio as usize;
            if carry == 0 {
                let group = end / c.compress_ratio as usize - 1;
                let groups_per_page = self.layout.page_bytes as usize / (dim * c.dtype.item_size());
                let page = self.page_ids_values
                    [token * self.layout.num_pages_per_request as usize + group / groups_per_page]
                    as usize;
                let offset = (page * groups_per_page + group % groups_per_page) * dim;
                expected_pages[offset..offset + dim].copy_from_slice(&reference.keys[group * dim..(group + 1) * dim]);
                changed_pages[offset..offset + dim].fill(true);
            } else {
                let offset = metadata.dst_state_slot as usize * (c.compress_ratio as usize - 1) * dim;
                expected_states[offset..offset + carry * dim]
                    .copy_from_slice(&reference.raw_keys[(end - carry) * dim..end * dim]);
            }
        }
        let actual_pages = values(&self.pages, c.dtype);
        for (index, &changed) in changed_pages.iter().enumerate() {
            if changed {
                assert!((actual_pages[index] - expected_pages[index]).abs() <= tolerance);
            } else {
                assert_eq!(actual_pages[index], before_pages[index]);
            }
        }
        assert_eq!(values(&self.states, c.dtype), expected_states);
    }
}

fn fixture_values(len: usize, mut state: u64, dtype: Dtype) -> Vec<f32> {
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = ((state % 1021) as f32 - 510.0) / 512.0;
            if dtype == Dtype::Bfloat16 {
                bf16::from_f32(value).to_f32()
            } else {
                value
            }
        })
        .collect()
}

fn buffer(device: &Device, values: &[f32], dtype: Dtype) -> Buffer {
    match dtype {
        Dtype::Float32 => Buffer::from_slice(device, values),
        _ => {
            Buffer::from_slice(
                device,
                &values.iter().map(|&x| bf16::from_f32(x).to_bits()).collect::<Vec<_>>(),
            )
        },
    }
}

fn values(buffer: &Buffer, dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Float32 => buffer.read_typed(0, buffer.len_bytes() / 4),
        _ => {
            buffer
                .read_typed::<u16>(0, buffer.len_bytes() / 2)
                .into_iter()
                .map(|x| bf16::from_bits(x).to_f32())
                .collect()
        },
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index={index}, actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}
