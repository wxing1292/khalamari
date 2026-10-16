use half::bf16;
use inference_executor_core::attn::qsa_indexer::core::QSAIndexerCore;
use inference_executor_core::attn::qsa_indexer::reference;
use inference_executor_core::attn::qsa_indexer::reference::Precision;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;
use crate::metal::Stream;
use crate::test_support::ReplayTestCache;

const ACTIVE: ReplayParameterKey = ReplayParameterKey::new("test.qsa_indexer.num_active_tokens");
const ACTIVE_SCORE_TASKS: ReplayParameterKey =
    ReplayParameterKey::new("test.qsa_indexer.num_active_score_task_templates");

#[test]
fn test_select_shrink() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        num_q_heads: 1,
        head_dim: 2,
        compress_ratio: 2,
        top_k: 2,
        rope_dim: 2,
        rope_theta: 10_000.0,
        norm_eps: 1e-6,
        dtype: Dtype::Float32,
    };
    let compute = Compute::new(&device, config);
    let shape = Shape {
        num_total_tokens: 1,
        num_total_sparse_queries: 1,
        num_total_request_sparse_queries: 1,
        num_total_groups: 8,
        num_total_score_task_templates: 1,
    };
    let layout = CacheLayout {
        num_req_slots: 1,
        num_pages_per_request: 1,
        num_state_slots: 1,
        page_bytes: 64,
    };
    let q = Buffer::from_slice(&device, &[1.0_f32, 0.0]);
    let tokens = Buffer::from_slice(
        &device,
        &Token {
            token_index: 15,
            ..Token::default()
        }
        .to_array(),
    );
    let score_tasks = Buffer::from_slice(
        &device,
        &ScoreTaskTemplate {
            query_start: 0,
            num_queries: 1,
        }
        .to_array(),
    );
    let keys = (1..=8).flat_map(|value| [value as f32, 0.0]).collect::<Vec<_>>();
    let pages = Buffer::from_slice(&device, &keys);
    let page_ids = Buffer::from_slice(&device, &[0_u32]);
    let group_ids = Buffer::new_zeroed_elements(&device, 2, Dtype::Uint32);
    let scores = Buffer::new_zeroed_elements(&device, 8, Dtype::Float32);

    let mut builder = stream.create_replay_program();
    builder.record(compute.select(
        shape,
        layout,
        ReplayU32::Fixed(1),
        ReplayU32::Fixed(1),
        SelectBuffers {
            q: &q,
            tokens: &tokens,
            score_task_templates: &score_tasks,
            pages: &pages,
            page_ids: &page_ids,
            group_ids: &group_ids,
        },
        &scores,
    ));
    let replay = builder.build();

    stream.submit_replay(&replay).wait();
    assert_eq!(group_ids.read_typed::<u32>(0, 2), [6, 7]);
    q.write_typed(0, &[0.0_f32, 0.0]);
    tokens.write_typed(
        0,
        &Token {
            token_index: 7,
            group_start: 1,
            ..Token::default()
        }
        .to_array(),
    );
    stream.submit_replay(&replay).wait();
    assert_eq!(group_ids.read_typed::<u32>(0, 2), [1, 2]);
}

#[test]
fn test_select_success() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        num_q_heads: 4,
        head_dim: 128,
        compress_ratio: 4,
        top_k: 2,
        rope_dim: 64,
        rope_theta: 10_000_000.0,
        norm_eps: 1e-6,
        dtype: Dtype::Bfloat16,
    };
    let compute = Compute::new(&device, config);
    let shape = Shape {
        num_total_tokens: 128,
        num_total_sparse_queries: 128,
        num_total_request_sparse_queries: 128,
        num_total_groups: 16_384,
        num_total_score_task_templates: 4,
    };
    let layout = CacheLayout {
        num_req_slots: 2,
        num_pages_per_request: 128,
        num_state_slots: 1,
        page_bytes: 32_768,
    };
    let rows = 18_usize;
    let q_bits = (0..rows * 4 * 128)
        .map(|index| bf16::from_f32(((index * 13 % 61) as f32 - 30.0) / 256.0).to_bits())
        .collect::<Vec<_>>();
    let q = Buffer::new_zeroed_elements(&device, shape.num_total_tokens as usize * 4 * 128, Dtype::Bfloat16);
    q.write_typed(0, &q_bits);

    let num_physical_pages = layout.num_req_slots as usize * layout.num_pages_per_request as usize;
    let keys_per_page = 128 * 128;
    let page_bits = (0..num_physical_pages * keys_per_page)
        .map(|index| bf16::from_f32(((index * 7 % 67) as f32 - 33.0) / 256.0).to_bits())
        .collect::<Vec<_>>();
    let pages = Buffer::from_slice(&device, &page_bits);
    let page_ids_host = (0..layout.num_req_slots as usize)
        .flat_map(|request| {
            (0..layout.num_pages_per_request as usize).map(move |page| {
                (request * layout.num_pages_per_request as usize + layout.num_pages_per_request as usize - page - 1)
                    as u32
            })
        })
        .collect::<Vec<_>>();
    let page_ids = Buffer::from_slice(&device, &page_ids_host);
    let group_starts = [125_u32, 127];
    let first_positions = [539_u32, 547];
    let mut token_words = vec![0_u32; shape.num_total_tokens as usize * 6];
    for row in 0..rows {
        let request = usize::from(row >= 10);
        let query_in_request = if request == 0 { row } else { row - 10 };
        let token = Token {
            batch_start: if request == 0 { 0 } else { 10 },
            token_index: first_positions[request] + query_in_request as u32,
            req_slot: request as u32,
            group_start: group_starts[request],
            ..Token::default()
        };
        token_words[row * 6..row * 6 + 6].copy_from_slice(&token.to_array());
    }
    let tokens = Buffer::from_slice(&device, &token_words);
    let task_words = [0_u32, 10, 10, 8, 0, 0, 0, 0];
    let score_tasks = Buffer::from_slice(&device, &task_words);
    let group_ids = Buffer::new_zeroed_elements(&device, shape.num_total_tokens as usize * 2, Dtype::Uint32);
    let scores = Buffer::new_zeroed_elements(
        &device,
        shape.num_total_tokens as usize * shape.num_total_groups as usize,
        Dtype::Float32,
    );
    let mut builder = stream.create_replay_program();
    builder.record(compute.select(
        shape,
        layout,
        ReplayU32::Parameter(ACTIVE),
        ReplayU32::Parameter(ACTIVE_SCORE_TASKS),
        SelectBuffers {
            q: &q,
            tokens: &tokens,
            score_task_templates: &score_tasks,
            pages: &pages,
            page_ids: &page_ids,
            group_ids: &group_ids,
        },
        &scores,
    ));
    let replay = builder.build();
    const ID_CANARY: u32 = 0xDEAD_BEEF;
    for (active_rows, active_tasks) in [(18_u32, 2_u32), (10, 1), (18, 2)] {
        group_ids.write_typed(0, &vec![ID_CANARY; rows * 2]);
        scores.write_typed(0, &vec![0.125_f32; rows * shape.num_total_groups as usize]);
        stream
            .submit_replay_with_arguments(
                &replay,
                &ReplayArguments::new()
                    .with_u32(ACTIVE, active_rows)
                    .with_u32(ACTIVE_SCORE_TASKS, active_tasks),
            )
            .wait();
        let actual_scores = scores.read_typed::<f32>(0, rows * shape.num_total_groups as usize);
        let actual_ids = group_ids.read_typed::<u32>(0, rows * 2);
        for row in active_rows as usize..rows {
            assert_eq!(&actual_ids[row * 2..row * 2 + 2], &[ID_CANARY; 2]);
            assert!(
                actual_scores[row * shape.num_total_groups as usize..(row + 1) * shape.num_total_groups as usize]
                    .iter()
                    .all(|&score| score == 0.125)
            );
        }
        for row in 0..active_rows as usize {
            let request = usize::from(row >= 10);
            let query_in_request = if request == 0 { row } else { row - 10 };
            let token_index = first_positions[request] + query_in_request as u32;
            let group_end = (token_index + 1) / 4;
            let visible_groups = group_end - group_starts[request];
            let score_row = &actual_scores[row * shape.num_total_groups as usize..];
            let mut expected_scores = Vec::new();
            for relative_group in 0..visible_groups {
                let absolute_group = group_starts[request] + relative_group;
                let physical_page = page_ids_host
                    [request * layout.num_pages_per_request as usize + absolute_group as usize / 128]
                    as usize;
                let key_base = (physical_page * 128 + absolute_group as usize % 128) * 128;
                let mut expected_score = 0.0_f32;
                for head in 0..4 {
                    let mut dot = 0.0_f32;
                    let q_base = (row * 4 + head) * 128;
                    for column in 0..128 {
                        dot += bf16::from_bits(q_bits[q_base + column]).to_f32()
                            * bf16::from_bits(page_bits[key_base + column]).to_f32();
                    }
                    expected_score += dot.max(0.0);
                }
                expected_score *= 128.0_f32.sqrt().recip();
                expected_scores.push(expected_score);
                assert!(
                    (score_row[relative_group as usize] - expected_score).abs() <= 2e-6,
                    "row={row} group={relative_group} GPU={} CPU={expected_score}",
                    score_row[relative_group as usize]
                );
            }
            let mut expected_ids: Vec<_> = (0..visible_groups).collect();
            expected_ids.sort_by(|&left, &right| {
                expected_scores[right as usize]
                    .total_cmp(&expected_scores[left as usize])
                    .then(left.cmp(&right))
            });
            expected_ids.truncate(2);
            expected_ids.sort_unstable();
            assert_eq!(
                &actual_ids[row * 2..row * 2 + 2],
                expected_ids
                    .iter()
                    .map(|&relative_group| group_starts[request] + relative_group)
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn test_prepare_select_bucketing() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    for (heads, dim, ratio, top_k, groups, dtype, zero_queries) in [
        (2, 7, 3, 3, 769, Dtype::Float32, false),
        (4, 128, 4, 512, 8193, Dtype::Bfloat16, false),
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
            num_total_sparse_queries: 4,
            num_total_request_sparse_queries: 4,
            num_total_groups: groups,
            num_total_score_task_templates: 4,
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
                ReplayU32::Parameter(ACTIVE_SCORE_TASKS),
                fixture.select_buffers(),
                &fixture.scores,
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
                .submit_replay_with_arguments(
                    replay,
                    &ReplayArguments::new().with_u32(ACTIVE, active).with_u32(
                        ACTIVE_SCORE_TASKS,
                        fixture.num_active_score_task_templates(active as usize),
                    ),
                )
                .wait();
            fixture.check(active as usize, &before_pages, &before_states);
        }
        let smaller = Shape {
            num_total_tokens: 2,
            num_total_sparse_queries: 2,
            num_total_request_sparse_queries: 2,
            num_total_score_task_templates: 2,
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
                        ReplayU32::Fixed(fixture.num_active_score_task_templates(2)),
                        fixture.select_buffers(),
                        &fixture.scores,
                    ));
                    builder.build()
                })
                .1
        );
    }
}

// Fixed ties exercise the exact full-range path when sampling cannot narrow candidates.
// Seeded dyadic inputs make dot products exact before scaling, independent of reduction order.
#[test]
fn test_select_random() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let config = Config {
        num_q_heads: 4,
        head_dim: 128,
        compress_ratio: 4,
        top_k: 512,
        rope_dim: 64,
        rope_theta: 1e7,
        norm_eps: 1e-6,
        dtype: Dtype::Bfloat16,
    };
    let compute = Compute::new(&device, config);
    for (request_tokens, groups, fixed, groups_per_page) in [
        (vec![1], 8192, false, 128),
        (vec![1], 8192, true, 128),
        (vec![2, 2], 4096, false, 64),
        (vec![1, 1, 1, 1, 1, 1, 1, 1], 1024, false, 256),
        (vec![3, 5], 600, false, 128),
        (vec![5, 5], 600, false, 128),
        (vec![32], 600, false, 128),
        (vec![33, 9], 600, false, 128),
        (vec![65, 40], 600, false, 128),
        (vec![32, 32], 600, true, 128),
    ] {
        let count: u32 = request_tokens.iter().sum();
        let shape = Shape {
            num_total_tokens: count,
            num_total_sparse_queries: count,
            num_total_request_sparse_queries: *request_tokens.iter().max().unwrap(),
            num_total_groups: groups + request_tokens.iter().max().unwrap().div_ceil(4),
            num_total_score_task_templates: count,
        };
        let num_pages_per_request = (shape.num_total_groups + 127).div_ceil(groups_per_page);
        let layout = CacheLayout {
            num_req_slots: request_tokens.len() as u32,
            num_pages_per_request,
            num_state_slots: 1,
            page_bytes: groups_per_page * 128 * 2,
        };
        let mut seed = 0x239A_8151_u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            bf16::from_f32(((seed % 127) as i32 - 63) as f32 / 256.0).to_bits()
        };
        let q_bits: Vec<_> = (0..count as usize * 512)
            .map(|_| if fixed { 0 } else { random() })
            .collect();
        let key_bits: Vec<_> =
            (0..layout.num_req_slots as usize * num_pages_per_request as usize * groups_per_page as usize * 128)
                .map(|_| random())
                .collect();
        let q = Buffer::from_slice(&device, &q_bits);
        let pages = Buffer::from_slice(&device, &key_bits);
        let page_ids_values: Vec<_> = (0..layout.num_req_slots * num_pages_per_request).rev().collect();
        let page_ids = Buffer::from_slice(&device, &page_ids_values);
        let mut metadata = Vec::new();
        let mut templates = Vec::new();
        let geometry = ScoreGeometry::for_shape(config.num_q_heads, config.head_dim, config.dtype, shape, layout);
        for (request, &num_queries) in request_tokens.iter().enumerate() {
            let batch_start = metadata.len() as u32;
            let task_width = geometry.num_queries_per_task(num_queries);
            for offset in (0..num_queries).step_by(task_width as usize) {
                templates.extend_from_slice(
                    &ScoreTaskTemplate {
                        query_start: batch_start + offset,
                        num_queries: (num_queries - offset).min(task_width),
                    }
                    .to_array(),
                );
            }
            for query in 0..num_queries {
                metadata.push(Token {
                    batch_start,
                    token_index: (127 + groups) * 4 - 1 + query,
                    req_slot: request as u32,
                    group_start: 127,
                    ..Token::default()
                });
            }
        }
        let num_score_tasks = (templates.len() / 2) as u32;
        templates.resize(count as usize * 2, 0);
        let tokens = Buffer::from_slice(&device, &metadata.iter().flat_map(|t| t.to_array()).collect::<Vec<_>>());
        let tasks = Buffer::from_slice(&device, &templates);
        let scores = Buffer::new_zeroed_elements(
            &device,
            count as usize * shape.num_total_groups as usize,
            Dtype::Float32,
        );
        let ids = Buffer::new_zeroed_elements(&device, count as usize * 512, Dtype::Uint32);
        let mut builder = stream.create_replay_program();
        builder.record(compute.select(
            shape,
            layout,
            ReplayU32::Fixed(count),
            ReplayU32::Fixed(num_score_tasks),
            SelectBuffers {
                q: &q,
                tokens: &tokens,
                score_task_templates: &tasks,
                pages: &pages,
                page_ids: &page_ids,
                group_ids: &ids,
            },
            &scores,
        ));
        stream.submit_replay(&builder.build()).wait();
        let actual_scores = scores.read_typed::<f32>(0, count as usize * shape.num_total_groups as usize);
        let actual_ids = ids.read_typed::<u32>(0, count as usize * 512);
        for (query, token) in metadata.iter().enumerate() {
            let end = (token.token_index + 1) / 4;
            let expected_scores: Vec<_> = (token.group_start..end)
                .map(|group| {
                    let page =
                        page_ids_values[(token.req_slot * num_pages_per_request + group / groups_per_page) as usize];
                    let key_base =
                        (page as usize * groups_per_page as usize + group as usize % groups_per_page as usize) * 128;
                    (0..4)
                        .map(|head| {
                            (0..128)
                                .map(|d| {
                                    bf16::from_bits(q_bits[(query * 4 + head) * 128 + d]).to_f32()
                                        * bf16::from_bits(key_bits[key_base + d]).to_f32()
                                })
                                .sum::<f32>()
                                .max(0.0)
                        })
                        .sum::<f32>()
                        / 128.0_f32.sqrt()
                })
                .collect();
            for (group, &expected) in expected_scores.iter().enumerate() {
                let actual = actual_scores[query * shape.num_total_groups as usize + group];
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "query={query}, group={group}: {actual} != {expected}"
                );
            }
            let mut expected_ids: Vec<_> = (token.group_start..end).collect();
            expected_ids.sort_unstable_by(|&a, &b| {
                expected_scores[(b - token.group_start) as usize]
                    .total_cmp(&expected_scores[(a - token.group_start) as usize])
                    .then(a.cmp(&b))
            });
            expected_ids.truncate(512);
            expected_ids.sort_unstable();
            assert_eq!(&actual_ids[query * 512..(query + 1) * 512], expected_ids);
        }
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
    score_tasks: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    pages: Buffer,
    page_ids: Buffer,
    states: Buffer,
    ids: Buffer,
    q: Buffer,
    scores: Buffer,
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
            num_total_sparse_queries: 4,
            num_total_request_sparse_queries: 4,
            num_total_groups: groups,
            num_total_score_task_templates: 4,
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
            score_tasks: {
                let mut words = Vec::new();
                for (query_start, token) in metadata.iter().enumerate() {
                    if (token.token_index + 1) / config.compress_ratio - token.group_start > config.top_k {
                        words.extend_from_slice(
                            &ScoreTaskTemplate {
                                query_start: query_start as u32,
                                num_queries: 1,
                            }
                            .to_array(),
                        );
                    }
                }
                words.resize(4 * 2, 0);
                Buffer::from_slice(device, &words)
            },
            q_norm: buffer(device, &q_norm_values, config.dtype),
            k_norm: buffer(device, &k_norm_values, config.dtype),
            pages: buffer(device, &page_values, config.dtype),
            page_ids: Buffer::from_slice(device, &(0..num_pages).rev().collect::<Vec<_>>()),
            states: buffer(device, &state_values, config.dtype),
            ids: Buffer::new_zeroed_elements(device, 4 * core.top_k, Dtype::Uint32),
            q: Buffer::new_zeroed_elements(device, 4 * core.num_q_heads * core.head_dim, config.dtype),
            scores: Buffer::new_zeroed_elements(device, 4 * groups as usize, Dtype::Float32),
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
            score_task_templates: &self.score_tasks,
            pages: &self.pages,
            page_ids: &self.page_ids,
            group_ids: &self.ids,
        }
    }

    fn num_active_score_task_templates(&self, active: usize) -> u32 {
        self.metadata[..active]
            .iter()
            .filter(|token| {
                (token.token_index + 1) / self.config.compress_ratio - token.group_start > self.config.top_k
            })
            .count() as u32
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
                expected.truncate(c.top_k as usize);
                expected.sort_unstable();
                assert_eq!(actual, expected.as_slice());
                let threshold = selected
                    .iter()
                    .map(|&id| reference.scores[0][id as usize - metadata.group_start as usize])
                    .fold(f32::INFINITY, f32::min);
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
