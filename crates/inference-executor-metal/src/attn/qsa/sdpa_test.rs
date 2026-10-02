use std::collections::HashSet;

use half::bf16;
use inference_backend_metal::components::gqa::fp8::bf16_to_f32;
use inference_backend_metal::components::gqa::fp8::bf16_to_fp8_e4m3;
use inference_backend_metal::components::gqa::fp8::f32_to_bf16;
use inference_backend_metal::components::gqa::fp8::fp8_e4m3_to_bf16;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_executor_core::attn::GQAPageTableLayout;

use super::batch_metadata::QSASDPAMetadataBuffers;
use super::sdpa::QSASDPA;
use super::sdpa::QSASDPAActive;
use super::sdpa::QSASDPAConfig;
use super::sdpa::QSASDPAInput;
use super::sdpa::QSASDPAReplay;
use super::sdpa::QSASDPARequest;
use super::sdpa::QSASDPAShape;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::ReplayRecorder;
use crate::replay::Replay;
use crate::replay::ReplayComponent;

const Q_HEADS: usize = 24;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const TOKENS_PER_PAGE: usize = 32;
const PAGE_BYTES: usize = 32_768;
const COMPRESS_RATIO: usize = 4;
const TOP_K: usize = 512;
const GROUP_START: u32 = 1;
const PAGE_TABLE_INDEX: u32 = 1;
const ROW_WIDTH: usize = Q_HEADS * HEAD_DIM;

#[test]
fn test_reference_attention_success() {
    let config = fixture_config();
    let q_values = vec![0_u16; ROW_WIDTH];
    let kv = (0..3)
        .map(|token| {
            let mut k = vec![token as f32; KV_HEADS * HEAD_DIM];
            let mut v = vec![0.0; KV_HEADS * HEAD_DIM];
            for head in 0..KV_HEADS {
                v[head * HEAD_DIM..(head + 1) * HEAD_DIM].fill(token as f32 * 2.0 + head as f32);
            }
            k[0] += 1.0;
            KVToken { k, v }
        })
        .collect::<Vec<_>>();

    let actual = reference_attention_from_kv(config, &q_values, 0, &kv);
    for q_head in 0..Q_HEADS {
        let expected = 2.0 + (q_head / (Q_HEADS / KV_HEADS)) as f32;
        assert!(
            actual[q_head * HEAD_DIM..(q_head + 1) * HEAD_DIM]
                .iter()
                .all(|&value| (value - expected).abs() <= 1.0e-6)
        );
    }
}

#[test]
fn test_replay_bucketing() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let runtime = MetalReplayRuntime::new(&stream);
    let config = fixture_config();
    let mut cache = KVCacheFixture::new(&device, config.page_table_layout);
    let sdpa = QSASDPA::new(&device, config);
    let mut replay = Replay::new("test QSA prepared-Q SDPA", TestQSASDPA(sdpa));
    let mut recorded_keys = HashSet::new();

    let shape = QSASDPAShape {
        num_total_tokens: 4,
        num_total_sparse_queries: 4,
    };
    let scratch = replay.component().0.new_scratch(shape);
    let metadata = QSASDPAMetadataBuffers::new(&device, config, shape);
    let q_values = fixture_q(shape.num_total_tokens as usize, 0xA17C_904D);
    let q = Buffer::from_slice(&device, &q_values);
    let selected_groups = fixture_selected_groups(shape.num_total_tokens as usize, GROUP_START);
    let selected_group_ids = Buffer::from_slice(&device, &selected_groups);
    let output = Buffer::new_zeroed_elements(&device, shape.num_total_tokens as usize * ROW_WIDTH, Dtype::Bfloat16);
    let bindings = TestBindings {
        config,
        shape,
        metadata: &metadata,
        scratch: &scratch,
        q: &q,
        q_values: &q_values,
        cache: &cache,
        selected_group_ids: &selected_group_ids,
        selected_groups: &selected_groups,
        output: &output,
    };

    // These active counts share one recorded capacity and cover dense-only,
    // mixed, and sparse-only replays. The mixed batch writes sparse row 0 and
    // dense rows 1 and 2, so both output maps must preserve original row IDs.
    let batches = [
        vec![request(2, 100, 1)],
        vec![request(0, 100, 4)],
        vec![request(1, 2055, 1), request(0, 100, 2)],
        vec![request(1, 2056, 2)],
    ];
    let expected_active = [(1, 1, 0), (4, 4, 0), (3, 2, 1), (2, 0, 2)];
    for (batch, expected) in batches.iter().zip(expected_active) {
        let active = bindings.run(&runtime, &mut replay, &mut recorded_keys, batch);
        assert_eq!(
            (
                active.num_active_tokens,
                active.num_active_dense_queries,
                active.num_active_sparse_queries
            ),
            expected
        );
        if active.num_active_dense_queries == 0 {
            assert_eq!(active.num_active_dense_tasks, 0);
        } else {
            assert!(active.num_active_dense_tasks > 0 && active.num_active_dense_tasks <= shape.num_total_tokens);
        }
    }

    // One request crosses the dense/sparse cutoff at position 2055. This
    // batch also places both dense and sparse results in original output rows.
    let crossing_and_tail_batch = vec![request(2, 2053, 3), request(0, 2056, 1)];
    let active = bindings.run(&runtime, &mut replay, &mut recorded_keys, &crossing_and_tail_batch);
    assert_eq!(active.num_active_tokens, 4);
    assert_eq!(active.num_active_dense_queries, 2);
    assert_eq!(active.num_active_sparse_queries, 2);

    // A 513-query sparse capacity selects S=1, which writes the final BF16
    // output directly. The sparse rows at 2055 and 2058 cover the complete
    // current group omission and the final partial-group residue.
    let single_split_shape = QSASDPAShape {
        num_total_tokens: 513,
        num_total_sparse_queries: 513,
    };
    let single_split_scratch = replay.component().0.new_scratch(single_split_shape);
    let single_split_metadata = QSASDPAMetadataBuffers::new(&device, config, single_split_shape);
    let single_split_q_values = fixture_q(single_split_shape.num_total_tokens as usize, 0x6A2D_F093);
    let single_split_q = Buffer::from_slice(&device, &single_split_q_values);
    let single_split_groups = fixture_selected_groups(single_split_shape.num_total_tokens as usize, GROUP_START);
    let single_split_group_ids = Buffer::from_slice(&device, &single_split_groups);
    let single_split_output = Buffer::new_zeroed_elements(
        &device,
        single_split_shape.num_total_tokens as usize * ROW_WIDTH,
        Dtype::Bfloat16,
    );
    let single_split_batch = vec![request(0, 100, 2), request(1, 2055, 1), request(2, 2058, 1)];
    let single_split_bindings = TestBindings {
        config,
        shape: single_split_shape,
        metadata: &single_split_metadata,
        scratch: &single_split_scratch,
        q: &single_split_q,
        q_values: &single_split_q_values,
        cache: &cache,
        selected_group_ids: &single_split_group_ids,
        selected_groups: &single_split_groups,
        output: &single_split_output,
    };
    single_split_bindings.run(&runtime, &mut replay, &mut recorded_keys, &single_split_batch);

    let row_start = 2 * ROW_WIDTH;
    let before_omitted_group = single_split_output.read_typed::<u16>(row_start, 2 * ROW_WIDTH);
    cache.overwrite_group(&device, 1, PAGE_TABLE_INDEX, 513, 0x7E);
    let single_split_bindings = TestBindings {
        config,
        shape: single_split_shape,
        metadata: &single_split_metadata,
        scratch: &single_split_scratch,
        q: &single_split_q,
        q_values: &single_split_q_values,
        cache: &cache,
        selected_group_ids: &single_split_group_ids,
        selected_groups: &single_split_groups,
        output: &single_split_output,
    };
    single_split_bindings.run(&runtime, &mut replay, &mut recorded_keys, &single_split_batch);
    let after_omitted_group = single_split_output.read_typed::<u16>(row_start, 2 * ROW_WIDTH);
    assert_eq!(
        before_omitted_group, after_omitted_group,
        "QSA sparse SDPA read the omitted complete group"
    );
}

#[derive(Clone, Copy)]
struct TestInput<'a> {
    input: QSASDPAInput<'a>,
    active: QSASDPAReplay,
}

struct TestQSASDPA(QSASDPA);

impl ReplayComponent for TestQSASDPA {
    type Key = (u32, u32);
    type Input<'a> = TestInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        let shape = input.input.metadata.shape();
        (shape.num_total_tokens, shape.num_total_sparse_queries)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.0.record(recorder, input.input, input.active);
    }
}

struct TestBindings<'a> {
    config: QSASDPAConfig,
    shape: QSASDPAShape,
    metadata: &'a QSASDPAMetadataBuffers,
    scratch: &'a super::sdpa::QSASDPAScratch,
    q: &'a Buffer,
    q_values: &'a [u16],
    cache: &'a KVCacheFixture,
    selected_group_ids: &'a Buffer,
    selected_groups: &'a [u32],
    output: &'a Buffer,
}

impl TestBindings<'_> {
    fn run(
        &self,
        runtime: &MetalReplayRuntime,
        replay: &mut Replay<TestQSASDPA>,
        recorded_keys: &mut HashSet<(u32, u32)>,
        requests: &[QSASDPARequest],
    ) -> QSASDPAActive {
        let active = self.metadata.update(requests);
        active.validate(self.shape);
        let expected = reference_outputs(self.config, self.cache, self.q_values, self.selected_groups, requests);
        let sentinel = f32_to_bf16(0.3125).to_bits();
        self.output
            .write_typed(0, &vec![sentinel; self.shape.num_total_tokens as usize * ROW_WIDTH]);

        let input = TestInput {
            input: QSASDPAInput {
                q: self.q,
                kv_pages: &self.cache.pages,
                page_ids: &self.cache.page_ids,
                selected_group_ids: self.selected_group_ids,
                output: self.output,
                metadata: self.metadata,
                scratch: self.scratch,
                page_table_index: ReplayU32::Fixed(PAGE_TABLE_INDEX),
            },
            active: QSASDPAReplay::parameters(),
        };
        let (key, cache_hit) = replay.record(runtime, &input);
        assert_eq!(cache_hit, !recorded_keys.insert(key));
        let mut arguments = ReplayArguments::new();
        active.add_replay_arguments(&mut arguments);
        runtime
            .submit_replay_with_arguments(replay.replay(&key), &arguments)
            .wait();

        let actual = self
            .output
            .read_typed::<u16>(0, self.shape.num_total_tokens as usize * ROW_WIDTH);
        for (row, expected_row) in expected.iter().enumerate() {
            let actual_row = &actual[row * ROW_WIDTH..(row + 1) * ROW_WIDTH];
            assert_close(
                &format!("QSA prepared-Q row {row}"),
                &decode_bf16(actual_row),
                expected_row,
                0.03125,
            );
        }
        for (row, actual_row) in actual
            .as_chunks::<ROW_WIDTH>()
            .0
            .iter()
            .enumerate()
            .skip(expected.len())
        {
            assert!(
                actual_row.iter().all(|&value| value == sentinel),
                "QSA SDPA changed inactive output row {row}"
            );
        }
        active
    }
}

fn reference_outputs(
    config: QSASDPAConfig,
    cache: &KVCacheFixture,
    q_values: &[u16],
    selected_groups: &[u32],
    requests: &[QSASDPARequest],
) -> Vec<Vec<f32>> {
    let mut outputs = Vec::new();
    for request in requests {
        for local_query in 0..request.num_tokens {
            let row = outputs.len();
            let position = request.token_index + local_query;
            let visible_complete_groups = (position + 1) / config.compress_ratio - request.group_start;
            let mut tokens = Vec::new();
            if visible_complete_groups <= config.top_k {
                let begin = request.group_start * config.compress_ratio;
                tokens.extend(begin..=position);
            } else {
                let row_groups = &selected_groups[row * TOP_K..(row + 1) * TOP_K];
                assert_eq!(row_groups.iter().copied().collect::<HashSet<_>>().len(), TOP_K);
                assert!(row_groups.iter().all(|&group| group >= request.group_start));
                for &group in row_groups {
                    let begin = group * config.compress_ratio;
                    tokens.extend(begin..begin + config.compress_ratio);
                }
                let complete_groups = (position + 1) / config.compress_ratio;
                let tail_tokens = (position + 1) % config.compress_ratio;
                let tail_begin = complete_groups * config.compress_ratio;
                tokens.extend(tail_begin..tail_begin + tail_tokens);
            }
            outputs.push(reference_attention_row(
                config,
                cache,
                q_values,
                row,
                request.req_slot,
                position,
                tokens,
            ));
        }
    }
    outputs
}

fn reference_attention_row(
    config: QSASDPAConfig,
    cache: &KVCacheFixture,
    q_values: &[u16],
    row: usize,
    req_slot: u32,
    position: u32,
    tokens: Vec<u32>,
) -> Vec<f32> {
    assert!(!tokens.is_empty());
    assert!(tokens.iter().all(|&token| token <= position));
    let kv = tokens
        .iter()
        .map(|&token| cache.read_token(config.page_table_layout, req_slot, PAGE_TABLE_INDEX, token))
        .collect::<Vec<_>>();
    reference_attention_from_kv(config, q_values, row, &kv)
}

fn reference_attention_from_kv(config: QSASDPAConfig, q_values: &[u16], row: usize, kv: &[KVToken]) -> Vec<f32> {
    assert!(!kv.is_empty());
    let mut output = vec![0.0_f32; ROW_WIDTH];
    for q_head in 0..Q_HEADS {
        let kv_head = q_head / (Q_HEADS / KV_HEADS);
        let q_begin = (row * Q_HEADS + q_head) * HEAD_DIM;
        let mut logits = Vec::with_capacity(kv.len());
        let mut max_logit = f32::NEG_INFINITY;
        for token in kv {
            let key_begin = kv_head * HEAD_DIM;
            let mut dot = 0.0_f32;
            for dim in 0..HEAD_DIM {
                let query = bf16::from_bits(q_values[q_begin + dim]).to_f32();
                dot = query.mul_add(token.k[key_begin + dim], dot);
            }
            let logit = dot * config.scale;
            max_logit = max_logit.max(logit);
            logits.push(logit);
        }
        let weights = logits
            .iter()
            .map(|&logit| (logit - max_logit).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f32>();
        for dim in 0..HEAD_DIM {
            let mut weighted_value = 0.0_f32;
            for (token, &weight) in kv.iter().zip(&weights) {
                weighted_value = weight.mul_add(token.v[kv_head * HEAD_DIM + dim], weighted_value);
            }
            output[q_head * HEAD_DIM + dim] = weighted_value / denominator;
        }
    }
    output
}

struct KVToken {
    k: Vec<f32>,
    v: Vec<f32>,
}

struct KVCacheFixture {
    pages: Buffer,
    page_ids: Buffer,
    layout: GQAPageTableLayout,
    page_id_values: Vec<u32>,
    page_values: Vec<u8>,
}

impl KVCacheFixture {
    fn new(device: &Device, layout: GQAPageTableLayout) -> Self {
        let num_pages = layout.num_page_ids();
        let mut page_id_values = (0..num_pages as u32).collect::<Vec<_>>();
        for upper in (1..num_pages).rev() {
            let selected = mix32(0x91F3_5D27 ^ upper as u32) as usize % (upper + 1);
            page_id_values.swap(upper, selected);
        }
        assert_ne!(page_id_values, (0..num_pages as u32).collect::<Vec<_>>());
        let mut page_values = vec![0_u8; num_pages * PAGE_BYTES];
        for physical_page in 0..num_pages {
            for plane_head in 0..KV_HEADS * 2 {
                for page_token in 0..TOKENS_PER_PAGE {
                    for dim in 0..HEAD_DIM {
                        let index = (physical_page * PAGE_BYTES)
                            + ((plane_head * TOKENS_PER_PAGE + page_token) * HEAD_DIM)
                            + dim;
                        let seed = (physical_page as u32).wrapping_mul(0x9E37_79B9)
                            ^ (plane_head as u32).wrapping_mul(0xA511_E9B3)
                            ^ (page_token as u32).wrapping_mul(0x63D8_3595)
                            ^ dim as u32;
                        page_values[index] = bf16_to_fp8_e4m3(f32_to_bf16(sample(seed) * 0.8));
                    }
                }
            }
        }
        Self {
            pages: Buffer::from_slice(device, &page_values),
            page_ids: Buffer::from_slice(device, &page_id_values),
            layout,
            page_id_values,
            page_values,
        }
    }

    fn read_token(&self, layout: GQAPageTableLayout, req_slot: u32, layer: u32, token: u32) -> KVToken {
        let page_slot = token as usize / TOKENS_PER_PAGE;
        let page_token = token as usize % TOKENS_PER_PAGE;
        let table_index = (((req_slot as usize * layout.num_gqa_layers as usize + layer as usize)
            * layout.num_blocks as usize)
            * layout.num_page_ids_per_block as usize)
            + page_slot;
        let physical_page = self.page_id_values[table_index] as usize;
        let mut k = vec![0.0_f32; KV_HEADS * HEAD_DIM];
        let mut v = vec![0.0_f32; KV_HEADS * HEAD_DIM];
        for head in 0..KV_HEADS {
            for dim in 0..HEAD_DIM {
                let source = physical_page * PAGE_BYTES + ((head * TOKENS_PER_PAGE + page_token) * HEAD_DIM) + dim;
                let value_source =
                    physical_page * PAGE_BYTES + (((KV_HEADS + head) * TOKENS_PER_PAGE + page_token) * HEAD_DIM) + dim;
                k[head * HEAD_DIM + dim] = bf16_to_f32(fp8_e4m3_to_bf16(self.page_values[source]));
                v[head * HEAD_DIM + dim] = bf16_to_f32(fp8_e4m3_to_bf16(self.page_values[value_source]));
            }
        }
        KVToken { k, v }
    }

    fn overwrite_group(&mut self, device: &Device, req_slot: u32, layer: u32, group: u32, value: u8) {
        let first_token = group as usize * COMPRESS_RATIO;
        for token in first_token..first_token + COMPRESS_RATIO {
            let page_slot = token / TOKENS_PER_PAGE;
            let page_token = token % TOKENS_PER_PAGE;
            let table_index = (((req_slot as usize * self.layout.num_gqa_layers as usize + layer as usize)
                * self.layout.num_blocks as usize)
                * self.layout.num_page_ids_per_block as usize)
                + page_slot;
            let physical_page = self.page_id_values[table_index] as usize;
            for plane_head in 0..KV_HEADS * 2 {
                let begin = physical_page * PAGE_BYTES + ((plane_head * TOKENS_PER_PAGE + page_token) * HEAD_DIM);
                self.page_values[begin..begin + HEAD_DIM].fill(value);
            }
        }
        self.pages.write_typed(0, &self.page_values);
    }
}

fn fixture_config() -> QSASDPAConfig {
    QSASDPAConfig {
        num_q_heads: Q_HEADS as u32,
        num_kv_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        scale: 0.041,
        page_bytes: PAGE_BYTES as u32,
        page_table_layout: GQAPageTableLayout {
            num_req_slots: 3,
            num_gqa_layers: 2,
            num_blocks: 1,
            num_page_ids_per_block: 65,
        },
        compress_ratio: COMPRESS_RATIO as u32,
        top_k: TOP_K as u32,
    }
}

fn request(req_slot: u32, token_index: u32, num_tokens: u32) -> QSASDPARequest {
    QSASDPARequest {
        req_slot,
        token_index,
        num_tokens,
        group_start: GROUP_START,
    }
}

fn fixture_q(num_rows: usize, seed: u32) -> Vec<u16> {
    let mut values: Vec<_> = (0..num_rows * ROW_WIDTH)
        .map(|index| f32_to_bf16(sample(seed ^ index as u32) * 0.55).to_bits())
        .collect();
    // One fixed query checks uniform attention beside the seeded random rows.
    values[..ROW_WIDTH].fill(0);
    values
}

fn fixture_selected_groups(num_rows: usize, group_start: u32) -> Vec<u32> {
    let mut selected = vec![0_u32; num_rows * TOP_K];
    for row in 0..num_rows {
        for group in 0..TOP_K {
            selected[row * TOP_K + group] = group_start + ((group * 73 + row * 17) % TOP_K) as u32;
        }
    }
    selected
}

fn sample(seed: u32) -> f32 {
    (mix32(seed) as f32 / u32::MAX as f32 - 0.5) * 2.0
}

fn mix32(mut value: u32) -> u32 {
    value = (value ^ (value >> 16)).wrapping_mul(0x7FEB_352D);
    value = (value ^ (value >> 15)).wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
}

fn decode_bf16(values: &[u16]) -> Vec<f32> {
    values.iter().map(|&value| bf16::from_bits(value).to_f32()).collect()
}

fn assert_close(stage: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs_error = 0.0_f32;
    let mut mean_abs_error = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "{stage} produced non-finite actual value at {index}: {actual}"
        );
        assert!(
            expected.is_finite(),
            "{stage} produced non-finite reference value at {index}: {expected}"
        );
        let error = (actual - expected).abs();
        max_abs_error = max_abs_error.max(error);
        mean_abs_error += error;
    }
    mean_abs_error /= actual.len() as f32;
    assert!(
        max_abs_error <= tolerance && mean_abs_error <= tolerance * 0.25,
        "{stage} mismatch: max_abs_error={max_abs_error} tolerance={tolerance} mean_abs_error={mean_abs_error} \
         mean_tolerance={}",
        tolerance * 0.25
    );
}
