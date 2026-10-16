use std::collections::HashSet;

use half::bf16;
use inference_backend_metal::components::gqa::fp8::bf16_to_f32;
use inference_backend_metal::components::gqa::fp8::bf16_to_fp8_e4m3;
use inference_backend_metal::components::gqa::fp8::f32_to_bf16;
use inference_backend_metal::components::gqa::fp8::fp8_e4m3_to_bf16;
use inference_backend_metal::components::qsa_indexer::CacheLayout;
use inference_backend_metal::components::qsa_indexer::Shape as QSAIndexerShape;
use inference_backend_metal::components::rms_norm_rope::RopeScaling;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_executor_core::attn::GQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::qsa_indexer::core::QSAIndexerCore;
use inference_executor_core::attn::qsa_indexer::reference as indexer_reference;
use inference_executor_core::attn::qsa_indexer::reference::Input as IndexerReferenceInput;
use inference_executor_core::attn::qsa_indexer::reference::Precision as IndexerPrecision;
use inference_executor_core::mlp::dense::reference::QuantizedAffineReferenceShape;
use inference_executor_core::mlp::dense::reference::quantized_affine_reference;

use super::QSA;
use super::QSAInput;
use super::QSAKVCacheBindings;
use super::QSAMetalConfig;
use super::QSAReplay;
use super::QSAReplayTopology;
use super::QSAWeights;
use crate::attn::qsa::batch_metadata::QSASDPAMetadataBuffers;
use crate::attn::qsa::sdpa::QSASDPAConfig;
use crate::attn::qsa::sdpa::QSASDPAShape;
use crate::attn::qsa_indexer::backend::QSAIndexerCacheBindings;
use crate::attn::qsa_indexer::batch_metadata::QSAIndexerMetadataBuffers;
use crate::attn::qsa_indexer::batch_metadata::QSAIndexerRequest;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::ReplayRecorder;
use crate::replay::Replay;
use crate::replay::ReplayComponent;

const HIDDEN_DIM: usize = 32;
const MAIN_HEAD_DIM: usize = 256;
const MAIN_Q_HEADS: usize = 24;
const MAIN_KV_HEADS: usize = 2;
const MAIN_KV_DIM: usize = MAIN_KV_HEADS * MAIN_HEAD_DIM;
const INDEX_HEAD_DIM: usize = 128;
const INDEX_Q_HEADS: usize = 2;
const GROUP_SIZE: usize = 32;
const BITS: usize = 8;
const NUM_TOKENS_PER_MAIN_PAGE: usize = 32;
const MAIN_PAGE_BYTES: u32 = 32_768;
const INDEX_GROUPS_PER_PAGE: usize = 4;
const INDEX_PAGE_BYTES: u32 = (INDEX_GROUPS_PER_PAGE * INDEX_HEAD_DIM * 2) as u32;
const INDEX_GROUP_CAPACITY: u32 = 513;
const NUM_MAIN_BLOCKS: u32 = 65;
const NUM_INDEX_PAGES_PER_REQUEST: u32 = 129;
const NUM_INDEX_STATE_SLOTS: u32 = 8;
const NORM_EPS: f32 = 1.0e-6;
const ROPE_THETA: f32 = 10_000_000.0;

#[test]
fn test_replay_bucketing() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let runtime = MetalReplayRuntime::new(&stream);
    let main_core = fixture_main_core();
    let main_config = fixture_main_config();
    let indexer_core = fixture_indexer_core();
    let indexer_config = fixture_indexer_config();
    let main_page_table = fixture_main_page_table();
    let sdpa_config = fixture_sdpa_config(main_core.clone(), indexer_core, main_config, main_page_table);
    let sdpa_shape = QSASDPAShape {
        num_total_tokens: 4,
        num_total_sparse_queries: 4,
    };
    let indexer_shape = QSAIndexerShape {
        num_total_tokens: sdpa_shape.num_total_tokens,
        num_total_sparse_queries: sdpa_shape.num_total_tokens,
        num_total_request_sparse_queries: sdpa_shape.num_total_tokens,
        num_total_score_task_templates: sdpa_shape.num_total_tokens,
        num_total_groups: INDEX_GROUP_CAPACITY,
    };
    let indexer_layout = CacheLayout {
        num_req_slots: main_page_table.num_req_slots,
        num_pages_per_request: NUM_INDEX_PAGES_PER_REQUEST,
        num_state_slots: NUM_INDEX_STATE_SLOTS,
        page_bytes: INDEX_PAGE_BYTES,
    };
    let qsa = QSA::new(
        &device,
        main_core.clone(),
        main_config,
        indexer_core,
        indexer_config,
        main_page_table,
    );
    let scratch = qsa.new_scratch(sdpa_shape, indexer_shape);
    let sdpa_metadata = QSASDPAMetadataBuffers::new(&device, sdpa_config, sdpa_shape);
    let indexer_metadata =
        QSAIndexerMetadataBuffers::new(&device, indexer_core, indexer_layout, sdpa_shape.num_total_tokens);
    let weights = FixtureWeights::new(&device, &main_core, indexer_core);
    let cache = FixtureCache::new(&device, main_page_table, indexer_layout, indexer_core, &weights);
    let hidden_state = Buffer::new_zeroed_elements(
        &device,
        sdpa_shape.num_total_tokens as usize * HIDDEN_DIM,
        Dtype::Bfloat16,
    );
    let next_hidden_state = Buffer::new_zeroed_elements(
        &device,
        sdpa_shape.num_total_tokens as usize * HIDDEN_DIM,
        Dtype::Bfloat16,
    );
    let scratch_bindings = scratch.bindings();
    let mut replay = Replay::new("test QSA", TestQSA(qsa));
    let mut recorded_keys = HashSet::new();
    let batches = [
        (
            vec![QSAIndexerRequest {
                req_slot: 0,
                token_index: 3,
                num_tokens: 1,
                group_start: 0,
                src_state_slot: 0,
            }],
            vec![2],
            (1, 1, 0),
        ),
        (
            vec![QSAIndexerRequest {
                req_slot: 1,
                token_index: 2050,
                num_tokens: 4,
                group_start: 0,
                src_state_slot: 1,
            }],
            vec![2, 3, 4, 5],
            (4, 1, 3),
        ),
        (
            vec![
                QSAIndexerRequest {
                    req_slot: 0,
                    token_index: 3,
                    num_tokens: 1,
                    group_start: 0,
                    src_state_slot: 0,
                },
                QSAIndexerRequest {
                    req_slot: 1,
                    token_index: 2050,
                    num_tokens: 2,
                    group_start: 0,
                    src_state_slot: 1,
                },
            ],
            vec![2, 3, 4],
            (3, 2, 1),
        ),
        (
            vec![QSAIndexerRequest {
                req_slot: 1,
                token_index: 2050,
                num_tokens: 2,
                group_start: 0,
                src_state_slot: 1,
            }],
            vec![2, 3],
            (2, 1, 1),
        ),
    ];

    for input_seed in [None, Some(0x82A1_1B53_u32)] {
        for (iteration, (requests, destinations, expected_active)) in batches.iter().enumerate() {
            let num_active_tokens: usize = requests.iter().map(|request| request.num_tokens as usize).sum();
            let hidden_values = match input_seed {
                None => vec![0.0; num_active_tokens * HIDDEN_DIM],
                Some(seed) => fixture_bf16_values(num_active_tokens * HIDDEN_DIM, seed + iteration as u32),
            };
            hidden_state.write_typed(0, &bf16_bits(&hidden_values));
            let active = replay.component().0.update_batch_metadata(
                &sdpa_metadata,
                &indexer_metadata,
                requests,
                destinations,
                &scratch,
            );
            assert_eq!(
                (
                    active.sdpa.num_active_tokens,
                    active.sdpa.num_active_dense_queries,
                    active.sdpa.num_active_sparse_queries,
                ),
                *expected_active
            );

            let input = QSAInput {
                batch_metadata: &sdpa_metadata,
                indexer_metadata: &indexer_metadata,
                page_table_index: ReplayU32::Fixed(0),
                hidden_state: &hidden_state,
                next_hidden_state: &next_hidden_state,
                kv_cache: QSAKVCacheBindings {
                    pages: &cache.main_pages,
                    page_ids: &cache.main_page_ids,
                },
                indexer_cache: QSAIndexerCacheBindings {
                    pages: &cache.indexer_pages,
                    page_ids: &cache.indexer_page_ids,
                    raw_states: &cache.indexer_states,
                },
                weights: weights.bindings(),
                scratch: scratch_bindings,
                active: QSAReplay::parameters(),
            };
            let reference = reference_qsa(QSAReferenceInput {
                main_core: &main_core,
                main_config,
                indexer_core,
                weights: &weights,
                hidden: &hidden_values,
                requests,
                main_history_k: &cache.main_history_k,
                main_history_v: &cache.main_history_v,
                indexer_raw_history: &cache.indexer_raw_history,
            });
            let (key, hit) = replay.record(&runtime, &input);
            assert_eq!(hit, !recorded_keys.insert(key));

            let before_states = cache
                .indexer_states
                .read_typed::<u16>(0, cache.indexer_states.len_bytes() / 2);
            let before_main_pages = cache.main_pages.read_typed::<u8>(0, cache.main_pages.len_bytes());
            let mut arguments = ReplayArguments::new();
            active.add_replay_arguments(&mut arguments);
            runtime
                .submit_replay_with_arguments(replay.replay(&key), &arguments)
                .wait();

            assert_close(
                "QGKV projection",
                &read_bf16_values(scratch_bindings.qgkv, num_active_tokens * main_core.qgkv_dim()),
                &reference.qgkv,
                0.015625,
            );
            assert_close(
                "main Q norm/RoPE",
                &read_bf16_values(scratch_bindings.q_norm_rope, num_active_tokens * main_core.q_dim()),
                &reference.main_q,
                0.015625,
            );
            assert_close(
                "main K norm/RoPE",
                &read_bf16_values(scratch_bindings.k_norm_rope, num_active_tokens * main_core.k_dim()),
                &reference.main_k,
                0.015625,
            );
            assert_close(
                "dense/sparse SDPA",
                &read_bf16_values(scratch_bindings.attention_output, num_active_tokens * main_core.q_dim()),
                &reference.attention,
                0.03125,
            );
            assert_close(
                "sigmoid gate",
                &read_bf16_values(
                    scratch_bindings.gated_attention_output,
                    num_active_tokens * main_core.q_dim(),
                ),
                &reference.gated,
                0.015625,
            );
            let actual_hidden = read_bf16_values(&next_hidden_state, reference.output.len());
            assert_close("output projection", &actual_hidden, &reference.output, 0.03125);
            assert_main_cache_writes(
                &cache.main_pages,
                &cache.main_page_id_values,
                &before_main_pages,
                requests,
                &reference.main_k_fp8,
                &reference.main_v_fp8,
            );
            assert_sparse_group_ids(scratch_bindings.selected_group_ids, requests, &reference.indexer_rows);
            assert_eq!(key.num_total_tokens, sdpa_shape.num_total_tokens);
            assert_eq!(key.num_total_sparse_queries, sdpa_shape.num_total_sparse_queries);
            assert_eq!(key.indexer_shape.num_total_groups, indexer_shape.num_total_groups);
            assert_candidate_states(
                &cache.indexer_states,
                &before_states,
                indexer_core,
                &reference.indexer_rows,
                requests,
                destinations,
            );
            if iteration == 0 {
                assert_indexer_group(
                    &cache.indexer_pages,
                    &cache.indexer_page_id_values,
                    indexer_layout,
                    0,
                    0,
                    &reference.indexer_rows[0].keys[..indexer_core.head_dim],
                    indexer_core.head_dim,
                );
            } else if iteration == 1 {
                assert_indexer_group(
                    &cache.indexer_pages,
                    &cache.indexer_page_id_values,
                    indexer_layout,
                    1,
                    512,
                    &reference.indexer_rows[0].keys[512 * indexer_core.head_dim..513 * indexer_core.head_dim],
                    indexer_core.head_dim,
                );
            }
        }
    }
}

struct TestQSA(QSA);

impl ReplayComponent for TestQSA {
    type Key = QSAReplayKey;
    type Input<'a> = QSAInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        QSAReplayKey {
            num_total_tokens: input.batch_metadata.shape().num_total_tokens,
            num_total_sparse_queries: input.batch_metadata.shape().num_total_sparse_queries,
            indexer_shape: input.scratch.indexer_shape,
            topology: self.0.replay_topology(input.batch_metadata.shape().num_total_tokens),
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        <QSA as ReplayLayer>::record(&self.0, recorder, *input);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct QSAReplayKey {
    num_total_tokens: u32,
    num_total_sparse_queries: u32,
    indexer_shape: QSAIndexerShape,
    topology: QSAReplayTopology,
}

struct FixtureWeights {
    qgkv_weight: Buffer,
    qgkv_scales: Buffer,
    qgkv_biases: Buffer,
    q_norm_scale: Buffer,
    k_norm_scale: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
    indexer_qk_weight: Buffer,
    indexer_qk_scales: Buffer,
    indexer_qk_biases: Buffer,
    indexer_q_norm: Buffer,
    indexer_k_norm: Buffer,
    qgkv_weight_values: Vec<u8>,
    qgkv_scale_values: Vec<f32>,
    qgkv_bias_values: Vec<f32>,
    q_norm_scale_values: Vec<f32>,
    k_norm_scale_values: Vec<f32>,
    output_weight_values: Vec<u8>,
    output_scale_values: Vec<f32>,
    output_bias_values: Vec<f32>,
    indexer_qk_weight_values: Vec<u8>,
    indexer_qk_scale_values: Vec<f32>,
    indexer_qk_bias_values: Vec<f32>,
    indexer_q_norm_values: Vec<f32>,
    indexer_k_norm_values: Vec<f32>,
}

impl FixtureWeights {
    fn new(device: &Device, main_core: &GQACore, indexer_core: QSAIndexerCore) -> Self {
        let qgkv_shape = affine_shape(1, main_core.qgkv_dim(), main_core.hidden_dim);
        let output_shape = affine_shape(1, main_core.hidden_dim, main_core.q_dim());
        let indexer_shape = affine_shape(1, indexer_core.qk_dim(), indexer_core.hidden_dim);
        let (qgkv_weight_values, qgkv_scale_values, qgkv_bias_values) = affine_values(qgkv_shape, 0x3D1C_4217);
        let (output_weight_values, output_scale_values, output_bias_values) = affine_values(output_shape, 0x9A41_88C3);
        let (indexer_qk_weight_values, indexer_qk_scale_values, indexer_qk_bias_values) =
            affine_values(indexer_shape, 0xC721_9D42);
        let q_norm_scale_values = (0..MAIN_HEAD_DIM)
            .map(|index| 1.0 + bf16::from_f32(-0.2 + (index % 9) as f32 * 0.05).to_f32())
            .collect::<Vec<_>>();
        let k_norm_scale_values = (0..MAIN_HEAD_DIM)
            .map(|index| 1.0 + bf16::from_f32(-0.15 + (index % 7) as f32 * 0.045).to_f32())
            .collect::<Vec<_>>();
        let indexer_q_norm_values = (0..indexer_core.head_dim)
            .map(|index| bf16::from_f32(-0.15 + (index % 7) as f32 * 0.04).to_f32())
            .collect::<Vec<_>>();
        let indexer_k_norm_values = (0..indexer_core.head_dim)
            .map(|index| bf16::from_f32(-0.1 + (index % 5) as f32 * 0.035).to_f32())
            .collect::<Vec<_>>();
        Self {
            qgkv_weight: Buffer::from_slice(device, &qgkv_weight_values),
            qgkv_scales: bf16_buffer(device, &qgkv_scale_values),
            qgkv_biases: bf16_buffer(device, &qgkv_bias_values),
            q_norm_scale: Buffer::from_slice(device, &q_norm_scale_values),
            k_norm_scale: Buffer::from_slice(device, &k_norm_scale_values),
            output_weight: Buffer::from_slice(device, &output_weight_values),
            output_scales: bf16_buffer(device, &output_scale_values),
            output_biases: bf16_buffer(device, &output_bias_values),
            indexer_qk_weight: Buffer::from_slice(device, &indexer_qk_weight_values),
            indexer_qk_scales: bf16_buffer(device, &indexer_qk_scale_values),
            indexer_qk_biases: bf16_buffer(device, &indexer_qk_bias_values),
            indexer_q_norm: bf16_buffer(device, &indexer_q_norm_values),
            indexer_k_norm: bf16_buffer(device, &indexer_k_norm_values),
            qgkv_weight_values,
            qgkv_scale_values,
            qgkv_bias_values,
            q_norm_scale_values,
            k_norm_scale_values,
            output_weight_values,
            output_scale_values,
            output_bias_values,
            indexer_qk_weight_values,
            indexer_qk_scale_values,
            indexer_qk_bias_values,
            indexer_q_norm_values,
            indexer_k_norm_values,
        }
    }

    fn bindings(&self) -> QSAWeights<'_> {
        QSAWeights {
            qgkv_weight: &self.qgkv_weight,
            qgkv_scales: &self.qgkv_scales,
            qgkv_biases: &self.qgkv_biases,
            q_norm_scale: &self.q_norm_scale,
            k_norm_scale: &self.k_norm_scale,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
            indexer: crate::attn::qsa_indexer::backend::QSAIndexerWeights {
                qk_weight: &self.indexer_qk_weight,
                qk_scales: &self.indexer_qk_scales,
                qk_biases: &self.indexer_qk_biases,
                q_norm: &self.indexer_q_norm,
                k_norm: &self.indexer_k_norm,
            },
        }
    }
}

struct FixtureCache {
    main_pages: Buffer,
    main_page_ids: Buffer,
    main_page_id_values: Vec<u32>,
    main_history_k: Vec<Vec<f32>>,
    main_history_v: Vec<Vec<f32>>,
    indexer_pages: Buffer,
    indexer_page_ids: Buffer,
    indexer_page_id_values: Vec<u32>,
    indexer_states: Buffer,
    indexer_raw_history: Vec<Vec<f32>>,
}

impl FixtureCache {
    fn new(
        device: &Device,
        main_layout: GQAPageTableLayout,
        indexer_layout: CacheLayout,
        indexer_core: QSAIndexerCore,
        weights: &FixtureWeights,
    ) -> Self {
        let main_page_id_values = (0..main_layout.num_req_slots)
            .flat_map(|req_slot| {
                (0..NUM_MAIN_BLOCKS).map(move |block| req_slot * NUM_MAIN_BLOCKS + NUM_MAIN_BLOCKS - 1 - block)
            })
            .collect::<Vec<_>>();
        let main_page_ids = Buffer::from_slice(device, &main_page_id_values);
        let num_main_physical_pages = main_layout.num_req_slots as usize * NUM_MAIN_BLOCKS as usize;
        let mut main_page_bytes = vec![0_u8; num_main_physical_pages * MAIN_PAGE_BYTES as usize];
        let history_lengths = [3_usize, 2050_usize];
        let mut main_history_k = Vec::with_capacity(history_lengths.len());
        let mut main_history_v = Vec::with_capacity(history_lengths.len());
        for (req_slot, &history_len) in history_lengths.iter().enumerate() {
            let raw_k = fixture_bf16_values(history_len * MAIN_KV_DIM, 0x7141_2345 + req_slot as u32);
            let raw_v = fixture_bf16_values(history_len * MAIN_KV_DIM, 0x9713_2E21 + req_slot as u32);
            let history_k = fp8_round_trip(&raw_k);
            let history_v = fp8_round_trip(&raw_v);
            for token in 0..history_len {
                let block = token / NUM_TOKENS_PER_MAIN_PAGE;
                let physical_page = main_page_id_values[req_slot * NUM_MAIN_BLOCKS as usize + block] as usize;
                write_host_main_token(
                    &mut main_page_bytes,
                    physical_page,
                    token % NUM_TOKENS_PER_MAIN_PAGE,
                    &history_k[token * MAIN_KV_DIM..(token + 1) * MAIN_KV_DIM],
                    &history_v[token * MAIN_KV_DIM..(token + 1) * MAIN_KV_DIM],
                );
            }
            main_history_k.push(history_k);
            main_history_v.push(history_v);
        }
        let main_pages = Buffer::from_slice(device, &main_page_bytes);

        let indexer_page_id_values = (0..indexer_layout.num_req_slots)
            .flat_map(|req_slot| {
                (0..indexer_layout.num_pages_per_request).map(move |page| {
                    req_slot * indexer_layout.num_pages_per_request + indexer_layout.num_pages_per_request - 1 - page
                })
            })
            .collect::<Vec<_>>();
        let indexer_page_ids = Buffer::from_slice(device, &indexer_page_id_values);
        let num_indexer_pages = indexer_layout.num_req_slots as usize * indexer_layout.num_pages_per_request as usize;
        let indexer_pages = Buffer::new_zeroed(device, num_indexer_pages * indexer_layout.page_bytes as usize);
        let indexer_raw_history = vec![
            fixture_bf16_values(3 * indexer_core.head_dim, 0x33B2_11C9),
            fixture_bf16_values(2050 * indexer_core.head_dim, 0xA81F_0723),
        ];
        let historical_indexer_keys = indexer_reference::projected_indexer(IndexerReferenceInput {
            core: indexer_core,
            precision: IndexerPrecision::Bf16,
            qk: &[],
            raw_history: &indexer_raw_history[1],
            group_start: 0,
            q_norm: &weights.indexer_q_norm_values,
            k_norm: &weights.indexer_k_norm_values,
        });
        for (group, key) in historical_indexer_keys
            .keys
            .chunks_exact(indexer_core.head_dim)
            .enumerate()
        {
            write_indexer_group(&indexer_pages, &indexer_page_id_values, indexer_layout, 1, group, key);
        }

        let mut state_values =
            vec![bf16::from_f32(0.125).to_bits(); indexer_layout.num_state_slots as usize * indexer_core.state_dim()];
        let state_dim = indexer_core.state_dim();
        state_values[..state_dim].copy_from_slice(&bf16_bits(&indexer_raw_history[0]));
        let source_carry = history_lengths[1] % indexer_core.compress_ratio;
        let carry_begin = (history_lengths[1] - source_carry) * indexer_core.head_dim;
        let slot_begin = state_dim;
        state_values[slot_begin..slot_begin + source_carry * indexer_core.head_dim]
            .copy_from_slice(&bf16_bits(&indexer_raw_history[1][carry_begin..]));
        let indexer_states = Buffer::from_slice(device, &state_values);
        Self {
            main_pages,
            main_page_ids,
            main_page_id_values,
            main_history_k,
            main_history_v,
            indexer_pages,
            indexer_page_ids,
            indexer_page_id_values,
            indexer_states,
            indexer_raw_history,
        }
    }
}

struct QSAReference {
    qgkv: Vec<f32>,
    main_q: Vec<f32>,
    main_k: Vec<f32>,
    attention: Vec<f32>,
    gated: Vec<f32>,
    output: Vec<f32>,
    main_k_fp8: Vec<f32>,
    main_v_fp8: Vec<f32>,
    indexer_rows: Vec<indexer_reference::Output>,
}

struct QSAReferenceInput<'a> {
    main_core: &'a GQACore,
    main_config: QSAMetalConfig,
    indexer_core: QSAIndexerCore,
    weights: &'a FixtureWeights,
    hidden: &'a [f32],
    requests: &'a [QSAIndexerRequest],
    main_history_k: &'a [Vec<f32>],
    main_history_v: &'a [Vec<f32>],
    indexer_raw_history: &'a [Vec<f32>],
}

fn reference_qsa(input: QSAReferenceInput<'_>) -> QSAReference {
    let QSAReferenceInput {
        main_core,
        main_config,
        indexer_core,
        weights,
        hidden,
        requests,
        main_history_k,
        main_history_v,
        indexer_raw_history,
    } = input;
    let num_rows = requests.iter().map(|request| request.num_tokens as usize).sum();
    let qgkv = bf16_round_trip(&quantized_affine_reference(
        affine_shape(num_rows, main_core.qgkv_dim(), main_core.hidden_dim),
        hidden,
        &weights.qgkv_weight_values,
        &weights.qgkv_scale_values,
        &weights.qgkv_bias_values,
    ));
    let mut q = Vec::with_capacity(num_rows * main_core.q_dim());
    let mut gate = Vec::with_capacity(num_rows * main_core.g_dim());
    let mut k = Vec::with_capacity(num_rows * main_core.k_dim());
    let mut v = Vec::with_capacity(num_rows * main_core.v_dim());
    for row in qgkv.chunks_exact(main_core.qgkv_dim()) {
        for head in 0..main_core.num_q_heads {
            let pair = &row[head * 2 * main_core.head_dim..(head + 1) * 2 * main_core.head_dim];
            q.extend_from_slice(&pair[..main_core.head_dim]);
            gate.extend_from_slice(&pair[main_core.head_dim..]);
        }
        let kv_begin = 2 * main_core.q_dim();
        k.extend_from_slice(&row[kv_begin..kv_begin + main_core.k_dim()]);
        v.extend_from_slice(&row[kv_begin + main_core.k_dim()..]);
    }
    let positions = requests
        .iter()
        .flat_map(|request| {
            (0..request.num_tokens).map(move |local_token| (request.token_index + local_token) as usize)
        })
        .collect::<Vec<_>>();
    let q = main_norm_rope(
        &q,
        num_rows,
        MAIN_Q_HEADS,
        MAIN_HEAD_DIM,
        &weights.q_norm_scale_values,
        positions.as_slice(),
        main_config,
    );
    let k = main_norm_rope(
        &k,
        num_rows,
        MAIN_KV_HEADS,
        MAIN_HEAD_DIM,
        &weights.k_norm_scale_values,
        positions.as_slice(),
        main_config,
    );
    let main_k_fp8 = fp8_round_trip(&k);
    let main_v_fp8 = fp8_round_trip(&v);

    let indexer_qk = bf16_round_trip(&quantized_affine_reference(
        affine_shape(num_rows, indexer_core.qk_dim(), indexer_core.hidden_dim),
        hidden,
        &weights.indexer_qk_weight_values,
        &weights.indexer_qk_scale_values,
        &weights.indexer_qk_bias_values,
    ));
    let mut indexer_rows = Vec::with_capacity(requests.len());
    let mut flat_begin = 0;
    for request in requests {
        let flat_end = flat_begin + request.num_tokens as usize;
        indexer_rows.push(indexer_reference::projected_indexer(IndexerReferenceInput {
            core: indexer_core,
            precision: IndexerPrecision::Bf16,
            qk: &indexer_qk[flat_begin * indexer_core.qk_dim()..flat_end * indexer_core.qk_dim()],
            raw_history: &indexer_raw_history[request.req_slot as usize],
            group_start: request.group_start as usize,
            q_norm: &weights.indexer_q_norm_values,
            k_norm: &weights.indexer_k_norm_values,
        }));
        flat_begin = flat_end;
    }

    let mut attention = vec![0.0_f32; num_rows * main_core.q_dim()];
    let mut flat_begin = 0;
    for (request, indexer_row) in requests.iter().zip(&indexer_rows) {
        let history_k = &main_history_k[request.req_slot as usize];
        let history_v = &main_history_v[request.req_slot as usize];
        assert_eq!(history_k.len(), request.token_index as usize * main_core.k_dim());
        assert_eq!(history_v.len(), request.token_index as usize * main_core.v_dim());
        for (local_token, group_ids) in indexer_row.group_ids.iter().enumerate() {
            let row = flat_begin + local_token;
            let position = request.token_index as usize + local_token;
            let query_context = if (position + 1) / indexer_core.compress_ratio - request.group_start as usize
                <= indexer_core.top_k
            {
                (request.group_start as usize * indexer_core.compress_ratio..=position).collect::<Vec<_>>()
            } else {
                let mut positions =
                    Vec::with_capacity(indexer_core.top_k * indexer_core.compress_ratio + indexer_core.compress_ratio);
                for &group in group_ids {
                    for token in
                        group as usize * indexer_core.compress_ratio..(group as usize + 1) * indexer_core.compress_ratio
                    {
                        assert!(token <= position);
                        positions.push(token);
                    }
                }
                let complete_groups = (position + 1) / indexer_core.compress_ratio;
                let tail_tokens = (position + 1) % indexer_core.compress_ratio;
                let tail_begin = complete_groups * indexer_core.compress_ratio;
                positions.extend(tail_begin..tail_begin + tail_tokens);
                positions
            };
            for q_head in 0..main_core.num_q_heads {
                let kv_head = q_head / (main_core.num_q_heads / main_core.num_kv_heads);
                let query = &q[row * main_core.q_dim() + q_head * main_core.head_dim
                    ..row * main_core.q_dim() + (q_head + 1) * main_core.head_dim];
                let mut logits = Vec::with_capacity(query_context.len());
                for &token in &query_context {
                    let key = if token < request.token_index as usize {
                        &history_k[token * main_core.k_dim() + kv_head * main_core.head_dim
                            ..token * main_core.k_dim() + (kv_head + 1) * main_core.head_dim]
                    } else {
                        let current_row = flat_begin + token - request.token_index as usize;
                        assert!(current_row <= row);
                        &main_k_fp8[current_row * main_core.k_dim() + kv_head * main_core.head_dim
                            ..current_row * main_core.k_dim() + (kv_head + 1) * main_core.head_dim]
                    };
                    logits.push(query.iter().zip(key).map(|(query, key)| query * key).sum::<f32>() * main_core.scale);
                }
                let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probabilities = logits.iter().map(|logit| (logit - max_logit).exp()).collect::<Vec<_>>();
                let denominator = probabilities.iter().sum::<f32>();
                probabilities
                    .iter_mut()
                    .for_each(|probability| *probability /= denominator);
                for dim in 0..main_core.head_dim {
                    let value = query_context
                        .iter()
                        .zip(&probabilities)
                        .map(|(&token, &probability)| {
                            let value = if token < request.token_index as usize {
                                history_v[token * main_core.v_dim() + kv_head * main_core.head_dim + dim]
                            } else {
                                let current_row = flat_begin + token - request.token_index as usize;
                                main_v_fp8[current_row * main_core.v_dim() + kv_head * main_core.head_dim + dim]
                            };
                            probability * value
                        })
                        .sum::<f32>();
                    attention[row * main_core.q_dim() + q_head * main_core.head_dim + dim] =
                        bf16_round_trip_value(value);
                }
            }
        }
        flat_begin += request.num_tokens as usize;
    }
    let gated = attention
        .iter()
        .zip(&gate)
        .map(|(&attention, &gate)| bf16_round_trip_value(attention * sigmoid_bf16(gate)))
        .collect::<Vec<_>>();
    let output = bf16_round_trip(&quantized_affine_reference(
        affine_shape(num_rows, main_core.hidden_dim, main_core.q_dim()),
        &gated,
        &weights.output_weight_values,
        &weights.output_scale_values,
        &weights.output_bias_values,
    ));
    QSAReference {
        qgkv,
        main_q: q,
        main_k: k,
        attention,
        gated,
        output,
        main_k_fp8,
        main_v_fp8,
        indexer_rows,
    }
}

fn main_norm_rope(
    input: &[f32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    norm_scale: &[f32],
    positions: &[usize],
    config: QSAMetalConfig,
) -> Vec<f32> {
    assert_eq!(input.len(), num_tokens * num_heads * head_dim);
    assert_eq!(norm_scale.len(), head_dim);
    assert_eq!(positions.len(), num_tokens);
    let rope_dim = config.rope_dim as usize;
    let rope_half = rope_dim / 2;
    let mut output = vec![0.0; input.len()];
    for (token, &position) in positions.iter().enumerate().take(num_tokens) {
        for head in 0..num_heads {
            let row_begin = (token * num_heads + head) * head_dim;
            let row = &input[row_begin..row_begin + head_dim];
            let inv_rms = (row.iter().map(|value| value * value).sum::<f32>() / head_dim as f32 + config.norm_eps)
                .sqrt()
                .recip();
            for dim in 0..rope_half {
                let other = dim + rope_half;
                let x = norm_scale[dim] * row[dim] * inv_rms;
                let y = norm_scale[other] * row[other] * inv_rms;
                let inverse_frequency = config.rope_theta.powf(-2.0 * dim as f32 / rope_dim as f32);
                let theta = position as f32 * inverse_frequency;
                let (sin, cos) = theta.sin_cos();
                output[row_begin + dim] = bf16_round_trip_value(x * cos - y * sin);
                output[row_begin + other] = bf16_round_trip_value(x * sin + y * cos);
            }
            for dim in rope_dim..head_dim {
                output[row_begin + dim] = bf16_round_trip_value(norm_scale[dim] * row[dim] * inv_rms);
            }
        }
    }
    output
}

fn assert_sparse_group_ids(
    selected_group_ids: &Buffer,
    requests: &[QSAIndexerRequest],
    references: &[indexer_reference::Output],
) {
    let num_tokens: usize = requests.iter().map(|request| request.num_tokens as usize).sum();
    let selected = selected_group_ids.read_typed::<u32>(0, num_tokens * 512);
    let mut flat_row = 0;
    for (request, reference) in requests.iter().zip(references) {
        for (local_token, group_ids) in reference.group_ids.iter().enumerate() {
            let position = request.token_index + local_token as u32;
            if (position + 1) / 4 - request.group_start > 512 {
                assert_eq!(group_ids.len(), 512);
                assert_eq!(&selected[flat_row * 512..(flat_row + 1) * 512], group_ids);
            }
            flat_row += 1;
        }
    }
}

fn assert_main_cache_writes(
    pages: &Buffer,
    page_ids: &[u32],
    before: &[u8],
    requests: &[QSAIndexerRequest],
    expected_k: &[f32],
    expected_v: &[f32],
) {
    let bytes = pages.read_typed::<u8>(0, pages.len_bytes());
    let mut expected = before.to_vec();
    let mut row = 0;
    for request in requests {
        for local_token in 0..request.num_tokens as usize {
            let token_index = request.token_index as usize + local_token;
            let block = token_index / NUM_TOKENS_PER_MAIN_PAGE;
            let page = page_ids[request.req_slot as usize * NUM_MAIN_BLOCKS as usize + block] as usize;
            let token = token_index % NUM_TOKENS_PER_MAIN_PAGE;
            for head in 0..MAIN_KV_HEADS {
                for dim in 0..MAIN_HEAD_DIM {
                    let source = row * MAIN_KV_DIM + head * MAIN_HEAD_DIM + dim;
                    let k_target = page * MAIN_PAGE_BYTES as usize
                        + (head * NUM_TOKENS_PER_MAIN_PAGE + token) * MAIN_HEAD_DIM
                        + dim;
                    let v_target = page * MAIN_PAGE_BYTES as usize
                        + ((MAIN_KV_HEADS + head) * NUM_TOKENS_PER_MAIN_PAGE + token) * MAIN_HEAD_DIM
                        + dim;
                    expected[k_target] = fp8_bits(expected_k[source]);
                    expected[v_target] = fp8_bits(expected_v[source]);
                }
            }
            row += 1;
        }
    }
    assert_eq!(
        bytes
            .iter()
            .zip(&expected)
            .position(|(actual, expected)| actual != expected),
        None
    );
}

fn assert_indexer_group(
    pages: &Buffer,
    page_ids: &[u32],
    layout: CacheLayout,
    req_slot: u32,
    group: usize,
    expected: &[f32],
    head_dim: usize,
) {
    let group_page_slot = group / INDEX_GROUPS_PER_PAGE;
    let physical_page = page_ids[req_slot as usize * layout.num_pages_per_request as usize + group_page_slot] as usize;
    let page_offset = physical_page * layout.page_bytes as usize / 2;
    let group_offset = (group % INDEX_GROUPS_PER_PAGE) * head_dim;
    let actual = pages.read_typed::<u16>(page_offset + group_offset, head_dim);
    assert_eq!(actual, bf16_bits(expected));
}

fn assert_candidate_states(
    states: &Buffer,
    before: &[u16],
    indexer_core: QSAIndexerCore,
    references: &[indexer_reference::Output],
    requests: &[QSAIndexerRequest],
    dst_state_slots: &[u32],
) {
    let after = states.read_typed::<u16>(0, before.len());
    let mut expected = before.to_vec();
    let state_dim = indexer_core.state_dim();
    let mut flat_row = 0;
    for (request, reference) in requests.iter().zip(references) {
        for local_token in 0..request.num_tokens as usize {
            let end_token = request.token_index as usize + local_token + 1;
            let carry = end_token % indexer_core.compress_ratio;
            let carry_begin = end_token - carry;
            let slot_begin = dst_state_slots[flat_row] as usize * state_dim;
            let raw = &reference.raw_keys[carry_begin * indexer_core.head_dim..end_token * indexer_core.head_dim];
            expected[slot_begin..slot_begin + carry * indexer_core.head_dim].copy_from_slice(&bf16_bits(raw));
            flat_row += 1;
        }
    }
    assert_eq!(after, expected);
}

fn fixture_main_core() -> GQACore {
    GQACore::new(
        0,
        HIDDEN_DIM,
        MAIN_HEAD_DIM,
        MAIN_Q_HEADS,
        MAIN_KV_HEADS,
        (MAIN_HEAD_DIM as f32).sqrt().recip(),
    )
}

fn fixture_main_config() -> QSAMetalConfig {
    QSAMetalConfig {
        group_size: GROUP_SIZE as u32,
        bits: BITS as u32,
        page_bytes: MAIN_PAGE_BYTES,
        rope_dim: 64,
        norm_eps: NORM_EPS,
        rope_theta: ROPE_THETA,
        rope_scaling: RopeScaling::Default,
        io_dtype: Dtype::Bfloat16,
    }
}

fn fixture_indexer_core() -> QSAIndexerCore {
    QSAIndexerCore {
        hidden_dim: HIDDEN_DIM,
        num_q_heads: INDEX_Q_HEADS,
        head_dim: INDEX_HEAD_DIM,
        compress_ratio: 4,
        top_k: 512,
        rope_dim: 64,
        rope_theta: ROPE_THETA,
        norm_eps: NORM_EPS,
    }
}

fn fixture_indexer_config() -> crate::attn::qsa_indexer::backend::QSAIndexerMetalConfig {
    crate::attn::qsa_indexer::backend::QSAIndexerMetalConfig {
        group_size: GROUP_SIZE as u32,
        bits: BITS as u32,
        io_dtype: Dtype::Bfloat16,
    }
}

fn fixture_main_page_table() -> GQAPageTableLayout {
    GQAPageTableLayout {
        num_req_slots: 2,
        num_gqa_layers: 1,
        num_blocks: NUM_MAIN_BLOCKS,
        num_page_ids_per_block: 1,
    }
}

fn fixture_sdpa_config(
    core: GQACore,
    indexer_core: QSAIndexerCore,
    config: QSAMetalConfig,
    page_table_layout: GQAPageTableLayout,
) -> QSASDPAConfig {
    QSASDPAConfig {
        num_q_heads: core.num_q_heads as u32,
        num_kv_heads: core.num_kv_heads as u32,
        head_dim: core.head_dim as u32,
        scale: core.scale,
        page_bytes: config.page_bytes,
        page_table_layout,
        compress_ratio: indexer_core.compress_ratio as u32,
        top_k: indexer_core.top_k as u32,
    }
}

fn affine_shape(num_rows: usize, output_dim: usize, input_dim: usize) -> QuantizedAffineReferenceShape {
    QuantizedAffineReferenceShape {
        num_rows,
        output_dim,
        input_dim,
        group_size: GROUP_SIZE,
        bits: BITS,
    }
}

fn affine_values(shape: QuantizedAffineReferenceShape, seed: u32) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut weight_state = seed;
    let weight = (0..shape.weight_bytes())
        .map(|_| {
            weight_state = weight_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (weight_state >> 24) as u8
        })
        .collect::<Vec<_>>();
    let mut parameter_state = seed ^ 0x9E37_79B9;
    let scales = (0..shape.affine_param_len())
        .map(|_| {
            parameter_state = parameter_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = parameter_state as f32 / u32::MAX as f32;
            bf16::from_f32(0.000_4 + value * 0.000_8).to_f32()
        })
        .collect::<Vec<_>>();
    let biases = (0..shape.affine_param_len())
        .map(|_| {
            parameter_state = parameter_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = parameter_state as f32 / u32::MAX as f32;
            bf16::from_f32((value - 0.5) * 0.02).to_f32()
        })
        .collect::<Vec<_>>();
    (weight, scales, biases)
}

fn write_host_main_token(pages: &mut [u8], page: usize, token: usize, k: &[f32], v: &[f32]) {
    for head in 0..MAIN_KV_HEADS {
        for dim in 0..MAIN_HEAD_DIM {
            let source = head * MAIN_HEAD_DIM + dim;
            let k_target =
                page * MAIN_PAGE_BYTES as usize + (head * NUM_TOKENS_PER_MAIN_PAGE + token) * MAIN_HEAD_DIM + dim;
            let v_target = page * MAIN_PAGE_BYTES as usize
                + ((MAIN_KV_HEADS + head) * NUM_TOKENS_PER_MAIN_PAGE + token) * MAIN_HEAD_DIM
                + dim;
            pages[k_target] = fp8_bits(k[source]);
            pages[v_target] = fp8_bits(v[source]);
        }
    }
}

fn write_indexer_group(
    pages: &Buffer,
    page_ids: &[u32],
    layout: CacheLayout,
    req_slot: u32,
    group: usize,
    key: &[f32],
) {
    let groups_per_page = layout.page_bytes as usize / (key.len() * 2);
    let page_slot = group / groups_per_page;
    let physical_page = page_ids[req_slot as usize * layout.num_pages_per_request as usize + page_slot] as usize;
    let page_offset = physical_page * layout.page_bytes as usize / 2;
    let group_offset = (group % groups_per_page) * key.len();
    pages.write_typed(page_offset + group_offset, &bf16_bits(key));
}

fn fixture_bf16_values(count: usize, mut state: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bf16_round_trip_value(((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0)
        })
        .collect()
}

fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    Buffer::from_slice(device, &bf16_bits(values))
}

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&value| bf16::from_f32(value).to_bits()).collect()
}

fn read_bf16_values(buffer: &Buffer, len: usize) -> Vec<f32> {
    buffer
        .read_typed::<u16>(0, len)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect()
}

fn bf16_round_trip(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&value| bf16_round_trip_value(value)).collect()
}

fn bf16_round_trip_value(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

fn fp8_bits(value: f32) -> u8 {
    bf16_to_fp8_e4m3(f32_to_bf16(value))
}

fn fp8_round_trip(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&value| bf16_to_f32(fp8_e4m3_to_bf16(fp8_bits(value))))
        .collect()
}

fn sigmoid_bf16(value: f32) -> f32 {
    let absolute = bf16_round_trip_value(value.abs());
    let exponential = bf16_round_trip_value((-absolute).exp());
    let denominator = bf16_round_trip_value(1.0 + exponential);
    let positive = bf16_round_trip_value(1.0 / denominator);
    if value < 0.0 {
        bf16_round_trip_value(1.0 - positive)
    } else {
        positive
    }
}

fn assert_close(stage: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "QSA {stage} produced non-finite values"
    );
    assert!(
        expected.iter().all(|value| value.is_finite()),
        "QSA {stage} reference produced non-finite values"
    );
    let mean_abs_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / actual.len() as f32;
    let max_abs_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let max_error_index = actual
        .iter()
        .zip(expected)
        .enumerate()
        .max_by(
            |(_, (actual_left, expected_left)), (_, (actual_right, expected_right))| {
                (*actual_left - *expected_left)
                    .abs()
                    .total_cmp(&(*actual_right - *expected_right).abs())
            },
        )
        .map(|(index, _)| index)
        .expect("QSA test outputs must be non-empty");
    assert!(
        max_abs_error <= tolerance && mean_abs_error <= tolerance * 0.25,
        "QSA {stage} mismatch: max_abs_error={max_abs_error} tolerance={tolerance} at {max_error_index} actual={} \
         expected={} mean_abs_error={mean_abs_error} mean_tolerance={}",
        actual[max_error_index],
        expected[max_error_index],
        tolerance * 0.25
    );
}
