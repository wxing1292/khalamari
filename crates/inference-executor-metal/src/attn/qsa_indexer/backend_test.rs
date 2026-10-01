use half::bf16;
use inference_backend_metal::components::qsa_indexer::CacheLayout;
use inference_backend_metal::components::qsa_indexer::Shape;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::Stream;
use inference_executor_core::attn::qsa_indexer::reference::Precision;
use inference_executor_core::attn::qsa_indexer::reference::{self};
use inference_executor_core::mlp::dense::reference::QuantizedAffineReferenceShape;
use inference_executor_core::mlp::dense::reference::quantized_affine_reference;

use super::*;
use crate::attn::qsa_indexer::batch_metadata::QSAIndexerRequest;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::ReplayRecorder;
use crate::replay::Replay;
use crate::replay::ReplayComponent;

const ACTIVE: ReplayParameterKey = ReplayParameterKey::new("test.indexer.num_active_tokens");

#[test]
fn test_replay_restore() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let runtime = MetalReplayRuntime::new(&stream);
    let core = QSAIndexerCore {
        hidden_dim: 32,
        num_q_heads: 2,
        head_dim: 8,
        compress_ratio: 4,
        top_k: 512,
        rope_dim: 4,
        rope_theta: 10_000_000.0,
        norm_eps: 1e-6,
    };
    let component = QSAIndexer::new(
        &device,
        core,
        QSAIndexerMetalConfig {
            group_size: 32,
            bits: 8,
            io_dtype: Dtype::Bfloat16,
        },
    );
    let layout = CacheLayout {
        num_req_slots: 3,
        num_pages_per_request: 129,
        num_state_slots: 12,
        page_bytes: 64,
    };
    let shape = Shape {
        num_total_tokens: 4,
        num_total_groups: 514,
    };
    let scratch = component.new_scratch(&device, shape.num_total_tokens, shape.num_total_groups);
    let metadata = QSAIndexerMetadataBuffers::new(&device, core, layout, 4);
    let hidden_values = fixture_values(4 * core.hidden_dim, 11);
    let hidden = bf16_buffer(&device, &hidden_values);
    let weight_values: Vec<u8> = (0..core.qk_dim() * core.hidden_dim)
        .map(|index| ((index * 17 + index / 11) % 16) as u8)
        .collect();
    let weight = Buffer::from_slice(&device, &weight_values);
    let scales = vec![0.03125; core.qk_dim()];
    let biases = vec![-0.25; core.qk_dim()];
    let scale = bf16_buffer(&device, &scales);
    let bias = bf16_buffer(&device, &biases);
    let norm_values = fixture_values(core.head_dim, 3);
    let norm = bf16_buffer(&device, &norm_values);
    let num_pages = layout.num_req_slots * layout.num_pages_per_request;
    let pages = bf16_buffer(
        &device,
        &vec![0.25; num_pages as usize * layout.page_bytes as usize / 2],
    );
    let page_ids_values: Vec<_> = (0..num_pages).rev().collect();
    let page_ids = Buffer::from_slice(&device, &page_ids_values);
    let states = bf16_buffer(
        &device,
        &vec![0.125; layout.num_state_slots as usize * core.state_dim()],
    );
    let initial_raw = fixture_values(core.head_dim, 31);
    states.write_typed(0, &bits(&initial_raw));
    let ids = Buffer::new_zeroed_elements(&device, 4 * core.top_k, Dtype::Uint32);
    let input = Input {
        shape: Shape {
            num_total_groups: core.top_k as u32,
            ..shape
        },
        num_active_tokens: ReplayU32::Parameter(ACTIVE),
        hidden_state: &hidden,
        batch_metadata: &metadata,
        cache: QSAIndexerCacheBindings {
            pages: &pages,
            page_ids: &page_ids,
            raw_states: &states,
        },
        weights: QSAIndexerWeights {
            qk_weight: &weight,
            qk_scales: &scale,
            qk_biases: &bias,
            q_norm: &norm,
            k_norm: &norm,
        },
        scratch: &scratch,
        group_ids: &ids,
    };
    let mut replay = Replay::new("test-indexer", TestIndexer(component));
    let project = |hidden: &[f32], count| {
        quantized_affine_reference(
            QuantizedAffineReferenceShape {
                num_rows: count,
                output_dim: core.qk_dim(),
                input_dim: core.hidden_dim,
                group_size: 32,
                bits: 8,
            },
            hidden,
            &weight_values,
            &scales,
            &biases,
        )
        .into_iter()
        .map(|x| bf16::from_f32(x).to_f32())
        .collect::<Vec<_>>()
    };
    let qk = project(&hidden_values, 4);
    let full_reference = reference::projected_indexer(reference::Input {
        core,
        precision: Precision::Bf16,
        qk: &qk,
        raw_history: &initial_raw,
        group_start: 0,
        q_norm: &norm_values,
        k_norm: &norm_values,
    });
    for (iteration, active) in [1_u32, 4, 3, 2].into_iter().enumerate() {
        metadata.update(
            &[QSAIndexerRequest {
                req_slot: 0,
                token_index: 1,
                num_tokens: active,
                group_start: 0,
                src_state_slot: 0,
            }],
            &(1..=active).collect::<Vec<_>>(),
            input.shape,
        );
        let before_states = values(&states);
        let (key, hit) = replay.record(&runtime, &input);
        assert_eq!(hit, iteration != 0);
        runtime
            .submit_replay_with_arguments(replay.replay(&key), &ReplayArguments::new().with_u32(ACTIVE, active))
            .wait();
        let result = reference::projected_indexer(reference::Input {
            core,
            precision: Precision::Bf16,
            qk: &qk[..active as usize * core.qk_dim()],
            raw_history: &initial_raw,
            group_start: 0,
            q_norm: &norm_values,
            k_norm: &norm_values,
        });
        check(core, input, &result, 1, &(1..=active).collect::<Vec<_>>());
        let after_states = values(&states);
        assert_eq!(&after_states[..core.state_dim()], &before_states[..core.state_dim()]);
        assert_eq!(
            &after_states[(active as usize + 1) * core.state_dim()..],
            &before_states[(active as usize + 1) * core.state_dim()..]
        );
    }
    let (smaller_key, hit) = replay.record(
        &runtime,
        &Input {
            shape: Shape {
                num_total_tokens: 2,
                ..input.shape
            },
            ..input
        },
    );
    assert!(!hit);
    assert_eq!(smaller_key.0.num_total_tokens, 2);

    // Accept only the first token. The rejected suffix already wrote a complete key.
    // Restore its earlier raw state without copying or overwriting the source slot.
    let continued_hidden = fixture_values(4 * core.hidden_dim, 29);
    hidden.write_typed(0, &bits(&continued_hidden));
    metadata.update(
        &[QSAIndexerRequest {
            req_slot: 0,
            token_index: 2,
            num_tokens: 3,
            group_start: 0,
            src_state_slot: 1,
        }],
        &[5, 6, 7],
        input.shape,
    );
    let restored_source = values(&states)[core.state_dim()..2 * core.state_dim()].to_vec();
    let (key, hit) = replay.record(&runtime, &input);
    assert!(hit);
    runtime
        .submit_replay_with_arguments(replay.replay(&key), &ReplayArguments::new().with_u32(ACTIVE, 3))
        .wait();
    let next_qk = project(&continued_hidden[..3 * core.hidden_dim], 3);
    let continued = reference::projected_indexer(reference::Input {
        core,
        precision: Precision::Bf16,
        qk: &next_qk,
        raw_history: &full_reference.raw_keys[..2 * core.head_dim],
        group_start: 0,
        q_norm: &norm_values,
        k_norm: &norm_values,
    });
    check(core, input, &continued, 2, &[5, 6, 7]);
    assert_eq!(
        &values(&states)[core.state_dim()..2 * core.state_dim()],
        restored_source
    );

    // Cross 2051 -> 2052 visible tokens and an index page boundary in one chunk.
    // Mix it with the restored request. Request slots differ from batch order.
    let other_history = fixture_values(2050 * core.head_dim, 43);
    let requests = [
        QSAIndexerRequest {
            req_slot: 2,
            token_index: 2050,
            num_tokens: 3,
            group_start: 0,
            src_state_slot: 2,
        },
        QSAIndexerRequest {
            req_slot: 0,
            token_index: 2,
            num_tokens: 1,
            group_start: 0,
            src_state_slot: 1,
        },
    ];
    let histories = [&other_history[..], &full_reference.raw_keys[..2 * core.head_dim]];
    let qk = project(&continued_hidden, 4);
    let groups_per_page = layout.page_bytes as usize / (2 * core.head_dim);
    let page_offset = |req_slot: u32, group: usize| {
        let page = page_ids_values[req_slot as usize * layout.num_pages_per_request as usize + group / groups_per_page];
        (page as usize * groups_per_page + group % groups_per_page) * core.head_dim
    };
    let initial = reference::projected_indexer(reference::Input {
        core,
        precision: Precision::Bf16,
        qk: &[],
        raw_history: &other_history,
        group_start: 0,
        q_norm: &norm_values,
        k_norm: &norm_values,
    });
    for (group, row) in initial.keys.chunks_exact(core.head_dim).enumerate() {
        pages.write_typed(page_offset(2, group), &bits(row));
    }
    states.write_typed(2 * core.state_dim(), &bits(&other_history[2048 * core.head_dim..]));
    let destinations = [8, 9, 10, 11];
    assert_eq!(metadata.update(&requests, &destinations, shape), 4);
    let mut expected_pages = values(&pages);
    let mut expected_states = values(&states);
    let input = Input { shape, ..input };
    let (key, hit) = replay.record(&runtime, &input);
    assert!(!hit);
    runtime
        .submit_replay_with_arguments(replay.replay(&key), &ReplayArguments::new().with_u32(ACTIVE, 4))
        .wait();
    let queries = values(scratch.q());
    let ids = ids.read_typed::<u32>(0, 4 * core.top_k);
    let mut flat_token = 0;
    let mut written_groups = Vec::new();
    for (request, history) in requests.iter().zip(histories) {
        let end = flat_token + request.num_tokens as usize;
        let reference = reference::projected_indexer(reference::Input {
            core,
            precision: Precision::Bf16,
            qk: &qk[flat_token * core.qk_dim()..end * core.qk_dim()],
            raw_history: history,
            group_start: request.group_start as usize,
            q_norm: &norm_values,
            k_norm: &norm_values,
        });
        assert_close(
            &queries[flat_token * core.num_q_heads * core.head_dim..end * core.num_q_heads * core.head_dim],
            &reference.q,
        );
        for token in 0..request.num_tokens as usize {
            let flat = flat_token + token;
            let selected = &ids[flat * core.top_k..(flat + 1) * core.top_k];
            if reference.scores[token].len() > core.top_k {
                assert_eq!(selected, &reference.group_ids[token]);
            } else {
                assert!(selected.iter().all(|&id| id == 0));
            }
            let position = request.token_index as usize + token + 1;
            let carry = position % core.compress_ratio;
            let offset = destinations[flat] as usize * core.state_dim();
            expected_states[offset..offset + carry * core.head_dim]
                .copy_from_slice(&reference.raw_keys[(position - carry) * core.head_dim..position * core.head_dim]);
            if carry == 0 {
                let group = position / core.compress_ratio - 1;
                let offset = page_offset(request.req_slot, group);
                let range = offset..offset + core.head_dim;
                expected_pages[range.clone()]
                    .copy_from_slice(&reference.keys[group * core.head_dim..(group + 1) * core.head_dim]);
                written_groups.push(range);
            }
        }
        flat_token = end;
    }
    let actual_pages = values(&pages);
    for range in written_groups {
        assert_close(&actual_pages[range.clone()], &expected_pages[range.clone()]);
        expected_pages[range.clone()].copy_from_slice(&actual_pages[range]);
    }
    assert_eq!(actual_pages, expected_pages);
    assert_eq!(values(&states), expected_states);
}

#[derive(Clone, Copy)]
struct Input<'a> {
    shape: Shape,
    num_active_tokens: ReplayU32,
    hidden_state: &'a Buffer,
    batch_metadata: &'a QSAIndexerMetadataBuffers,
    cache: QSAIndexerCacheBindings<'a>,
    weights: QSAIndexerWeights<'a>,
    scratch: &'a QSAIndexerScratch,
    group_ids: &'a Buffer,
}

struct TestIndexer(QSAIndexer);
impl ReplayComponent for TestIndexer {
    type Key = (Shape, CacheLayout, affine_quantized::KernelKind);
    type Input<'a> = Input<'a>;
    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        (
            input.shape,
            input.batch_metadata.layout(),
            self.0.replay_topology(input.shape.num_total_tokens),
        )
    }
    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        self.0.prepare(
            recorder,
            QSAIndexerPrepareInput {
                num_total_tokens: input.shape.num_total_tokens,
                num_active_tokens: input.num_active_tokens,
                hidden_state: input.hidden_state,
                batch_metadata: input.batch_metadata,
                cache: input.cache,
                weights: input.weights,
                scratch: input.scratch,
            },
        );
        self.0.select(
            recorder,
            QSAIndexerSelectInput {
                shape: input.shape,
                num_active_tokens: input.num_active_tokens,
                batch_metadata: input.batch_metadata,
                pages: input.cache.pages,
                page_ids: input.cache.page_ids,
                scratch: input.scratch,
                group_ids: input.group_ids,
            },
        );
    }
}

fn check(core: QSAIndexerCore, input: Input<'_>, reference: &reference::Output, position: usize, slots: &[u32]) {
    let queries = values(input.scratch.q());
    assert_close(&queries[..reference.q.len()], &reference.q);
    let ids = input.group_ids.read_typed::<u32>(0, slots.len() * core.top_k);
    let states = values(input.cache.raw_states);
    for (token, &slot) in slots.iter().enumerate() {
        let selected = &ids[token * core.top_k..(token + 1) * core.top_k];
        if reference.scores[token].len() > core.top_k {
            assert_eq!(selected, &reference.group_ids[token]);
        } else {
            assert!(selected.iter().all(|&id| id == 0));
        }
        let end = position + token + 1;
        let carry = end % core.compress_ratio;
        assert_eq!(
            &states[slot as usize * core.state_dim()..slot as usize * core.state_dim() + carry * core.head_dim],
            &reference.raw_keys[(end - carry) * core.head_dim..end * core.head_dim]
        );
    }
    // This request's logical page zero holds its first complete groups.
    let pages = values(input.cache.pages);
    let page = input.cache.page_ids.read_typed::<u32>(0, 1)[0] as usize;
    let offset = page * input.batch_metadata.layout().page_bytes as usize / 2;
    let written_groups = (position + slots.len()) / core.compress_ratio;
    assert_close(
        &pages[offset..offset + written_groups * core.head_dim],
        &reference.keys[..written_groups * core.head_dim],
    );
}

fn fixture_values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (((index * seed + index / 5) % 31) as f32 - 15.0) / 32.0)
        .collect()
}
fn bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&x| bf16::from_f32(x).to_bits()).collect()
}
fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
    Buffer::from_slice(device, &bits(values))
}
fn values(buffer: &Buffer) -> Vec<f32> {
    buffer
        .read_typed::<u16>(0, buffer.len_bytes() / 2)
        .into_iter()
        .map(|x| bf16::from_bits(x).to_f32())
        .collect()
}
fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 0.03125,
            "index={index}, actual={actual}, expected={expected}"
        );
    }
}
