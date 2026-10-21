use half::bf16;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::Stream;
use inference_executor_core::mlp::dense::reference::QuantizedAffineReferenceShape;
use inference_executor_core::mlp::dense::reference::quantized_affine_reference;
use inference_executor_core::ngram_embedding::reference as hash_reference;
use inference_executor_core::ple::reference;

use super::*;
use crate::ple::batch_metadata::PLERequest;

const TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.ple.tokens");
const REQUESTS: ReplayParameterKey = ReplayParameterKey::new("test.ple.requests");

#[test]
fn test_prepare_apply_commit_replay() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let runtime = &stream;
    let ngram = NGramEmbeddingCore {
        vocab_size: 32,
        eos_token_id: 0,
        ngram_size: 3,
        heads_per_ngram: 2,
        embedding_dim: 128,
    };
    let core = PLECore {
        num_channels: 3,
        hidden_dim: 37,
        embedding_dim: 128,
        conv_kernel_size: 4,
        conv_dilation: 3,
        norm_eps: 1e-6,
    };
    let parameters = HashParameters::new(ngram, 11, 0, 1234);
    let table_rows = parameters.num_rows() as usize;
    let component = PLE::new(
        &device,
        core,
        ngram,
        HashParameters::new(ngram, 11, 0, 1234),
        PLEMetalConfig {
            key: QuantizedAffineLayout {
                group_size: 32,
                bits: 4,
                scale_bias_dtype: Dtype::Bfloat16,
            },
            value: QuantizedAffineLayout {
                group_size: 32,
                bits: 4,
                scale_bias_dtype: Dtype::Bfloat16,
            },
            io_dtype: Dtype::Bfloat16,
            table_scale: 1.0,
        },
    );
    let scratch = PLEScratch::new(&device, core, ngram, 8);
    let mut metadata = PLEMetadataBuffers::new(&device, ngram, 8, 2, 4);
    let key = Affine::new(&device, core.residual_dim() as usize, 128, 13);
    let value = Affine::new(&device, 37, 128, 17);
    let wide = core.residual_dim() as usize;
    let tail = core.state_len() as usize;
    let kn = fixture(wide, 19);
    let qn = fixture(wide, 23);
    let cn = fixture(wide, 29);
    let cw = fixture(wide * 4, 31);
    let kn_buf = buffer(&device, &kn);
    let qn_buf = buffer(&device, &qn);
    let cn_buf = buffer(&device, &cn);
    let cw_buf = buffer(&device, &cw);
    let weights = PLEWeights {
        key: key.bindings(),
        value: value.bindings(),
        key_norm: &kn_buf,
        query_norm: &qn_buf,
        conv_norm: &cn_buf,
        conv: &cw_buf,
    };
    let table = fixture(table_rows * 32, 37);
    let table_buffer = buffer(&device, &table);
    let shards = [ngram_embedding::Shard {
        row_start: 0,
        num_rows: table_rows as u32,
        weights: ngram_embedding::TableWeights::Bf16(&table_buffer),
    }];
    let residual = Buffer::new_zeroed(&device, 8 * wide * 2);
    let output = Buffer::new_zeroed(&device, 8 * wide * 2);
    let initial = fixture(4 * tail * wide, 41);
    let states = buffer(&device, &initial);
    let mut recorder = ReplayRecorder::new(runtime.create_replay_program());
    component.prepare(
        &mut recorder,
        PLEPrepareInput {
            num_total_tokens: metadata.max_tokens(),
            num_active_tokens: ReplayU32::Parameter(TOKENS),
            metadata: &metadata,
            scratch: &scratch,
            shards: &shards,
            weights,
        },
    );
    let prepare = recorder.build();
    let mut recorder = ReplayRecorder::new(runtime.create_replay_program());
    component.apply(
        &mut recorder,
        PLEApplyInput {
            num_total_tokens: metadata.max_tokens(),
            num_active_tokens: ReplayU32::Parameter(TOKENS),
            metadata: &metadata,
            scratch: &scratch,
            residual: &residual,
            output: &output,
            states: &states,
            weights,
        },
    );
    let apply = recorder.build();
    let mut recorder = ReplayRecorder::new(runtime.create_replay_program());
    component.commit(
        &mut recorder,
        PLECommitInput {
            num_active_requests: ReplayU32::Parameter(REQUESTS),
            metadata: &metadata,
            scratch: &scratch,
            states: &states,
        },
    );
    let commit = recorder.build();
    for fixed in [true, false] {
        states.write_typed(0, &bits(&initial));
        let mut cpu_states: Vec<_> = initial.chunks_exact(tail * wide).map(|s| s.to_vec()).collect();
        let a = [1, 2, 3];
        let b = [5, 0, 6, 7];
        let continuation = [3, 4];
        let short = [8];
        let h0 = [0, 0];
        let h1 = [9, 0];
        let restored = [1, 2];
        let batches = [
            (
                vec![
                    PLERequest {
                        token_ids: &a,
                        previous_tokens: &h0,
                        src_state_slot: 0,
                        dst_state_slot: 2,
                    },
                    PLERequest {
                        token_ids: &b,
                        previous_tokens: &h1,
                        src_state_slot: 1,
                        dst_state_slot: 3,
                    },
                ],
                vec![2_u32, 1],
            ),
            (
                vec![PLERequest {
                    token_ids: &continuation,
                    previous_tokens: &restored,
                    src_state_slot: 2,
                    dst_state_slot: 0,
                }],
                vec![2],
            ),
            (
                vec![
                    PLERequest {
                        token_ids: &short,
                        previous_tokens: &h0,
                        src_state_slot: 0,
                        dst_state_slot: 2,
                    },
                    PLERequest {
                        token_ids: &b,
                        previous_tokens: &h1,
                        src_state_slot: 3,
                        dst_state_slot: 1,
                    },
                ],
                vec![0, 3],
            ),
        ];
        for (requests, accepted) in batches {
            let active = metadata.update(&requests);
            let count = active.num_active_tokens as usize;
            let x = if fixed {
                vec![0.0; count * wide]
            } else {
                fixture(count * wide, 43)
            };
            residual.write_typed(0, &bits(&x));
            output.write_typed(0, &vec![bf16::from_f32(-99.0).to_bits(); 8 * wide]);
            let args = ReplayArguments::new().with_u32(TOKENS, active.num_active_tokens);
            let before = states.read_typed::<u16>(0, initial.len());
            // Separate submissions prove that preparation has no residual or convolution-state dependency.
            runtime.submit_replay_with_arguments(&prepare, &args).wait();
            assert_eq!(states.read_typed::<u16>(0, initial.len()), before);
            let expected_ids: Vec<_> = requests
                .iter()
                .flat_map(|r| hash_reference::row_ids(ngram, &parameters, r.previous_tokens, r.token_ids))
                .collect();
            assert_eq!(scratch.row_ids.read_typed::<u32>(0, count * 4), expected_ids);
            let embedding: Vec<_> = expected_ids
                .iter()
                .flat_map(|&id| table[id as usize * 32..(id as usize + 1) * 32].iter().copied())
                .collect();
            assert_eq!(
                scratch.embedding.read_typed::<u16>(0, embedding.len()),
                bits(&embedding)
            );
            let keys = key.project(count, &embedding);
            let values = value.project(count, &embedding);
            close(&read(&scratch.key, keys.len()), &keys, 0.015625);
            close(&read(&scratch.value, values.len()), &values, 0.015625);
            runtime.submit_replay_with_arguments(&apply, &args).wait();
            assert_eq!(states.read_typed::<u16>(0, initial.len()), before);
            let mut start = 0;
            let mut refs = Vec::new();
            for r in &requests {
                let n = r.token_ids.len();
                let result = reference::apply(reference::Input {
                    core,
                    num_tokens: n,
                    residual: &x[start * wide..(start + n) * wide],
                    keys: &keys[start * wide..(start + n) * wide],
                    values: &values[start * 37..(start + n) * 37],
                    state: &cpu_states[r.src_state_slot as usize],
                    weights: reference::Weights {
                        key_norm: &kn,
                        query_norm: &qn,
                        conv_norm: &cn,
                        conv: &cw,
                    },
                });
                close(
                    &read(&output, count * wide)[start * wide..(start + n) * wide],
                    &result.residual,
                    0.03125,
                );
                close(
                    &read(&scratch.normalized_gated, count * wide)[start * wide..(start + n) * wide],
                    &result.normalized,
                    0.03125,
                );
                refs.push(result);
                start += n;
            }
            assert!(
                output.read_typed::<u16>(0, 8 * wide)[count * wide..]
                    .iter()
                    .all(|&v| v == bf16::from_f32(-99.0).to_bits())
            );
            metadata.set_accepted_tokens(&accepted);
            runtime
                .submit_replay_with_arguments(
                    &commit,
                    &ReplayArguments::new().with_u32(REQUESTS, active.num_active_requests),
                )
                .wait();
            for (i, r) in requests.iter().enumerate() {
                let n = accepted[i] as usize;
                let history = [
                    cpu_states[r.src_state_slot as usize].clone(),
                    refs[i].normalized[..n * wide].to_vec(),
                ]
                .concat();
                cpu_states[r.dst_state_slot as usize] = history[history.len() - tail * wide..].to_vec();
                let actual = read(&states, initial.len());
                close(
                    &actual[r.dst_state_slot as usize * tail * wide..(r.dst_state_slot as usize + 1) * tail * wide],
                    &cpu_states[r.dst_state_slot as usize],
                    0.03125,
                );
            }
            let after = states.read_typed::<u16>(0, initial.len());
            for slot in 0..4 {
                if !requests.iter().any(|r| r.dst_state_slot as usize == slot) {
                    assert_eq!(
                        &after[slot * tail * wide..(slot + 1) * tail * wide],
                        &before[slot * tail * wide..(slot + 1) * tail * wide]
                    );
                }
            }
        }
    }
}

struct Affine {
    n: usize,
    k: usize,
    packed: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    weight: Buffer,
    scale: Buffer,
    bias: Buffer,
}
impl Affine {
    fn new(device: &Device, n: usize, k: usize, seed: usize) -> Self {
        let mut packed = vec![0; n * k / 2];
        for index in 0..n * k {
            packed[index / 2] |= (((index * seed + index / k) % 16) as u8) << ((index % 2) * 4);
        }
        let scales = vec![1.0 / 128.0; n * k / 32];
        let biases = vec![-7.0 / 128.0; scales.len()];
        Self {
            n,
            k,
            weight: Buffer::from_slice(device, &packed),
            scale: buffer(device, &scales),
            bias: buffer(device, &biases),
            packed,
            scales,
            biases,
        }
    }
    fn bindings(&self) -> QuantizedAffineWeights<'_> {
        QuantizedAffineWeights::new(&self.weight, &self.scale, &self.bias)
    }
    fn project(&self, rows: usize, x: &[f32]) -> Vec<f32> {
        quantized_affine_reference(
            QuantizedAffineReferenceShape {
                num_rows: rows,
                output_dim: self.n,
                input_dim: self.k,
                group_size: 32,
                bits: 4,
            },
            x,
            &self.packed,
            &self.scales,
            &self.biases,
        )
        .into_iter()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect()
    }
}
fn fixture(n: usize, seed: usize) -> Vec<f32> {
    let mut random = seed as u64;
    (0..n)
        .map(|_| {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            bf16::from_f32(((random % 127) as f32 - 63.0) / 256.0).to_f32()
        })
        .collect()
}
fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&v| bf16::from_f32(v).to_bits()).collect()
}
fn buffer(device: &Device, v: &[f32]) -> Buffer {
    Buffer::from_slice(device, &bits(v))
}
fn read(buffer: &Buffer, n: usize) -> Vec<f32> {
    buffer
        .read_typed::<u16>(0, n)
        .into_iter()
        .map(|v| bf16::from_bits(v).to_f32())
        .collect()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tolerance, "index={i}: {a} != {e}");
    }
}
