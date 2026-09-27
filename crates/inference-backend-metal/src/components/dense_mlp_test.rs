use half::bf16;
use inference_executor_core::mlp::dense::DenseMLPCore;
use inference_executor_core::mlp::dense::reference::QuantizedDenseMLPReferenceGeometry;
use inference_executor_core::mlp::dense::reference::QuantizedDenseMLPReferenceWeights;
use inference_executor_core::mlp::dense::reference::quantized_dense_mlp_reference;

use super::*;
use crate::metal::Buffer;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayProgram;
use crate::metal::Stream;
use crate::test_support::ReplayTestCache;

const NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("test.dense_mlp.num_active_tokens");

#[test]
fn test_replay_bucketing() {
    for (config, cases) in [
        (
            standard_config(),
            &[
                (4_u32, &[1_u32, 4, 3, 2][..], 0x1000_0001_u32),
                (8, &[1_u32, 8, 7, 3][..], 0x1000_0002),
                (12, &[9_u32, 12, 10, 11][..], 0x1000_0003),
                (20, &[17_u32, 20, 18, 19][..], 0x1000_0004),
                (33, &[32_u32, 33, 1, 31, 32][..], 0x1000_0005),
            ][..],
        ),
        (
            Config {
                hidden_dim: 512,
                ..standard_config()
            },
            &[(4_u32, &[1_u32, 4, 3, 2][..], 0x3000_0001_u32)][..],
        ),
        (
            mixed_layout_config(),
            &[
                (8_u32, &[1_u32, 8, 7, 3][..], 0x2000_0001_u32),
                (20, &[1_u32, 20, 18, 3][..], 0x2000_0002),
                (33, &[32_u32, 33, 1, 31, 32][..], 0x2000_0003),
            ][..],
        ),
    ] {
        let fixture = DenseMLPFixture::new(33, config);
        let mut cache = ReplayTestCache::new();
        for &(num_total_tokens, active_sequence, seed) in cases {
            let key = (num_total_tokens, fixture.compute.topology(num_total_tokens));
            let (_, cache_hit) = cache.record(key, || fixture.replay(num_total_tokens));
            assert!(!cache_hit);
            for (case_index, &num_active_tokens) in active_sequence.iter().enumerate() {
                let hidden = fixture.write_hidden(num_active_tokens, seed.wrapping_add(case_index as u32));
                let (replay, cache_hit) = cache.record(key, || unreachable!());
                assert!(cache_hit);
                fixture.submit(replay, num_active_tokens);
                fixture.assert_active_output(&hidden, num_active_tokens);
            }
        }
    }
}

struct DenseMLPFixture {
    stream: Stream,
    config: Config,
    compute: Compute,
    num_allocated_tokens: u32,
    hidden_state: Buffer,
    next_hidden_state: Buffer,
    swiglu: Buffer,
    weights: DenseMLPWeights,
}

impl DenseMLPFixture {
    fn new(num_allocated_tokens: u32, config: Config) -> Self {
        let shape = Shape {
            num_total_tokens: num_allocated_tokens,
        };
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let compute = Compute::new(&device, config);
        let weights = DenseMLPWeights::new(&device, config);
        Self {
            hidden_state: Buffer::new_zeroed(&device, config.input_bytes(shape)),
            next_hidden_state: Buffer::new_zeroed(&device, config.output_bytes(shape)),
            swiglu: Buffer::new_zeroed(&device, config.swiglu_bytes(shape)),
            stream,
            config,
            compute,
            num_allocated_tokens,
            weights,
        }
    }

    fn replay(&self, num_total_tokens: u32) -> ReplayProgram {
        let mut builder = self.stream.create_replay_program();
        builder.record(self.compute.invoke(
            Shape { num_total_tokens },
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            Buffers {
                hidden_state: &self.hidden_state,
                next_hidden_state: &self.next_hidden_state,
            },
            Scratch { swiglu: &self.swiglu },
            self.weights.bindings(),
        ));
        builder.build()
    }

    fn write_hidden(&self, num_active_tokens: u32, seed: u32) -> Vec<f32> {
        assert!(num_active_tokens <= self.num_allocated_tokens);
        let num_values = self.num_allocated_tokens as usize * self.config.hidden_dim as usize;
        let values = bf16_values(&generated_values(num_values, seed));
        self.hidden_state.write_typed(
            0,
            &values
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>(),
        );
        values[..num_active_tokens as usize * self.config.hidden_dim as usize].to_vec()
    }

    fn submit(&self, replay: &ReplayProgram, num_active_tokens: u32) {
        let arguments = ReplayArguments::new().with_u32(NUM_ACTIVE_TOKENS, num_active_tokens);
        self.stream.submit_replay_with_arguments(replay, &arguments).wait();
    }

    fn assert_active_output(&self, hidden: &[f32], num_active_tokens: u32) {
        let expected = quantized_dense_mlp_reference(
            &DenseMLPCore {
                model_layer_index: 0,
                hidden_dim: self.config.hidden_dim as usize,
                intermediate_dim: self.config.intermediate_dim as usize,
            },
            hidden,
            num_active_tokens as usize,
            QuantizedDenseMLPReferenceGeometry {
                gate_up_group_size: self.config.gate_up_group_size as usize,
                gate_up_bits: self.config.gate_up_bits as usize,
                down_group_size: self.config.down_group_size as usize,
                down_bits: self.config.down_bits as usize,
            },
            self.weights.reference(),
        )
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
        let actual = self
            .next_hidden_state
            .read_typed::<u16>(0, expected.len())
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        assert_close_rel(&actual, &expected, 2.0e-5, 8.0e-3);
    }
}

struct DenseMLPWeights {
    gate_up_weight: Buffer,
    gate_up_scales: Buffer,
    gate_up_biases: Buffer,
    down_weight: Buffer,
    down_scales: Buffer,
    down_biases: Buffer,
    gate_up_weight_values: Vec<u8>,
    gate_up_scale_values: Vec<f32>,
    gate_up_bias_values: Vec<f32>,
    down_weight_values: Vec<u8>,
    down_scale_values: Vec<f32>,
    down_bias_values: Vec<f32>,
}

impl DenseMLPWeights {
    fn new(device: &Device, config: Config) -> Self {
        let gate_up = config.gate_up_config();
        let down = config.down_config();
        let gate_up_weight_values = generated_bytes(gate_up.weight_bytes(), 0x3000_0001);
        let gate_up_scale_values = stored_affine_values(
            generated_scales(
                gate_up.scale_or_bias_bytes() / config.gate_up_scale_bias_dtype.item_size(),
                0x3000_0002,
            ),
            config.gate_up_scale_bias_dtype,
        );
        let gate_up_bias_values = stored_affine_values(
            generated_biases(
                gate_up.scale_or_bias_bytes() / config.gate_up_scale_bias_dtype.item_size(),
                0x3000_0003,
            ),
            config.gate_up_scale_bias_dtype,
        );
        let down_weight_values = generated_bytes(down.weight_bytes(), 0x3000_0004);
        let down_scale_values = stored_affine_values(
            generated_scales(
                down.scale_or_bias_bytes() / config.down_scale_bias_dtype.item_size(),
                0x3000_0005,
            ),
            config.down_scale_bias_dtype,
        );
        let down_bias_values = stored_affine_values(
            generated_biases(
                down.scale_or_bias_bytes() / config.down_scale_bias_dtype.item_size(),
                0x3000_0006,
            ),
            config.down_scale_bias_dtype,
        );
        Self {
            gate_up_weight: Buffer::from_slice(device, &gate_up_weight_values),
            gate_up_scales: affine_buffer(device, &gate_up_scale_values, config.gate_up_scale_bias_dtype),
            gate_up_biases: affine_buffer(device, &gate_up_bias_values, config.gate_up_scale_bias_dtype),
            down_weight: Buffer::from_slice(device, &down_weight_values),
            down_scales: affine_buffer(device, &down_scale_values, config.down_scale_bias_dtype),
            down_biases: affine_buffer(device, &down_bias_values, config.down_scale_bias_dtype),
            gate_up_weight_values,
            gate_up_scale_values,
            gate_up_bias_values,
            down_weight_values,
            down_scale_values,
            down_bias_values,
        }
    }

    fn bindings(&self) -> Weights<'_> {
        Weights {
            gate_up_weight: &self.gate_up_weight,
            gate_up_scales: &self.gate_up_scales,
            gate_up_biases: &self.gate_up_biases,
            down_weight: &self.down_weight,
            down_scales: &self.down_scales,
            down_biases: &self.down_biases,
        }
    }

    fn reference(&self) -> QuantizedDenseMLPReferenceWeights<'_> {
        QuantizedDenseMLPReferenceWeights {
            gate_up_weight: &self.gate_up_weight_values,
            gate_up_scales: &self.gate_up_scale_values,
            gate_up_biases: &self.gate_up_bias_values,
            down_weight: &self.down_weight_values,
            down_scales: &self.down_scale_values,
            down_biases: &self.down_bias_values,
        }
    }
}

fn standard_config() -> Config {
    Config {
        hidden_dim: 64,
        intermediate_dim: 4160,
        gate_up_group_size: 32,
        gate_up_bits: 4,
        gate_up_scale_bias_dtype: Dtype::Bfloat16,
        down_group_size: 32,
        down_bits: 4,
        down_scale_bias_dtype: Dtype::Bfloat16,
        dtype: Dtype::Bfloat16,
    }
}

fn mixed_layout_config() -> Config {
    Config {
        hidden_dim: 64,
        intermediate_dim: 64,
        gate_up_group_size: 32,
        gate_up_bits: 4,
        gate_up_scale_bias_dtype: Dtype::Float32,
        down_group_size: 32,
        down_bits: 6,
        down_scale_bias_dtype: Dtype::Float32,
        dtype: Dtype::Bfloat16,
    }
}

fn affine_buffer(device: &Device, values: &[f32], dtype: Dtype) -> Buffer {
    match dtype {
        Dtype::Bfloat16 => {
            Buffer::from_slice(
                device,
                &values
                    .iter()
                    .map(|value| bf16::from_f32(*value).to_bits())
                    .collect::<Vec<_>>(),
            )
        },
        Dtype::Float32 => Buffer::from_slice(device, values),
        _ => panic!("unsupported dense MLP affine parameter dtype {dtype:?}"),
    }
}

fn stored_affine_values(values: Vec<f32>, dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::Bfloat16 => bf16_values(&values),
        Dtype::Float32 => values,
        _ => panic!("unsupported dense MLP affine parameter dtype {dtype:?}"),
    }
}

fn bf16_values(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| bf16::from_f32(*value).to_f32()).collect()
}

fn generated_values(count: usize, mut state: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

fn generated_scales(count: usize, random_seed: u32) -> Vec<f32> {
    generated_values(count, random_seed)
        .into_iter()
        .map(|value| 0.0005 + value.abs() * 0.001)
        .collect()
}

fn generated_biases(count: usize, random_seed: u32) -> Vec<f32> {
    generated_values(count, random_seed)
        .into_iter()
        .map(|value| value * 0.0002)
        .collect()
}

fn generated_bytes(count: usize, mut state: u32) -> Vec<u8> {
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        })
        .collect()
}

fn assert_close_rel(actual: &[f32], expected: &[f32], abs_tolerance: f32, rel_tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        let tolerance = abs_tolerance.max(expected.abs() * rel_tolerance);
        assert!(
            diff <= tolerance,
            "dense MLP output mismatch at {index}: expected={expected} actual={actual} diff={diff} \
             tolerance={tolerance}"
        );
    }
}
