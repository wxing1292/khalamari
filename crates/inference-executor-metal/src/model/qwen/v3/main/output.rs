use std::rc::Rc;

use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;

use crate::def::layer::ReplayLayer;
use crate::model::gather::Gather;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedInput;
use crate::replay::ReplayComponent;

const QWEN3_GATHER_UNEMBED_NUM_ACTIVE_ROWS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3.gather_unembed.num_active_rows");

pub struct Qwen3GatherUnembed {
    gather: Gather,
    unembed: Option<Rc<Unembed>>,
}

#[derive(Clone, Copy)]
pub struct Qwen3GatherUnembedArgs<'a> {
    pub num_rows: u32,
    pub hidden_input: &'a Buffer,
    pub row_indices: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub logits: &'a Buffer,
}

impl Qwen3GatherUnembed {
    pub fn new(device: &Device, hidden_dim: u32, unembed: Rc<Unembed>) -> Self {
        Self {
            gather: Gather::new(device, hidden_dim),
            unembed: Some(unembed),
        }
    }

    pub fn unembed(&self) -> Rc<Unembed> {
        Rc::clone(
            self.unembed
                .as_ref()
                .expect("qwen3 GatherUnembed weights must be loaded before use"),
        )
    }

    pub fn load_weights(&mut self, unembed: Rc<Unembed>) {
        assert!(self.unembed.is_none(), "qwen3 GatherUnembed weights are already loaded");
        self.unembed = Some(unembed);
    }

    pub fn unload_weights(&mut self) -> Rc<Unembed> {
        self.unembed.take().expect("qwen3 GatherUnembed weights are not loaded")
    }

    fn loaded_unembed(&self) -> &Unembed {
        self.unembed
            .as_deref()
            .expect("qwen3 GatherUnembed weights must be loaded before execution")
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_rows: u32,
        num_active_rows: ReplayU32,
        args: Qwen3GatherUnembedArgs<'a>,
    ) -> &'a Buffer {
        if let ReplayU32::Fixed(value) = num_active_rows {
            assert_eq!(value, args.num_rows);
            assert_eq!(value, num_total_rows);
        } else {
            assert_eq!(args.num_rows, num_total_rows);
        }
        self.gather.record(
            recorder,
            num_total_rows,
            num_active_rows,
            args.hidden_input,
            args.row_indices,
            args.hidden_output,
        );
        <Unembed as ReplayLayer>::record(
            self.loaded_unembed(),
            recorder,
            UnembedInput {
                num_total_rows,
                num_active_rows,
                hidden: args.hidden_output,
                logits: args.logits,
            },
        )
    }

    pub fn prepare_replay(&self, num_active_rows: u32) -> (Qwen3GatherUnembedReplayKey, ReplayArguments) {
        let key = self.replay_key_for_active_rows(num_active_rows);
        let arguments = ReplayArguments::new().with_u32(QWEN3_GATHER_UNEMBED_NUM_ACTIVE_ROWS, num_active_rows);
        (key, arguments)
    }

    fn replay_key_for_active_rows(&self, num_active_rows: u32) -> Qwen3GatherUnembedReplayKey {
        assert!(
            num_active_rows > 0,
            "qwen3 GatherUnembed replay requires Main output rows"
        );
        Qwen3GatherUnembedReplayKey {
            num_total_rows: num_active_rows,
            unembed_topology: self.loaded_unembed().replay_topology(num_active_rows),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3GatherUnembedReplayKey {
    num_total_rows: u32,
    unembed_topology: affine_quantized::KernelKind,
}

impl ReplayComponent for Qwen3GatherUnembed {
    type Key = Qwen3GatherUnembedReplayKey;
    type Input<'a> = Qwen3GatherUnembedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.replay_key_for_active_rows(input.num_rows)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        Qwen3GatherUnembed::record(
            self,
            recorder,
            key.num_total_rows,
            ReplayU32::Parameter(QWEN3_GATHER_UNEMBED_NUM_ACTIVE_ROWS),
            *input,
        );
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use inference_backend_metal::metal::Dtype;
    use inference_backend_metal::metal::Stream;
    use inference_executor_core::mlp::dense::reference::QuantizedAffineReferenceShape;
    use inference_executor_core::mlp::dense::reference::quantized_affine_reference;

    use super::*;
    use crate::model::unembedding::UnembedConfig;
    use crate::model::unembedding::fixture_unembed;
    use crate::replay::Replay;

    const MAX_ROWS: u32 = 32;
    const VOCAB_SIZE: u32 = 32;
    const HIDDEN_DIM: u32 = 32;
    const GROUP_SIZE: u32 = 32;
    const NUM_TOTAL_ROWS: u32 = 8;

    #[test]
    fn test_replay_matches_cpu_reference_across_active_counts() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = &stream;
        let (component, weights) = fixture_component(&device);
        let buffers = TestBuffers::new(&device);
        let input = buffers.input(NUM_TOTAL_ROWS);
        let mut replay = Replay::new("qwen3 GatherUnembed component test", component);
        let (recorded_key, cache_hit) = replay.record(runtime, &input);
        assert!(!cache_hit);

        for num_active_rows in [1_u32, 8, 3, 7, 2, 6, 4, 5] {
            assert_eq!(replay.record(runtime, &input), (recorded_key.clone(), true));
            runtime
                .submit_replay_with_arguments(
                    replay.replay(&recorded_key),
                    &ReplayArguments::new().with_u32(QWEN3_GATHER_UNEMBED_NUM_ACTIVE_ROWS, num_active_rows),
                )
                .wait();

            let gathered = buffers.gathered_reference(num_active_rows as usize);
            let hidden_values = num_active_rows as usize * HIDDEN_DIM as usize;
            let actual_hidden = buffers.hidden_output.read_typed::<u16>(0, hidden_values);
            let expected_hidden = gathered
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>();
            assert_eq!(actual_hidden, expected_hidden);

            let expected_logits = quantized_affine_reference(
                QuantizedAffineReferenceShape {
                    num_rows: num_active_rows as usize,
                    output_dim: VOCAB_SIZE as usize,
                    input_dim: HIDDEN_DIM as usize,
                    group_size: GROUP_SIZE as usize,
                    bits: 8,
                },
                &gathered,
                &weights.weight,
                &weights.scales,
                &weights.biases,
            );
            let actual_logits = buffers
                .logits
                .read_typed::<u16>(0, num_active_rows as usize * VOCAB_SIZE as usize)
                .into_iter()
                .map(|bits| bf16::from_bits(bits).to_f32())
                .collect::<Vec<_>>();
            assert_close(&actual_logits, &expected_logits, 0.125);
        }
    }

    struct TestBuffers {
        hidden_input: Buffer,
        row_indices: Buffer,
        hidden_output: Buffer,
        logits: Buffer,
        hidden_values: Vec<f32>,
        row_index_values: Vec<u32>,
    }

    impl TestBuffers {
        fn new(device: &Device) -> Self {
            let hidden_values = (0..MAX_ROWS as usize * HIDDEN_DIM as usize)
                .map(|index| bf16::from_f32(((index * 19 + 7) % 113) as f32 * 0.015_625 - 0.75).to_f32())
                .collect::<Vec<_>>();
            let hidden_bits = hidden_values
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>();
            let row_index_values = vec![7, 1, 15, 3, 20, 0, 31, 9];
            Self {
                hidden_input: Buffer::from_slice(device, &hidden_bits),
                row_indices: Buffer::from_slice(device, &row_index_values),
                hidden_output: Buffer::new_zeroed_elements(
                    device,
                    MAX_ROWS as usize * HIDDEN_DIM as usize,
                    Dtype::Bfloat16,
                ),
                logits: Buffer::new_zeroed_elements(device, MAX_ROWS as usize * VOCAB_SIZE as usize, Dtype::Bfloat16),
                hidden_values,
                row_index_values,
            }
        }

        fn input(&self, num_rows: u32) -> Qwen3GatherUnembedArgs<'_> {
            Qwen3GatherUnembedArgs {
                num_rows,
                hidden_input: &self.hidden_input,
                row_indices: &self.row_indices,
                hidden_output: &self.hidden_output,
                logits: &self.logits,
            }
        }

        fn gathered_reference(&self, num_active_rows: usize) -> Vec<f32> {
            let mut output = Vec::with_capacity(num_active_rows * HIDDEN_DIM as usize);
            for &row_index in &self.row_index_values[..num_active_rows] {
                let begin = row_index as usize * HIDDEN_DIM as usize;
                output.extend_from_slice(&self.hidden_values[begin..begin + HIDDEN_DIM as usize]);
            }
            output
        }
    }

    struct TestUnembedWeights {
        weight: Vec<u8>,
        scales: Vec<f32>,
        biases: Vec<f32>,
    }

    fn fixture_component(device: &Device) -> (Qwen3GatherUnembed, TestUnembedWeights) {
        let config = UnembedConfig {
            max_tokens: MAX_ROWS,
            vocab_size: VOCAB_SIZE,
            hidden_dim: HIDDEN_DIM,
            group_size: GROUP_SIZE,
            bits: 8,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let (unembed, weight, scales, biases) = fixture_unembed(device, config);
        (
            Qwen3GatherUnembed::new(device, config.hidden_dim, Rc::new(unembed)),
            TestUnembedWeights { weight, scales, biases },
        )
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
