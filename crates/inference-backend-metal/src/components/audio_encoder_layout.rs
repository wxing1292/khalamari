//! Layout transforms and sinusoidal positions for the Qwen3-ASR Audio Tower.

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;

const SOURCE: &str = include_str!("metal/audio_encoder_layout.metal");
const ROWS_PER_COMPLETE_CHUNK: u32 = 13;

pub struct Compute {
    chunk_log_mel: CompiledKernel,
    flatten_conv: CompiledKernel,
    compact_position: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device) -> Self {
        Self {
            chunk_log_mel: CompiledKernel::new(device, SOURCE, "audio_chunk_log_mel_f32_to_bf16"),
            flatten_conv: CompiledKernel::new(device, SOURCE, "audio_flatten_conv_bf16"),
            compact_position: CompiledKernel::new(device, SOURCE, "audio_compact_position_bf16"),
        }
    }

    pub fn invoke_chunk_log_mel<'a>(
        &'a self,
        shape: LogMelChunkShape,
        source: &'a Buffer,
        output: &'a Buffer,
    ) -> LogMelChunkInvocation<'a> {
        shape.validate();
        assert!(
            source.len_bytes() >= shape.source_bytes(),
            "log-Mel source buffer is too small"
        );
        assert!(
            output.len_bytes() >= shape.output_bytes(shape.output_values()),
            "chunked log-Mel output buffer is too small"
        );
        LogMelChunkInvocation {
            kernel: &self.chunk_log_mel,
            shape,
            source,
            output,
        }
    }

    pub fn invoke_flatten_conv<'a>(
        &'a self,
        shape: FlattenConvShape,
        source: &'a Buffer,
        output: &'a Buffer,
    ) -> FlattenConvInvocation<'a> {
        shape.validate();
        let tensor_bytes = shape.num_values() as usize * Dtype::Bfloat16.item_size();
        assert!(
            source.len_bytes() >= tensor_bytes,
            "audio Conv2D flatten source buffer is too small"
        );
        assert!(
            output.len_bytes() >= tensor_bytes,
            "audio Conv2D flatten output buffer is too small"
        );
        FlattenConvInvocation {
            kernel: &self.flatten_conv,
            shape,
            source,
            output,
        }
    }

    pub fn invoke_compact_position<'a>(
        &'a self,
        shape: CompactPositionShape,
        source: &'a Buffer,
        output: &'a Buffer,
    ) -> CompactPositionInvocation<'a> {
        shape.validate();
        let source_bytes = shape.source_values() as usize * Dtype::Bfloat16.item_size();
        let output_bytes = shape.num_values() as usize * Dtype::Bfloat16.item_size();
        assert!(
            source.len_bytes() >= source_bytes,
            "audio position source buffer is too small"
        );
        assert!(
            output.len_bytes() >= output_bytes,
            "audio position output buffer is too small"
        );
        CompactPositionInvocation {
            kernel: &self.compact_position,
            shape,
            source,
            output,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogMelChunkShape {
    pub num_mel_bins: u32,
    pub num_frames: u32,
    pub frames_per_chunk: u32,
}

impl LogMelChunkShape {
    pub fn num_chunks(self) -> u32 {
        self.num_frames.div_ceil(self.frames_per_chunk)
    }

    pub fn output_values(self) -> u32 {
        self.num_chunks() * self.num_mel_bins * self.frames_per_chunk
    }

    fn validate(self) {
        assert!(self.num_mel_bins > 0, "log-Mel bin count must be positive");
        assert!(self.num_frames > 0, "log-Mel frame count must be positive");
        assert!(self.frames_per_chunk > 0, "log-Mel chunk width must be positive");
        self.num_chunks()
            .checked_mul(self.num_mel_bins)
            .and_then(|values| values.checked_mul(self.frames_per_chunk))
            .expect("chunked log-Mel output value count must fit the shader u32 count domain");
    }

    fn source_bytes(self) -> usize {
        self.num_mel_bins as usize * self.num_frames as usize * Dtype::Float32.item_size()
    }

    fn output_bytes(self, output_values: u32) -> usize {
        output_values as usize * Dtype::Bfloat16.item_size()
    }
}

pub struct LogMelChunkInvocation<'a> {
    kernel: &'a CompiledKernel,
    shape: LogMelChunkShape,
    source: &'a Buffer,
    output: &'a Buffer,
}

impl Operator for LogMelChunkInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let output_values = self.shape.output_values();
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.source, 0);
        recorder.set_buffer_write(1, self.output, 0);
        recorder.set_u32(2, self.shape.num_mel_bins);
        recorder.set_u32(3, self.shape.num_frames);
        recorder.set_u32(4, self.shape.frames_per_chunk);
        recorder.set_u32(5, self.shape.num_chunks());
        recorder.dispatch_1d(output_values as usize, 256);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlattenConvShape {
    pub batch: u32,
    pub height: u32,
    pub width: u32,
    pub channels: u32,
}

impl FlattenConvShape {
    pub fn num_values(self) -> u32 {
        self.batch * self.height * self.width * self.channels
    }

    fn validate(self) {
        assert!(self.batch > 0, "audio Conv2D batch size must be positive");
        assert!(self.height > 0, "audio Conv2D height must be positive");
        assert!(self.width > 0, "audio Conv2D width must be positive");
        assert!(self.channels > 0, "audio Conv2D channel count must be positive");
        self.batch
            .checked_mul(self.height)
            .and_then(|values| values.checked_mul(self.width))
            .and_then(|values| values.checked_mul(self.channels))
            .expect("audio Conv2D flatten value count must fit the shader u32 count domain");
    }
}

pub struct FlattenConvInvocation<'a> {
    kernel: &'a CompiledKernel,
    shape: FlattenConvShape,
    source: &'a Buffer,
    output: &'a Buffer,
}

impl Operator for FlattenConvInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let num_values = self.shape.num_values();
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.source, 0);
        recorder.set_buffer_write(1, self.output, 0);
        recorder.set_u32(2, self.shape.batch);
        recorder.set_u32(3, self.shape.height);
        recorder.set_u32(4, self.shape.width);
        recorder.set_u32(5, self.shape.channels);
        recorder.dispatch_1d(num_values as usize, 256);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactPositionShape {
    pub num_rows: u32,
    pub source_rows_per_chunk: u32,
    pub hidden_dim: u32,
}

impl CompactPositionShape {
    fn validate(self) {
        assert!(self.num_rows > 0, "audio position row count must be positive");
        assert!(
            self.source_rows_per_chunk > 0 && self.source_rows_per_chunk <= ROWS_PER_COMPLETE_CHUNK,
            "audio position source rows per chunk must be in 1..={ROWS_PER_COMPLETE_CHUNK}"
        );
        assert!(
            self.num_rows <= self.source_rows_per_chunk || self.source_rows_per_chunk == ROWS_PER_COMPLETE_CHUNK,
            "multi-chunk audio positions require {ROWS_PER_COMPLETE_CHUNK} source rows per chunk"
        );
        assert!(
            self.hidden_dim >= 4 && self.hidden_dim.is_multiple_of(2),
            "audio position hidden dimension must be even and at least four"
        );
        self.num_rows
            .checked_mul(self.hidden_dim)
            .expect("audio position output value count must fit the shader u32 count domain");
        self.num_rows
            .div_ceil(ROWS_PER_COMPLETE_CHUNK)
            .checked_mul(self.source_rows_per_chunk)
            .and_then(|rows| rows.checked_mul(self.hidden_dim))
            .expect("audio position padded source value count must fit u32");
    }

    fn num_values(self) -> u32 {
        self.num_rows * self.hidden_dim
    }

    fn source_values(self) -> u32 {
        self.num_rows.div_ceil(ROWS_PER_COMPLETE_CHUNK) * self.source_rows_per_chunk * self.hidden_dim
    }
}

pub struct CompactPositionInvocation<'a> {
    kernel: &'a CompiledKernel,
    shape: CompactPositionShape,
    source: &'a Buffer,
    output: &'a Buffer,
}

impl Operator for CompactPositionInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let num_values = self.shape.num_values();
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.source, 0);
        recorder.set_buffer_write(1, self.output, 0);
        recorder.set_u32(2, self.shape.num_rows);
        recorder.set_u32(3, self.shape.source_rows_per_chunk);
        recorder.set_u32(4, self.shape.hidden_dim);
        recorder.dispatch_1d(num_values as usize, 256);
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    #[test]
    fn test_invoke_chunk_log_mel_tail_padding() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = LogMelChunkShape {
            num_mel_bins: 2,
            num_frames: 5,
            frames_per_chunk: 3,
        };
        let source_values = (0..shape.num_mel_bins * shape.num_frames)
            .map(|index| index as f32 + 1.0)
            .collect::<Vec<_>>();
        let source = Buffer::from_slice(&device, &source_values);
        let output = Buffer::new_zeroed_elements(&device, shape.output_values(), Dtype::Bfloat16);
        let compute = Compute::new(&device);
        let mut recorder = stream.create_replay_program();
        recorder.record(compute.invoke_chunk_log_mel(shape, &source, &output));
        stream
            .submit_replay_with_arguments(&recorder.build(), &ReplayArguments::new())
            .wait();

        let actual = output
            .read_typed::<u16>(0, shape.output_values() as usize)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 4.0, 5.0, 0.0, 9.0, 10.0, 0.0]
        );
    }

    #[test]
    fn test_invoke_flatten_conv_layout() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = FlattenConvShape {
            batch: 2,
            height: 2,
            width: 2,
            channels: 2,
        };
        let source_values = (0..shape.num_values())
            .map(|index| bf16::from_f32(index as f32 + 1.0))
            .collect::<Vec<_>>();
        let source = Buffer::from_slice(
            &device,
            &source_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        );
        let output = Buffer::new_zeroed_elements(&device, shape.num_values(), Dtype::Bfloat16);
        let compute = Compute::new(&device);
        let mut recorder = stream.create_replay_program();
        recorder.record(compute.invoke_flatten_conv(shape, &source, &output));
        stream
            .submit_replay_with_arguments(&recorder.build(), &ReplayArguments::new())
            .wait();

        let actual = output
            .read_typed::<u16>(0, shape.num_values() as usize)
            .into_iter()
            .map(bf16::from_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, reference_flatten_conv(shape, &source_values));
    }

    #[test]
    fn test_invoke_compact_position_chunk_boundary() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let shape = CompactPositionShape {
            num_rows: 14,
            source_rows_per_chunk: 13,
            hidden_dim: 4,
        };
        let source_values = (0..shape.source_values())
            .map(|index| bf16::from_f32(index as f32 * 0.01))
            .collect::<Vec<_>>();
        let source = Buffer::from_slice(
            &device,
            &source_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        );
        let output = Buffer::new_zeroed_elements(&device, shape.num_values(), Dtype::Bfloat16);
        let compute = Compute::new(&device);
        let mut recorder = stream.create_replay_program();
        recorder.record(compute.invoke_compact_position(shape, &source, &output));
        stream
            .submit_replay_with_arguments(&recorder.build(), &ReplayArguments::new())
            .wait();

        let actual = output
            .read_typed::<u16>(0, shape.num_values() as usize)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        let expected = reference_compact_position(shape, &source_values);
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 0.01,
                "index={index} actual={actual} expected={expected}"
            );
        }
    }

    fn reference_flatten_conv(shape: FlattenConvShape, source: &[bf16]) -> Vec<bf16> {
        let mut output = vec![bf16::ZERO; source.len()];
        for batch in 0..shape.batch {
            for width in 0..shape.width {
                for channel in 0..shape.channels {
                    for height in 0..shape.height {
                        let source_index = (((batch * shape.height + height) * shape.width + width) * shape.channels
                            + channel) as usize;
                        let output_index = (((batch * shape.width + width) * shape.channels + channel) * shape.height
                            + height) as usize;
                        output[output_index] = source[source_index];
                    }
                }
            }
        }
        output
    }

    fn reference_compact_position(shape: CompactPositionShape, source: &[bf16]) -> Vec<f32> {
        let mut output = vec![0.0; shape.num_values() as usize];
        let half_dim = shape.hidden_dim / 2;
        let increment = 10000.0_f32.ln() / (half_dim - 1) as f32;
        for row in 0..shape.num_rows {
            let chunk = row / ROWS_PER_COMPLETE_CHUNK;
            let local_row = row % ROWS_PER_COMPLETE_CHUNK;
            for dim in 0..shape.hidden_dim {
                let frequency_index = if dim < half_dim { dim } else { dim - half_dim };
                let phase = local_row as f32 * (-increment * frequency_index as f32).exp();
                let position = if dim < half_dim { phase.sin() } else { phase.cos() };
                let source_index =
                    ((chunk * shape.source_rows_per_chunk + local_row) * shape.hidden_dim + dim) as usize;
                output[(row * shape.hidden_dim + dim) as usize] = source[source_index].to_f32() + position;
            }
        }
        output
    }
}
