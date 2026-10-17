//! Explicit diagnostic checks for active rows of dense floating-point buffers.

use std::rc::Rc;

use objc2_metal::MTLComputePipelineState;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;
use crate::metal::stream::check::SubmissionCheck;

pub struct Compute {
    kernel: CompiledKernel,
    dtype: Dtype,
    row_width: u32,
}

impl Compute {
    pub fn new(device: &Device, dtype: Dtype, row_width: u32) -> Self {
        assert!(row_width > 0);
        let name = match dtype {
            Dtype::Float32 => "check_finite_f32",
            Dtype::Bfloat16 => "check_finite_bf16",
            _ => panic!("finite check requires F32 or BF16"),
        };
        Self {
            kernel: CompiledKernel::new(device, include_str!("metal/check_finite.metal"), name),
            dtype,
            row_width,
        }
    }

    pub fn invoke<'a>(&'a self, input: Input<'a>, label: String) -> impl Operator + 'a {
        input.num_active_rows.validate_range(1, input.num_total_rows);
        let elements = input
            .num_total_rows
            .checked_mul(self.row_width)
            .expect("finite-check index must fit u32");
        let bytes = elements as usize * self.dtype.item_size();
        assert!(input.buffer.len_bytes() >= bytes, "finite-check input is too short");
        Invocation {
            compute: self,
            input,
            label,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Input<'a> {
    pub buffer: &'a Buffer,
    pub num_total_rows: u32,
    pub num_active_rows: ReplayU32,
}

struct Invocation<'a> {
    compute: &'a Compute,
    input: Input<'a>,
    label: String,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let Input {
            buffer,
            num_total_rows,
            num_active_rows,
        } = self.input;
        let dtype = self.compute.dtype;
        let row_width = self.compute.row_width;
        let elements = num_total_rows * row_width;
        let device = Device::from_raw_retained(self.compute.kernel.as_raw().device());
        let check = Rc::new(FiniteCheck {
            status: Buffer::from_slice(&device, &[0_u32; 4]),
            label: self.label,
            dtype,
            row_width,
        });
        recorder.set_kernel(&self.compute.kernel);
        recorder.set_buffer_read(0, buffer, 0);
        match num_active_rows {
            ReplayU32::Fixed(rows) => recorder.set_u32(1, rows),
            ReplayU32::Parameter(key) => recorder.bind_u32(1, key, 1, num_total_rows),
        }
        recorder.set_u32(2, row_width);
        recorder.set_buffer_read_write(3, &check.status, 0);
        recorder.set_submission_check(check);
        recorder.dispatch_1d(elements as usize, 256);
    }
}

#[derive(Debug)]
struct FiniteCheck {
    status: Buffer,
    label: String,
    dtype: Dtype,
    row_width: u32,
}

impl SubmissionCheck for FiniteCheck {
    fn reset(&self) {
        self.status.write_typed(0, &[0_u32]);
    }

    fn assert_success(&self) {
        let mut bytes = [0_u8; 16];
        self.status.read_bytes(0, &mut bytes[..4]);
        if u32::from_ne_bytes(bytes[..4].try_into().unwrap()) == 0 {
            return;
        }
        self.status.read_bytes(0, &mut bytes);
        let [_, row, column, bits] =
            std::array::from_fn(|index| u32::from_ne_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap()));
        panic!(
            "Metal non-finite value: {}; row={row} column={column} row_width={} dtype={:?} value={} \
             f32_bits=0x{bits:08x}",
            self.label,
            self.row_width,
            self.dtype,
            f32::from_bits(bits)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::ReplayParameterKey;
    use crate::metal::Stream;

    #[test]
    fn test_active_rows_and_snapshot() {
        const ROWS: ReplayParameterKey = ReplayParameterKey::new("check_finite.rows");
        let device = Device::system_default();
        let stream = Stream::new(&device);
        struct Clear<'a> {
            kernel: &'a CompiledKernel,
            buffer: &'a Buffer,
        }
        impl Operator for Clear<'_> {
            fn record(self, recorder: &CommandRecorder<'_>) {
                recorder.set_kernel(self.kernel);
                recorder.set_buffer_write(0, self.buffer, 0);
                recorder.dispatch_1d(self.buffer.len_bytes(), 32);
            }
        }
        let clear = CompiledKernel::new(
            &device,
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void clear(device uchar* values [[buffer(0)]], uint index [[thread_position_in_grid]]) {
                values[index] = 0;
            }
        "#,
            "clear",
        );
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let compute = Compute::new(&device, dtype, 4);
            let input = Buffer::new_zeroed_elements(&device, 12_u32, dtype);
            let mut builder = stream.create_replay_program();
            builder.record(compute.invoke(
                Input {
                    buffer: &input,
                    num_total_rows: 3,
                    num_active_rows: ReplayU32::Parameter(ROWS),
                },
                "Main layer=3 attention.output".to_owned(),
            ));
            builder.record(Clear {
                kernel: &clear,
                buffer: &input,
            });
            builder.record(compute.invoke(
                Input {
                    buffer: &input,
                    num_total_rows: 3,
                    num_active_rows: ReplayU32::Parameter(ROWS),
                },
                "later scratch reuse".to_owned(),
            ));
            let replay = builder.build();
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let write_bad = || {
                    match dtype {
                        Dtype::Float32 => input.write_typed(6, &[bad]),
                        Dtype::Bfloat16 => input.write_typed(6, &[half::bf16::from_f32(bad).to_bits()]),
                        _ => unreachable!(),
                    }
                };
                write_bad();
                // Inactive scratch may contain arbitrary values.
                stream
                    .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(ROWS, 1))
                    .wait();
                write_bad();
                let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    stream
                        .submit_replay_with_arguments(&replay, &ReplayArguments::new().with_u32(ROWS, 2))
                        .wait();
                }))
                .expect_err("active non-finite value must fail");
                let message = failure.downcast_ref::<String>().unwrap();
                assert!(
                    message.contains("Main layer=3 attention.output; row=1 column=2"),
                    "{message}"
                );
                assert!(
                    message.contains(&format!("f32_bits=0x{:08x}", bad.to_bits())),
                    "{message}"
                );
                assert!(
                    input
                        .read_typed::<u8>(0, input.len_bytes())
                        .iter()
                        .all(|value| *value == 0)
                );
            }
        }
    }
}
