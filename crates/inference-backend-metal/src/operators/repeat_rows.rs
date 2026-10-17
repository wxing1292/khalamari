//! Repeat each input row into adjacent output rows.
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Operator;
use crate::metal::ReplayU32;

pub struct Compute {
    kernel: CompiledKernel,
    num_columns: u32,
    num_repeats: u32,
}
impl Compute {
    pub fn new(device: &Device, num_columns: u32, num_repeats: u32) -> Self {
        assert!(num_columns > 0 && num_repeats > 0);
        num_columns
            .checked_mul(num_repeats)
            .expect("repeated row width must fit u32");
        Self {
            kernel: CompiledKernel::new(device, include_str!("metal/repeat_rows.metal"), "repeat_rows"),
            num_columns,
            num_repeats,
        }
    }
    pub fn invoke<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        input: &'a Buffer,
        output: &'a Buffer,
    ) -> Invocation<'a> {
        assert!(num_total_tokens > 0);
        let bytes = u64::from(num_total_tokens) * u64::from(self.num_columns) * 2;
        assert!(input.len_bytes_u64() >= bytes && output.len_bytes_u64() >= bytes * u64::from(self.num_repeats));
        Invocation {
            compute: self,
            num_total_tokens,
            num_active_tokens,
            input,
            output,
        }
    }
}
pub struct Invocation<'a> {
    compute: &'a Compute,
    num_total_tokens: u32,
    num_active_tokens: ReplayU32,
    input: &'a Buffer,
    output: &'a Buffer,
}
impl Operator for Invocation<'_> {
    fn record(self, r: &CommandRecorder<'_>) {
        r.set_kernel(&self.compute.kernel);
        r.set_buffer_read(0, self.input, 0);
        r.set_buffer_write(1, self.output, 0);
        r.set_u32(2, self.compute.num_columns);
        r.set_u32(3, self.compute.num_repeats);
        match self.num_active_tokens {
            ReplayU32::Fixed(v) => r.set_u32(4, v),
            ReplayU32::Parameter(k) => r.bind_u32(4, k, 1, self.num_total_tokens),
        }
        r.dispatch_threadblocks(
            (
                (self.compute.num_columns as usize * self.compute.num_repeats as usize).div_ceil(256),
                self.num_total_tokens as usize,
                1,
            ),
            (256, 1, 1),
        );
    }
}
