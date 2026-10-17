use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const THREADS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub num_channels: u32,
    pub hidden_dim: u32,
    pub conv_kernel_size: u32,
    pub conv_dilation: u32,
    pub norm_eps: f32,
}
impl Config {
    pub fn validate(self) {
        assert!(self.num_channels > 0 && self.hidden_dim > 0);
        assert!(self.conv_kernel_size > 1 && self.conv_dilation > 0);
        assert!(self.norm_eps.is_finite() && self.norm_eps > 0.0);
        self.num_channels
            .checked_mul(self.hidden_dim)
            .expect("PLE width must fit u32");
        (self.conv_kernel_size - 1)
            .checked_mul(self.conv_dilation)
            .expect("PLE state length must fit u32");
        assert_u32_count_domain(
            checked_product("PLE state values", &[self.tail() as usize, self.wide() as usize]),
            "PLE state values",
        );
    }
    fn wide(self) -> u32 {
        self.num_channels * self.hidden_dim
    }
    fn tail(self) -> u32 {
        (self.conv_kernel_size - 1) * self.conv_dilation
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Token {
    pub batch_start: u32,
    pub src_state_slot: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Request {
    pub query_start: u32,
    pub src_state_slot: u32,
    pub dst_state_slot: u32,
}

#[derive(Clone, Copy)]
pub struct NormBuffers<'a> {
    pub input: &'a Buffer,
    pub weight: &'a Buffer,
    pub output: &'a Buffer,
}
#[derive(Clone, Copy)]
pub struct GateBuffers<'a> {
    pub key: &'a Buffer,
    pub query: &'a Buffer,
    pub value: &'a Buffer,
    pub output: &'a Buffer,
}
#[derive(Clone, Copy)]
pub struct ConvBuffers<'a> {
    pub normalized: &'a Buffer,
    pub gated: &'a Buffer,
    pub residual: &'a Buffer,
    pub weight: &'a Buffer,
    pub tokens: &'a Buffer,
    pub states: &'a Buffer,
    pub output: &'a Buffer,
}
#[derive(Clone, Copy)]
pub struct CommitBuffers<'a> {
    pub normalized: &'a Buffer,
    pub requests: &'a Buffer,
    pub accepted_tokens: &'a Buffer,
    pub states: &'a Buffer,
}

pub struct Compute {
    config: Config,
    norm: CompiledKernel,
    gate: CompiledKernel,
    conv: CompiledKernel,
    commit: CompiledKernel,
}
impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let source = format!(
            "#include <metal_stdlib>\nusing namespace metal;\nconstant uint channels={}u, hidden={}u, wide={}u, \
             taps={}u, dilation={}u, tail={}u, threads={THREADS}u;\nconstant float eps={:.9e}f;\n{}",
            config.num_channels,
            config.hidden_dim,
            config.wide(),
            config.conv_kernel_size,
            config.conv_dilation,
            config.tail(),
            config.norm_eps,
            include_str!("metal/ple.metal")
        );
        Self {
            config,
            norm: CompiledKernel::new(device, &source, "ple_norm"),
            gate: CompiledKernel::new(device, &source, "ple_gate"),
            conv: CompiledKernel::new(device, &source, "ple_conv"),
            commit: CompiledKernel::new(device, &source, "ple_commit"),
        }
    }
    pub fn norm<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        buffers: NormBuffers<'a>,
    ) -> NormInvocation<'a> {
        self.validate_rows(num_total_tokens, num_active_tokens, &[buffers.input, buffers.output]);
        assert!(buffers.weight.len_bytes() >= self.config.wide() as usize * 2);
        NormInvocation {
            compute: self,
            num_total_tokens,
            num_active_tokens,
            buffers,
        }
    }
    pub fn gate<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        buffers: GateBuffers<'a>,
    ) -> GateInvocation<'a> {
        self.validate_rows(
            num_total_tokens,
            num_active_tokens,
            &[buffers.key, buffers.query, buffers.output],
        );
        assert!(buffers.value.len_bytes() >= bytes(num_total_tokens, self.config.hidden_dim as usize * 2));
        GateInvocation {
            compute: self,
            num_total_tokens,
            num_active_tokens,
            buffers,
        }
    }
    pub fn conv<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        num_state_slots: u32,
        buffers: ConvBuffers<'a>,
    ) -> ConvInvocation<'a> {
        self.validate_rows(
            num_total_tokens,
            num_active_tokens,
            &[buffers.normalized, buffers.gated, buffers.residual, buffers.output],
        );
        self.validate_states(num_state_slots, buffers.states);
        assert!(buffers.tokens.len_bytes() >= bytes(num_total_tokens, size_of::<Token>()));
        assert!(buffers.weight.len_bytes() >= bytes(self.config.wide(), self.config.conv_kernel_size as usize * 2));
        ConvInvocation {
            compute: self,
            num_total_tokens,
            num_active_tokens,
            buffers,
        }
    }
    pub fn commit<'a>(
        &'a self,
        num_total_tokens: u32,
        num_total_requests: u32,
        num_active_requests: ReplayU32,
        num_state_slots: u32,
        buffers: CommitBuffers<'a>,
    ) -> CommitInvocation<'a> {
        validate_active(num_total_requests, num_active_requests);
        assert!(
            num_total_tokens > 0
                && buffers.normalized.len_bytes() >= bytes(num_total_tokens, self.config.wide() as usize * 2)
        );
        assert!(buffers.requests.len_bytes() >= bytes(num_total_requests, size_of::<Request>()));
        assert!(buffers.accepted_tokens.len_bytes() >= bytes(num_total_requests, 4));
        self.validate_states(num_state_slots, buffers.states);
        CommitInvocation {
            compute: self,
            num_total_requests,
            num_active_requests,
            buffers,
        }
    }
    fn validate_rows(&self, num_total_tokens: u32, num_active_tokens: ReplayU32, buffers: &[&Buffer]) {
        validate_active(num_total_tokens, num_active_tokens);
        let values = checked_product("PLE values", &[num_total_tokens as usize, self.config.wide() as usize]);
        let bytes = values.checked_mul(2).expect("PLE bytes must fit usize");
        assert!(buffers.iter().all(|buffer| buffer.len_bytes() >= bytes));
    }
    fn validate_states(&self, slots: u32, states: &Buffer) {
        assert!(slots > 0);
        assert!(
            states.len_bytes()
                >= checked_product(
                    "PLE state bytes",
                    &[
                        slots as usize,
                        self.config.tail() as usize,
                        self.config.wide() as usize,
                        2
                    ]
                )
        );
    }
}

pub struct NormInvocation<'a> {
    compute: &'a Compute,
    num_total_tokens: u32,
    num_active_tokens: ReplayU32,
    buffers: NormBuffers<'a>,
}
impl Operator for NormInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.norm);
        recorder.set_buffer_read(0, self.buffers.input, 0);
        recorder.set_buffer_read(1, self.buffers.weight, 0);
        recorder.set_buffer_write(2, self.buffers.output, 0);
        bind_active(recorder, 3, self.num_active_tokens, self.num_total_tokens);
        recorder.dispatch_threadblocks(
            (
                self.compute.config.num_channels as usize,
                self.num_total_tokens as usize,
                1,
            ),
            (THREADS, 1, 1),
        );
    }
}
pub struct GateInvocation<'a> {
    compute: &'a Compute,
    num_total_tokens: u32,
    num_active_tokens: ReplayU32,
    buffers: GateBuffers<'a>,
}
impl Operator for GateInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.gate);
        recorder.set_buffer_read(0, self.buffers.key, 0);
        recorder.set_buffer_read(1, self.buffers.query, 0);
        recorder.set_buffer_read(2, self.buffers.value, 0);
        recorder.set_buffer_write(3, self.buffers.output, 0);
        bind_active(recorder, 4, self.num_active_tokens, self.num_total_tokens);
        recorder.dispatch_threadblocks(
            (
                self.compute.config.num_channels as usize,
                self.num_total_tokens as usize,
                1,
            ),
            (THREADS, 1, 1),
        );
    }
}
pub struct ConvInvocation<'a> {
    compute: &'a Compute,
    num_total_tokens: u32,
    num_active_tokens: ReplayU32,
    buffers: ConvBuffers<'a>,
}
impl Operator for ConvInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.conv);
        for (i, b) in [
            self.buffers.normalized,
            self.buffers.gated,
            self.buffers.residual,
            self.buffers.weight,
            self.buffers.tokens,
            self.buffers.states,
        ]
        .into_iter()
        .enumerate()
        {
            recorder.set_buffer_read(i, b, 0);
        }
        recorder.set_buffer_write(6, self.buffers.output, 0);
        bind_active(recorder, 7, self.num_active_tokens, self.num_total_tokens);
        recorder.dispatch_threadblocks(
            (
                (self.compute.config.wide() as usize).div_ceil(256),
                self.num_total_tokens as usize,
                1,
            ),
            (256, 1, 1),
        );
    }
}
pub struct CommitInvocation<'a> {
    compute: &'a Compute,
    num_total_requests: u32,
    num_active_requests: ReplayU32,
    buffers: CommitBuffers<'a>,
}
impl Operator for CommitInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.commit);
        recorder.set_buffer_read(0, self.buffers.normalized, 0);
        recorder.set_buffer_read(1, self.buffers.requests, 0);
        recorder.set_buffer_read(2, self.buffers.accepted_tokens, 0);
        recorder.set_buffer_read_write(3, self.buffers.states, 0);
        bind_active(recorder, 4, self.num_active_requests, self.num_total_requests);
        recorder.dispatch_threadblocks(
            (
                (self.compute.config.wide() as usize * self.compute.config.tail() as usize).div_ceil(256),
                self.num_total_requests as usize,
                1,
            ),
            (256, 1, 1),
        );
    }
}
fn bytes(rows: u32, width: usize) -> usize {
    checked_product("PLE buffer bytes", &[rows as usize, width])
}
fn validate_active(capacity: u32, num_active_items: ReplayU32) {
    assert!(capacity > 0);
    if let ReplayU32::Fixed(n) = num_active_items {
        assert!(n > 0 && n <= capacity);
    }
}
fn bind_active(recorder: &CommandRecorder<'_>, index: usize, num_active_items: ReplayU32, capacity: u32) {
    match num_active_items {
        ReplayU32::Fixed(n) => recorder.set_u32(index, n),
        ReplayU32::Parameter(k) => recorder.bind_u32(index, k, 1, capacity),
    }
}

#[cfg(test)]
#[path = "ple_test.rs"]
mod tests;
