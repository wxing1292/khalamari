use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../metal/gdn_state_replay.metal");
const REQUIRED_THREADS: usize = 256;

/// Shared geometry of the all-layer replay log and state arenas.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub num_gdn_layers: u32,
    pub num_state_slots: u32,
    pub max_replay_tokens: u32,
    pub num_qk_heads: u32,
    pub qk_head_dim: u32,
    pub num_v_heads: u32,
    pub v_head_dim: u32,
    pub conv_state_len: u32,
}

impl Config {
    pub fn validate(self) {
        assert!(self.num_gdn_layers > 0 && self.num_state_slots > 0 && self.max_replay_tokens > 0);
        assert!(self.num_qk_heads > 0 && self.qk_head_dim > 0 && self.num_v_heads > 0 && self.v_head_dim > 0);
        assert_eq!(self.num_v_heads % self.num_qk_heads, 0);
        assert!(self.conv_state_len > 0);
        u32::try_from(self.recurrent_state_values()).expect("GDN replay recurrent stride must fit u32");
        u32::try_from(self.conv_state_values()).expect("GDN replay convolution stride must fit u32");
    }

    pub fn alpha_bytes(self) -> usize {
        self.log_bytes(self.num_v_heads as usize, size_of::<f32>())
    }
    pub fn k_bytes(self) -> usize {
        self.log_bytes(self.num_qk_heads as usize * self.qk_head_dim as usize, size_of::<f32>())
    }
    pub fn u_bytes(self) -> usize {
        self.log_bytes(self.num_v_heads as usize * self.v_head_dim as usize, size_of::<f32>())
    }
    pub fn qkv_bytes(self) -> usize {
        self.log_bytes(self.qkv_dim(), size_of::<u16>())
    }

    fn log_bytes(self, width: usize, value_bytes: usize) -> usize {
        checked_product(
            "GDN replay log bytes",
            &[
                self.num_gdn_layers as usize,
                self.max_replay_tokens as usize,
                width,
                value_bytes,
            ],
        )
    }
    fn qkv_dim(self) -> usize {
        self.num_qk_heads
            .checked_mul(self.qk_head_dim)
            .and_then(|dim| dim.checked_mul(2))
            .and_then(|dim| {
                self.num_v_heads
                    .checked_mul(self.v_head_dim)
                    .and_then(|v_dim| dim.checked_add(v_dim))
            })
            .expect("GDN replay concatenated Q/K/V dimension must fit u32") as usize
    }
    fn recurrent_state_values(self) -> usize {
        checked_product(
            "GDN replay recurrent state values",
            &[
                self.num_v_heads as usize,
                self.v_head_dim as usize,
                self.qk_head_dim as usize,
            ],
        )
    }
    fn recurrent_state_values_per_thread(self) -> usize {
        [16, 4, 1]
            .into_iter()
            .find(|&count| (self.qk_head_dim as usize).is_multiple_of(count))
            .unwrap()
    }
    fn conv_state_values(self) -> usize {
        checked_product(
            "GDN replay convolution state values",
            &[self.qkv_dim(), self.conv_state_len as usize],
        )
    }
    fn arena_bytes(self, state_values: usize) -> usize {
        checked_product(
            "GDN replay arena bytes",
            &[
                self.num_gdn_layers as usize,
                self.num_state_slots as usize,
                state_values,
                size_of::<u16>(),
            ],
        )
    }
}

/// One accepted state version, materialized across every GDN layer.
/// Source slots must remain live and distinct from all destination slots.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Job {
    pub src_recurrent_state_slot: u32,
    pub src_conv_state_slot: u32,
    pub dst_recurrent_state_slot: u32,
    pub dst_conv_state_slot: u32,
    pub replay_token_begin: u32,
    pub num_tokens: u32,
}

/// Packs materialization jobs in the shader ABI field order.
pub fn write_jobs(buffer: &Buffer, jobs: &[Job]) {
    let values = jobs
        .iter()
        .flat_map(|job| {
            [
                job.src_recurrent_state_slot,
                job.src_conv_state_slot,
                job.dst_recurrent_state_slot,
                job.dst_conv_state_slot,
                job.replay_token_begin,
                job.num_tokens,
            ]
        })
        .collect::<Vec<_>>();
    buffer.write_typed(0, &values);
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub recurrent_states: &'a Buffer,
    pub conv_states: &'a Buffer,
    pub alpha: &'a Buffer,
    pub k: &'a Buffer,
    pub u: &'a Buffer,
    pub qkv: &'a Buffer,
    pub jobs: &'a Buffer,
}

pub struct Commit {
    config: Config,
    recurrent: CompiledKernel,
    conv: CompiledKernel,
}

impl Commit {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let num_values_per_thread = config.recurrent_state_values_per_thread();
        let vector_width = num_values_per_thread.min(4);
        let vector_suffix = if vector_width == 1 { "" } else { "4" };
        let constants = format!(
            "using namespace metal;\nconstant uint num_qk_heads = {}u;\nconstant uint qk_head_dim = {}u;\nconstant \
             uint num_v_heads = {}u;\nconstant uint v_head_dim = {}u;\nconstant uint qkv_dim = {}u;\nconstant uint \
             conv_state_len = {}u;\nconstant uint recurrent_state_vector_width = {vector_width}u;\nconstant uint \
             recurrent_state_num_vectors = {}u;\nusing RecurrentStateVector = float{vector_suffix};\nusing \
             RecurrentStateStorage = bfloat{vector_suffix};\n",
            config.num_qk_heads,
            config.qk_head_dim,
            config.num_v_heads,
            config.v_head_dim,
            config.qkv_dim(),
            config.conv_state_len,
            num_values_per_thread / vector_width,
        );
        let source = SOURCE.replacen("using namespace metal;", &constants, 1);
        Self {
            config,
            recurrent: CompiledKernel::new(device, &source, "gdn_state_replay_recurrent_bf16"),
            conv: CompiledKernel::new(device, &source, "gdn_state_replay_conv_bf16"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        num_total_jobs: u32,
        num_active_jobs: ReplayU32,
        buffers: Buffers<'a>,
    ) -> Invocation<'a> {
        assert!(num_total_jobs > 0);
        num_active_jobs.validate_range(0, num_total_jobs);
        let config = self.config;
        assert!(buffers.jobs.len_bytes() >= num_total_jobs as usize * size_of::<Job>());
        assert!(buffers.recurrent_states.len_bytes() >= config.arena_bytes(config.recurrent_state_values()));
        assert!(buffers.conv_states.len_bytes() >= config.arena_bytes(config.conv_state_values()));
        assert!(buffers.alpha.len_bytes() >= config.alpha_bytes());
        assert!(buffers.k.len_bytes() >= config.k_bytes());
        assert!(buffers.u.len_bytes() >= config.u_bytes());
        assert!(buffers.qkv.len_bytes() >= config.qkv_bytes());
        Invocation {
            kernel: self,
            num_total_jobs,
            num_active_jobs,
            buffers,
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a Commit,
    num_total_jobs: u32,
    num_active_jobs: ReplayU32,
    buffers: Buffers<'a>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        recorder.set_kernel(&self.kernel.recurrent);
        recorder.set_buffer_read_write(0, self.buffers.recurrent_states, 0);
        recorder.set_buffer_read(1, self.buffers.alpha, 0);
        recorder.set_buffer_read(2, self.buffers.k, 0);
        recorder.set_buffer_read(3, self.buffers.u, 0);
        recorder.set_buffer_read(4, self.buffers.jobs, 0);
        self.bind_active_jobs(recorder, 5);
        recorder.set_u32(6, config.num_state_slots);
        recorder.set_u32(7, config.max_replay_tokens);
        self.dispatch(
            recorder,
            config.recurrent_state_values() / config.recurrent_state_values_per_thread(),
        );
        recorder.set_kernel(&self.kernel.conv);
        recorder.set_buffer_read_write(0, self.buffers.conv_states, 0);
        recorder.set_buffer_read(1, self.buffers.qkv, 0);
        recorder.set_buffer_read(2, self.buffers.jobs, 0);
        self.bind_active_jobs(recorder, 3);
        recorder.set_u32(4, config.num_state_slots);
        recorder.set_u32(5, config.max_replay_tokens);
        self.dispatch(recorder, config.conv_state_values());
    }
}

impl Invocation<'_> {
    fn bind_active_jobs(&self, recorder: &CommandRecorder<'_>, index: usize) {
        match self.num_active_jobs {
            ReplayU32::Fixed(value) => recorder.set_u32(index, value),
            ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 0, self.num_total_jobs),
        }
    }
    fn dispatch(&self, recorder: &CommandRecorder<'_>, num_state_values: usize) {
        recorder.dispatch_threadblocks(
            (
                num_state_values.div_ceil(REQUIRED_THREADS),
                self.num_total_jobs as usize,
                self.kernel.config.num_gdn_layers as usize,
            ),
            (REQUIRED_THREADS, 1, 1),
        );
    }
}

#[cfg(test)]
#[path = "state_replay_test.rs"]
mod tests;
