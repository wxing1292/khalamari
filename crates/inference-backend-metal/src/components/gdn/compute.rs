use crate::components::assert_u32_count_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Operator;
use crate::metal::ReplayU32;

mod chunkwise;
mod recurrent;
mod replay;

pub use replay::ReplayBuffers;

const GDN_COMPUTE_SOURCE: &str = include_str!("../metal/gdn_compute.metal");

// Initial short-input policy. This limit is independent of the chunkwise tile width.
const MAX_FIXED_REPLAY_TOKENS: usize = 8;

/// Maximum replay inputs per request, including fixed and speculative tokens.
/// Call at initialization or input validation, where the token-count domain is established.
pub fn max_replay_tokens_per_request(num_spec_tokens: usize) -> usize {
    MAX_FIXED_REPLAY_TOKENS
        .checked_add(num_spec_tokens)
        .expect("GDN replay token capacity must fit usize")
}

/// Selects chunkwise execution for a committed segment with no selectable suffix.
pub fn uses_chunkwise(num_tokens: usize, num_spec_tokens: usize) -> bool {
    num_spec_tokens == 0 && num_tokens > MAX_FIXED_REPLAY_TOKENS
}

const SHORT_CONV_REQUIRED_THREADS: u32 = 256;
const CHUNKWISE_TOKEN_CHUNK_SIZE: u32 = 16;
const CHUNKWISE_NUM_V_ROWS_PER_SIMDGROUP: u32 = 8;
const CHUNKWISE_NUM_QK_DIM_THREADS: u32 = 32;
const FINAL_RECURRENT_STATE_NUM_QK_DIM_THREADS: u32 = 32;
const CANDIDATE_RECURRENT_STATE_NUM_QK_DIM_THREADS: u32 = 32;
const CANDIDATE_RECURRENT_STATE_NUM_V_ROWS_PER_SIMDGROUP: u32 = 2;
const CANDIDATE_RECURRENT_STATE_NUM_SIMDGROUPS: u32 = 2;
const OUTPUT_NORM_GATE_REQUIRED_THREADS: u32 = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants<T> {
    thread_block: T,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FinalRecurrentStateThreadBlockConstants {
    num_qk_dim_threads: u32,
    num_v_rows: u32,
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkwiseStateThreadBlockConstants {
    num_v_rows: u32,
    num_state_qk_columns: u32,
}

impl ChunkwiseStateThreadBlockConstants {
    fn num_simdgroups(self) -> u32 {
        self.num_v_rows.div_ceil(CHUNKWISE_NUM_V_ROWS_PER_SIMDGROUP)
    }

    fn threadblock_memory_bytes(self, qk_head_dim: u32) -> u64 {
        // Match the static arrays in gdn_compute_chunkwise.metal. Four-row
        // tiles still reserve eight V rows for their single SIMDgroup.
        let num_tokens = CHUNKWISE_TOKEN_CHUNK_SIZE as u64;
        let num_v_rows = self.num_simdgroups() as u64 * CHUNKWISE_NUM_V_ROWS_PER_SIMDGROUP as u64;
        let qk_head_dim = qk_head_dim as u64;
        let num_bf16_values =
            2 * num_tokens * qk_head_dim + num_tokens * num_v_rows + num_v_rows * qk_head_dim + num_tokens * num_tokens;
        let num_f32_values = 2 * num_tokens * num_tokens + 3 * num_tokens;
        num_bf16_values * 2 + num_f32_values * 4
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CandidateRecurrentStateSimdgroupConstants {
    num_v_rows: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CandidateRecurrentStateThreadBlockConstants {
    num_qk_dim_threads: u32,
    num_simdgroups: u32,
    simdgroup: CandidateRecurrentStateSimdgroupConstants,
    required_threads: u32,
}

impl CandidateRecurrentStateThreadBlockConstants {
    fn num_v_rows(self) -> u32 {
        self.num_simdgroups * self.simdgroup.num_v_rows
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ModelGeometry {
    num_qk_heads: u32,
    qk_head_dim: u32,
    num_v_heads: u32,
    v_head_dim: u32,
    conv_kernel_size: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelSetConstants {
    short_conv: KernelConstants<ThreadBlockConstants>,
    candidate_conv_state: KernelConstants<ThreadBlockConstants>,
    final_recurrent_state: KernelConstants<FinalRecurrentStateThreadBlockConstants>,
    chunkwise_state: KernelConstants<ChunkwiseStateThreadBlockConstants>,
    candidate_recurrent_state: KernelConstants<CandidateRecurrentStateThreadBlockConstants>,
    output_norm_gate: KernelConstants<ThreadBlockConstants>,
}

/// Compile-time model and kernel geometry for the mixed compute graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ComputeConstants {
    model: ModelGeometry,
    kernels: KernelSetConstants,
    output_gate: OutputGate,
}

/// Initialization-time activation for the projected output gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputGate {
    Silu,
    Sigmoid,
}

/// Construction input for the generic GDN compute graph.
///
/// The core consumes projected `qkv`, `a`, `b`, and `z` tensors over flat token
/// axis `T`. Q/K use `[T, Hqk, Dqk]`; V and recurrent output use
/// `[T, Hv, Dv]`; recurrent state uses `[S, Hv, Dv, Dqk]`. `Cqkv` is the
/// concatenated channel width `2 * Hqk * Dqk + Hv * Dv` at projection and
/// short-convolution boundaries; it is unrelated to convolution-kernel width.
///
/// ```text
/// qkv + conv_state + conv_weight
///   -> causal depthwise short_conv -> SiLU -> conv_qkv
/// old conv_state + qkv
///   -> take final Ks inputs -> next_conv_state
/// ```
///
/// `conv_qkv` is the post-SiLU recurrent-core input, not the raw convolution
/// accumulation. `next_conv_state` contains the final `Ks = Kc - 1` inputs for
/// the next invocation. Model geometry determines `ComputeConstants` during
/// initialization. `q_scale` and `norm_eps` remain kernel arguments.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub num_qk_heads: u32,
    pub qk_head_dim: u32,
    pub num_v_heads: u32,
    pub v_head_dim: u32,
    pub conv_kernel_size: u32,
    pub q_scale: f32,
    pub norm_eps: f32,
    pub output_gate: OutputGate,
}

impl Config {
    fn model_geometry(self) -> ModelGeometry {
        ModelGeometry {
            num_qk_heads: self.num_qk_heads,
            qk_head_dim: self.qk_head_dim,
            num_v_heads: self.num_v_heads,
            v_head_dim: self.v_head_dim,
            conv_kernel_size: self.conv_kernel_size,
        }
    }

    pub fn qkv_dim(self) -> u32 {
        self.model_geometry().qkv_dim()
    }

    pub fn conv_state_len(self) -> u32 {
        self.model_geometry().conv_state_len()
    }

    pub fn recurrent_state_stride(self) -> usize {
        self.model_geometry().recurrent_state_stride()
    }

    pub fn num_recurrent_output_values(self, shape: Shape) -> usize {
        self.model_geometry().num_recurrent_output_values(shape)
    }

    pub fn num_qkv_values(self, shape: Shape) -> usize {
        self.model_geometry().num_qkv_values(shape)
    }

    pub fn num_conv_state_values(self, shape: Shape) -> usize {
        self.model_geometry().num_conv_state_values(shape)
    }

    fn num_conv_weight_values(self) -> usize {
        self.model_geometry().num_conv_weight_values()
    }

    fn validate(self) {
        assert!(self.num_qk_heads > 0);
        assert!(self.qk_head_dim > 0);
        assert!(self.num_v_heads > 0);
        assert!(self.v_head_dim > 0);
        assert_eq!(self.num_v_heads % self.num_qk_heads, 0);
        assert!(self.conv_kernel_size > 1);
        assert!(self.q_scale.is_finite() && self.q_scale > 0.0);
        assert!(self.norm_eps.is_finite() && self.norm_eps > 0.0);
        assert_eq!(
            self.qk_head_dim % CANDIDATE_RECURRENT_STATE_NUM_QK_DIM_THREADS,
            0,
            "GDN candidate register-V requires Dqk divisible by the SIMDgroup width"
        );
        assert_eq!(
            self.v_head_dim % CANDIDATE_RECURRENT_STATE_NUM_V_ROWS_PER_SIMDGROUP,
            0,
            "GDN candidate recurrent-state constants require Dv divisible by the SIMDgroup V-row count"
        );
        assert_eq!(
            (self.v_head_dim / CANDIDATE_RECURRENT_STATE_NUM_V_ROWS_PER_SIMDGROUP)
                % CANDIDATE_RECURRENT_STATE_NUM_SIMDGROUPS,
            0,
            "GDN candidate recurrent thread-block V-row count must divide Dv"
        );
        let _ = self.qkv_dim();
        let _ = self.recurrent_state_stride();
        let _ = self.num_conv_weight_values();
    }
}

impl ComputeConstants {
    fn from_config(config: Config, max_threadblock_memory_bytes: usize) -> Self {
        config.validate();
        let final_recurrent_state_num_v_rows = [8, 4]
            .into_iter()
            .find(|num_v_rows| {
                config.v_head_dim.is_multiple_of(*num_v_rows)
                    && *num_v_rows * FINAL_RECURRENT_STATE_NUM_QK_DIM_THREADS <= 1024
            })
            .expect("GDN recurrent V-dimension constants require Dv divisible by four");
        let candidate_recurrent_state_thread_block = CandidateRecurrentStateThreadBlockConstants {
            num_qk_dim_threads: CANDIDATE_RECURRENT_STATE_NUM_QK_DIM_THREADS,
            num_simdgroups: CANDIDATE_RECURRENT_STATE_NUM_SIMDGROUPS,
            simdgroup: CandidateRecurrentStateSimdgroupConstants {
                num_v_rows: CANDIDATE_RECURRENT_STATE_NUM_V_ROWS_PER_SIMDGROUP,
            },
            required_threads: CANDIDATE_RECURRENT_STATE_NUM_QK_DIM_THREADS * CANDIDATE_RECURRENT_STATE_NUM_SIMDGROUPS,
        };
        // Keep each cooperative state update within 128 Q/K columns.
        let num_state_qk_columns = [128, 96, 64, 32]
            .into_iter()
            .find(|num_columns| config.qk_head_dim.is_multiple_of(*num_columns))
            .unwrap();
        let chunkwise_state_thread_block = [64, 32, 16, 8, 4]
            .into_iter()
            .map(|num_v_rows| {
                ChunkwiseStateThreadBlockConstants {
                    num_v_rows,
                    num_state_qk_columns,
                }
            })
            .find(|thread_block| {
                config.v_head_dim.is_multiple_of(thread_block.num_v_rows)
                    && thread_block.threadblock_memory_bytes(config.qk_head_dim) <= max_threadblock_memory_bytes as u64
            })
            .expect("GDN chunkwise V-row tile must divide Dv and fit device shared memory");
        let constants = Self {
            model: config.model_geometry(),
            output_gate: config.output_gate,
            kernels: KernelSetConstants {
                short_conv: KernelConstants {
                    thread_block: ThreadBlockConstants {
                        required_threads: SHORT_CONV_REQUIRED_THREADS,
                    },
                },
                candidate_conv_state: KernelConstants {
                    thread_block: ThreadBlockConstants {
                        required_threads: SHORT_CONV_REQUIRED_THREADS,
                    },
                },
                final_recurrent_state: KernelConstants {
                    thread_block: FinalRecurrentStateThreadBlockConstants {
                        num_qk_dim_threads: FINAL_RECURRENT_STATE_NUM_QK_DIM_THREADS,
                        num_v_rows: final_recurrent_state_num_v_rows,
                        required_threads: FINAL_RECURRENT_STATE_NUM_QK_DIM_THREADS * final_recurrent_state_num_v_rows,
                    },
                },
                chunkwise_state: KernelConstants {
                    thread_block: chunkwise_state_thread_block,
                },
                candidate_recurrent_state: KernelConstants {
                    thread_block: candidate_recurrent_state_thread_block,
                },
                output_norm_gate: KernelConstants {
                    thread_block: ThreadBlockConstants {
                        required_threads: OUTPUT_NORM_GATE_REQUIRED_THREADS,
                    },
                },
            },
        };
        constants.validate();
        constants
    }

    fn validate(self) {
        assert!(self.kernels.short_conv.thread_block.required_threads > 0);
        assert!(self.kernels.candidate_conv_state.thread_block.required_threads > 0);
        let final_recurrent_state = self.kernels.final_recurrent_state.thread_block;
        assert_eq!(
            final_recurrent_state.required_threads,
            final_recurrent_state.num_qk_dim_threads * final_recurrent_state.num_v_rows
        );
        assert_eq!(self.model.v_head_dim % final_recurrent_state.num_v_rows, 0);
        let chunkwise_state = self.kernels.chunkwise_state.thread_block;
        assert_eq!(self.model.v_head_dim % chunkwise_state.num_v_rows, 0);
        let candidate_recurrent_state = self.kernels.candidate_recurrent_state.thread_block;
        assert_eq!(
            candidate_recurrent_state.required_threads,
            candidate_recurrent_state.num_qk_dim_threads * candidate_recurrent_state.num_simdgroups
        );
        assert_eq!(
            candidate_recurrent_state.num_qk_dim_threads, final_recurrent_state.num_qk_dim_threads,
            "GDN recurrent kernels share one generated Q/K-dimension thread count"
        );
        assert_eq!(self.model.v_head_dim % candidate_recurrent_state.num_v_rows(), 0);
        assert!(self.kernels.output_norm_gate.thread_block.required_threads > 0);
    }
}

impl ModelGeometry {
    fn qkv_dim(self) -> u32 {
        self.num_qk_heads
            .checked_mul(self.qk_head_dim)
            .and_then(|dim| dim.checked_mul(2))
            .and_then(|dim| {
                self.num_v_heads
                    .checked_mul(self.v_head_dim)
                    .and_then(|v_dim| dim.checked_add(v_dim))
            })
            .expect("GDN concatenated Q/K/V dimension must fit u32")
    }

    fn conv_state_len(self) -> u32 {
        self.conv_kernel_size - 1
    }

    fn recurrent_state_stride(self) -> usize {
        checked_product(
            "GDN recurrent state stride",
            &[
                self.num_v_heads as usize,
                self.v_head_dim as usize,
                self.qk_head_dim as usize,
            ],
        )
    }

    fn num_recurrent_output_values(self, shape: Shape) -> usize {
        checked_product(
            "GDN output element count",
            &[
                shape.num_total_tokens as usize,
                self.num_v_heads as usize,
                self.v_head_dim as usize,
            ],
        )
    }

    fn num_qkv_values(self, shape: Shape) -> usize {
        checked_product(
            "GDN convolution element count",
            &[shape.num_total_tokens as usize, self.qkv_dim() as usize],
        )
    }

    fn num_conv_state_values(self, shape: Shape) -> usize {
        checked_product(
            "GDN convolution state element count",
            &[
                shape.num_total_reqs as usize,
                self.qkv_dim() as usize,
                self.conv_state_len() as usize,
            ],
        )
    }

    fn num_candidate_conv_state_values(self, shape: Shape) -> usize {
        checked_product(
            "GDN candidate convolution state element count",
            &[
                shape.num_total_tokens as usize,
                self.qkv_dim() as usize,
                self.conv_state_len() as usize,
            ],
        )
    }

    fn num_conv_weight_values(self) -> usize {
        checked_product(
            "GDN convolution weight element count",
            &[self.qkv_dim() as usize, self.conv_kernel_size as usize],
        )
    }
}

impl ComputeConstants {
    fn total_output_norm_gate_threads(self, shape: Shape) -> usize {
        checked_product(
            "GDN output norm + gate thread count",
            &[
                shape.num_total_tokens as usize,
                self.model.num_v_heads as usize,
                self.kernels.output_norm_gate.thread_block.required_threads as usize,
            ],
        )
    }

    fn validate_shape(self, shape: Shape) {
        shape.validate();
        for (name, num_elements) in [
            ("GDN convolution", self.model.num_qkv_values(shape)),
            ("GDN output", self.model.num_recurrent_output_values(shape)),
            ("GDN convolution state", self.model.num_conv_state_values(shape)),
            ("GDN convolution weights", self.model.num_conv_weight_values()),
            (
                "GDN output norm + gate threads",
                self.total_output_norm_gate_threads(shape),
            ),
            ("GDN recurrent state stride", self.model.recurrent_state_stride()),
        ] {
            assert_u32_count_domain(num_elements, name);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub num_total_reqs: u32,
    pub num_total_tokens: u32,
}

impl Shape {
    fn validate(self) {
        assert!(self.num_total_reqs > 0);
        assert!(self.num_total_tokens > 0);
    }
}

fn source(constants: ComputeConstants) -> String {
    let model = constants.model;
    let source_constants = format!(
        "using namespace metal;\n\nconstant uint num_qk_heads = {num_qk_heads}u;\nconstant uint qk_head_dim = \
         {qk_head_dim}u;\nconstant uint num_v_heads = {num_v_heads}u;\nconstant uint v_head_dim = \
         {v_head_dim}u;\nconstant uint conv_kernel_size = {conv_kernel_size}u;\nconstant uint qkv_dim = \
         {qkv_dim}u;\nconstant uint conv_state_len = {conv_state_len}u;\nconstant uint \
         output_norm_gate_required_threads = {output_norm_gate_required_threads}u;\nconstant bool \
         output_gate_is_sigmoid = {output_gate_is_sigmoid};",
        num_qk_heads = model.num_qk_heads,
        qk_head_dim = model.qk_head_dim,
        num_v_heads = model.num_v_heads,
        v_head_dim = model.v_head_dim,
        conv_kernel_size = model.conv_kernel_size,
        qkv_dim = model.qkv_dim(),
        conv_state_len = model.conv_state_len(),
        output_norm_gate_required_threads = constants.kernels.output_norm_gate.thread_block.required_threads,
        output_gate_is_sigmoid = constants.output_gate == OutputGate::Sigmoid,
    );
    let common_source = GDN_COMPUTE_SOURCE.replacen("using namespace metal;", &source_constants, 1);
    format!(
        "{common_source}\n{}\n{}",
        recurrent::source(constants),
        replay::source(constants),
    )
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub qkv: &'a Buffer,
    pub a: &'a Buffer,
    pub b: &'a Buffer,
    pub z: &'a Buffer,
    pub conv_weight: &'a Buffer,
    pub norm_weight: &'a Buffer,
    pub a_log: &'a Buffer,
    pub dt_bias: &'a Buffer,
    pub cu_tokens: &'a Buffer,
    pub src_recurrent_state_slots: &'a Buffer,
    pub src_conv_state_slots: &'a Buffer,
    /// Persistent recurrent state slot for each forward row.
    ///
    /// `u32::MAX` keeps the row output but discards the row's recurrent state.
    pub flat_recurrent_state_write_slots: &'a Buffer,
    /// Persistent convolution state slot for each forward row.
    ///
    /// `u32::MAX` keeps the row output but discards the row's convolution state.
    pub flat_conv_state_write_slots: &'a Buffer,
    pub conv_state: &'a Buffer,
    pub conv_state_offset_bytes: u64,
    pub next_conv_state: &'a Buffer,
    pub next_conv_state_offset_bytes: u64,
    pub recurrent_state_arena: &'a Buffer,
    pub recurrent_state_arena_offset_bytes: u64,
    pub conv_qkv: &'a Buffer,
    pub recurrent_output: &'a Buffer,
    pub norm_gated_output: &'a Buffer,
}

pub struct Compute {
    constants: ComputeConstants,
    q_scale: f32,
    norm_eps: f32,
    short_conv: CompiledKernel,
    short_conv_replay: CompiledKernel,
    replay: CompiledKernel,
    candidate_conv_state: CompiledKernel,
    final_recurrent_state: CompiledKernel,
    chunkwise_state: CompiledKernel,
    candidate_recurrent_state: CompiledKernel,
    output_norm_gate: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        let constants = ComputeConstants::from_config(config, device.max_threadblock_memory_length());
        let source = source(constants);
        let chunkwise_state = CompiledKernel::new_tensor_ops(
            device,
            &format!("{source}\n{}", chunkwise::source(constants)),
            "gdn_compute_chunkwise_state_bf16",
        );
        assert!(chunkwise_state.static_threadblock_memory_length() <= device.max_threadblock_memory_length());
        assert!(
            (constants.kernels.chunkwise_state.thread_block.num_simdgroups() * CHUNKWISE_NUM_QK_DIM_THREADS) as usize
                <= chunkwise_state.max_total_threads_per_threadblock(),
            "GDN chunkwise threadblock exceeds the pipeline thread limit"
        );
        Self {
            constants,
            q_scale: config.q_scale,
            norm_eps: config.norm_eps,
            short_conv: CompiledKernel::new(device, &source, "gdn_compute_short_conv_bf16"),
            short_conv_replay: CompiledKernel::new(device, &source, "gdn_compute_short_conv_replay_bf16"),
            replay: CompiledKernel::new(device, &source, "gdn_compute_replay_bf16"),
            candidate_conv_state: CompiledKernel::new(device, &source, "gdn_compute_candidate_conv_state_bf16"),
            final_recurrent_state: CompiledKernel::new(device, &source, "gdn_compute_final_recurrent_state_bf16"),
            chunkwise_state,
            candidate_recurrent_state: CompiledKernel::new(
                device,
                &source,
                "gdn_compute_candidate_recurrent_state_bf16",
            ),
            output_norm_gate: CompiledKernel::new(device, &source, "gdn_compute_output_norm_gate_bf16"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        shape: Shape,
        buffers: Buffers<'a>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
    ) -> Invocation<'a> {
        Invocation {
            compute: self,
            shape,
            buffers,
            num_active_reqs,
            num_active_tokens,
        }
    }

    pub fn invoke_with_candidate_state_update<'a>(
        &'a self,
        shape: Shape,
        buffers: Buffers<'a>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
    ) -> CandidateStateUpdateInvocation<'a> {
        CandidateStateUpdateInvocation {
            compute: self,
            shape,
            buffers,
            num_active_reqs,
            num_active_tokens,
        }
    }

    /// Record a fixed mixed graph whose chunkwise and recurrent cores can overlap.
    ///
    /// The first `num_active_chunkwise_requests` requests use chunkwise execution. Both
    /// branches retain the full recorded request capacity. Either branch may
    /// have zero active requests. Each active request must have a token.
    /// Candidate mode writes every supplied state destination. Final-only
    /// mode ignores nonfinal destinations in both branches.
    pub fn invoke_mixed<'a>(
        &'a self,
        shape: Shape,
        buffers: Buffers<'a>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
        num_active_chunkwise_requests: ReplayU32,
        write_candidate_states: bool,
    ) -> MixedInvocation<'a> {
        MixedInvocation {
            compute: self,
            shape,
            buffers,
            num_active_reqs,
            num_active_tokens,
            num_active_chunkwise_requests,
            write_candidate_states,
        }
    }

    fn record_short_conv(
        &self,
        recorder: &CommandRecorder,
        shape: Shape,
        buffers: &Buffers<'_>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
        write_final_conv_state: bool,
    ) {
        recorder.set_kernel(&self.short_conv);
        self.record_short_conv_bindings(
            recorder,
            shape,
            buffers,
            num_active_reqs,
            num_active_tokens,
            write_final_conv_state,
        );
        recorder.set_buffer_write(13, buffers.conv_qkv, 0);
        recorder.set_u64(14, 0);
        recorder.set_u32(15, 0);
        self.dispatch_short_conv(recorder, shape);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_short_conv_bindings(
        &self,
        recorder: &CommandRecorder,
        shape: Shape,
        buffers: &Buffers<'_>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
        write_final_conv_state: bool,
    ) {
        recorder.set_buffer_write(0, buffers.conv_qkv, 0);
        recorder.set_buffer_write(1, buffers.next_conv_state, 0);
        recorder.set_buffer_read(2, buffers.qkv, 0);
        recorder.set_buffer_read(3, buffers.conv_state, 0);
        recorder.set_buffer_read(4, buffers.conv_weight, 0);
        recorder.set_buffer_read(5, buffers.src_conv_state_slots, 0);
        recorder.set_buffer_read(6, buffers.flat_conv_state_write_slots, 0);
        recorder.set_buffer_read(7, buffers.cu_tokens, 0);
        set_batch_args(recorder, shape, 8, num_active_reqs, num_active_tokens);
        recorder.set_u64(10, buffers.conv_state_offset_bytes);
        recorder.set_u64(11, buffers.next_conv_state_offset_bytes);
        recorder.set_u32(12, u32::from(write_final_conv_state));
    }

    fn dispatch_short_conv(&self, recorder: &CommandRecorder, shape: Shape) {
        let total_short_conv_threads = self
            .constants
            .model
            .num_qkv_values(shape)
            .max(self.constants.model.num_conv_state_values(shape));
        recorder.dispatch_1d(
            total_short_conv_threads,
            self.constants.kernels.short_conv.thread_block.required_threads as usize,
        );
    }

    fn record_candidate_conv_state(
        &self,
        recorder: &CommandRecorder,
        shape: Shape,
        buffers: &Buffers<'_>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
    ) {
        recorder.set_kernel(&self.candidate_conv_state);
        recorder.set_barrier_before();
        recorder.set_buffer_write(0, buffers.next_conv_state, 0);
        recorder.set_buffer_read(1, buffers.qkv, 0);
        recorder.set_buffer_read(2, buffers.conv_state, 0);
        recorder.set_buffer_read(3, buffers.src_conv_state_slots, 0);
        recorder.set_buffer_read(4, buffers.flat_conv_state_write_slots, 0);
        recorder.set_buffer_read(5, buffers.cu_tokens, 0);
        set_batch_args(recorder, shape, 6, num_active_reqs, num_active_tokens);
        recorder.set_u64(8, buffers.conv_state_offset_bytes);
        recorder.set_u64(9, buffers.next_conv_state_offset_bytes);
        recorder.dispatch_1d(
            self.constants.model.num_candidate_conv_state_values(shape),
            self.constants
                .kernels
                .candidate_conv_state
                .thread_block
                .required_threads as usize,
        );
    }

    /// Output norm + gate execution:
    ///
    /// ```text
    /// recurrent_output [T, Hv, Dv] -> RMS norm * configured output gate(z)
    ///   -> norm_gated_output [T, Hv, Dv]
    /// OutputNormGateThreadBlockTask / threadblock:
    ///   { flat_token_index, v_head_index }
    /// grid: (T * Hv, 1, 1); threadblock: (128, 1, 1)
    /// reduce: Dv; produces: one normalized/gated [Dv] vector
    /// ```
    ///
    /// Both task fields are grid-derived. The kernel does not require a
    /// materialized task buffer.
    fn record_output_norm_gate(
        &self,
        recorder: &CommandRecorder,
        shape: Shape,
        buffers: &Buffers<'_>,
        num_active_tokens: ReplayU32,
    ) {
        recorder.set_kernel(&self.output_norm_gate);
        recorder.set_barrier_before();
        recorder.set_buffer_write(0, buffers.norm_gated_output, 0);
        recorder.set_buffer_read(1, buffers.recurrent_output, 0);
        recorder.set_buffer_read(2, buffers.z, 0);
        recorder.set_buffer_read(3, buffers.norm_weight, 0);
        recorder.set_f32(4, self.norm_eps);
        set_replay_u32(
            recorder,
            5,
            num_active_tokens,
            shape.num_total_tokens,
            "GDN active token count",
        );
        recorder.dispatch_1d(
            self.constants.total_output_norm_gate_threads(shape),
            self.constants.kernels.output_norm_gate.thread_block.required_threads as usize,
        );
    }
}

pub struct Invocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_reqs: ReplayU32,
    num_active_tokens: ReplayU32,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let compute = self.compute;
        compute.constants.validate_shape(self.shape);
        validate_buffers(compute.constants, self.shape, &self.buffers);
        compute.record_short_conv(
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            self.num_active_tokens,
            true,
        );
        recurrent::record_final_recurrent_state(
            compute,
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            ReplayU32::Fixed(0),
        );
        compute.record_output_norm_gate(recorder, self.shape, &self.buffers, self.num_active_tokens);
    }
}

pub struct CandidateStateUpdateInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_reqs: ReplayU32,
    num_active_tokens: ReplayU32,
}

impl Operator for CandidateStateUpdateInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let compute = self.compute;
        compute.constants.validate_shape(self.shape);
        assert_u32_count_domain(
            compute.constants.model.num_candidate_conv_state_values(self.shape),
            "GDN candidate convolution state",
        );
        validate_buffers(compute.constants, self.shape, &self.buffers);
        compute.record_short_conv(
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            self.num_active_tokens,
            false,
        );
        compute.record_candidate_conv_state(
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            self.num_active_tokens,
        );
        recurrent::record_candidate_recurrent_state(
            compute,
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            ReplayU32::Fixed(0),
        );
        compute.record_output_norm_gate(recorder, self.shape, &self.buffers, self.num_active_tokens);
    }
}

pub struct MixedInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
    num_active_reqs: ReplayU32,
    num_active_tokens: ReplayU32,
    num_active_chunkwise_requests: ReplayU32,
    write_candidate_states: bool,
}

impl Operator for MixedInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let compute = self.compute;
        compute.constants.validate_shape(self.shape);
        validate_buffers(compute.constants, self.shape, &self.buffers);
        if let (ReplayU32::Fixed(num_active_reqs), ReplayU32::Fixed(num_active_chunkwise_requests)) =
            (self.num_active_reqs, self.num_active_chunkwise_requests)
        {
            assert!(num_active_chunkwise_requests <= num_active_reqs);
        }
        compute.record_short_conv(
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            self.num_active_tokens,
            !self.write_candidate_states,
        );
        if self.write_candidate_states {
            assert_u32_count_domain(
                compute.constants.model.num_candidate_conv_state_values(self.shape),
                "GDN candidate convolution state",
            );
            compute.record_candidate_conv_state(
                recorder,
                self.shape,
                &self.buffers,
                self.num_active_reqs,
                self.num_active_tokens,
            );
        }
        // The request prefix and suffix own disjoint output rows and state
        // slots. State slots may be scattered, but each has one request owner.
        recorder.record_disjoint_buffers(
            &[self.buffers.recurrent_output, self.buffers.recurrent_state_arena],
            || {
                chunkwise::record_chunkwise_state(
                    compute,
                    recorder,
                    self.shape,
                    &self.buffers,
                    self.num_active_chunkwise_requests,
                    self.write_candidate_states,
                );
                if self.write_candidate_states {
                    recurrent::record_candidate_recurrent_state(
                        compute,
                        recorder,
                        self.shape,
                        &self.buffers,
                        self.num_active_reqs,
                        self.num_active_chunkwise_requests,
                    );
                } else {
                    recurrent::record_final_recurrent_state(
                        compute,
                        recorder,
                        self.shape,
                        &self.buffers,
                        self.num_active_reqs,
                        self.num_active_chunkwise_requests,
                    );
                }
            },
        );
        compute.record_output_norm_gate(recorder, self.shape, &self.buffers, self.num_active_tokens);
    }
}

fn set_chunkwise_count(recorder: &CommandRecorder<'_>, index: usize, value: ReplayU32, max_value: u32) {
    match value {
        ReplayU32::Fixed(value) => {
            assert!(
                value <= max_value,
                "GDN chunkwise request count exceeds recorded capacity"
            );
            recorder.set_u32(index, value);
        },
        ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 0, max_value),
    }
}

fn set_batch_args(
    recorder: &CommandRecorder,
    shape: Shape,
    start_index: usize,
    num_active_reqs: ReplayU32,
    num_active_tokens: ReplayU32,
) {
    set_replay_u32(
        recorder,
        start_index,
        num_active_reqs,
        shape.num_total_reqs,
        "GDN active request count",
    );
    set_replay_u32(
        recorder,
        start_index + 1,
        num_active_tokens,
        shape.num_total_tokens,
        "GDN active token count",
    );
}

fn set_replay_u32(recorder: &CommandRecorder<'_>, index: usize, value: ReplayU32, max_value: u32, name: &str) {
    match value {
        ReplayU32::Fixed(value) => {
            assert!(value > 0, "{name} must be positive");
            assert!(value <= max_value, "{name} exceeds recorded capacity");
            recorder.set_u32(index, value);
        },
        ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, max_value),
    }
}

fn validate_buffers(constants: ComputeConstants, shape: Shape, buffers: &Buffers<'_>) {
    let model = constants.model;
    let bf16_bytes = size_of::<u16>() as u64;
    for (name, offset_bytes) in [
        ("conv_state", buffers.conv_state_offset_bytes),
        ("next_conv_state", buffers.next_conv_state_offset_bytes),
        ("recurrent_state", buffers.recurrent_state_arena_offset_bytes),
    ] {
        assert_eq!(
            offset_bytes % bf16_bytes,
            0,
            "GDN {name} byte offset must be BF16-aligned"
        );
    }
    assert!(
        buffers.qkv.len_bytes() >= model.num_qkv_values(shape) * size_of::<u16>(),
        "GDN qkv buffer is too small"
    );
    assert!(
        buffers.a.len_bytes() >= shape.num_total_tokens as usize * model.num_v_heads as usize * size_of::<u16>(),
        "GDN a buffer is too small"
    );
    assert!(
        buffers.b.len_bytes() >= shape.num_total_tokens as usize * model.num_v_heads as usize * size_of::<u16>(),
        "GDN b buffer is too small"
    );
    assert!(
        buffers.z.len_bytes() >= model.num_recurrent_output_values(shape) * size_of::<u16>(),
        "GDN z buffer is too small"
    );
    assert_eq!(
        buffers.conv_weight.len_bytes(),
        model.num_conv_weight_values() * size_of::<u16>()
    );
    assert_eq!(
        buffers.norm_weight.len_bytes(),
        model.v_head_dim as usize * size_of::<u16>()
    );
    assert_eq!(buffers.a_log.len_bytes(), model.num_v_heads as usize * size_of::<u16>());
    assert_eq!(
        buffers.dt_bias.len_bytes(),
        model.num_v_heads as usize * size_of::<u16>()
    );
    assert!(
        buffers.cu_tokens.len_bytes_u64() >= (shape.num_total_reqs as u64 + 1) * size_of::<u32>() as u64,
        "GDN cu_tokens buffer is too small"
    );
    assert!(
        buffers.src_recurrent_state_slots.len_bytes() >= shape.num_total_reqs as usize * size_of::<u32>(),
        "GDN src_recurrent_state_slots buffer is too small"
    );
    assert!(
        buffers.src_conv_state_slots.len_bytes() >= shape.num_total_reqs as usize * size_of::<u32>(),
        "GDN src_conv_state_slots buffer is too small"
    );
    assert!(
        buffers.flat_recurrent_state_write_slots.len_bytes() >= shape.num_total_tokens as usize * size_of::<u32>(),
        "GDN flat_recurrent_state_write_slots buffer is too small"
    );
    assert!(
        buffers.flat_conv_state_write_slots.len_bytes() >= shape.num_total_tokens as usize * size_of::<u32>(),
        "GDN flat_conv_state_write_slots buffer is too small"
    );
    let conv_state_region_bytes = model.num_conv_state_values(shape) as u64 * bf16_bytes;
    let recurrent_state_region_bytes = model.recurrent_state_stride() as u64 * bf16_bytes;
    assert!(
        buffers.conv_state.len_bytes_u64()
            >= buffers
                .conv_state_offset_bytes
                .checked_add(conv_state_region_bytes)
                .expect("GDN conv_state region size overflow"),
        "GDN conv_state buffer is too small"
    );
    assert!(
        buffers.next_conv_state.len_bytes_u64()
            >= buffers
                .next_conv_state_offset_bytes
                .checked_add(conv_state_region_bytes)
                .expect("GDN next_conv_state region size overflow"),
        "GDN next_conv_state buffer is too small"
    );
    assert!(
        buffers.recurrent_state_arena.len_bytes_u64()
            >= buffers
                .recurrent_state_arena_offset_bytes
                .checked_add(recurrent_state_region_bytes)
                .expect("GDN recurrent state region size overflow")
    );
    assert!(
        buffers.conv_qkv.len_bytes() >= model.num_qkv_values(shape) * size_of::<u16>(),
        "GDN conv_qkv buffer is too small"
    );
    assert!(
        buffers.recurrent_output.len_bytes() >= model.num_recurrent_output_values(shape) * size_of::<u16>(),
        "GDN recurrent_output buffer is too small"
    );
    assert!(
        buffers.norm_gated_output.len_bytes() >= model.num_recurrent_output_values(shape) * size_of::<u16>(),
        "GDN norm_gated_output buffer is too small"
    );
}

#[cfg(test)]
#[path = "compute_test.rs"]
mod tests;
