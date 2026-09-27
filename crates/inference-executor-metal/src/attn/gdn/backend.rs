use inference_backend_metal::components::gdn::compute as backend_compute;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GDNReplayShape;
use inference_executor_core::attn::gdn::GDNOutputGate;
use inference_executor_core::backend::recorder::Recorder;

use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::batch_metadata::GDNReplayBucketPolicy;
use crate::attn::gdn::scratch::GDNScratch;
use crate::attn::gdn::scratch::GDNScratchBindings;
use crate::attn::gdn::state_table::GDNPreparedRequestState;
use crate::def::layer::ReplayLayer;
use crate::def::replay_op::ReplayOp;

pub const GDN_NUM_ACTIVE_REQUESTS: ReplayParameterKey = ReplayParameterKey::new("gdn.num_active_requests");
pub const GDN_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("gdn.num_active_tokens");
pub const GDN_NUM_ACTIVE_CHUNKWISE_REQUESTS: ReplayParameterKey =
    ReplayParameterKey::new("gdn.num_active_chunkwise_requests");

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GDNReplayTopology {
    pub qkvabz_affine: affine_quantized::KernelKind,
    pub output_affine: affine_quantized::KernelKind,
}

pub fn add_gdn_replay_arguments(
    shape: GDNReplayShape,
    num_active_chunkwise_requests: u32,
    arguments: &mut ReplayArguments,
) {
    add_gdn_private_replay_arguments(shape, num_active_chunkwise_requests, arguments);
    arguments.set_u32(GDN_NUM_ACTIVE_TOKENS, shape.num_tokens);
}

pub fn add_gdn_private_replay_arguments(
    shape: GDNReplayShape,
    num_active_chunkwise_requests: u32,
    arguments: &mut ReplayArguments,
) {
    shape.validate();
    debug_assert!(num_active_chunkwise_requests <= shape.num_reqs);
    arguments.set_u32(GDN_NUM_ACTIVE_REQUESTS, shape.num_reqs);
    arguments.set_u32(GDN_NUM_ACTIVE_CHUNKWISE_REQUESTS, num_active_chunkwise_requests);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GDNMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub norm_eps: f32,
    pub input_dtype: Dtype,
    pub output_dtype: Dtype,
    pub qkvabz_scale_bias_dtype: Dtype,
    pub output_scale_bias_dtype: Dtype,
}

impl GDNMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(self.norm_eps > 0.0);
        validate_boundary_dtype("input", self.input_dtype);
        validate_boundary_dtype("output", self.output_dtype);
        assert!(matches!(self.qkvabz_scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(matches!(self.output_scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
    }
}

fn validate_boundary_dtype(name: &str, dtype: Dtype) {
    match dtype {
        Dtype::Bfloat16 => {},
        Dtype::Float32 => todo!("F32 GDN {name} boundary is not supported"),
        dtype => panic!("unsupported GDN {name} boundary dtype {dtype:?}"),
    }
}

#[derive(Clone, Copy)]
pub struct GDNLayerStateBindings<'a> {
    pub conv_state: &'a Buffer,
    pub conv_state_offset_bytes: u64,
    pub next_conv_state: &'a Buffer,
    pub next_conv_state_offset_bytes: u64,
    pub recurrent_state_arena: &'a Buffer,
    pub recurrent_state_arena_offset_bytes: u64,
    pub replay: backend_compute::ReplayBuffers<'a>,
}

#[derive(Clone, Copy)]
pub struct GDNWeights<'a> {
    pub qkvabz_weight: &'a Buffer,
    pub qkvabz_scales: &'a Buffer,
    pub qkvabz_biases: &'a Buffer,
    pub conv_weight: &'a Buffer,
    pub norm_weight: &'a Buffer,
    pub a_log: &'a Buffer,
    pub dt_bias: &'a Buffer,
    pub output_weight: &'a Buffer,
    pub output_scales: &'a Buffer,
    pub output_biases: &'a Buffer,
}

/// Borrowed bindings for one GDN replay recording. The replay shape belongs to
/// `batch_metadata` and is read from it during recording.
#[derive(Clone, Copy)]
pub struct GDNInput<'a> {
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub scratch: GDNScratchBindings<'a>,
    pub batch_metadata: &'a GDNMetadataBuffers,
    pub state: GDNLayerStateBindings<'a>,
    pub weights: GDNWeights<'a>,
    pub num_active_tokens: ReplayU32,
}

/// The caller-owned next-hidden-state buffer returned by one GDN recording.
pub type GDNOutput<'a> = &'a Buffer;

/// GDN data flow:
///
/// ```text
/// hidden_state (BF16)
///   -> qkvabz
///   -> scratch.qkvabz (BF16)
///   -> QKV/A/B/Z column views (offset + row stride, no copy)
///             |
///             v
///          compute (BF16 storage, F32 arithmetic)
///      short_conv -> ragged_recurrent -> output_norm_gate
///             |
///             v
///      scratch.norm_gated_output (BF16)
///             |
///             v
///          output
///             |
///             v
///      next_hidden_state (BF16)
/// ```
pub struct GDN {
    device: Device,
    core: GDNCore,
    qkvabz: affine_quantized::Matmul,
    compute: backend_compute::Compute,
    output: affine_quantized::Matmul,
}

impl GDN {
    pub fn max_replay_tokens_per_request(num_spec_tokens: usize) -> usize {
        backend_compute::max_replay_tokens_per_request(num_spec_tokens)
    }

    pub fn uses_chunkwise(&self, num_tokens: usize, num_spec_tokens: usize) -> bool {
        backend_compute::uses_chunkwise(num_tokens, num_spec_tokens)
    }

    pub fn new(device: &Device, core: GDNCore, config: GDNMetalConfig) -> Self {
        core.validate();
        config.validate();
        let qkvabz_dim = core.qkvabz_dim();
        Self {
            device: device.clone(),
            core: core.clone(),
            qkvabz: affine_quantized::Matmul::new(
                device,
                affine_config(
                    qkvabz_dim,
                    core.hidden_dim,
                    config.input_dtype,
                    Dtype::Bfloat16,
                    config.qkvabz_scale_bias_dtype,
                    config,
                ),
            ),
            compute: backend_compute::Compute::new(device, compute_config(&core, config)),
            output: affine_quantized::Matmul::new(
                device,
                affine_config(
                    core.hidden_dim,
                    core.v_dim(),
                    Dtype::Bfloat16,
                    config.output_dtype,
                    config.output_scale_bias_dtype,
                    config,
                ),
            ),
        }
    }

    pub fn new_scratch(&self, max_tokens: usize) -> GDNScratch {
        GDNScratch::new(&self.device, &self.core, max_tokens)
    }

    pub fn prepare(
        &self,
        metadata: &GDNMetadataBuffers,
        cu_tokens: &[u32],
        num_active_chunkwise_requests: u32,
        state: &GDNPreparedRequestState,
        policy: &GDNReplayBucketPolicy,
        num_total_tokens: u32,
    ) -> GDNReplayShape {
        let num_tokens = cu_tokens.last().copied().unwrap_or_default();
        assert_eq!(metadata.max_requests(), policy.max_requests() as usize);
        assert_eq!(metadata.max_tokens(), policy.max_tokens() as usize);
        assert!(
            num_total_tokens <= policy.max_tokens(),
            "GDN total token count must not exceed the metadata capacity"
        );
        self.validate_token_capacity(num_tokens, num_total_tokens);
        let num_active_requests =
            u32::try_from(state.src_recurrent_state_slots.len()).expect("GDN active request count must fit u32");
        metadata.update(
            cu_tokens,
            num_active_chunkwise_requests,
            &state.src_recurrent_state_slots,
            &state.src_conv_state_slots,
            &state.flat_recurrent_state_write_slots,
            &state.flat_conv_state_write_slots,
            policy.num_total_requests(num_active_requests),
            num_total_tokens,
        )
    }

    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        let mut boundaries = self.qkvabz.topology_boundaries().into_vec();
        boundaries.extend(self.output.topology_boundaries());
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.into_boxed_slice()
    }

    pub fn replay_bucket_policy(&self, max_requests: u32, max_tokens: u32) -> GDNReplayBucketPolicy {
        let boundaries = self.replay_token_topology_boundaries();
        GDNReplayBucketPolicy::new(max_requests, max_tokens, &boundaries)
    }

    pub fn replay_topology(&self, batch_metadata: &GDNMetadataBuffers) -> GDNReplayTopology {
        let shape = batch_metadata.replay_shape();
        shape.validate();
        self.replay_topology_for_token_capacity(shape.num_total_tokens)
    }

    fn replay_topology_for_token_capacity(&self, num_total_tokens: u32) -> GDNReplayTopology {
        assert!(num_total_tokens > 0, "GDN replay topology requires token capacity");
        GDNReplayTopology {
            qkvabz_affine: self.qkvabz.topology(num_total_tokens),
            output_affine: self.output.topology(num_total_tokens),
        }
    }

    fn validate_token_capacity(&self, num_tokens: u32, num_total_tokens: u32) {
        assert!(num_tokens > 0, "GDN replay requires active tokens");
        assert!(
            num_tokens <= num_total_tokens,
            "GDN active token count must not exceed the total token count"
        );
        let active_topology = self.replay_topology_for_token_capacity(num_tokens);
        let selected_topology = self.replay_topology_for_token_capacity(num_total_tokens);
        assert_eq!(
            active_topology.qkvabz_affine, selected_topology.qkvabz_affine,
            "GDN total token count must preserve the QKVABZ affine topology"
        );
        assert_eq!(
            active_topology.output_affine, selected_topology.output_affine,
            "GDN total token count must preserve the output affine topology"
        );
    }
}

impl ReplayLayer for GDN {
    type Input<'a> = GDNInput<'a>;
    type Output<'a> = GDNOutput<'a>;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shape = input.batch_metadata.replay_shape();
        shape.validate();
        match input.num_active_tokens {
            ReplayU32::Fixed(num_active_tokens) => {
                assert_eq!(num_active_tokens, shape.num_tokens);
                assert_eq!(shape.num_reqs, shape.num_total_reqs);
                assert_eq!(shape.num_tokens, shape.num_total_tokens);
            },
            ReplayU32::Parameter(key) => {
                assert_ne!(
                    key, GDN_NUM_ACTIVE_REQUESTS,
                    "GDN active-token key must differ from the private active-request key"
                );
                assert_ne!(
                    key, GDN_NUM_ACTIVE_CHUNKWISE_REQUESTS,
                    "GDN active-token key must differ from the private prefill-request key"
                );
                self.validate_token_capacity(shape.num_tokens, shape.num_total_tokens);
            },
        }
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let scratch = input.scratch;
        let batch_metadata = input.batch_metadata;
        let state = input.state;
        let weights = input.weights;
        let active_reqs = if matches!(input.num_active_tokens, ReplayU32::Parameter(_)) {
            ReplayU32::Parameter(GDN_NUM_ACTIVE_REQUESTS)
        } else {
            ReplayU32::Fixed(shape.num_reqs)
        };
        let active_tokens = input.num_active_tokens;
        let active_chunkwise_requests = if matches!(active_tokens, ReplayU32::Parameter(_)) {
            ReplayU32::Parameter(GDN_NUM_ACTIVE_CHUNKWISE_REQUESTS)
        } else {
            ReplayU32::Fixed(batch_metadata.num_active_chunkwise_requests())
        };
        let qkvabz = self.qkvabz.invoke(
            shape.num_total_tokens,
            active_tokens,
            scratch.qkvabz,
            0,
            hidden_state,
            0,
            weights.qkvabz_weight,
            0,
            weights.qkvabz_scales,
            0,
            weights.qkvabz_biases,
            0,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(qkvabz));
        let compute_buffers = backend_compute::Buffers {
            qkv: scratch.qkvabz,
            a: scratch.qkvabz,
            b: scratch.qkvabz,
            z: scratch.qkvabz,
            conv_weight: weights.conv_weight,
            norm_weight: weights.norm_weight,
            a_log: weights.a_log,
            dt_bias: weights.dt_bias,
            cu_tokens: batch_metadata.cu_tokens(),
            src_recurrent_state_slots: batch_metadata.src_recurrent_state_slots(),
            src_conv_state_slots: batch_metadata.src_conv_state_slots(),
            flat_recurrent_state_write_slots: batch_metadata.flat_recurrent_state_write_slots(),
            flat_conv_state_write_slots: batch_metadata.flat_conv_state_write_slots(),
            conv_state: state.conv_state,
            conv_state_offset_bytes: state.conv_state_offset_bytes,
            next_conv_state: state.next_conv_state,
            next_conv_state_offset_bytes: state.next_conv_state_offset_bytes,
            recurrent_state_arena: state.recurrent_state_arena,
            recurrent_state_arena_offset_bytes: state.recurrent_state_arena_offset_bytes,
            conv_qkv: scratch.conv_qkv,
            recurrent_output: scratch.recurrent_output,
            norm_gated_output: scratch.norm_gated_output,
        };
        let compute_shape = compute_shape(shape);
        let compute = self.compute.invoke_with_replay(
            compute_shape,
            compute_buffers,
            state.replay,
            active_reqs,
            active_tokens,
            active_chunkwise_requests,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(compute));
        let output = self.output.invoke(
            shape.num_total_tokens,
            active_tokens,
            next_hidden_state,
            0,
            scratch.norm_gated_output,
            0,
            weights.output_weight,
            0,
            weights.output_scales,
            0,
            weights.output_biases,
            0,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(output));
        next_hidden_state
    }
}

fn compute_config(core: &GDNCore, config: GDNMetalConfig) -> backend_compute::Config {
    backend_compute::Config {
        projection_layout: backend_compute::ProjectionLayout::PackedQkvabz,
        num_qk_heads: core.num_qk_heads.try_into().expect("GDN query/key heads must fit u32"),
        qk_head_dim: core.qk_head_dim.try_into().expect("GDN qk_head_dim must fit u32"),
        num_v_heads: core.num_v_heads.try_into().expect("GDN num_v_heads must fit u32"),
        v_head_dim: core.v_head_dim.try_into().expect("GDN v_head_dim must fit u32"),
        conv_kernel_size: core
            .conv_kernel_size
            .try_into()
            .expect("GDN conv_kernel_size must fit u32"),
        q_scale: core.q_scale,
        norm_eps: config.norm_eps,
        output_gate: match core.output_gate {
            GDNOutputGate::Silu => backend_compute::OutputGate::Silu,
            GDNOutputGate::Sigmoid => backend_compute::OutputGate::Sigmoid,
        },
    }
}

fn compute_shape(shape: GDNReplayShape) -> backend_compute::Shape {
    backend_compute::Shape {
        num_total_reqs: shape.num_total_reqs,
        num_total_tokens: shape.num_total_tokens,
    }
}

fn affine_config(
    n: usize,
    k: usize,
    input_dtype: Dtype,
    output_dtype: Dtype,
    scale_bias_dtype: Dtype,
    config: GDNMetalConfig,
) -> affine_quantized::Config {
    affine_quantized::Config {
        n: n.try_into().expect("GDN affine n must fit i32"),
        k: k.try_into().expect("GDN affine k must fit i32"),
        group_size: config.group_size.try_into().expect("GDN group size must fit i32"),
        bits: config.bits.try_into().expect("GDN bits must fit i32"),
        input_dtype,
        output_dtype,
        scale_bias_dtype,
    }
}

#[cfg(test)]
#[path = "backend_full_test.rs"]
mod full_tests;
