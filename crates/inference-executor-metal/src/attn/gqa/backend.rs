use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gqa::activation_gate as backend_activation_gate;
use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::gqa::qgkv_split as backend_qgkv_split;
use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::gqa::sdpa::ExecutionVariant;
use inference_backend_metal::components::gqa::sdpa::sliced_kv as backend_sliced_kv;
use inference_backend_metal::components::gqa::sdpa::sliced_q as backend_sliced_q;
use inference_backend_metal::components::rms_norm_rope;
use inference_backend_metal::components::rms_norm_rope::RopeScaling;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::GQACore;

use super::gqa_sdpa_config;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;
use crate::attn::gqa::batch_metadata::GQAReplayShape;
use crate::attn::gqa::request_page_table::GQAPageTableLayout;
use crate::attn::gqa::scratch::GQAScratch;
use crate::attn::gqa::scratch::GQAScratchBindings;
use crate::attn::gqa::sdpa::RequestShape;
use crate::attn::gqa::sdpa::Selector;
use crate::def::layer::ReplayLayer;

pub const GQA_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("gqa.num_active_tokens");
pub const GQA_NUM_ACTIVE_Q_TOKEN_TILES: ReplayParameterKey = ReplayParameterKey::new("gqa.num_active_q_token_tiles");
pub const GQA_NUM_ACTIVE_KV_SPLITS: ReplayParameterKey = ReplayParameterKey::new("gqa.num_active_kv_splits");

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GQAReplayTopology {
    pub sdpa_execution: ExecutionVariant,
    pub sdpa_direct_output: bool,
    pub qgkv_affine: affine_quantized::KernelKind,
    pub output_affine: affine_quantized::KernelKind,
}

pub fn add_gqa_replay_arguments(shape: GQAReplayShape, topology: GQAReplayTopology, arguments: &mut ReplayArguments) {
    add_gqa_private_replay_arguments(shape, topology, arguments);
    arguments.set_u32(GQA_NUM_ACTIVE_TOKENS, shape.num_tokens);
}

pub fn add_gqa_private_replay_arguments(
    shape: GQAReplayShape,
    topology: GQAReplayTopology,
    arguments: &mut ReplayArguments,
) {
    shape.validate();
    if topology.sdpa_execution.map.kind != backend_sdpa::MapKernelKind::SlicedKv {
        arguments.set_u32(GQA_NUM_ACTIVE_Q_TOKEN_TILES, shape.num_q_token_tiles);
    }
    if !topology.sdpa_direct_output {
        arguments.set_u32(GQA_NUM_ACTIVE_KV_SPLITS, shape.num_sdpa_map_task_templates);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GQAMetalConfig {
    pub group_size: u32,
    pub bits: u32,
    pub page_bytes: u32,
    pub rope_dim: u32,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
    pub io_dtype: Dtype,
}

impl GQAMetalConfig {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(self.page_bytes > 0);
        assert!(self.rope_dim > 0);
        assert_eq!(self.rope_dim % 2, 0);
        assert!(self.norm_eps.is_finite() && self.norm_eps > 0.0);
        assert!(self.rope_theta.is_finite() && self.rope_theta > 0.0);
        self.rope_scaling.validate();
        assert!(
            !matches!(self.rope_scaling, RopeScaling::Yarn { .. }) || self.rope_theta > 1.0,
            "Yarn rope_theta must be greater than 1"
        );
        match self.io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 GQA model boundary is not supported"),
            dtype => panic!("unsupported GQA model boundary dtype {dtype:?}"),
        }
    }

    pub fn num_tokens_per_page(self, core: &GQACore) -> u32 {
        gqa_sdpa_config(self, core.num_q_heads, core.num_kv_heads, core.head_dim).tokens_per_page
    }
}

#[derive(Clone, Copy)]
pub struct GQAKVCacheBindings<'a> {
    pub kv_pages: &'a Buffer,
    pub page_ids: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct GQAWeights<'a> {
    pub qgkv_weight: &'a Buffer,
    pub qgkv_scales: &'a Buffer,
    pub qgkv_biases: &'a Buffer,
    pub q_norm_weight: &'a Buffer,
    pub k_norm_weight: &'a Buffer,
    pub output_weight: &'a Buffer,
    pub output_scales: &'a Buffer,
    pub output_biases: &'a Buffer,
}

/// Borrowed bindings for one GQA replay recording. The replay shape belongs to
/// `batch_metadata` and is read from it during recording.
#[derive(Clone, Copy)]
pub struct GQAInput<'a> {
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: ReplayU32,
    pub batch_metadata: &'a GQAMetadataBuffers,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub kv_cache: GQAKVCacheBindings<'a>,
    pub weights: GQAWeights<'a>,
    pub scratch: GQAScratchBindings<'a>,
    pub num_active_tokens: ReplayU32,
}

/// The caller-owned next-hidden-state buffer returned by one GQA recording.
pub type GQAOutput<'a> = &'a Buffer;

/// Gated GQA data flow:
///
/// ```text
/// hidden_state
///   -> qgkv
///   -> scratch.qgkv
///   -> qgkv_to_q_g_k_v
///      |- scratch.q -> q_norm_rope -> scratch.q_norm_rope -----------+
///      |- scratch.g -------------------------------------------------|
///      |- scratch.k -> k_norm_rope -> scratch.k_norm_rope --+        |
///      `- scratch.v -----------------------------------------+        |
///                                                           |        |
///                                                           v        |
///                                                    kv_page_write    |
///                                                           |        |
///                                                           v        |
///                                                   kv_cache.kv_pages |
///                                                           |        |
///                                                           +--------+
///                                                                    |
///                                                                    v
///                              backend_sliced_kv::Compute or backend_sliced_q::Compute
///                                -> scratch.sdpa_partial_exp_sums
///                                -> scratch.sdpa_partial_max_logits
///                                -> scratch.sdpa_partial_output
///                                -> scratch.attention_output
///                                -> gate with scratch.g
///                                -> scratch.gated_attention_output
///                                -> output
///                                -> next_hidden_state
/// ```
pub struct GQA {
    device: Device,
    core: GQACore,
    config: GQAMetalConfig,
    sdpa_selector: Selector,
    qgkv: affine_quantized::Matmul,
    qgkv_to_q_g_k_v: backend_qgkv_split::Compute,
    q_norm_rope: rms_norm_rope::Compute,
    k_norm_rope: rms_norm_rope::Compute,
    kv_page_write: backend_kv_page_write::Compute,
    gate: backend_activation_gate::Compute,
    output: affine_quantized::Matmul,
}

impl GQA {
    fn validate_input(&self, input: &GQAInput<'_>) {
        let shape = input.batch_metadata.replay_shape();
        shape.validate();
        match input.num_active_tokens {
            ReplayU32::Fixed(num_active_tokens) => {
                assert_eq!(num_active_tokens, shape.num_tokens);
                assert_eq!(shape.num_tokens, shape.num_total_tokens);
                assert_eq!(shape.num_q_token_tiles, shape.num_total_q_token_tiles);
            },
            ReplayU32::Parameter(key) => {
                assert_ne!(
                    key, GQA_NUM_ACTIVE_Q_TOKEN_TILES,
                    "GQA active-token key must differ from the private Q-token-range key"
                );
                assert_ne!(
                    key, GQA_NUM_ACTIVE_KV_SPLITS,
                    "GQA active-token key must differ from the private SDPA map TaskTemplate key"
                );
                self.validate_token_capacity_topology(shape.num_tokens, shape.num_total_tokens);
            },
        }
        input.page_table_layout.validate();
        if let ReplayU32::Fixed(index) = input.gqa_layer_index {
            assert!(index < input.page_table_layout.num_gqa_layers);
        }
    }

    pub fn new(
        device: &Device,
        core: GQACore,
        config: GQAMetalConfig,
        max_tokens: usize,
        page_table_layout: GQAPageTableLayout,
    ) -> Self {
        core.validate();
        validate_config_for_core(&core, config);
        let qgkv = core.qgkv_shape();
        let output = core.output_shape();
        let sdpa_config = gqa_sdpa_config(config, core.num_q_heads, core.num_kv_heads, core.head_dim);
        let sdpa_selector = Selector::for_lookup(sdpa_config, max_tokens, page_table_layout);
        Self {
            device: device.clone(),
            core: core.clone(),
            config,
            sdpa_selector,
            qgkv: affine_quantized::Matmul::new(device, affine_config(qgkv.out_dim, qgkv.in_dim, config)),
            qgkv_to_q_g_k_v: backend_qgkv_split::Compute::new(device, qgkv_to_q_g_k_v_config(&core, config)),
            q_norm_rope: rms_norm_rope::Compute::new(device, norm_rope_config(&core, config, core.num_q_heads)),
            k_norm_rope: rms_norm_rope::Compute::new(device, norm_rope_config(&core, config, core.num_kv_heads)),
            kv_page_write: backend_kv_page_write::Compute::new(device, kv_page_write_config(&core, config)),
            gate: backend_activation_gate::Compute::new(device, gate_config(&core, config)),
            output: affine_quantized::Matmul::new(device, affine_config(output.out_dim, output.in_dim, config)),
        }
    }

    pub fn num_tokens_per_page(&self) -> u32 {
        self.config.num_tokens_per_page(&self.core)
    }

    pub fn new_scratch(&self) -> GQAScratch {
        GQAScratch::new(&self.device, &self.core, self.config, &self.sdpa_selector)
    }

    pub fn new_metadata(&self) -> GQAMetadataBuffers {
        let limits = self.sdpa_selector.limits();
        GQAMetadataBuffers::new(
            &self.device,
            limits.max_tokens as usize,
            limits.max_map_task_templates as usize,
        )
    }

    pub fn prepare(
        &self,
        batch_metadata: &GQAMetadataBuffers,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
        policy: &GQAReplayBucketPolicy,
        num_total_tokens: u32,
    ) -> GQAReplayShape {
        let num_tokens = cu_tokens.last().copied().unwrap_or_default();
        assert!(num_tokens > 0, "GQA replay requires active tokens");
        assert!(
            num_tokens <= num_total_tokens,
            "GQA active token count must not exceed the total token count"
        );
        assert!(
            num_total_tokens <= policy.max_tokens(),
            "GQA total token count must not exceed the metadata capacity"
        );
        self.validate_token_capacity_topology(num_tokens, num_total_tokens);
        assert_eq!(
            batch_metadata.max_tokens(),
            self.sdpa_selector.limits().max_tokens as usize
        );
        assert_eq!(
            batch_metadata.max_map_task_templates(),
            self.sdpa_selector.limits().max_map_task_templates as usize
        );
        let request_shapes = RequestShape::from_batch(token_indices, cu_tokens);
        let selection = self.sdpa_selector.select(&request_shapes, policy, num_total_tokens);
        batch_metadata.update(req_slots, token_indices, cu_tokens, &selection)
    }

    pub fn replay_bucket_policy(&self) -> GQAReplayBucketPolicy {
        GQAReplayBucketPolicy::new(
            self.sdpa_selector.limits().max_tokens,
            self.sdpa_selector.limits().max_map_task_templates,
        )
    }

    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        let mut boundaries = self.qgkv.topology_boundaries().into_vec();
        boundaries.extend(self.output.topology_boundaries());
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.into_boxed_slice()
    }

    fn validate_token_capacity_topology(&self, num_tokens: u32, num_total_tokens: u32) {
        assert_eq!(
            self.qgkv.topology(num_tokens),
            self.qgkv.topology(num_total_tokens),
            "GQA caller-owned token capacity must preserve the active QGKV affine topology"
        );
        assert_eq!(
            self.output.topology(num_tokens),
            self.output.topology(num_total_tokens),
            "GQA caller-owned token capacity must preserve the active output affine topology"
        );
    }

    pub fn replay_topology(&self, batch_metadata: &GQAMetadataBuffers) -> GQAReplayTopology {
        let shape = batch_metadata.replay_shape();
        shape.validate();
        GQAReplayTopology {
            sdpa_execution: batch_metadata.variant(),
            sdpa_direct_output: batch_metadata.direct_output(),
            qgkv_affine: self.qgkv.topology(shape.num_total_tokens),
            output_affine: self.output.topology(shape.num_total_tokens),
        }
    }
}

impl ReplayLayer for GQA {
    type Input<'a> = GQAInput<'a>;
    type Output<'a> = GQAOutput<'a>;

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: Self::Input<'a>) -> Self::Output<'a> {
        self.validate_input(&input);
        let shape = input.batch_metadata.replay_shape();
        let page_table_layout = input.page_table_layout;
        let page_table_index = input.gqa_layer_index;
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let kv_cache = input.kv_cache;
        let weights = input.weights;
        let batch_metadata = input.batch_metadata;
        let scratch = input.scratch;
        let active_tokens = input.num_active_tokens;
        let qgkv = self.qgkv.invoke(
            shape.num_total_tokens,
            active_tokens,
            scratch.qgkv,
            0,
            hidden_state,
            0,
            weights.qgkv_weight,
            0,
            weights.qgkv_scales,
            0,
            weights.qgkv_biases,
            0,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(qgkv));
        let qgkv_to_q_g_k_v_shape = self.qgkv_to_q_g_k_v_shape(shape);
        let qgkv_to_q_g_k_v_buffers = backend_qgkv_split::Buffers {
            qgkv: scratch.qgkv,
            q: scratch.q,
            g: scratch.g,
            k: scratch.k,
            v: scratch.v,
        };
        let qgkv_to_q_g_k_v =
            self.qgkv_to_q_g_k_v
                .invoke(qgkv_to_q_g_k_v_shape, qgkv_to_q_g_k_v_buffers, active_tokens);
        recorder.record_with_barrier_before(ReplayOp::opaque(qgkv_to_q_g_k_v));
        let q_norm_rope_shape = self.norm_rope_shape(shape);
        let q_norm_rope_buffers = rms_norm_rope::Buffers {
            input: scratch.q,
            norm_weight: weights.q_norm_weight,
            flat_token_indices: batch_metadata.flat_token_indices(),
            output: scratch.q_norm_rope,
        };
        let q_norm_rope = self
            .q_norm_rope
            .invoke(q_norm_rope_shape, q_norm_rope_buffers, active_tokens);
        recorder.record_with_barrier_before(ReplayOp::opaque(q_norm_rope));
        let k_norm_rope_shape = self.norm_rope_shape(shape);
        let k_norm_rope_buffers = rms_norm_rope::Buffers {
            input: scratch.k,
            norm_weight: weights.k_norm_weight,
            flat_token_indices: batch_metadata.flat_token_indices(),
            output: scratch.k_norm_rope,
        };
        let k_norm_rope = self
            .k_norm_rope
            .invoke(k_norm_rope_shape, k_norm_rope_buffers, active_tokens);
        recorder.record(ReplayOp::opaque(k_norm_rope));
        let kv_page_write_shape = self.kv_page_write_shape(shape, page_table_layout);
        let kv_page_write_buffers = backend_kv_page_write::Buffers {
            pages: kv_cache.kv_pages,
            flat_k: scratch.k_norm_rope,
            flat_v: scratch.v,
            req_slots: batch_metadata.req_slots(),
            flat_token_indices: batch_metadata.flat_token_indices(),
            page_ids: kv_cache.page_ids,
        };
        let kv_page_write = self.kv_page_write.invoke(
            kv_page_write_shape,
            kv_page_write_buffers,
            active_tokens,
            page_table_index,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(kv_page_write));
        let attention_output = self.record_sdpa(recorder, input);
        let gate_shape = self.gate_shape(shape);
        let gate_buffers = backend_activation_gate::Buffers {
            attention_output,
            g: scratch.g,
            output: scratch.gated_attention_output,
        };
        let gate = self.gate.invoke(gate_shape, gate_buffers, active_tokens);
        recorder.record_with_barrier_before(ReplayOp::opaque(gate));
        let output = self.output.invoke(
            shape.num_total_tokens,
            active_tokens,
            next_hidden_state,
            0,
            scratch.gated_attention_output,
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

impl GQA {
    fn record_sdpa<'a>(&'a self, recorder: &mut ReplayRecorder, input: GQAInput<'a>) -> &'a Buffer {
        let shape = input.batch_metadata.replay_shape();
        let page_table_layout = input.page_table_layout;
        let page_table_index = input.gqa_layer_index;
        let batch_metadata = input.batch_metadata;
        let kv_cache = input.kv_cache;
        let scratch = input.scratch;
        let active_tokens = input.num_active_tokens;
        let has_active_parameter = matches!(active_tokens, ReplayU32::Parameter(_));
        let active_q_token_tiles = if has_active_parameter {
            ReplayU32::Parameter(GQA_NUM_ACTIVE_Q_TOKEN_TILES)
        } else {
            ReplayU32::Fixed(shape.num_q_token_tiles)
        };
        let active_kv_splits = if has_active_parameter {
            ReplayU32::Parameter(GQA_NUM_ACTIVE_KV_SPLITS)
        } else {
            ReplayU32::Fixed(shape.num_sdpa_map_task_templates)
        };
        let execution = batch_metadata.variant();
        if execution.map.kind == backend_sdpa::MapKernelKind::SlicedKv {
            let sdpa_config = self.sdpa_sliced_kv_config(page_table_layout);
            let sdpa =
                backend_sliced_kv::Compute::new(&self.device, sdpa_config, execution, self.sdpa_sliced_kv_shape(shape));
            let map_buffers = backend_sliced_kv::MapBuffers {
                q: scratch.q_norm_rope,
                kv_pages: kv_cache.kv_pages,
                req_slots: batch_metadata.req_slots(),
                page_ids: kv_cache.page_ids,
                sdpa_map_task_templates: batch_metadata.sdpa_map_task_templates(),
                partial_exp_sums: scratch.sdpa_partial_exp_sums,
                partial_max_logits: scratch.sdpa_partial_max_logits,
                partial_output: scratch.sdpa_partial_output,
            };
            let map = sdpa.invoke_map(map_buffers, page_table_index, active_tokens, active_kv_splits);
            recorder.record_with_barrier_before(ReplayOp::opaque(map));
            let reduce_buffers = backend_sliced_kv::ReduceBuffers {
                partial_exp_sums: scratch.sdpa_partial_exp_sums,
                partial_max_logits: scratch.sdpa_partial_max_logits,
                partial_output: scratch.sdpa_partial_output,
                cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                output: scratch.attention_output,
            };
            let reduce = sdpa.invoke_reduce(reduce_buffers, active_tokens);
            recorder.record_with_barrier_before(ReplayOp::opaque(reduce));
        } else if batch_metadata.direct_output() {
            let sdpa = backend_sliced_q::Compute::new_direct(
                &self.device,
                self.sdpa_sliced_q_config(page_table_layout),
                execution,
                self.sdpa_sliced_q_shape(shape),
            );
            let direct = sdpa.invoke_direct(
                backend_sliced_q::DirectBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: kv_cache.kv_pages,
                    req_slots: batch_metadata.req_slots(),
                    page_ids: kv_cache.page_ids,
                    visible_kv_token_ranges: batch_metadata.visible_kv_token_ranges(),
                    q_token_ranges: batch_metadata.q_token_ranges(),
                    sdpa_map_task_templates: batch_metadata.sdpa_map_task_templates(),
                    output: scratch.attention_output,
                },
                page_table_index,
                active_tokens,
                active_q_token_tiles,
            );
            recorder.record_with_barrier_before(ReplayOp::opaque(direct));
        } else {
            let sdpa_config = self.sdpa_sliced_q_config(page_table_layout);
            let sdpa =
                backend_sliced_q::Compute::new(&self.device, sdpa_config, execution, self.sdpa_sliced_q_shape(shape));
            let map_buffers = backend_sliced_q::MapBuffers {
                q: scratch.q_norm_rope,
                kv_pages: kv_cache.kv_pages,
                req_slots: batch_metadata.req_slots(),
                page_ids: kv_cache.page_ids,
                visible_kv_token_ranges: batch_metadata.visible_kv_token_ranges(),
                q_token_ranges: batch_metadata.q_token_ranges(),
                sdpa_map_task_templates: batch_metadata.sdpa_map_task_templates(),
                partial_output: scratch.sdpa_partial_output,
                partial_exp_sums: scratch.sdpa_partial_exp_sums,
                partial_max_logits: scratch.sdpa_partial_max_logits,
            };
            let map = sdpa.invoke_map(
                map_buffers,
                page_table_index,
                active_tokens,
                active_q_token_tiles,
                active_kv_splits,
            );
            recorder.record_with_barrier_before(ReplayOp::opaque(map));
            let reduce_buffers = backend_sliced_q::ReduceBuffers {
                partial_output: scratch.sdpa_partial_output,
                partial_exp_sums: scratch.sdpa_partial_exp_sums,
                partial_max_logits: scratch.sdpa_partial_max_logits,
                q_token_ranges: batch_metadata.q_token_ranges(),
                cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                output: scratch.attention_output,
            };
            let reduce = sdpa.invoke_reduce(reduce_buffers, active_q_token_tiles);
            recorder.record_with_barrier_before(ReplayOp::opaque(reduce));
        }
        scratch.attention_output
    }

    fn qgkv_to_q_g_k_v_shape(&self, shape: GQAReplayShape) -> backend_qgkv_split::Shape {
        backend_qgkv_split::Shape {
            num_total_tokens: shape.num_total_tokens,
        }
    }

    fn norm_rope_shape(&self, shape: GQAReplayShape) -> rms_norm_rope::Shape {
        rms_norm_rope::Shape {
            num_total_tokens: shape.num_total_tokens,
        }
    }

    fn kv_page_write_shape(
        &self,
        shape: GQAReplayShape,
        page_table_layout: GQAPageTableLayout,
    ) -> backend_kv_page_write::Shape {
        backend_kv_page_write::Shape {
            num_total_token_writes: shape.num_total_tokens,
            page_table_layout: backend_page_table_layout(page_table_layout),
        }
    }

    fn sdpa_sliced_kv_config(&self, page_table_layout: GQAPageTableLayout) -> backend_sliced_kv::Config {
        debug_assert!(u32::try_from(self.core.num_q_heads).is_ok());
        debug_assert!(u32::try_from(self.core.num_kv_heads).is_ok());
        debug_assert!(u32::try_from(self.core.head_dim).is_ok());
        backend_sliced_kv::Config {
            num_q_heads: self.core.num_q_heads as u32,
            num_kv_heads: self.core.num_kv_heads as u32,
            head_dim: self.core.head_dim as u32,
            scale: self.core.scale,
            page_bytes: self.config.page_bytes,
            page_table_layout: backend_page_table_layout(page_table_layout),
            dtype: self.config.io_dtype,
        }
    }

    fn sdpa_sliced_kv_shape(&self, shape: GQAReplayShape) -> backend_sliced_kv::Shape {
        backend_sliced_kv::Shape {
            num_total_tokens: shape.num_total_tokens,
            num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
        }
    }

    fn sdpa_sliced_q_config(&self, page_table_layout: GQAPageTableLayout) -> backend_sliced_q::Config {
        debug_assert!(u32::try_from(self.core.num_q_heads).is_ok());
        debug_assert!(u32::try_from(self.core.num_kv_heads).is_ok());
        debug_assert!(u32::try_from(self.core.head_dim).is_ok());
        backend_sliced_q::Config {
            num_q_heads: self.core.num_q_heads as u32,
            num_kv_heads: self.core.num_kv_heads as u32,
            head_dim: self.core.head_dim as u32,
            scale: self.core.scale,
            page_bytes: self.config.page_bytes,
            dtype: self.config.io_dtype,
            page_table_layout: backend_page_table_layout(page_table_layout),
        }
    }

    fn sdpa_sliced_q_shape(&self, shape: GQAReplayShape) -> backend_sliced_q::Shape {
        backend_sliced_q::Shape {
            num_total_tokens: shape.num_total_tokens,
            num_total_q_token_tiles: shape.num_total_q_token_tiles,
            num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
        }
    }

    fn gate_shape(&self, shape: GQAReplayShape) -> backend_activation_gate::Shape {
        backend_activation_gate::Shape {
            num_total_tokens: shape.num_total_tokens,
        }
    }
}

fn backend_page_table_layout(shape: GQAPageTableLayout) -> backend_kv_page_write::PageTableLayout {
    backend_kv_page_write::PageTableLayout {
        num_req_slots: shape.num_req_slots,
        num_blocks: shape.num_blocks,
        num_gqa_layers: shape.num_gqa_layers,
        num_page_ids_per_block: shape.num_page_ids_per_block,
    }
}

fn qgkv_to_q_g_k_v_config(core: &GQACore, config: GQAMetalConfig) -> backend_qgkv_split::Config {
    let num_q_heads = core.num_q_heads.try_into().expect("GQA q heads must fit u32");
    let num_kv_heads = core.num_kv_heads.try_into().expect("GQA KV heads must fit u32");
    let head_dim = core.head_dim.try_into().expect("GQA head_dim must fit u32");
    match config.io_dtype {
        Dtype::Float32 => backend_qgkv_split::Config::f32(num_q_heads, num_kv_heads, head_dim),
        Dtype::Bfloat16 => backend_qgkv_split::Config::bf16(num_q_heads, num_kv_heads, head_dim),
        dtype => panic!("unsupported GQA dtype {dtype:?}"),
    }
}

fn norm_rope_config(core: &GQACore, config: GQAMetalConfig, num_heads: usize) -> rms_norm_rope::Config {
    let num_heads_u32 = num_heads.try_into().expect("GQA head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("GQA head_dim must fit u32");
    let norm_rope = match config.io_dtype {
        Dtype::Float32 => {
            rms_norm_rope::Config::f32(
                num_heads_u32,
                head_dim,
                config.rope_dim,
                config.norm_eps,
                config.rope_theta,
            )
        },
        Dtype::Bfloat16 => {
            rms_norm_rope::Config::bf16(
                num_heads_u32,
                head_dim,
                config.rope_dim,
                config.norm_eps,
                config.rope_theta,
            )
        },
        dtype => panic!("unsupported GQA dtype {dtype:?}"),
    };
    norm_rope.with_rope_scaling(config.rope_scaling)
}

fn kv_page_write_config(core: &GQACore, config: GQAMetalConfig) -> backend_kv_page_write::Config {
    backend_kv_page_write::Config {
        num_kv_heads: core.num_kv_heads.try_into().expect("GQA KV heads must fit u32"),
        head_dim: core.head_dim.try_into().expect("GQA head_dim must fit u32"),
        page_bytes: config.page_bytes,
        dtype: config.io_dtype,
    }
}

fn gate_config(core: &GQACore, config: GQAMetalConfig) -> backend_activation_gate::Config {
    let num_q_heads = core.num_q_heads.try_into().expect("GQA q heads must fit u32");
    let head_dim = core.head_dim.try_into().expect("GQA head_dim must fit u32");
    match config.io_dtype {
        Dtype::Float32 => backend_activation_gate::Config::f32(num_q_heads, head_dim),
        Dtype::Bfloat16 => backend_activation_gate::Config::bf16(num_q_heads, head_dim),
        dtype => panic!("unsupported GQA dtype {dtype:?}"),
    }
}

fn validate_config_for_core(core: &GQACore, config: GQAMetalConfig) {
    config.validate();
    assert!(config.rope_dim as usize <= core.head_dim);
    assert!(config.num_tokens_per_page(core) > 0);
}

fn affine_config(n: usize, k: usize, config: GQAMetalConfig) -> affine_quantized::Config {
    affine_quantized::Config {
        n: n.try_into().expect("GQA affine n must fit i32"),
        k: k.try_into().expect("GQA affine k must fit i32"),
        group_size: config.group_size.try_into().expect("GQA group_size must fit i32"),
        bits: config.bits.try_into().expect("GQA bits must fit i32"),
        input_dtype: config.io_dtype,
        output_dtype: config.io_dtype,
        scale_bias_dtype: config.io_dtype,
    }
}

#[cfg(test)]
#[path = "backend_full_test.rs"]
mod full_tests;

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::rms_norm_rope::RopeScaling;
    use inference_backend_metal::metal::Dtype;
    use inference_executor_core::attn::GQACore;

    use super::GQAMetalConfig;

    #[test]
    #[should_panic(expected = "GQA Q-head count must fit u32")]
    fn test_tokens_per_page_rejects_head_count_outside_shader_domain() {
        let num_heads = usize::MAX / 6;
        let core = GQACore::new(0, 1, 1, num_heads, num_heads, 1.0);
        GQAMetalConfig {
            group_size: 64,
            bits: 4,
            page_bytes: 65536,
            rope_dim: 64,
            norm_eps: 1.0e-6,
            rope_theta: 10_000_000.0,
            rope_scaling: RopeScaling::Default,
            io_dtype: Dtype::Bfloat16,
        }
        .num_tokens_per_page(&core);
    }
}
