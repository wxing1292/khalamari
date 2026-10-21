use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::gqa::qkv_split as backend_qkv_split;
use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::gqa::sdpa::sliced_kv as backend_sliced_kv;
use inference_backend_metal::components::gqa::sdpa::sliced_q as backend_sliced_q;
use inference_backend_metal::components::rms_norm_rope;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::UngatedGQACore;

use super::gqa_sdpa_config;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;
use crate::attn::gqa::batch_metadata::GQAReplayShape;
use crate::attn::gqa::request_page_table::GQAPageTableLayout;
use crate::attn::gqa::sdpa::RequestShape;
use crate::attn::gqa::sdpa::Selector;
use crate::attn::gqa::ungated_scratch::UngatedGQAScratch;
use crate::attn::gqa::ungated_scratch::UngatedGQAScratchBindings;
use crate::def::layer::ReplayLayer;

pub const UNGATED_GQA_NUM_ACTIVE_Q_TOKEN_TILES: ReplayParameterKey =
    ReplayParameterKey::new("ungated_gqa.num_active_q_token_tiles");
pub const UNGATED_GQA_NUM_ACTIVE_KV_SPLITS: ReplayParameterKey =
    ReplayParameterKey::new("ungated_gqa.num_active_kv_splits");

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UngatedGQAReplayTopology {
    pub sdpa_execution: backend_sdpa::ExecutionVariant,
    pub sdpa_direct_output: bool,
    pub qkv_affine: affine_quantized::KernelKind,
    pub output_affine: affine_quantized::KernelKind,
}

pub fn add_ungated_gqa_private_replay_arguments(
    shape: GQAReplayShape,
    topology: UngatedGQAReplayTopology,
    arguments: &mut ReplayArguments,
) {
    shape.validate();
    if topology.sdpa_execution.map.kind != backend_sdpa::MapKernelKind::SlicedKv {
        arguments.set_u32(UNGATED_GQA_NUM_ACTIVE_Q_TOKEN_TILES, shape.num_q_token_tiles);
    }
    if !topology.sdpa_direct_output {
        arguments.set_u32(UNGATED_GQA_NUM_ACTIVE_KV_SPLITS, shape.num_sdpa_map_task_templates);
    }
}

#[derive(Clone, Copy)]
pub struct UngatedGQAWeights<'a> {
    pub qkv_weight: &'a Buffer,
    pub qkv_scales: &'a Buffer,
    pub qkv_biases: &'a Buffer,
    pub q_norm_weight: &'a Buffer,
    pub k_norm_weight: &'a Buffer,
    pub output_weight: &'a Buffer,
    pub output_scales: &'a Buffer,
    pub output_biases: &'a Buffer,
}

/// Borrowed bindings for one ungated GQA replay recording. The replay shape
/// belongs to `batch_metadata` and is read from it during recording.
#[derive(Clone, Copy)]
pub struct UngatedGQAInput<'a> {
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: ReplayU32,
    pub batch_metadata: &'a GQAMetadataBuffers,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub kv_cache: GQAKVCacheBindings<'a>,
    pub weights: UngatedGQAWeights<'a>,
    pub scratch: UngatedGQAScratchBindings<'a>,
    pub num_active_tokens: ReplayU32,
}

/// The caller-owned next-hidden-state buffer returned by one ungated GQA
/// recording.
pub type UngatedGQAOutput<'a> = &'a Buffer;

/// Ungated GQA data flow:
///
/// ```text
/// hidden_state
///   -> qkv
///   -> scratch.qkv
///   -> qkv_to_q_k_v
///      |- scratch.q -> q_norm_rope -> scratch.q_norm_rope -----------+
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
///                                -> output
///                                -> next_hidden_state
/// ```
pub struct UngatedGQA {
    device: Device,
    core: UngatedGQACore,
    config: GQAMetalConfig,
    sdpa_selector: Selector,
    qkv: affine_quantized::Matmul,
    qkv_to_q_k_v: backend_qkv_split::Compute,
    q_norm_rope: rms_norm_rope::Compute,
    k_norm_rope: rms_norm_rope::Compute,
    kv_page_write: backend_kv_page_write::Compute,
    output: affine_quantized::Matmul,
}

impl GQAMetalConfig {
    pub fn num_ungated_tokens_per_page(self, core: &UngatedGQACore) -> u32 {
        gqa_sdpa_config(self, core.num_q_heads, core.num_kv_heads, core.head_dim).tokens_per_page
    }
}

impl UngatedGQA {
    fn validate_input(&self, input: &UngatedGQAInput<'_>) {
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
                    key, UNGATED_GQA_NUM_ACTIVE_Q_TOKEN_TILES,
                    "ungated GQA active-token key must differ from the private Q-token-tile key"
                );
                assert_ne!(
                    key, UNGATED_GQA_NUM_ACTIVE_KV_SPLITS,
                    "ungated GQA active-token key must differ from the private KV-split key"
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
        core: UngatedGQACore,
        config: GQAMetalConfig,
        max_tokens: usize,
        page_table_layout: GQAPageTableLayout,
    ) -> Self {
        core.validate();
        validate_config_for_core(&core, config);
        let qkv = core.qkv_shape();
        let output = core.output_shape();
        let sdpa_config = gqa_sdpa_config(config, core.num_q_heads, core.num_kv_heads, core.head_dim);
        let sdpa_selector = Selector::for_lookup(sdpa_config, max_tokens, page_table_layout);
        Self {
            device: device.clone(),
            core: core.clone(),
            config,
            sdpa_selector,
            qkv: affine_quantized::Matmul::new(device, affine_config(qkv.out_dim, qkv.in_dim, config)),
            qkv_to_q_k_v: backend_qkv_split::Compute::new(device, qkv_to_q_k_v_config(&core, config)),
            q_norm_rope: rms_norm_rope::Compute::new(device, norm_rope_config(&core, config, core.num_q_heads)),
            k_norm_rope: rms_norm_rope::Compute::new(device, norm_rope_config(&core, config, core.num_kv_heads)),
            kv_page_write: backend_kv_page_write::Compute::new(device, kv_page_write_config(&core, config)),
            output: affine_quantized::Matmul::new(device, affine_config(output.out_dim, output.in_dim, config)),
        }
    }

    pub fn num_tokens_per_page(&self) -> u32 {
        self.config.num_ungated_tokens_per_page(&self.core)
    }

    pub fn new_scratch(&self) -> UngatedGQAScratch {
        UngatedGQAScratch::new(&self.device, &self.core, self.config, &self.sdpa_selector)
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
        assert!(num_tokens > 0, "ungated GQA replay requires active tokens");
        assert!(
            num_tokens <= num_total_tokens,
            "ungated GQA active token count must not exceed the total token count"
        );
        assert!(
            num_total_tokens <= policy.max_tokens(),
            "ungated GQA total token count must not exceed the metadata capacity"
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

    fn validate_token_capacity_topology(&self, num_tokens: u32, num_total_tokens: u32) {
        assert_eq!(
            self.qkv.topology(num_tokens),
            self.qkv.topology(num_total_tokens),
            "ungated GQA caller-owned token capacity must preserve the active QKV affine topology"
        );
        assert_eq!(
            self.output.topology(num_tokens),
            self.output.topology(num_total_tokens),
            "ungated GQA caller-owned token capacity must preserve the active output affine topology"
        );
    }

    pub fn replay_topology(&self, batch_metadata: &GQAMetadataBuffers) -> UngatedGQAReplayTopology {
        let shape = batch_metadata.replay_shape();
        shape.validate();
        UngatedGQAReplayTopology {
            sdpa_execution: batch_metadata.variant(),
            sdpa_direct_output: batch_metadata.direct_output(),
            qkv_affine: self.qkv.topology(shape.num_total_tokens),
            output_affine: self.output.topology(shape.num_total_tokens),
        }
    }
}

impl ReplayLayer for UngatedGQA {
    type Input<'a> = UngatedGQAInput<'a>;
    type Output<'a> = UngatedGQAOutput<'a>;

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: Self::Input<'a>) -> Self::Output<'a> {
        self.validate_input(&input);
        let shape = input.batch_metadata.replay_shape();
        let page_table_layout = input.page_table_layout;
        let gqa_layer_index = input.gqa_layer_index;
        let hidden_state = input.hidden_state;
        let next_hidden_state = input.next_hidden_state;
        let kv_cache = input.kv_cache;
        let weights = input.weights;
        let batch_metadata = input.batch_metadata;
        let scratch = input.scratch;
        let active_tokens = input.num_active_tokens;
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qkv.invoke(
            shape.num_total_tokens,
            active_tokens,
            scratch.qkv,
            0,
            hidden_state,
            0,
            weights.qkv_weight,
            0,
            weights.qkv_scales,
            0,
            weights.qkv_biases,
            0,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.qkv_to_q_k_v.invoke(
            self.qkv_to_q_k_v_shape(shape),
            backend_qkv_split::Buffers {
                qkv: scratch.qkv,
                q: scratch.q,
                k: scratch.k,
                v: scratch.v,
            },
            active_tokens,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.q_norm_rope.invoke(
            self.norm_rope_shape(shape),
            rms_norm_rope::Buffers {
                input: scratch.q,
                norm_weight: weights.q_norm_weight,
                flat_token_indices: batch_metadata.flat_token_indices(),
                output: scratch.q_norm_rope,
            },
            active_tokens,
        )));
        recorder.record(ReplayOp::opaque(self.k_norm_rope.invoke(
            self.norm_rope_shape(shape),
            rms_norm_rope::Buffers {
                input: scratch.k,
                norm_weight: weights.k_norm_weight,
                flat_token_indices: batch_metadata.flat_token_indices(),
                output: scratch.k_norm_rope,
            },
            active_tokens,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kv_page_write.invoke(
            self.kv_page_write_shape(shape, page_table_layout),
            backend_kv_page_write::Buffers {
                pages: kv_cache.kv_pages,
                flat_k: scratch.k_norm_rope,
                flat_v: scratch.v,
                req_slots: batch_metadata.req_slots(),
                flat_token_indices: batch_metadata.flat_token_indices(),
                page_ids: kv_cache.page_ids,
            },
            active_tokens,
            gqa_layer_index,
        )));
        let attention_output = self.record_sdpa(recorder, input);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.output.invoke(
            shape.num_total_tokens,
            active_tokens,
            next_hidden_state,
            0,
            attention_output,
            0,
            weights.output_weight,
            0,
            weights.output_scales,
            0,
            weights.output_biases,
            0,
        )));
        next_hidden_state
    }
}

impl UngatedGQA {
    fn record_sdpa<'a>(&'a self, recorder: &mut ReplayRecorder, input: UngatedGQAInput<'a>) -> &'a Buffer {
        let shape = input.batch_metadata.replay_shape();
        let page_table_layout = input.page_table_layout;
        let gqa_layer_index = input.gqa_layer_index;
        let batch_metadata = input.batch_metadata;
        let kv_cache = input.kv_cache;
        let scratch = input.scratch;
        let active_tokens = input.num_active_tokens;
        let has_active_parameter = matches!(active_tokens, ReplayU32::Parameter(_));
        let active_q_token_tiles = if has_active_parameter {
            ReplayU32::Parameter(UNGATED_GQA_NUM_ACTIVE_Q_TOKEN_TILES)
        } else {
            ReplayU32::Fixed(shape.num_q_token_tiles)
        };
        let active_kv_splits = if has_active_parameter {
            ReplayU32::Parameter(UNGATED_GQA_NUM_ACTIVE_KV_SPLITS)
        } else {
            ReplayU32::Fixed(shape.num_sdpa_map_task_templates)
        };
        let execution = batch_metadata.variant();
        if execution.map.kind == backend_sdpa::MapKernelKind::SlicedKv {
            let sdpa_config = self.sdpa_sliced_kv_config(page_table_layout);
            let sdpa =
                backend_sliced_kv::Compute::new(&self.device, sdpa_config, execution, self.sdpa_sliced_kv_shape(shape));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(
                backend_sliced_kv::MapBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: kv_cache.kv_pages,
                    req_slots: batch_metadata.req_slots(),
                    page_ids: kv_cache.page_ids,
                    sdpa_map_task_templates: batch_metadata.sdpa_map_task_templates(),
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    partial_output: scratch.sdpa_partial_output,
                },
                gqa_layer_index,
                active_tokens,
                active_kv_splits,
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(
                backend_sliced_kv::ReduceBuffers {
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    partial_output: scratch.sdpa_partial_output,
                    cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                },
                active_tokens,
            )));
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
                gqa_layer_index,
                active_tokens,
                active_q_token_tiles,
            );
            recorder.record_with_barrier_before(ReplayOp::opaque(direct));
        } else {
            let sdpa_config = self.sdpa_sliced_q_config(page_table_layout);
            let sdpa =
                backend_sliced_q::Compute::new(&self.device, sdpa_config, execution, self.sdpa_sliced_q_shape(shape));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(
                backend_sliced_q::MapBuffers {
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
                },
                gqa_layer_index,
                active_tokens,
                active_q_token_tiles,
                active_kv_splits,
            )));
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(
                backend_sliced_q::ReduceBuffers {
                    partial_output: scratch.sdpa_partial_output,
                    partial_exp_sums: scratch.sdpa_partial_exp_sums,
                    partial_max_logits: scratch.sdpa_partial_max_logits,
                    q_token_ranges: batch_metadata.q_token_ranges(),
                    cu_sdpa_partial_outputs: batch_metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                },
                active_q_token_tiles,
            )));
        }
        scratch.attention_output
    }

    fn qkv_to_q_k_v_shape(&self, shape: GQAReplayShape) -> backend_qkv_split::Shape {
        backend_qkv_split::Shape {
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
        backend_sliced_kv::Config {
            num_q_heads: self
                .core
                .num_q_heads
                .try_into()
                .expect("ungated GQA q heads must fit u32"),
            num_kv_heads: self
                .core
                .num_kv_heads
                .try_into()
                .expect("ungated GQA KV heads must fit u32"),
            head_dim: self
                .core
                .head_dim
                .try_into()
                .expect("ungated GQA head_dim must fit u32"),
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
        backend_sliced_q::Config {
            num_q_heads: self
                .core
                .num_q_heads
                .try_into()
                .expect("ungated GQA q heads must fit u32"),
            num_kv_heads: self
                .core
                .num_kv_heads
                .try_into()
                .expect("ungated GQA KV heads must fit u32"),
            head_dim: self
                .core
                .head_dim
                .try_into()
                .expect("ungated GQA head_dim must fit u32"),
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
}

fn backend_page_table_layout(shape: GQAPageTableLayout) -> backend_kv_page_write::PageTableLayout {
    backend_kv_page_write::PageTableLayout {
        num_req_slots: shape.num_req_slots,
        num_blocks: shape.num_blocks,
        num_gqa_layers: shape.num_gqa_layers,
        num_page_ids_per_block: shape.num_page_ids_per_block,
    }
}

fn qkv_to_q_k_v_config(core: &UngatedGQACore, config: GQAMetalConfig) -> backend_qkv_split::Config {
    let num_q_heads = core.num_q_heads.try_into().expect("ungated GQA q heads must fit u32");
    let num_kv_heads = core.num_kv_heads.try_into().expect("ungated GQA KV heads must fit u32");
    let head_dim = core.head_dim.try_into().expect("ungated GQA head_dim must fit u32");
    match config.io_dtype {
        Dtype::Float32 => backend_qkv_split::Config::f32(num_q_heads, num_kv_heads, head_dim),
        Dtype::Bfloat16 => backend_qkv_split::Config::bf16(num_q_heads, num_kv_heads, head_dim),
        dtype => panic!("unsupported ungated GQA dtype {dtype:?}"),
    }
}

fn norm_rope_config(core: &UngatedGQACore, config: GQAMetalConfig, num_heads: usize) -> rms_norm_rope::Config {
    let num_heads_u32 = num_heads.try_into().expect("ungated GQA head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("ungated GQA head_dim must fit u32");
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
        dtype => panic!("unsupported ungated GQA dtype {dtype:?}"),
    };
    norm_rope.with_rope_scaling(config.rope_scaling)
}

fn kv_page_write_config(core: &UngatedGQACore, config: GQAMetalConfig) -> backend_kv_page_write::Config {
    backend_kv_page_write::Config {
        num_kv_heads: core.num_kv_heads.try_into().expect("ungated GQA KV heads must fit u32"),
        head_dim: core.head_dim.try_into().expect("ungated GQA head_dim must fit u32"),
        page_bytes: config.page_bytes,
        dtype: config.io_dtype,
    }
}

fn validate_config_for_core(core: &UngatedGQACore, config: GQAMetalConfig) {
    config.validate();
    assert!(config.rope_dim as usize <= core.head_dim);
    assert!(config.num_ungated_tokens_per_page(core) > 0);
}

fn affine_config(n: usize, k: usize, config: GQAMetalConfig) -> affine_quantized::Config {
    affine_quantized::Config {
        n: n.try_into().expect("ungated GQA affine n must fit i32"),
        k: k.try_into().expect("ungated GQA affine k must fit i32"),
        group_size: config
            .group_size
            .try_into()
            .expect("ungated GQA group_size must fit i32"),
        bits: config.bits.try_into().expect("ungated GQA bits must fit i32"),
        input_dtype: config.io_dtype,
        output_dtype: config.io_dtype,
        scale_bias_dtype: config.io_dtype,
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use inference_backend_metal::components::gqa::fp8::bf16_to_fp8_e4m3;
    use inference_backend_metal::components::rms_norm_rope::RopeScaling;
    use inference_backend_metal::metal::Stream;

    use super::*;
    use crate::attn::gqa::backend::GQA_NUM_ACTIVE_TOKENS;

    #[test]
    fn test_fixed_partitions_and_direct_sdpa_preserve_causal_output_and_active_tokens() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = &stream;
        let core = UngatedGQACore::new(0, 32, 128, 6, 1, 128_f32.sqrt().recip());
        let config = GQAMetalConfig {
            group_size: 32,
            bits: 8,
            page_bytes: 2 * 8 * 128,
            rope_dim: 128,
            norm_eps: 1.0e-6,
            rope_theta: 10_000.0,
            rope_scaling: RopeScaling::Default,
            io_dtype: Dtype::Bfloat16,
        };
        let mut component = UngatedGQA::new(
            &device,
            core.clone(),
            config,
            8,
            GQAPageTableLayout {
                num_req_slots: 1,
                num_blocks: 1,
                num_gqa_layers: 1,
                num_page_ids_per_block: 1,
            },
        );
        let sdpa_config = gqa_sdpa_config(config, 6, 1, 128);
        let variants = [
            backend_sdpa::ExecutionVariant::sliced_kv(sdpa_config, 128, 128, 6),
            backend_sdpa::ExecutionVariant::sliced_q_per_head(sdpa_config, 8, 16, 6),
            backend_sdpa::ExecutionVariant::sliced_q(
                sdpa_config,
                backend_sdpa::MapKernelKind::QHeadsPerToken,
                4,
                16,
                6,
            ),
            backend_sdpa::ExecutionVariant::sliced_q(
                sdpa_config,
                backend_sdpa::MapKernelKind::QTokensAndHeads,
                4,
                16,
                6,
            ),
        ];
        component.sdpa_selector = Selector::for_partitions(
            backend_sdpa::Registry::from_variants(sdpa_config, variants.to_vec()),
            8,
            1,
            |_| 8_u32.div_ceil(3),
        );
        let scratch = component.new_scratch();
        let metadata = component.new_metadata();
        let attention_output = scratch.bindings().attention_output;
        let policy = component.replay_bucket_policy();
        let mut pages = vec![0_u8; config.page_bytes as usize];
        for token in 0..8 {
            let value = bf16_to_fp8_e4m3(bf16::from_f32(token as f32 * 0.25));
            pages[(8 + token) * 128..(9 + token) * 128].fill(value);
        }
        let kv_pages = Buffer::from_slice(&device, &pages);
        let page_ids = Buffer::from_slice(&device, &[0_u32]);
        // This test records only the private SDPA stage. Projection bindings are unused.
        let unused = Buffer::new_zeroed_elements(&device, 1, Dtype::Uint32);
        let page_table_layout = GQAPageTableLayout {
            num_req_slots: 1,
            num_blocks: 1,
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        };
        let routes = variants
            .into_iter()
            .flat_map(|variant| [8, 3, 5].map(|slice| (variant, slice, false)))
            .chain(
                variants
                    .into_iter()
                    .filter(|variant| variant.map.kind != backend_sdpa::MapKernelKind::SlicedKv)
                    .map(|variant| (variant, 8, true)),
            );
        for (variant, kv_slice, direct_output) in routes {
            for num_active_tokens in [3_u32, 8, 1, 5] {
                let requests = RequestShape::from_batch(&[0], &[0, num_active_tokens]);
                let selection =
                    component
                        .sdpa_selector
                        .select_partitioned(variant, kv_slice, direct_output, &requests, &policy, 8);
                let shape = metadata.update(&[0], &[0], &[0, num_active_tokens], &selection);
                attention_output.zero_bytes(0, attention_output.len_bytes());
                let mut recorder = ReplayRecorder::new(runtime.create_replay_program());
                component.record_sdpa(
                    &mut recorder,
                    UngatedGQAInput {
                        page_table_layout,
                        gqa_layer_index: ReplayU32::Fixed(0),
                        batch_metadata: &metadata,
                        hidden_state: &unused,
                        next_hidden_state: &unused,
                        kv_cache: GQAKVCacheBindings {
                            kv_pages: &kv_pages,
                            page_ids: &page_ids,
                        },
                        weights: UngatedGQAWeights {
                            qkv_weight: &unused,
                            qkv_scales: &unused,
                            qkv_biases: &unused,
                            q_norm_weight: &unused,
                            k_norm_weight: &unused,
                            output_weight: &unused,
                            output_scales: &unused,
                            output_biases: &unused,
                        },
                        scratch: scratch.bindings(),
                        num_active_tokens: ReplayU32::Parameter(GQA_NUM_ACTIVE_TOKENS),
                    },
                );
                let replay = recorder.build();
                let mut arguments = ReplayArguments::new();
                arguments.set_u32(GQA_NUM_ACTIVE_TOKENS, num_active_tokens);
                add_ungated_gqa_private_replay_arguments(shape, component.replay_topology(&metadata), &mut arguments);
                runtime.submit_replay_with_arguments(&replay, &arguments).wait();
                let actual = attention_output.read_typed::<u16>(0, 8 * core.q_dim());
                for (token, q_token_heads) in actual.chunks_exact(core.q_dim()).enumerate() {
                    // Q and K are zero. Softmax is uniform over tokens 0..=token.
                    let expected = if token < num_active_tokens as usize {
                        token as f32 * 0.125
                    } else {
                        0.0
                    };
                    for &bits in q_token_heads {
                        assert!(
                            (bf16::from_bits(bits).to_f32() - expected).abs() <= 0.015625,
                            "route={variant:?}, slice={kv_slice}, direct={direct_output}, active={num_active_tokens}, \
                             token={token}"
                        );
                    }
                }
            }
        }
    }
}
