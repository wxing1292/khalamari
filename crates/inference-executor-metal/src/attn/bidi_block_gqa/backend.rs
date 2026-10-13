//! Shared history-plus-block GQA recording.

use std::mem::size_of;

use inference_backend_metal::components::gqa::bidi_block_sdpa as backend_bidi_block_sdpa;
use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::gqa::sdpa::ExecutionVariant;
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
use inference_executor_core::attn::BiDiBlockGQACore;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::attn::GQAReplayShape;
use inference_executor_core::backend::recorder::Recorder;

use crate::attn::bidi_block_gqa::metadata::BiDiBlockGQAMetadataBuffers;
use crate::attn::bidi_block_gqa::scratch::BiDiBlockGQAScratchBindings;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::def::layer::ReplayLayer;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::def::quantized_affine::QuantizedAffineWeights;
use crate::def::replay_op::ReplayOp;

pub const BIDI_BLOCK_GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES: ReplayParameterKey =
    ReplayParameterKey::new("bidi_block_gqa.num_active_sdpa_map_task_templates");
pub const BIDI_BLOCK_GQA_NUM_ACTIVE_Q_TOKEN_RANGES: ReplayParameterKey =
    ReplayParameterKey::new("bidi_block_gqa.num_active_q_token_ranges");

pub fn add_bidi_block_gqa_replay_arguments(shape: GQAReplayShape, arguments: &mut ReplayArguments) {
    shape.validate();
    arguments.set_u32(BIDI_BLOCK_GQA_NUM_ACTIVE_Q_TOKEN_RANGES, shape.num_q_token_tiles);
    arguments.set_u32(
        BIDI_BLOCK_GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES,
        shape.num_sdpa_map_task_templates,
    );
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiDiBlockGQAMetalConfig {
    pub q: QuantizedAffineLayout,
    pub k: QuantizedAffineLayout,
    pub v: QuantizedAffineLayout,
    pub output: QuantizedAffineLayout,
    pub page_bytes: u32,
    pub rope_dim: u32,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: rms_norm_rope::RopeScaling,
    pub io_dtype: Dtype,
    pub norm_weight_dtype: Dtype,
}

impl BiDiBlockGQAMetalConfig {
    pub fn validate(self) {
        self.q.validate();
        self.k.validate();
        self.v.validate();
        self.output.validate();
        assert!(self.page_bytes > 0);
        assert!(self.rope_dim > 0 && self.rope_dim.is_multiple_of(2));
        assert!(self.norm_eps.is_finite() && self.norm_eps > 0.0);
        assert!(self.rope_theta.is_finite() && self.rope_theta > 0.0);
        self.rope_scaling.validate();
        assert_eq!(self.io_dtype, Dtype::Bfloat16, "BiDiBlockGQA requires BF16 model IO");
        assert!(matches!(self.norm_weight_dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn num_tokens_per_page(self, core: &inference_executor_core::attn::UngatedGQACore) -> u32 {
        self.validate();
        core.validate();
        let bytes_per_token = core
            .num_kv_heads
            .checked_mul(core.head_dim)
            .and_then(|values| values.checked_mul(2))
            .and_then(|values| values.checked_mul(size_of::<u8>()))
            .expect("BiDiBlockGQA KV bytes per token must fit usize");
        let page_bytes = self.page_bytes as usize;
        assert!(
            page_bytes.is_multiple_of(bytes_per_token),
            "BiDiBlockGQA page must contain whole KV tokens"
        );
        (page_bytes / bytes_per_token)
            .try_into()
            .expect("BiDiBlockGQA tokens per page must fit u32")
    }
}

#[derive(Clone, Copy)]
pub struct BiDiBlockGQAWeights<'a> {
    pub q: QuantizedAffineWeights<'a>,
    pub k: QuantizedAffineWeights<'a>,
    pub v: QuantizedAffineWeights<'a>,
    pub q_norm_weight: &'a Buffer,
    pub k_norm_weight: &'a Buffer,
    pub output: QuantizedAffineWeights<'a>,
}

#[derive(Clone, Copy)]
pub struct BiDiBlockGQAInput<'a> {
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: u32,
    pub metadata: &'a BiDiBlockGQAMetadataBuffers,
    pub hidden_state: &'a Buffer,
    pub next_hidden_state: &'a Buffer,
    pub kv_cache: GQAKVCacheBindings<'a>,
    pub weights: BiDiBlockGQAWeights<'a>,
    pub scratch: BiDiBlockGQAScratchBindings<'a>,
    pub num_active_tokens: ReplayU32,
}

pub struct BiDiBlockGQA {
    device: Device,
    core: BiDiBlockGQACore,
    metal: BiDiBlockGQAMetalConfig,
    q: affine_quantized::Matmul,
    k: affine_quantized::Matmul,
    v: affine_quantized::Matmul,
    q_norm_rope: rms_norm_rope::Compute,
    k_norm_rope: rms_norm_rope::Compute,
    sdpa_execution: ExecutionVariant,
    bidi_block_sdpa: backend_bidi_block_sdpa::Compute,
    output: affine_quantized::Matmul,
}

impl BiDiBlockGQA {
    pub fn new(
        device: &Device,
        core: BiDiBlockGQACore,
        metal: BiDiBlockGQAMetalConfig,
        sdpa_execution: ExecutionVariant,
    ) -> Self {
        core.validate();
        metal.validate();
        let attention = &core.attention;
        assert!(metal.rope_dim as usize <= attention.head_dim);
        assert!(metal.num_tokens_per_page(attention) > 0);
        let sdpa_config = backend_sdpa::Config {
            io_dtype: metal.io_dtype,
            num_q_heads: attention
                .num_q_heads
                .try_into()
                .expect("BiDiBlockGQA Q-head count must fit u32"),
            num_kv_heads: attention
                .num_kv_heads
                .try_into()
                .expect("BiDiBlockGQA KV-head count must fit u32"),
            head_dim: attention
                .head_dim
                .try_into()
                .expect("BiDiBlockGQA head_dim must fit u32"),
            tokens_per_page: metal.num_tokens_per_page(attention),
        };
        assert!(
            sdpa_execution.supports(sdpa_config),
            "BiDiBlockGQA history attention execution must support the layer geometry"
        );
        let output = attention.output_shape();
        Self {
            device: device.clone(),
            q: affine_quantized::Matmul::new(
                device,
                metal.q.config(attention.q_dim(), attention.hidden_dim, metal.io_dtype),
            ),
            k: affine_quantized::Matmul::new(
                device,
                metal.k.config(attention.k_dim(), attention.hidden_dim, metal.io_dtype),
            ),
            v: affine_quantized::Matmul::new(
                device,
                metal.v.config(attention.v_dim(), attention.hidden_dim, metal.io_dtype),
            ),
            q_norm_rope: rms_norm_rope::Compute::new(device, norm_rope_config(attention, metal, attention.num_q_heads)),
            k_norm_rope: rms_norm_rope::Compute::new(
                device,
                norm_rope_config(attention, metal, attention.num_kv_heads),
            ),
            sdpa_execution,
            bidi_block_sdpa: backend_bidi_block_sdpa::Compute::new(
                device,
                backend_bidi_block_sdpa::Config {
                    block_size: core
                        .block_size
                        .try_into()
                        .expect("BiDiBlockGQA block size must fit u32"),
                    max_q_tokens: sdpa_execution.map.thread_block.max_q_tokens,
                    num_q_heads: attention
                        .num_q_heads
                        .try_into()
                        .expect("BiDiBlockGQA Q-head count must fit u32"),
                    num_kv_heads: attention
                        .num_kv_heads
                        .try_into()
                        .expect("BiDiBlockGQA KV-head count must fit u32"),
                    head_dim: attention
                        .head_dim
                        .try_into()
                        .expect("BiDiBlockGQA head_dim must fit u32"),
                    scale: attention.scale,
                    dtype: metal.io_dtype,
                },
            ),
            output: affine_quantized::Matmul::new(
                device,
                metal.output.config(output.out_dim, output.in_dim, metal.io_dtype),
            ),
            core,
            metal,
        }
    }

    fn validate_input(&self, input: &BiDiBlockGQAInput<'_>) -> GQAReplayShape {
        input.page_table_layout.validate();
        assert!(
            input.gqa_layer_index < input.page_table_layout.num_gqa_layers,
            "BiDiBlockGQA layer index exceeds the page table"
        );
        let shape = input.metadata.replay_shape();
        let sdpa_execution = input.metadata.sdpa_execution();
        shape.validate();
        assert!(shape.reduce_sdpa_partial_outputs);
        assert!(
            shape.num_total_tokens as usize <= input.scratch.capacity.block.max_tokens,
            "BiDiBlockGQA replay token capacity exceeds scratch"
        );
        assert!(
            shape.num_total_sdpa_map_task_templates as usize <= input.scratch.capacity.max_sdpa_map_task_templates,
            "BiDiBlockGQA replay partial count exceeds scratch"
        );
        assert_eq!(
            input.scratch.capacity.block.block_size, self.core.block_size,
            "BiDiBlockGQA scratch block size must match the backend"
        );
        assert_eq!(
            sdpa_execution, self.sdpa_execution,
            "BiDiBlockGQA history attention metadata must match the frozen backend execution"
        );
        assert_eq!(
            self.sdpa_execution.map.thread_block.max_q_tokens as usize, input.scratch.capacity.max_q_tokens,
            "BiDiBlockGQA history attention execution must match scratch capacity"
        );
        shape
    }

    fn sdpa_sliced_kv_config(&self, page_table_layout: GQAPageTableLayout) -> backend_sliced_kv::Config {
        let attention = &self.core.attention;
        backend_sliced_kv::Config {
            num_q_heads: attention
                .num_q_heads
                .try_into()
                .expect("BiDiBlockGQA Q-head count must fit u32"),
            num_kv_heads: attention
                .num_kv_heads
                .try_into()
                .expect("BiDiBlockGQA KV-head count must fit u32"),
            head_dim: attention
                .head_dim
                .try_into()
                .expect("BiDiBlockGQA head_dim must fit u32"),
            scale: attention.scale,
            page_bytes: self.metal.page_bytes,
            page_table_layout: backend_page_table_layout(page_table_layout),
            dtype: self.metal.io_dtype,
        }
    }

    fn sdpa_sliced_q_config(&self, page_table_layout: GQAPageTableLayout) -> backend_sliced_q::Config {
        let attention = &self.core.attention;
        backend_sliced_q::Config {
            num_q_heads: attention
                .num_q_heads
                .try_into()
                .expect("BiDiBlockGQA Q-head count must fit u32"),
            num_kv_heads: attention
                .num_kv_heads
                .try_into()
                .expect("BiDiBlockGQA KV-head count must fit u32"),
            head_dim: attention
                .head_dim
                .try_into()
                .expect("BiDiBlockGQA head_dim must fit u32"),
            scale: attention.scale,
            page_bytes: self.metal.page_bytes,
            dtype: self.metal.io_dtype,
            page_table_layout: backend_page_table_layout(page_table_layout),
        }
    }

    fn record_bidi_block_sdpa<'a, R>(
        &'a self,
        recorder: &mut R,
        shape: GQAReplayShape,
        metadata: &'a BiDiBlockGQAMetadataBuffers,
        scratch: BiDiBlockGQAScratchBindings<'a>,
    ) where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record(ReplayOp::opaque(self.bidi_block_sdpa.invoke(
            backend_bidi_block_sdpa::Shape {
                num_total_tokens: shape.num_total_tokens,
                num_total_q_token_ranges: shape.num_total_q_token_tiles,
                num_total_partial_output_slots: shape.num_total_sdpa_map_task_templates,
            },
            ReplayU32::Parameter(BIDI_BLOCK_GQA_NUM_ACTIVE_Q_TOKEN_RANGES),
            backend_bidi_block_sdpa::Buffers {
                q: scratch.q_norm_rope,
                local_k: scratch.k_norm_rope,
                local_v: scratch.v,
                q_token_ranges: metadata.q_token_ranges(),
                cu_sdpa_partial_outputs: metadata.cu_sdpa_partial_outputs(),
                partial_exp_sums: scratch.partial_exp_sums,
                partial_max_logits: scratch.partial_max_logits,
                partial_output: scratch.partial_output,
            },
        )));
    }
}

impl ReplayLayer for BiDiBlockGQA {
    type Input<'a> = BiDiBlockGQAInput<'a>;
    type Output<'a> = &'a Buffer;

    fn record<'a, R>(&'a self, recorder: &mut R, input: Self::Input<'a>) -> Self::Output<'a>
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let shape = self.validate_input(&input);
        let sdpa_execution = self.sdpa_execution;
        let attention = &self.core.attention;
        let scratch = input.scratch;
        let active_tokens = input.num_active_tokens;
        recorder.record_with_barrier_before(ReplayOp::opaque(self.q.invoke(
            shape.num_total_tokens,
            active_tokens,
            scratch.q,
            0,
            input.hidden_state,
            0,
            input.weights.q.weight,
            input.weights.q.weight_offset,
            input.weights.q.scales,
            input.weights.q.scales_offset,
            input.weights.q.biases,
            input.weights.q.biases_offset,
        )));
        recorder.record(ReplayOp::opaque(self.k.invoke(
            shape.num_total_tokens,
            active_tokens,
            scratch.k,
            0,
            input.hidden_state,
            0,
            input.weights.k.weight,
            input.weights.k.weight_offset,
            input.weights.k.scales,
            input.weights.k.scales_offset,
            input.weights.k.biases,
            input.weights.k.biases_offset,
        )));
        recorder.record(ReplayOp::opaque(self.v.invoke(
            shape.num_total_tokens,
            active_tokens,
            scratch.v,
            0,
            input.hidden_state,
            0,
            input.weights.v.weight,
            input.weights.v.weight_offset,
            input.weights.v.scales,
            input.weights.v.scales_offset,
            input.weights.v.biases,
            input.weights.v.biases_offset,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.q_norm_rope.invoke(
            rms_norm_rope::Shape {
                num_total_tokens: shape.num_total_tokens,
            },
            rms_norm_rope::Buffers {
                input: scratch.q,
                norm_weight: input.weights.q_norm_weight,
                flat_token_indices: input.metadata.flat_token_indices(),
                output: scratch.q_norm_rope,
            },
            active_tokens,
        )));
        recorder.record(ReplayOp::opaque(self.k_norm_rope.invoke(
            rms_norm_rope::Shape {
                num_total_tokens: shape.num_total_tokens,
            },
            rms_norm_rope::Buffers {
                input: scratch.k,
                norm_weight: input.weights.k_norm_weight,
                flat_token_indices: input.metadata.flat_token_indices(),
                output: scratch.k_norm_rope,
            },
            active_tokens,
        )));

        if sdpa_execution.map.kind == backend_sdpa::MapKernelKind::SlicedKv {
            let sdpa_config = self.sdpa_sliced_kv_config(input.page_table_layout);
            let sdpa_shape = backend_sliced_kv::Shape {
                num_total_tokens: shape.num_total_tokens,
                num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
            };
            let sdpa = backend_sliced_kv::Compute::new(&self.device, sdpa_config, sdpa_execution, sdpa_shape);
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(
                backend_sliced_kv::MapBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: input.kv_cache.kv_pages,
                    req_slots: input.metadata.req_slots(),
                    page_ids: input.kv_cache.page_ids,
                    sdpa_map_task_templates: input.metadata.sdpa_map_task_templates(),
                    partial_exp_sums: scratch.partial_exp_sums,
                    partial_max_logits: scratch.partial_max_logits,
                    partial_output: scratch.partial_output,
                },
                ReplayU32::Fixed(input.gqa_layer_index),
                active_tokens,
                ReplayU32::Parameter(BIDI_BLOCK_GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES),
            )));
            self.record_bidi_block_sdpa(recorder, shape, input.metadata, scratch);
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(
                backend_sliced_kv::ReduceBuffers {
                    partial_exp_sums: scratch.partial_exp_sums,
                    partial_max_logits: scratch.partial_max_logits,
                    partial_output: scratch.partial_output,
                    cu_sdpa_partial_outputs: input.metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                },
                active_tokens,
            )));
        } else {
            let sdpa_config = self.sdpa_sliced_q_config(input.page_table_layout);
            let sdpa_shape = backend_sliced_q::Shape {
                num_total_tokens: shape.num_total_tokens,
                num_total_q_token_tiles: shape.num_total_q_token_tiles,
                num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
            };
            let sdpa = backend_sliced_q::Compute::new(&self.device, sdpa_config, sdpa_execution, sdpa_shape);
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_map(
                backend_sliced_q::MapBuffers {
                    q: scratch.q_norm_rope,
                    kv_pages: input.kv_cache.kv_pages,
                    req_slots: input.metadata.req_slots(),
                    page_ids: input.kv_cache.page_ids,
                    visible_kv_token_ranges: input.metadata.visible_kv_token_ranges(),
                    q_token_ranges: input.metadata.q_token_ranges(),
                    sdpa_map_task_templates: input.metadata.sdpa_map_task_templates(),
                    partial_output: scratch.partial_output,
                    partial_exp_sums: scratch.partial_exp_sums,
                    partial_max_logits: scratch.partial_max_logits,
                },
                ReplayU32::Fixed(input.gqa_layer_index),
                active_tokens,
                ReplayU32::Parameter(BIDI_BLOCK_GQA_NUM_ACTIVE_Q_TOKEN_RANGES),
                ReplayU32::Parameter(BIDI_BLOCK_GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES),
            )));
            self.record_bidi_block_sdpa(recorder, shape, input.metadata, scratch);
            recorder.record_with_barrier_before(ReplayOp::opaque(sdpa.invoke_reduce(
                backend_sliced_q::ReduceBuffers {
                    partial_output: scratch.partial_output,
                    partial_exp_sums: scratch.partial_exp_sums,
                    partial_max_logits: scratch.partial_max_logits,
                    q_token_ranges: input.metadata.q_token_ranges(),
                    cu_sdpa_partial_outputs: input.metadata.cu_sdpa_partial_outputs(),
                    output: scratch.attention_output,
                },
                ReplayU32::Parameter(BIDI_BLOCK_GQA_NUM_ACTIVE_Q_TOKEN_RANGES),
            )));
        }
        recorder.record_with_barrier_before(ReplayOp::opaque(self.output.invoke(
            shape.num_total_tokens,
            active_tokens,
            input.next_hidden_state,
            0,
            scratch.attention_output,
            0,
            input.weights.output.weight,
            input.weights.output.weight_offset,
            input.weights.output.scales,
            input.weights.output.scales_offset,
            input.weights.output.biases,
            input.weights.output.biases_offset,
        )));
        input.next_hidden_state
    }
}

fn backend_page_table_layout(layout: GQAPageTableLayout) -> backend_kv_page_write::PageTableLayout {
    backend_kv_page_write::PageTableLayout {
        num_req_slots: layout.num_req_slots,
        num_blocks: layout.num_blocks,
        num_gqa_layers: layout.num_gqa_layers,
        num_page_ids_per_block: layout.num_page_ids_per_block,
    }
}

fn norm_rope_config(
    core: &inference_executor_core::attn::UngatedGQACore,
    metal: BiDiBlockGQAMetalConfig,
    num_heads: usize,
) -> rms_norm_rope::Config {
    let num_heads = num_heads.try_into().expect("BiDiBlockGQA head count must fit u32");
    let head_dim = core.head_dim.try_into().expect("BiDiBlockGQA head_dim must fit u32");
    let norm_rope = match metal.io_dtype {
        Dtype::Float32 => {
            rms_norm_rope::Config::f32(num_heads, head_dim, metal.rope_dim, metal.norm_eps, metal.rope_theta)
        },
        Dtype::Bfloat16 => {
            rms_norm_rope::Config::bf16(num_heads, head_dim, metal.rope_dim, metal.norm_eps, metal.rope_theta)
        },
        dtype => panic!("unsupported BiDiBlockGQA dtype {dtype:?}"),
    };
    norm_rope
        .with_rope_scaling(metal.rope_scaling)
        .with_norm_weight_dtype(metal.norm_weight_dtype)
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::ReplayArguments;
    use inference_executor_core::attn::GQAReplayShape;

    use super::BIDI_BLOCK_GQA_NUM_ACTIVE_Q_TOKEN_RANGES;
    use super::BIDI_BLOCK_GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES;
    use super::add_bidi_block_gqa_replay_arguments;

    #[test]
    fn test_replay_arguments_success() {
        let shape = GQAReplayShape {
            num_tokens: 8,
            num_total_tokens: 8,
            num_q_token_tiles: 1,
            num_total_q_token_tiles: 1,
            num_sdpa_map_task_templates: 7,
            num_total_sdpa_map_task_templates: 8,
            reduce_sdpa_partial_outputs: true,
        };
        let mut shorter_history_arguments = ReplayArguments::new();
        add_bidi_block_gqa_replay_arguments(shape, &mut shorter_history_arguments);
        let mut longer_history = shape;
        longer_history.num_sdpa_map_task_templates = 8;
        let mut longer_history_arguments = ReplayArguments::new();
        add_bidi_block_gqa_replay_arguments(longer_history, &mut longer_history_arguments);

        assert_eq!(
            shorter_history_arguments,
            ReplayArguments::new()
                .with_u32(BIDI_BLOCK_GQA_NUM_ACTIVE_Q_TOKEN_RANGES, 1)
                .with_u32(BIDI_BLOCK_GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES, 7)
        );
        assert_eq!(
            longer_history_arguments,
            ReplayArguments::new()
                .with_u32(BIDI_BLOCK_GQA_NUM_ACTIVE_Q_TOKEN_RANGES, 1)
                .with_u32(BIDI_BLOCK_GQA_NUM_ACTIVE_SDPA_MAP_TASK_TEMPLATES, 8)
        );
    }
}
