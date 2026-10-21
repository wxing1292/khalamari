//! Persistent history-KV projection and cache write.

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::rms_norm_rope;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::BiDiBlockGQACore;

use crate::attn::bidi_block_gqa::backend::BiDiBlockGQAMetalConfig;
use crate::attn::bidi_block_gqa::backend::BiDiBlockGQAWeights;
use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::request_page_table::GQAPageTableLayout;

pub struct BiDiBlockGQAKVCacheWriteScratch {
    max_tokens: usize,
    k: Buffer,
    v: Buffer,
    k_norm_rope: Buffer,
}

#[derive(Clone, Copy)]
pub struct BiDiBlockGQAKVCacheWriteScratchBindings<'a> {
    pub max_tokens: usize,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub k_norm_rope: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct BiDiBlockGQAKVCacheWriteInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_layer_index: u32,
    pub main_feature: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub flat_token_indices: &'a Buffer,
    pub kv_cache: GQAKVCacheBindings<'a>,
    pub weights: BiDiBlockGQAWeights<'a>,
    pub scratch: BiDiBlockGQAKVCacheWriteScratchBindings<'a>,
}

pub struct BiDiBlockGQAKVCacheWriter {
    k: affine_quantized::Matmul,
    v: affine_quantized::Matmul,
    k_norm_rope: rms_norm_rope::Compute,
    kv_page_write: backend_kv_page_write::Compute,
}

impl BiDiBlockGQAKVCacheWriteScratch {
    pub fn new(device: &Device, core: &BiDiBlockGQACore, io_dtype: Dtype, max_tokens: usize) -> Self {
        core.validate();
        match io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 BiDiBlockGQA model boundary is not supported"),
            dtype => panic!("unsupported BiDiBlockGQA model boundary dtype {dtype:?}"),
        }
        assert!(max_tokens > 0, "BiDiBlockGQA KV-cache-write scratch requires tokens");
        let kv_elements = max_tokens
            .checked_mul(core.attention.k_dim())
            .expect("BiDiBlockGQA KV-cache-write scratch K/V element count must fit usize");
        Self {
            max_tokens,
            k: Buffer::new_zeroed_elements(device, kv_elements, io_dtype),
            v: Buffer::new_zeroed_elements(device, kv_elements, io_dtype),
            k_norm_rope: Buffer::new_zeroed_elements(device, kv_elements, io_dtype),
        }
    }

    pub fn bindings(&self) -> BiDiBlockGQAKVCacheWriteScratchBindings<'_> {
        BiDiBlockGQAKVCacheWriteScratchBindings {
            max_tokens: self.max_tokens,
            k: &self.k,
            v: &self.v,
            k_norm_rope: &self.k_norm_rope,
        }
    }
}

impl BiDiBlockGQAKVCacheWriter {
    pub fn new(device: &Device, core: BiDiBlockGQACore, metal: BiDiBlockGQAMetalConfig) -> Self {
        core.validate();
        metal.validate();
        let attention = &core.attention;
        assert!(metal.rope_dim as usize <= attention.head_dim);
        Self {
            k: affine_quantized::Matmul::new(
                device,
                metal.k.config(attention.k_dim(), attention.hidden_dim, metal.io_dtype),
            ),
            v: affine_quantized::Matmul::new(
                device,
                metal.v.config(attention.v_dim(), attention.hidden_dim, metal.io_dtype),
            ),
            k_norm_rope: rms_norm_rope::Compute::new(device, k_norm_rope_config(attention, metal)),
            kv_page_write: backend_kv_page_write::Compute::new(
                device,
                backend_kv_page_write::Config {
                    num_kv_heads: attention
                        .num_kv_heads
                        .try_into()
                        .expect("BiDiBlockGQA KV-cache-write KV-head count must fit u32"),
                    head_dim: attention
                        .head_dim
                        .try_into()
                        .expect("BiDiBlockGQA KV-cache-write head_dim must fit u32"),
                    page_bytes: metal.page_bytes,
                    dtype: metal.io_dtype,
                },
            ),
        }
    }

    pub fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: BiDiBlockGQAKVCacheWriteInput<'a>) {
        assert!(
            input.num_total_tokens > 0,
            "BiDiBlockGQA KV-cache write requires tokens"
        );
        assert!(
            input.num_total_tokens as usize <= input.scratch.max_tokens,
            "BiDiBlockGQA KV-cache write exceeds scratch"
        );
        input.page_table_layout.validate();
        assert!(
            input.gqa_layer_index < input.page_table_layout.num_gqa_layers,
            "BiDiBlockGQA KV-cache-write layer index exceeds the page table"
        );
        let num_total_tokens = input.num_total_tokens;
        recorder.record_with_barrier_before(ReplayOp::opaque(self.k.invoke(
            num_total_tokens,
            input.num_active_tokens,
            input.scratch.k,
            0,
            input.main_feature,
            0,
            input.weights.k.weight,
            input.weights.k.weight_offset_bytes,
            input.weights.k.scales,
            input.weights.k.scales_offset_bytes,
            input.weights.k.biases,
            input.weights.k.biases_offset_bytes,
        )));
        recorder.record(ReplayOp::opaque(self.v.invoke(
            num_total_tokens,
            input.num_active_tokens,
            input.scratch.v,
            0,
            input.main_feature,
            0,
            input.weights.v.weight,
            input.weights.v.weight_offset_bytes,
            input.weights.v.scales,
            input.weights.v.scales_offset_bytes,
            input.weights.v.biases,
            input.weights.v.biases_offset_bytes,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.k_norm_rope.invoke(
            rms_norm_rope::Shape { num_total_tokens },
            rms_norm_rope::Buffers {
                input: input.scratch.k,
                norm_weight: input.weights.k_norm_weight,
                flat_token_indices: input.flat_token_indices,
                output: input.scratch.k_norm_rope,
            },
            input.num_active_tokens,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.kv_page_write.invoke(
            backend_kv_page_write::Shape {
                num_total_token_writes: num_total_tokens,
                page_table_layout: backend_kv_page_write::PageTableLayout {
                    num_req_slots: input.page_table_layout.num_req_slots,
                    num_gqa_layers: input.page_table_layout.num_gqa_layers,
                    num_blocks: input.page_table_layout.num_blocks,
                    num_page_ids_per_block: input.page_table_layout.num_page_ids_per_block,
                },
            },
            backend_kv_page_write::Buffers {
                pages: input.kv_cache.kv_pages,
                flat_k: input.scratch.k_norm_rope,
                flat_v: input.scratch.v,
                req_slots: input.req_slots,
                flat_token_indices: input.flat_token_indices,
                page_ids: input.kv_cache.page_ids,
            },
            input.num_active_tokens,
            ReplayU32::Fixed(input.gqa_layer_index),
        )));
    }
}

fn k_norm_rope_config(
    core: &inference_executor_core::attn::UngatedGQACore,
    metal: BiDiBlockGQAMetalConfig,
) -> rms_norm_rope::Config {
    let num_heads = core
        .num_kv_heads
        .try_into()
        .expect("BiDiBlockGQA KV-cache-write KV-head count must fit u32");
    let head_dim = core
        .head_dim
        .try_into()
        .expect("BiDiBlockGQA KV-cache-write head_dim must fit u32");
    let norm_rope = match metal.io_dtype {
        Dtype::Float32 => {
            rms_norm_rope::Config::f32(num_heads, head_dim, metal.rope_dim, metal.norm_eps, metal.rope_theta)
        },
        Dtype::Bfloat16 => {
            rms_norm_rope::Config::bf16(num_heads, head_dim, metal.rope_dim, metal.norm_eps, metal.rope_theta)
        },
        dtype => panic!("unsupported BiDiBlockGQA KV-cache-write dtype {dtype:?}"),
    };
    norm_rope
        .with_rope_scaling(metal.rope_scaling)
        .with_norm_weight_dtype(metal.norm_weight_dtype)
}

#[cfg(test)]
mod tests {
    // Context projection shares the same explicit K/V affine layouts as block execution.
}
