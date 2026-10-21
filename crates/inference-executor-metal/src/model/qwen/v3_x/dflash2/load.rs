use std::num::NonZeroUsize;
use std::path::Path;
use std::rc::Rc;

use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::BiDiBlockCapacity;
use inference_executor_core::checkpoint::QuantizedTensorBindings;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2WeightBindings;
use inference_executor_core::model::qwen::v3_x::dflash2::resolve_qwen3x_dflash2_weight_bindings;

use crate::attn::bidi_block_gqa::state::BiDiBlockGQAState;
use crate::attn::gqa::request_page_table::GQAPageTableLayout;
use crate::checkpoint::SafeTensorStore;
use crate::model::embedding::Embed;
use crate::model::qwen::v3_x::dflash2::attention::qwen3x_dflash2_gqa_core;
use crate::model::qwen::v3_x::dflash2::attention::qwen3x_dflash2_gqa_sdpa_config;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2Model;
use crate::model::qwen::v3_x::dflash2::output::Qwen3xDFlash2Output;
use crate::model::unembedding::Unembed;
use crate::sampling::sampling_params::SamplingParamsStore;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3xDFlash2LoadConfig {
    pub num_spec_tokens: NonZeroUsize,
    pub page_size_bytes: usize,
    pub max_position_embeddings: usize,
    pub max_requests: usize,
    pub max_tokens: usize,
    pub num_cache_pages: usize,
    pub num_tokens_per_block: usize,
}

pub struct Qwen3xDFlash2Loaded {
    pub page_table_layout: GQAPageTableLayout,
    pub gqa_state: BiDiBlockGQAState,
    pub model: Rc<Qwen3xDFlash2Model>,
    pub embed: Rc<Embed>,
    pub output: Qwen3xDFlash2Output,
    pub num_spec_tokens: usize,
    pub mask_token_id: i32,
    pub sliding_window: usize,
    pub page_bytes: usize,
    pub max_anchor_position: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn load_qwen3x_dflash2(
    device: &Device,
    model_dir: &Path,
    config: &Qwen3xDFlash2Config,
    load_config: Qwen3xDFlash2LoadConfig,
    main_text_embed: Rc<Embed>,
    main_unembed: Rc<Unembed>,
    sampling_params: Rc<SamplingParamsStore>,
) -> Result<Qwen3xDFlash2Loaded, ModelExecutorError> {
    let num_spec_tokens = load_config.num_spec_tokens.get();
    let spec_block_size = num_spec_tokens + 1;
    let mut store = SafeTensorStore::from_model_dir(model_dir)?;
    let Qwen3xDFlash2WeightBindings {
        main_feature,
        layers,
        final_norm_weight,
        selector,
    } = resolve_qwen3x_dflash2_weight_bindings(config, store.index().tensor_names())?;
    let scale_bias_dtype = load_scale_bias_dtype(&mut store, &selector.hidden_projection)?;
    let attention_core = qwen3x_dflash2_gqa_core(config, num_spec_tokens, 0);
    let attention_sdpa_config = qwen3x_dflash2_gqa_sdpa_config(config, load_config.page_size_bytes)?;
    let tokens_per_page = attention_sdpa_config.tokens_per_page as usize;
    let page_ids_per_block = num_page_ids_per_block(load_config.num_tokens_per_block, tokens_per_page);
    let page_table_layout = GQAPageTableLayout {
        num_req_slots: load_config
            .max_requests
            .try_into()
            .expect("Qwen3x DFlash2 max_requests must fit u32"),
        num_blocks: load_config
            .max_position_embeddings
            .div_ceil(load_config.num_tokens_per_block)
            .max(1)
            .try_into()
            .expect("Qwen3x DFlash2 block capacity must fit u32"),
        num_gqa_layers: config
            .num_hidden_layers
            .try_into()
            .expect("Qwen3x DFlash2 layer count must fit u32"),
        num_page_ids_per_block: page_ids_per_block
            .try_into()
            .expect("Qwen3x DFlash2 pages per block must fit u32"),
    };
    let capacity = BiDiBlockCapacity::new(load_config.max_requests, spec_block_size);
    let gqa_state = BiDiBlockGQAState::new(
        device,
        attention_core,
        attention_sdpa_config,
        page_table_layout,
        capacity,
        load_config.max_tokens,
        load_config.num_cache_pages,
    );
    let max_query_tokens = capacity.max_tokens;
    let max_proposal_tokens = load_config
        .max_requests
        .checked_mul(num_spec_tokens)
        .expect("Qwen3x DFlash2 proposal token capacity must fit usize");
    let embed = Rc::new(
        main_text_embed.with_max_tokens(
            max_query_tokens
                .try_into()
                .expect("Qwen3x DFlash2 embed row capacity must fit u32"),
        ),
    );
    let unembed = Rc::new(
        main_unembed.with_max_tokens(
            max_proposal_tokens
                .try_into()
                .expect("Qwen3x DFlash2 unembed row capacity must fit u32"),
        ),
    );
    let mut output = Qwen3xDFlash2Output::new(
        device,
        config,
        num_spec_tokens,
        load_config.max_requests,
        unembed,
        &selector,
        sampling_params,
        scale_bias_dtype,
    )?;
    output.load_weights(device, &mut store, selector)?;
    let mut model = Qwen3xDFlash2Model::new(
        device,
        config,
        num_spec_tokens,
        load_config.page_size_bytes,
        &main_feature,
        &layers,
        &gqa_state,
        load_config.max_tokens,
        load_config.max_requests,
        max_query_tokens,
        scale_bias_dtype,
    )?;
    model.load_weights(device, &mut store, config, &main_feature, layers, final_norm_weight)?;
    Ok(Qwen3xDFlash2Loaded {
        page_table_layout,
        gqa_state,
        model: Rc::new(model),
        embed,
        output,
        num_spec_tokens,
        mask_token_id: config
            .mask_token_id
            .try_into()
            .map_err(|_| ModelExecutorError::custom("Qwen3x DFlash2 MASK token ID must fit i32"))?,
        sliding_window: config.sliding_window,
        page_bytes: load_config.page_size_bytes,
        max_anchor_position: load_config
            .max_position_embeddings
            .try_into()
            .expect("Qwen3x DFlash2 maximum anchor position must fit u32"),
    })
}

fn load_scale_bias_dtype(
    store: &mut SafeTensorStore,
    bindings: &QuantizedTensorBindings,
) -> Result<Dtype, ModelExecutorError> {
    let tensors = store.load_tensors([bindings.scales.as_str(), bindings.biases.as_str()])?;
    let scales = tensors
        .get(&bindings.scales)
        .expect("requested DFlash2 scale tensor must be loaded");
    let biases = tensors
        .get(&bindings.biases)
        .expect("requested DFlash2 bias tensor must be loaded");
    scale_bias_dtype(scales.dtype(), biases.dtype())
}

fn scale_bias_dtype(scales: safetensors::Dtype, biases: safetensors::Dtype) -> Result<Dtype, ModelExecutorError> {
    if scales != biases {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3x DFlash2 affine scales and biases must use one dtype, got scales={scales:?} biases={biases:?}"
        )));
    }
    match scales {
        safetensors::Dtype::BF16 => Ok(Dtype::Bfloat16),
        safetensors::Dtype::F32 => Ok(Dtype::Float32),
        dtype => {
            Err(ModelExecutorError::custom(format!(
                "Qwen3x DFlash2 affine scales and biases must use BF16 or F32, got {dtype:?}"
            )))
        },
    }
}

fn num_page_ids_per_block(num_tokens_per_block: usize, num_tokens_per_page: usize) -> usize {
    assert!(num_tokens_per_block > 0);
    assert!(num_tokens_per_page > 0);
    assert!(
        num_tokens_per_block.is_multiple_of(num_tokens_per_page),
        "Qwen3x DFlash2 GQA tokens per block must be divisible by tokens per page"
    );
    num_tokens_per_block / num_tokens_per_page
}
