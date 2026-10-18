use std::mem::size_of;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3::QWEN3_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3::init_qwen3_model_config;
use inference_executor_core::model::qwen::v3::weight_layout::Qwen3ModelWeightBindings;
use inference_executor_core::model::qwen::v3::weight_layout::resolve_qwen3_model_weight_bindings;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRModelConfig;
use inference_executor_core::model::qwen::v3_asr::audio_output_rows;
use inference_executor_core::model::qwen::v3_asr::init_qwen3_asr_config;
use inference_executor_core::model::qwen::v3_asr::weight_layout::resolve_qwen3_asr_weight_bindings;
use inference_executor_core::sampling::HFGenerationConfig;
use inference_executor_core::sampling::MAX_TOP_K;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_runtime_core::runtime::Token;

use crate::checkpoint::SafeTensorStore;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedConfig;
use crate::model::main_resource_embed::MainResourceEmbed;
use crate::model::page_arena::PageArena;
use crate::model::qwen::v3::executor::Qwen3Checkpoint;
use crate::model::qwen::v3::executor::Qwen3Executor;
use crate::model::qwen::v3::executor::Qwen3PendingTransactions;
use crate::model::qwen::v3::executor::num_page_ids_per_block;
use crate::model::qwen::v3::main::Qwen3Main;
use crate::model::qwen::v3::main::component_config::Qwen3MainConfig;
use crate::model::qwen::v3::main::component_config::derive_qwen3_dense_mlp_configs;
use crate::model::qwen::v3::main::component_config::derive_qwen3_gqa_configs;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQAState;
use crate::model::qwen::v3::main::layer::Qwen3MainLayerScratch;
use crate::model::qwen::v3::main::output::Qwen3GatherUnembed;
use crate::model::qwen::v3::main::text_embed::Qwen3MainTextEmbed;
use crate::model::qwen::v3_asr::Qwen3ASRAudioProcessor;
use crate::model::resource_arena::MetalResourceArena;
use crate::model::resource_arena::new_resource_arena;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedConfig;
use crate::replay::Replay;
use crate::sampling::sampling_params::SamplingParamsStore;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3ExecutorConfig {
    pub max_requests: usize,
    pub max_tokens: usize,
    pub max_tokens_per_request: usize,
    pub num_cache_pages: usize,
    pub num_tokens_per_block: usize,
}

impl Qwen3ExecutorConfig {
    pub fn validate(self) {
        assert!(self.max_requests > 0, "qwen3 replay executor requires max_requests > 0");
        assert!(self.max_tokens > 0, "qwen3 replay executor requires max_tokens > 0");
        assert!(
            self.max_tokens_per_request > 0,
            "qwen3 replay executor requires max_tokens_per_request > 0"
        );
        assert!(
            self.num_cache_pages > 0,
            "qwen3 replay executor requires num_cache_pages > 0"
        );
        assert!(
            self.num_tokens_per_block > 0,
            "qwen3 replay executor requires num_tokens_per_block > 0"
        );
        assert!(
            u32::try_from(self.max_requests).is_ok(),
            "qwen3 max_requests must fit the u32 request-slot domain"
        );
        assert!(
            i32::try_from(self.max_tokens).is_ok(),
            "qwen3 max_tokens must fit the i32 flattened-token domain"
        );
        assert!(
            u32::try_from(self.max_tokens_per_request).is_ok(),
            "qwen3 max_tokens_per_request must fit the u32 position domain"
        );
        assert!(
            u32::try_from(self.num_tokens_per_block).is_ok(),
            "qwen3 num_tokens_per_block must fit the u32 cache-block domain"
        );
        assert!(
            u32::try_from(self.num_cache_pages - 1).is_ok(),
            "qwen3 cache page IDs must fit u32"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen3ModelLayout {
    pub max_tokens: u32,
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub scale_bias_dtype: Dtype,
    pub hidden_dtype: Dtype,
    pub rms_norm_eps: f32,
}

impl Qwen3ModelLayout {
    fn from_main_config(config: Qwen3MainConfig<'_>, max_tokens: usize) -> Result<Self, ModelExecutorError> {
        let text = config.text;
        let quant = config.quantization;
        Ok(Self {
            max_tokens: max_tokens as u32,
            vocab_size: text
                .vocab_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3 vocab_size must fit u32"))?,
            hidden_dim: text
                .hidden_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3 hidden_size must fit u32"))?,
            group_size: quant
                .group_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3 quantization group_size must fit u32"))?,
            bits: quant
                .bits
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3 quantization bits must fit u32"))?,
            scale_bias_dtype: Dtype::Bfloat16,
            hidden_dtype: Dtype::Bfloat16,
            rms_norm_eps: text.rms_norm_eps,
        })
    }

    fn validate(self) {
        assert!(self.max_tokens > 0, "qwen3 replay model requires positive max_tokens");
        assert!(self.vocab_size > 0, "qwen3 replay model requires positive vocab_size");
        assert!(self.hidden_dim > 0, "qwen3 replay model requires positive hidden_dim");
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.hidden_dim % self.group_size, 0);
        assert!(matches!(self.scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert_eq!(self.hidden_dtype, Dtype::Bfloat16);
        assert!(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0);
        i32::try_from(self.vocab_size).expect("qwen3 vocab index must fit i32");
        i32::try_from(self.hidden_dim).expect("qwen3 hidden dimension must fit i32");
        self.max_tokens
            .checked_mul(self.hidden_dim)
            .expect("qwen3 flattened hidden tensor index must fit u32");
    }

    fn embedding_config(self) -> EmbedConfig {
        EmbedConfig {
            max_tokens: self.max_tokens,
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            group_size: self.group_size,
            bits: self.bits,
            scale_bias_dtype: self.scale_bias_dtype,
            output_dtype: self.hidden_dtype,
        }
    }

    fn unembed_config(self, max_sampling_inputs: u32) -> UnembedConfig {
        UnembedConfig {
            max_tokens: max_sampling_inputs,
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            group_size: self.group_size,
            bits: self.bits,
            input_dtype: self.hidden_dtype,
            output_dtype: self.hidden_dtype,
            scale_bias_dtype: self.scale_bias_dtype,
        }
    }

    fn hidden_bytes(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(self.hidden_dim as usize)
            .and_then(|elements| elements.checked_mul(self.hidden_dtype.item_size()))
            .expect("qwen3 hidden buffer byte length must fit usize")
    }

    fn token_id_bytes(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(size_of::<i32>())
            .expect("qwen3 token ID buffer byte length must fit usize")
    }
}

enum Qwen3InitMode {
    Vanilla,
    Asr {
        config: Box<Qwen3ASRModelConfig>,
        resource_arena: Arc<MetalResourceArena>,
    },
}

pub fn init_qwen_3_model(
    model_dir: impl AsRef<Path>,
    config: Qwen3ExecutorConfig,
) -> Result<Qwen3Executor, ModelExecutorError> {
    config.validate();
    init_qwen_3_model_inner(model_dir.as_ref(), Qwen3InitMode::Vanilla, config)
}

pub struct Qwen3ASRLoaded {
    pub executor: Qwen3Executor,
    pub audio_processor: Arc<Qwen3ASRAudioProcessor>,
    pub resource_arena: Arc<MetalResourceArena>,
}

pub fn init_qwen3_asr_model(
    model_dir: impl AsRef<Path>,
    config: Qwen3ExecutorConfig,
) -> Result<Qwen3ASRLoaded, ModelExecutorError> {
    config.validate();
    let model_dir = model_dir.as_ref();
    let model_config = init_qwen3_asr_config(model_dir)?;
    let store = SafeTensorStore::from_model_dir(model_dir)?;
    let bindings = resolve_qwen3_asr_weight_bindings(&model_config, store.index().tensor_names())?;
    let max_resource_tokens = audio_output_rows(model_config.preprocessor.max_frames);
    let hidden_dim_bytes = model_config
        .text
        .hidden_size
        .checked_mul(Dtype::Bfloat16.item_size())
        .ok_or_else(|| ModelExecutorError::custom("Qwen3-ASR hidden row byte length must fit usize"))?;
    let arena_capacity_bytes = config
        .max_requests
        .checked_mul(max_resource_tokens)
        .and_then(|tokens| tokens.checked_mul(hidden_dim_bytes))
        .ok_or_else(|| ModelExecutorError::custom("Qwen3-ASR resource arena capacity must fit usize"))?;
    let resource_arena = Arc::new(new_resource_arena(&Device::system_default(), arena_capacity_bytes));
    let audio_processor =
        Qwen3ASRAudioProcessor::load(model_dir, &model_config, &bindings, Arc::clone(&resource_arena))?;
    let executor = init_qwen_3_model_inner(
        model_dir,
        Qwen3InitMode::Asr {
            config: Box::new(model_config),
            resource_arena: Arc::clone(&resource_arena),
        },
        config,
    )?;
    Ok(Qwen3ASRLoaded {
        executor,
        audio_processor,
        resource_arena,
    })
}

fn init_qwen_3_model_inner(
    model_dir: &Path,
    init_mode: Qwen3InitMode,
    config: Qwen3ExecutorConfig,
) -> Result<Qwen3Executor, ModelExecutorError> {
    let checkpoint = match &init_mode {
        Qwen3InitMode::Asr { config, .. } => Qwen3Checkpoint::Asr(config.clone()),
        Qwen3InitMode::Vanilla => Qwen3Checkpoint::Text(init_qwen3_model_config(model_dir)?),
    };
    let eos_token_ids = match &checkpoint {
        Qwen3Checkpoint::Text(model_config) => {
            let generation_config = HFGenerationConfig::load(model_dir)?;
            if generation_config.eos_token_ids().is_empty() {
                model_config.eos_token_ids().to_vec()
            } else {
                generation_config.eos_token_ids().to_vec()
            }
        },
        Qwen3Checkpoint::Asr(model_config) => model_config.generation.eos_token_ids.clone(),
    };
    let default_stop_sequences = eos_token_ids
        .iter()
        .map(|&token_id| vec![Token::new(token_id)])
        .collect();

    let text = checkpoint.text();
    let quantization = match &checkpoint {
        Qwen3Checkpoint::Text(model_config) => {
            model_config
                .quantization
                .as_ref()
                .ok_or_else(|| ModelExecutorError::custom("qwen3 replay model requires quantization config"))?
        },
        Qwen3Checkpoint::Asr(model_config) => &model_config.quantization,
    };
    let main_config = Qwen3MainConfig {
        text,
        quantization,
        page_size_bytes: QWEN3_PAGE_SIZE_BYTES,
    };
    let layout = Qwen3ModelLayout::from_main_config(main_config, config.max_tokens)?;
    layout.validate();

    let (main_gqa_core, main_gqa_metal) = derive_qwen3_gqa_configs(0, main_config)?;
    let gqa_tokens_per_page = main_gqa_metal.num_ungated_tokens_per_page(&main_gqa_core) as usize;
    let main_page_ids_per_block = num_page_ids_per_block(config.num_tokens_per_block, gqa_tokens_per_page);
    let gqa_page_table_layout = GQAPageTableLayout {
        num_req_slots: config.max_requests as u32,
        num_blocks: text
            .max_position_embeddings
            .div_ceil(config.num_tokens_per_block)
            .max(1)
            .try_into()
            .expect("qwen3 GQA block capacity must fit u32"),
        num_gqa_layers: text
            .num_hidden_layers
            .try_into()
            .expect("qwen3 GQA layer count must fit u32"),
        num_page_ids_per_block: main_page_ids_per_block as u32,
    };
    let device = Device::system_default();
    let runtime = MetalRuntime::new(device.clone());
    let mut store = SafeTensorStore::from_model_dir(model_dir)?;
    let (embed_bindings, main_bindings, unembed_bindings) = match &checkpoint {
        Qwen3Checkpoint::Text(model_config) => {
            let Qwen3ModelWeightBindings { embed, main, unembed } =
                resolve_qwen3_model_weight_bindings(model_config, store.index().tensor_names())?;
            (embed, main, Some(unembed))
        },
        Qwen3Checkpoint::Asr(model_config) => {
            let bindings = resolve_qwen3_asr_weight_bindings(model_config, store.index().tensor_names())?;
            (bindings.embed, bindings.text, None)
        },
    };
    let main_gqa_state = Qwen3MainGQAState::new(
        &device,
        main_gqa_core,
        main_gqa_metal,
        gqa_page_table_layout,
        config.max_tokens,
        config.num_cache_pages,
    );
    let layer_scratch = Rc::new(Qwen3MainLayerScratch::new(
        &device,
        config.max_tokens,
        layout.hidden_dim as usize,
    ));
    let (dense_core, dense_metal) = derive_qwen3_dense_mlp_configs(0, main_config)?;
    let dense_scratch = Rc::new(DenseMLPScratch::new(
        &device,
        &dense_core,
        dense_metal.io_dtype,
        config.max_tokens,
    ));

    let mut embed = Embed::new(&device, layout.embedding_config());
    embed.load_weights(&device, &mut store, embed_bindings)?;
    let embed = Rc::new(embed);
    let token_hidden_input = Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes()));
    let hidden_output = Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes()));
    // Only the last token of each request requires vocabulary projection and sampling.
    let max_sampling_inputs = config.max_requests.min(config.max_tokens) as u32;
    let unembed_config = layout.unembed_config(max_sampling_inputs);
    let mut unembed = Unembed::new(&device, unembed_config);
    match unembed_bindings {
        Some(unembed_bindings) => unembed.load_weights(&device, &mut store, unembed_bindings)?,
        None => unembed.load_tied_weights(embed.tied_weights()),
    }
    let gather_unembed = Qwen3GatherUnembed::new(&device, unembed_config.hidden_dim, Rc::new(unembed));
    let sampler_bounds = TopKSamplingBounds {
        max_sampling_inputs,
        vocab_size: layout.vocab_size,
        top_k: MAX_TOP_K as u32,
    };
    sampler_bounds.validate();
    let sampling_params = Rc::new(SamplingParamsStore::new(
        &device,
        sampler_bounds,
        config.max_requests as u32,
    ));
    let mut main = Qwen3Main::new(&device, main_config, &main_gqa_state, layer_scratch, &dense_scratch)?;
    main.load_weights(&device, &mut store, text, main_bindings)?;
    drop(store);

    let sampler = Rc::new(TopKSampling::new(&device, sampling_params));
    let main_resource_embed = match &init_mode {
        Qwen3InitMode::Vanilla => None,
        Qwen3InitMode::Asr { resource_arena, .. } => {
            Some(MainResourceEmbed::new(
                &device,
                inference_backend_metal::components::resource_embed::Config {
                    hidden_dim: layout.hidden_dim,
                    io_dtype: layout.hidden_dtype,
                },
                Arc::clone(resource_arena),
                config.max_tokens,
            ))
        },
    };
    let model_name = match &checkpoint {
        Qwen3Checkpoint::Text(_) => "qwen3",
        Qwen3Checkpoint::Asr(_) => "qwen3_asr",
    };
    Ok(Qwen3Executor {
        model_name: model_name.to_string(),
        model_dir: model_dir.to_path_buf(),
        checkpoint,
        default_stop_sequences,
        config,
        runtime,
        layout,
        token_ids: Buffer::new_zeroed(&device, layout.token_id_bytes()),
        token_hidden_input,
        hidden_output,
        gather_flat_indices: Buffer::new_zeroed_elements(&device, max_sampling_inputs, Dtype::Uint32),
        unembed_hidden: Buffer::new_zeroed_elements(&device, max_sampling_inputs * layout.hidden_dim, Dtype::Bfloat16),
        unembed_logits: Buffer::new_zeroed(&device, unembed_config.logits_bytes()),
        main_text_embed: Replay::new("qwen3 MainTextEmbed", Qwen3MainTextEmbed::new(embed)),
        main_resource_embed,
        main: Replay::new("qwen3 Main", main),
        gather_unembed: Replay::new("qwen3 GatherUnembed", gather_unembed),
        sampling: Replay::new("qwen3 sampling", Sampling::new(Rc::clone(&sampler))),
        sampler,
        sampler_output: TopKSamplingOutputBuffers::new(&device, sampler_bounds),
        request_sampling: RequestSamplingState::new(config.max_requests),
        main_gqa_state,
        pages: PageArena::new(&device, config.num_cache_pages, QWEN3_PAGE_SIZE_BYTES),
        pending_transactions: Qwen3PendingTransactions::new(),
        gqa_page_table_layout,
        unloaded_embed: None,
        unloaded_unembed: None,
    })
}
