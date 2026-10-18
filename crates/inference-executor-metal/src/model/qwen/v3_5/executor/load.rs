use std::mem::size_of;
use std::num::NonZeroUsize;
use std::path::Path;
use std::rc::Rc;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::QWEN35_PAGE_SIZE_BYTES;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::Qwen35PendingTransactions;
use inference_executor_core::model::qwen::v3_5::init_qwen35_model_config;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MTPWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::resolve_qwen35_model_weight_bindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::resolve_qwen35_mtp_weight_bindings;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2MainConfig;
use inference_executor_core::model::qwen::v3_x::dflash2::init_qwen3x_dflash2_config;
use inference_executor_core::sampling::HFGenerationConfig;
use inference_executor_core::sampling::MAX_TOP_K;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::TopKSamplingBounds;
use inference_runtime_core::runtime::Token;

use crate::attn::gdn::backend::GDN;
use crate::attn::gdn::state_table::GDNStateCapacity;
use crate::checkpoint::SafeTensorStore;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedConfig;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::page_arena::PageArena;
use crate::model::qwen::v3_5::component_config::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::component_config::derive_qwen35_dense_mlp_configs;
use crate::model::qwen::v3_5::component_config::derive_qwen35_gdn_configs;
use crate::model::qwen::v3_5::component_config::derive_qwen35_gqa_configs;
use crate::model::qwen::v3_5::component_config::derive_qwen35_moe_configs;
use crate::model::qwen::v3_5::component_config::qwen35_layer_counts;
use crate::model::qwen::v3_5::component_config::validate_qwen35_mtp_config;
use crate::model::qwen::v3_5::executor::Qwen35DFlash2Speculator;
use crate::model::qwen::v3_5::executor::Qwen35Executor;
use crate::model::qwen::v3_5::executor::Qwen35MTPExecution;
use crate::model::qwen::v3_5::executor::Qwen35MTPSpeculator;
use crate::model::qwen::v3_5::executor::Qwen35SpeculativeResources;
use crate::model::qwen::v3_5::executor::Qwen35Speculator;
use crate::model::qwen::v3_5::executor::Qwen35WeightSource;
use crate::model::qwen::v3_5::executor::num_page_ids_per_block;
use crate::model::qwen::v3_5::main::Qwen35Main;
use crate::model::qwen::v3_5::main::layer::Qwen35MainLayerScratch;
use crate::model::qwen::v3_5::main::output::Qwen35GatherUnembed;
use crate::model::qwen::v3_5::main::text_embed::Qwen35MainTextEmbed;
use crate::model::qwen::v3_5::mtp::Qwen35MTP;
use crate::model::qwen::v3_5::mtp::embed::Qwen35MTPEmbed;
use crate::model::qwen::v3_5::mtp::hidden_state_cache::Qwen35MTPHiddenStateCache;
use crate::model::qwen::v3_5::mtp::hidden_state_transfer::Qwen35MTPHiddenStateTransfer;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPLayerScratch;
use crate::model::qwen::v3_x::dflash2::execution::Qwen3xDFlash2Execution;
use crate::model::qwen::v3_x::dflash2::load::Qwen3xDFlash2LoadConfig;
use crate::model::qwen::v3_x::dflash2::load::Qwen3xDFlash2Loaded;
use crate::model::qwen::v3_x::dflash2::load::load_qwen3x_dflash2;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedConfig;
use crate::replay::Replay;
use crate::sampling::rejection_replay::RejectionSampler;
use crate::sampling::rejection_replay::RejectionSampling;
use crate::sampling::sampling_params::SamplingParamsStore;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_replay::DraftSampling;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use crate::trace;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35ExecutorConfig {
    pub max_requests: usize,
    pub max_tokens: usize,
    pub max_tokens_per_request: usize,
    pub num_cache_pages: usize,
    pub num_tokens_per_block: usize,
}

impl Qwen35ExecutorConfig {
    pub fn validate(self) {
        assert!(
            self.max_requests > 0,
            "qwen3.5 replay executor requires max_requests > 0"
        );
        assert!(self.max_tokens > 0, "qwen3.5 replay executor requires max_tokens > 0");
        assert!(
            self.max_tokens_per_request > 0,
            "qwen3.5 replay executor requires max_tokens_per_request > 0"
        );
        assert!(
            self.num_cache_pages > 0,
            "qwen3.5 replay executor requires num_cache_pages > 0"
        );
        assert!(
            self.num_tokens_per_block > 0,
            "qwen3.5 replay executor requires num_tokens_per_block > 0"
        );
        assert!(
            u32::try_from(self.max_requests).is_ok(),
            "qwen3.5 max_requests must fit the u32 request-slot domain"
        );
        assert!(
            i32::try_from(self.max_tokens).is_ok(),
            "qwen3.5 max_tokens must fit the i32 flattened-token domain"
        );
        assert!(
            u32::try_from(self.max_tokens_per_request).is_ok(),
            "qwen3.5 max_tokens_per_request must fit the u32 position/state-version domain"
        );
        assert!(
            u32::try_from(self.num_tokens_per_block).is_ok(),
            "qwen3.5 num_tokens_per_block must fit the u32 cache-block domain"
        );
        assert!(
            u32::try_from(self.num_cache_pages - 1).is_ok(),
            "qwen3.5 cache page IDs must fit u32"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35ModelLayout {
    pub max_tokens: u32,
    pub vocab_size: u32,
    pub hidden_dim: u32,
    pub group_size: u32,
    pub bits: u32,
    pub scale_bias_dtype: Dtype,
    pub hidden_dtype: Dtype,
    pub rms_norm_eps: f32,
}

impl Qwen35ModelLayout {
    fn from_model_config(model_config: &Qwen35ModelConfig, max_tokens: usize) -> Result<Self, ModelExecutorError> {
        let text = &model_config.text_config;
        let quant = model_config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("qwen3.5 replay model requires quantization config"))?;
        Ok(Self {
            max_tokens: max_tokens as u32,
            vocab_size: text
                .vocab_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 vocab_size must fit u32"))?,
            hidden_dim: text
                .hidden_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 hidden_size must fit u32"))?,
            group_size: quant
                .group_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 quantization group_size must fit u32"))?,
            bits: quant
                .bits
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 quantization bits must fit u32"))?,
            scale_bias_dtype: Dtype::Bfloat16,
            hidden_dtype: Dtype::Bfloat16,
            rms_norm_eps: text.rms_norm_eps,
        })
    }

    fn validate(self) {
        assert!(self.max_tokens > 0, "qwen3.5 replay model requires positive max_tokens");
        assert!(self.vocab_size > 0, "qwen3.5 replay model requires positive vocab_size");
        assert!(self.hidden_dim > 0, "qwen3.5 replay model requires positive hidden_dim");
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.hidden_dim % self.group_size, 0);
        assert!(matches!(self.scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert_eq!(self.hidden_dtype, Dtype::Bfloat16);
        assert!(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0);
        i32::try_from(self.vocab_size).expect("qwen3.5 vocab index must fit i32");
        i32::try_from(self.hidden_dim).expect("qwen3.5 hidden dimension must fit i32");
        i32::try_from(self.group_size).expect("qwen3.5 quantization group size must fit i32");
        i32::try_from(self.bits).expect("qwen3.5 quantization bits must fit i32");
        self.max_tokens
            .checked_mul(self.hidden_dim)
            .expect("qwen3.5 flattened hidden tensor index must fit u32");
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
            .expect("qwen3.5 hidden buffer byte length must fit usize")
    }

    fn token_id_bytes(self) -> usize {
        (self.max_tokens as usize)
            .checked_mul(size_of::<i32>())
            .expect("qwen3.5 token ID buffer byte length must fit usize")
    }
}

#[allow(clippy::upper_case_acronyms)]
enum Qwen35InitMode<'a> {
    Vanilla,
    MTP {
        model_dir: &'a Path,
        config: Box<Qwen35ModelConfig>,
        num_spec_tokens: NonZeroUsize,
    },
    DFlash2 {
        model_dir: &'a Path,
        config: Box<Qwen3xDFlash2Config>,
        num_spec_tokens: NonZeroUsize,
    },
}

struct Qwen35MTPLoad {
    config: Qwen35ModelConfig,
    store: SafeTensorStore,
    bindings: Qwen35MTPWeightBindings,
    gqa_state: Qwen3xGQAState,
    num_spec_tokens: NonZeroUsize,
}

#[allow(clippy::upper_case_acronyms)]
enum Qwen35SpecSource<'a> {
    Vanilla,
    MTP(Box<Qwen35MTPLoad>),
    DFlash2 {
        model_dir: &'a Path,
        config: Box<Qwen3xDFlash2Config>,
        num_spec_tokens: NonZeroUsize,
    },
}

#[allow(clippy::upper_case_acronyms)]
enum Qwen35SpecLoad {
    Vanilla,
    MTP(Box<Qwen35MTPLoad>),
    DFlash2(Box<Qwen3xDFlash2Loaded>),
}

fn qwen35_gdn_state_capacity(
    max_tokens_per_request: usize,
    num_tokens_per_block: usize,
    max_spec_tokens: usize,
) -> GDNStateCapacity {
    assert!(max_tokens_per_request > 0, "qwen3.5 GDN state requires request tokens");
    assert!(num_tokens_per_block > 0, "qwen3.5 GDN state requires tokens per block");
    let max_publish_jobs_per_req = max_tokens_per_request.div_ceil(num_tokens_per_block);
    // Only the accepted final state and crossed cache boundaries need full slots.
    let max_materialized_states_per_req = max_publish_jobs_per_req
        .checked_add(1)
        .expect("qwen3.5 GDN materialized-state count must fit usize");
    let num_state_slots_per_req = max_materialized_states_per_req
        .checked_add(1)
        .expect("qwen3.5 GDN state-slot count must fit usize");
    GDNStateCapacity::new(
        num_state_slots_per_req,
        max_materialized_states_per_req,
        max_publish_jobs_per_req,
        GDN::max_replay_tokens_per_request(max_spec_tokens),
    )
}

pub fn init_qwen_3_5_model(
    model_dir: impl AsRef<Path>,
    config: Qwen35ExecutorConfig,
) -> Result<Qwen35Executor, ModelExecutorError> {
    init_qwen_3_5_model_inner(model_dir.as_ref(), Qwen35InitMode::Vanilla, config)
}

pub fn init_qwen_3_5_model_with_mtp(
    model_dir: impl AsRef<Path>,
    mtp_model_dir: impl AsRef<Path>,
    num_spec_tokens: NonZeroUsize,
    mut config: Qwen35ExecutorConfig,
) -> Result<Qwen35Executor, ModelExecutorError> {
    // Runtime retains the Main token budget. Shared Main/MTP workspaces must
    // also fit a mixed batch whose Decode requests submit no drafts.
    config.max_tokens = qwen35_mtp_token_capacity(config.max_tokens, config.max_requests, num_spec_tokens);
    let mtp_model_dir = mtp_model_dir.as_ref();
    let mtp_config = init_qwen35_model_config(mtp_model_dir)?;
    init_qwen_3_5_model_inner(
        model_dir.as_ref(),
        Qwen35InitMode::MTP {
            model_dir: mtp_model_dir,
            config: Box::new(mtp_config),
            num_spec_tokens,
        },
        config,
    )
}

fn qwen35_mtp_token_capacity(max_tokens: usize, max_requests: usize, num_spec_tokens: NonZeroUsize) -> usize {
    // Ordinary Prefill has equal Main/MTP row counts. Decode or tail-repair
    // Prefill can add at most K - 1 rows per request beyond the Main input.
    // Every request consumes at least one Main row.
    let extra_rows = max_requests
        .min(max_tokens)
        .checked_mul(num_spec_tokens.get() - 1)
        .expect("qwen3.5 MTP extra token capacity must fit usize");
    max_tokens
        .checked_add(extra_rows)
        .expect("qwen3.5 MTP token capacity must fit usize")
}

pub fn init_qwen_3_5_model_with_dflash2(
    model_dir: impl AsRef<Path>,
    dflash2_model_dir: impl AsRef<Path>,
    num_spec_tokens: NonZeroUsize,
    config: Qwen35ExecutorConfig,
) -> Result<Qwen35Executor, ModelExecutorError> {
    let dflash2_model_dir = dflash2_model_dir.as_ref();
    let dflash2_config = init_qwen3x_dflash2_config(dflash2_model_dir)?;
    if num_spec_tokens.get() >= dflash2_config.max_position_embeddings
        || num_spec_tokens.get() >= dflash2_config.sliding_window - 1
    {
        return Err(ModelExecutorError::custom(
            "DFlash2 num_spec_tokens must be smaller than max_position_embeddings and sliding_window - 1",
        ));
    }
    init_qwen_3_5_model_inner(
        model_dir.as_ref(),
        Qwen35InitMode::DFlash2 {
            model_dir: dflash2_model_dir,
            config: Box::new(dflash2_config),
            num_spec_tokens,
        },
        config,
    )
}

fn init_qwen_3_5_model_inner(
    model_dir: &Path,
    init_mode: Qwen35InitMode<'_>,
    config: Qwen35ExecutorConfig,
) -> Result<Qwen35Executor, ModelExecutorError> {
    config.validate();
    let weight_source = match &init_mode {
        Qwen35InitMode::Vanilla => Qwen35WeightSource::Vanilla,
        Qwen35InitMode::MTP { model_dir, config, .. } => {
            Qwen35WeightSource::MTP {
                model_dir: model_dir.to_path_buf(),
                config: config.clone(),
            }
        },
        Qwen35InitMode::DFlash2 { model_dir, config, .. } => {
            Qwen35WeightSource::DFlash2 {
                model_dir: model_dir.to_path_buf(),
                config: config.clone(),
            }
        },
    };
    if let Qwen35InitMode::MTP { num_spec_tokens, .. } = &init_mode {
        assert!(
            config.max_tokens_per_request >= num_spec_tokens.get(),
            "qwen3.5 MTP requires max_tokens_per_request >= num_spec_tokens"
        );
    }
    let model_config = init_qwen35_model_config(model_dir)?;
    match &init_mode {
        Qwen35InitMode::Vanilla => {},
        Qwen35InitMode::MTP { config: mtp_config, .. } => validate_qwen35_mtp_config(&model_config, mtp_config)?,
        Qwen35InitMode::DFlash2 {
            config: dflash2_config, ..
        } => {
            let text = &model_config.text_config;
            dflash2_config.validate_main(Qwen3xDFlash2MainConfig {
                hidden_size: text.hidden_size,
                num_hidden_layers: text.num_hidden_layers,
                vocab_size: text.vocab_size,
                max_position_embeddings: text.max_position_embeddings,
                rope_theta: text.rope_theta,
            })?;
        },
    }
    let generation_config = HFGenerationConfig::load(model_dir)?;
    let sampler_config = generation_config.sampler();
    let default_stop_sequences = generation_config
        .eos_token_ids()
        .iter()
        .map(|&token_id| vec![Token::new(token_id)])
        .collect();
    let device = Device::system_default();
    let runtime = MetalRuntime::new(device.clone());
    let mut store = SafeTensorStore::from_model_dir(model_dir)?;
    let weight_bindings = resolve_qwen35_model_weight_bindings(&model_config, store.index().tensor_names())?;
    let layer_counts = qwen35_layer_counts(&model_config)?;
    assert!(layer_counts.gqa > 0, "qwen3.5 Main requires at least one GQA layer");
    assert!(layer_counts.gdn > 0, "qwen3.5 Main requires at least one GDN layer");
    let metal_defaults = Qwen35MetalDefaults::from_quantization(model_config.quantization.as_ref())?;
    let layout = Qwen35ModelLayout::from_model_config(&model_config, config.max_tokens)?;
    layout.validate();
    let max_spec_tokens = match &init_mode {
        Qwen35InitMode::Vanilla => 0,
        Qwen35InitMode::MTP { num_spec_tokens, .. } | Qwen35InitMode::DFlash2 { num_spec_tokens, .. } => {
            num_spec_tokens.get()
        },
    };
    // Output rows contain one target per draft and one final target per request.
    // Prefill workspace rows do not require vocabulary projection or sampling.
    let max_sampling_inputs = config
        .max_requests
        .checked_mul(
            max_spec_tokens
                .checked_add(1)
                .expect("qwen3.5 target rows per request must fit usize"),
        )
        .expect("qwen3.5 sampling row capacity must fit usize")
        .min(config.max_tokens) as u32;
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
    trace::qwen35_state(|| {
        format!(
            "event=sampler_config temperature={} top_k={} top_p={} bounds_top_k={} max_sampling_inputs={} \
             vocab_size={}",
            sampler_config.temperature,
            sampler_config.top_k,
            sampler_config.top_p,
            sampler_bounds.top_k,
            sampler_bounds.max_sampling_inputs,
            sampler_bounds.vocab_size
        )
    });
    let unembed_config = layout.unembed_config(max_sampling_inputs);
    let first_gqa_layer = (0..model_config.text_config.num_hidden_layers)
        .find(|&index| {
            model_config
                .layer_type_at(index)
                .is_ok_and(|kind| kind == inference_executor_core::model::qwen::v3_5::LayerType::FullAttention)
        })
        .expect("qwen3.5 Main requires a GQA layer");
    let (main_gqa_core, main_gqa_metal) =
        derive_qwen35_gqa_configs(first_gqa_layer, &model_config.text_config, metal_defaults)?;
    let gqa_tokens_per_page = main_gqa_metal.num_tokens_per_page(&main_gqa_core) as usize;
    let main_page_ids_per_block = num_page_ids_per_block(config.num_tokens_per_block, gqa_tokens_per_page);
    let main_gqa_page_table_layout = GQAPageTableLayout {
        num_req_slots: config.max_requests as u32,
        num_blocks: model_config
            .text_config
            .max_position_embeddings
            .div_ceil(config.num_tokens_per_block)
            .max(1)
            .try_into()
            .expect("qwen3.5 GQA block capacity must fit u32"),
        num_gqa_layers: layer_counts
            .gqa
            .try_into()
            .expect("qwen3.5 GQA layer count must fit u32"),
        num_page_ids_per_block: main_page_ids_per_block as u32,
    };
    let gqa_page_table_layout = main_gqa_page_table_layout;
    let main_gqa_state = Qwen3xGQAState::new(
        &device,
        main_gqa_core,
        main_gqa_metal,
        gqa_page_table_layout,
        config.max_tokens,
        config.num_cache_pages,
    );
    let spec_source = match init_mode {
        Qwen35InitMode::Vanilla => Qwen35SpecSource::Vanilla,
        Qwen35InitMode::MTP {
            model_dir: mtp_model_dir,
            config: mtp_model_config,
            num_spec_tokens,
        } => {
            let mtp_store = SafeTensorStore::from_model_dir(mtp_model_dir)?;
            let mtp_weight_bindings =
                resolve_qwen35_mtp_weight_bindings(&mtp_model_config, mtp_store.index().tensor_names())?;
            let (mtp_gqa_core, mtp_gqa_metal) = derive_qwen35_gqa_configs(
                model_config.text_config.num_hidden_layers,
                &mtp_model_config.text_config,
                Qwen35MetalDefaults::from_quantization(mtp_model_config.quantization.as_ref())?,
            )?;
            let mtp_gqa_page_table_layout = GQAPageTableLayout {
                num_req_slots: config.max_requests as u32,
                num_blocks: model_config
                    .text_config
                    .max_position_embeddings
                    .div_ceil(config.num_tokens_per_block)
                    .max(1)
                    .try_into()
                    .expect("qwen3.5 MTP GQA block capacity must fit u32"),
                num_gqa_layers: num_spec_tokens.get() as u32,
                num_page_ids_per_block: num_page_ids_per_block(
                    config.num_tokens_per_block,
                    mtp_gqa_metal.num_tokens_per_page(&mtp_gqa_core) as usize,
                ) as u32,
            };
            let mtp_gqa_state = Qwen3xGQAState::new(
                &device,
                mtp_gqa_core,
                mtp_gqa_metal,
                mtp_gqa_page_table_layout,
                config.max_tokens,
                config.num_cache_pages,
            );
            Qwen35SpecSource::MTP(Box::new(Qwen35MTPLoad {
                config: *mtp_model_config,
                store: mtp_store,
                bindings: mtp_weight_bindings,
                gqa_state: mtp_gqa_state,
                num_spec_tokens,
            }))
        },
        Qwen35InitMode::DFlash2 {
            model_dir: dflash2_model_dir,
            config: dflash2_config,
            num_spec_tokens,
        } => {
            Qwen35SpecSource::DFlash2 {
                model_dir: dflash2_model_dir,
                config: dflash2_config,
                num_spec_tokens,
            }
        },
    };
    let gdn_layers = (0..model_config.text_config.num_hidden_layers)
        .filter(|&index| {
            model_config
                .layer_type_at(index)
                .is_ok_and(|kind| kind == inference_executor_core::model::qwen::v3_5::LayerType::GDN)
        })
        .collect::<Vec<_>>();
    let gdn_cores = gdn_layers
        .iter()
        .map(|&index| derive_qwen35_gdn_configs(index, &model_config.text_config, metal_defaults).map(|pair| pair.0))
        .collect::<Result<Vec<_>, _>>()?;
    let (_, gdn_metal) = derive_qwen35_gdn_configs(gdn_layers[0], &model_config.text_config, metal_defaults)?;
    let gdn_state_capacity = qwen35_gdn_state_capacity(
        config.max_tokens_per_request,
        config.num_tokens_per_block,
        max_spec_tokens,
    );
    let main_gdn_state = Qwen3xGDNState::new(
        &device,
        runtime.stream().new_on_same_queue(),
        &gdn_cores,
        gdn_metal,
        config.max_requests,
        gdn_state_capacity,
        config.max_tokens,
        config.num_tokens_per_block,
        config.num_cache_pages,
        QWEN35_PAGE_SIZE_BYTES,
    );
    let main_layer_scratch = std::rc::Rc::new(Qwen35MainLayerScratch::new(
        &device,
        config.max_tokens,
        layout.hidden_dim as usize,
    ));
    let mtp_uses_dense = matches!(
        &spec_source,
        Qwen35SpecSource::MTP(mtp) if !mtp.config.layer_uses_moe(0)
    );
    let dense_mlp_scratch = if layer_counts.has_dense_mlp || mtp_uses_dense {
        let source = (0..model_config.text_config.num_hidden_layers)
            .find(|&index| !model_config.layer_uses_moe(index))
            .map(|index| (&model_config, index))
            .or(match &spec_source {
                Qwen35SpecSource::MTP(mtp) => Some((&mtp.config, 0)),
                Qwen35SpecSource::Vanilla | Qwen35SpecSource::DFlash2 { .. } => None,
            })
            .expect("qwen3.5 dense scratch requires a dense layer");
        let defaults = Qwen35MetalDefaults::from_quantization(source.0.quantization.as_ref())?;
        let (core, metal) = derive_qwen35_dense_mlp_configs(source.1, &source.0.text_config, defaults)?;
        Some(std::rc::Rc::new(DenseMLPScratch::new(
            &device,
            &core,
            metal.io_dtype,
            config.max_tokens,
        )))
    } else {
        None
    };
    let mtp_uses_moe = matches!(
        &spec_source,
        Qwen35SpecSource::MTP(mtp) if mtp.config.layer_uses_moe(0)
    );
    let moe_scratch = if layer_counts.has_moe || mtp_uses_moe {
        let source = (0..model_config.text_config.num_hidden_layers)
            .find(|&index| model_config.layer_uses_moe(index))
            .map(|index| (&model_config, index))
            .or(match &spec_source {
                Qwen35SpecSource::MTP(mtp) => Some((&mtp.config, 0)),
                Qwen35SpecSource::Vanilla | Qwen35SpecSource::DFlash2 { .. } => None,
            })
            .expect("qwen3.5 MoE scratch requires an MoE layer");
        let defaults = Qwen35MetalDefaults::from_quantization(source.0.quantization.as_ref())?;
        let (core, metal) = derive_qwen35_moe_configs(&format!("layers.{}", source.1), source.1, source.0, defaults)?;
        Some(std::rc::Rc::new(MoEScratch::new(
            &device,
            &core,
            metal,
            config.max_tokens,
        )))
    } else {
        None
    };
    let inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35ModelWeightBindings {
        embed: embed_bindings,
        main: main_bindings,
        unembed: unembed_bindings,
    } = weight_bindings;
    let mut embed = Embed::new(&device, layout.embedding_config());
    embed.load_weights(&device, &mut store, embed_bindings)?;
    let embed = std::rc::Rc::new(embed);
    let token_hidden_input = Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes()));
    let hidden_output = Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes()));
    let mut unembed = Unembed::new(&device, unembed_config);
    unembed.load_weights(&device, &mut store, unembed_bindings)?;
    let gather_unembed = Qwen35GatherUnembed::new(&device, unembed_config.hidden_dim, Rc::new(unembed));
    assert_eq!(
        gather_unembed.max_rows() as usize,
        max_sampling_inputs as usize,
        "qwen3.5 GatherUnembed policy and executor output-row capacities must match"
    );
    let spec_load = match spec_source {
        Qwen35SpecSource::Vanilla => Qwen35SpecLoad::Vanilla,
        Qwen35SpecSource::MTP(mtp) => Qwen35SpecLoad::MTP(mtp),
        Qwen35SpecSource::DFlash2 {
            model_dir: dflash2_model_dir,
            config: dflash2_config,
            num_spec_tokens,
        } => {
            Qwen35SpecLoad::DFlash2(Box::new(load_qwen3x_dflash2(
                &device,
                dflash2_model_dir,
                &dflash2_config,
                Qwen3xDFlash2LoadConfig {
                    num_spec_tokens,
                    page_size_bytes: QWEN35_PAGE_SIZE_BYTES,
                    max_position_embeddings: model_config.text_config.max_position_embeddings,
                    max_requests: config.max_requests,
                    max_tokens: config.max_tokens,
                    num_cache_pages: config.num_cache_pages,
                    num_tokens_per_block: config.num_tokens_per_block,
                },
                Rc::clone(&embed),
                gather_unembed.unembed(),
                Rc::clone(&sampling_params),
            )?))
        },
    };
    let residual_capture = match &spec_load {
        Qwen35SpecLoad::Vanilla | Qwen35SpecLoad::MTP(_) => None,
        Qwen35SpecLoad::DFlash2(loaded) => {
            let capture: Rc<dyn MainResidualCapture> = loaded.model.main_feature_projector();
            Some(capture)
        },
    };
    let mut main = Qwen35Main::new(
        &device,
        &model_config,
        config.max_tokens,
        metal_defaults,
        &main_gqa_state,
        &main_gdn_state,
        residual_capture,
        std::rc::Rc::clone(&main_layer_scratch),
        dense_mlp_scratch.as_ref(),
        moe_scratch.as_ref(),
    )?;
    main.load_weights(&device, &mut store, &model_config, main_bindings)?;
    drop(store);
    let sampler = Rc::new(TopKSampling::new(&device, sampling_params));
    let speculative_resources = || {
        let rejection_sampler = Rc::new(RejectionSampler::new(
            &device,
            max_spec_tokens,
            config.max_requests,
            max_sampling_inputs as usize,
            sampler_bounds.top_k,
        ));
        Qwen35SpeculativeResources {
            rejection_sampling: Replay::new(
                "qwen3.5 rejection sampling",
                RejectionSampling::new(Rc::clone(&sampler), rejection_sampler),
            ),
            spec_probs: SpecProbsStore::new(
                &device,
                max_spec_tokens,
                config.max_requests,
                max_sampling_inputs as usize,
                sampler_bounds.top_k as usize,
            ),
            target_distribution_indices: Buffer::from_slice(&device, &(0..max_sampling_inputs).collect::<Vec<_>>()),
        }
    };
    let speculator = match spec_load {
        Qwen35SpecLoad::Vanilla => Qwen35Speculator::Vanilla,
        Qwen35SpecLoad::MTP(mtp_load) => {
            let Qwen35MTPLoad {
                config: mtp_model_config,
                mut store,
                bindings: mtp_bindings,
                gqa_state,
                num_spec_tokens,
            } = *mtp_load;
            let mtp_layer_scratch = Rc::new(Qwen35MTPLayerScratch::new(
                &device,
                config.max_tokens,
                layout.hidden_dim as usize,
            ));
            let Qwen35MTPWeightBindings {
                embed: mtp_embed_bindings,
                body,
                final_norm_weight,
            } = mtp_bindings;
            let mut mtp_embed = Qwen35MTPEmbed::new(&device, &mtp_model_config, Rc::clone(&embed), config.max_tokens)?;
            mtp_embed.load_weights(&device, &mut store, &mtp_model_config, mtp_embed_bindings)?;
            let mtp_defaults = Qwen35MetalDefaults::from_quantization(mtp_model_config.quantization.as_ref())?;
            let mut mtp = Qwen35MTP::new(
                &device,
                &model_config,
                &mtp_model_config,
                config.max_tokens,
                mtp_defaults,
                &gqa_state,
                config.num_cache_pages,
                mtp_layer_scratch,
                dense_mlp_scratch.as_ref(),
                moe_scratch.as_ref(),
            )?;
            mtp.load_weights(&device, &mut store, &mtp_model_config, body, final_norm_weight)?;
            Qwen35Speculator::MTP(Box::new(Qwen35MTPSpeculator {
                common: speculative_resources(),
                num_spec_tokens: num_spec_tokens.get(),
                hidden_input: Rc::new(Buffer::new_zeroed(&device, layout.hidden_bytes())),
                previous_hidden_routes: Buffer::new_zeroed_elements(&device, config.max_tokens * 2, Dtype::Uint32),
                hidden_state_cache_write_routes: Buffer::new_zeroed_elements(
                    &device,
                    config.max_tokens * 2,
                    Dtype::Uint32,
                ),
                draft_distribution_indices: Buffer::new_zeroed_elements(&device, config.max_requests, Dtype::Uint32),
                previous_hidden: Buffer::new_zeroed(&device, layout.hidden_bytes()),
                hidden_state_cache: Qwen35MTPHiddenStateCache::new(
                    &device,
                    config.max_requests,
                    num_spec_tokens.get(),
                    layout.hidden_dim as usize,
                ),
                hidden_state_transfers: (0..num_spec_tokens.get())
                    .map(|_| {
                        Replay::new(
                            "qwen3.5 MTP hidden-state transfer",
                            Qwen35MTPHiddenStateTransfer::new(&device, layout.hidden_dim as usize, config.max_tokens),
                        )
                    })
                    .collect(),
                embed: Replay::new("qwen3.5 MTPEmbed", mtp_embed),
                body: Replay::new("qwen3.5 MTP", mtp),
                sampling: Replay::new(
                    "qwen3.5 draft sampling",
                    DraftSampling::new(Rc::clone(&sampler), config.max_requests as u32),
                ),
                gqa_state,
                execution: Qwen35MTPExecution::new(config.max_requests, num_spec_tokens.get()),
            }))
        },
        Qwen35SpecLoad::DFlash2(loaded) => {
            Qwen35Speculator::DFlash2(Box::new(Qwen35DFlash2Speculator {
                common: speculative_resources(),
                execution: Qwen3xDFlash2Execution::new(&device, *loaded),
            }))
        },
    };
    let num_main_gqa_page_ids_per_block = (gqa_page_table_layout.num_gqa_layers as usize)
        .checked_mul(gqa_page_table_layout.num_page_ids_per_block as usize)
        .expect("qwen3.5 Main page IDs per block must fit usize");
    let num_speculator_gqa_page_ids_per_block = match &speculator {
        Qwen35Speculator::Vanilla | Qwen35Speculator::MTP(_) => 0,
        Qwen35Speculator::DFlash2(dflash2) => dflash2.execution.num_gqa_page_ids_per_block(),
    };
    let num_gqa_page_ids_per_main_lane_block = num_main_gqa_page_ids_per_block
        .checked_add(num_speculator_gqa_page_ids_per_block)
        .expect("qwen3.5 Main cache-lane page IDs per block must fit usize");
    let pages = PageArena::new(&device, config.num_cache_pages, QWEN35_PAGE_SIZE_BYTES);
    let model = Qwen35Executor {
        model_name: model_config.model_type.clone(),
        model_dir: model_dir.to_path_buf(),
        model_config,
        weight_source,
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
        main_text_embed: Replay::new("qwen3.5 MainTextEmbed", Qwen35MainTextEmbed::new(Rc::clone(&embed))),
        main_resource_embed: None,
        main: Replay::new("qwen3.5 Main", main),
        gather_unembed: Replay::new("qwen3.5 GatherUnembed", gather_unembed),
        sampling: Replay::new("qwen3.5 sampling", Sampling::new(Rc::clone(&sampler))),
        sampler: Rc::clone(&sampler),
        sampler_bounds,
        sampler_output: TopKSamplingOutputBuffers::new(&device, sampler_bounds),
        request_sampling: RequestSamplingState::new(config.max_requests),
        main_gqa_state,
        main_gdn_state,
        speculator,
        pages,
        pending_transactions: Qwen35PendingTransactions::new(),
        gqa_page_table_layout,
        num_gqa_page_ids_per_main_lane_block,
        unloaded_embed: None,
        unloaded_unembed: None,
    };
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdn_capacity_adds_current_commit_and_cache_boundary_states() {
        assert_eq!(qwen35_gdn_state_capacity(16, 8, 4), GDNStateCapacity::new(4, 3, 2, 12));
        assert_eq!(qwen35_gdn_state_capacity(7, 8, 0), GDNStateCapacity::new(3, 2, 1, 8));
    }

    #[test]
    fn test_mtp_token_capacity_covers_mixed_prefill_and_zero_draft_decode() {
        assert_eq!(qwen35_mtp_token_capacity(128, 4, NonZeroUsize::new(1).unwrap()), 128);
        assert_eq!(qwen35_mtp_token_capacity(128, 4, NonZeroUsize::new(3).unwrap()), 136);
        for max_tokens in 1..16 {
            for max_requests in 1..8 {
                for num_spec_tokens in 1..8 {
                    let capacity = qwen35_mtp_token_capacity(
                        max_tokens,
                        max_requests,
                        NonZeroUsize::new(num_spec_tokens).unwrap(),
                    );
                    for num_decode_reqs in 0..=max_tokens.min(max_requests) {
                        let prefill_rows = max_tokens - num_decode_reqs;
                        let mtp_rows = prefill_rows + num_decode_reqs * num_spec_tokens;
                        assert!(mtp_rows <= capacity);
                        let max_prefill_reqs = (max_requests - num_decode_reqs).min(prefill_rows);
                        let repaired_mtp_rows = mtp_rows + max_prefill_reqs * (num_spec_tokens - 1);
                        assert!(repaired_mtp_rows <= capacity);
                    }
                }
            }
        }
    }
}
