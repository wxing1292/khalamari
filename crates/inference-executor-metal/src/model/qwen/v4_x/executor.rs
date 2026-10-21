use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::ModelOutputTiming;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_core::model::qwen::v3_5::Qwen35DecodeDecision;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelBatchRequest;
use inference_executor_core::model::qwen::v3_5::Qwen35PendingTransactions;
use inference_executor_core::model::qwen::v3_5::Qwen35SampledTokens;
use inference_executor_core::model::qwen::v3_5::gather_flat_indices;
use inference_executor_core::model::qwen::v3_5::sample_decisions_from_sampled_tokens;
use inference_executor_core::model::qwen::v3_5::sample_req_slots;
use inference_executor_core::model::qwen::v3_5::sample_sampler_configs;
use inference_executor_core::model::qwen::v3_5::sample_token_positions;
use inference_executor_core::model::qwen::v3_5::to_core_batch_resp;
use inference_executor_core::model::qwen::v4_x::Qwen4xModelConfig;
use inference_executor_core::model::qwen::v4_x::init_qwen4x_model_config;
use inference_executor_core::model::qwen::v4_x::weight_layout::Qwen4xAttentionWeightBindings;
use inference_executor_core::model::qwen::v4_x::weight_layout::Qwen4xModelWeightBindings;
use inference_executor_core::model::qwen::v4_x::weight_layout::resolve_qwen4x_model_weight_bindings;
use inference_executor_core::sampling::HFGenerationConfig;
use inference_executor_core::sampling::MAX_TOP_K;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::SamplerConfig;
use inference_runtime_core::compute::BatchDevReq;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DevReq;
use inference_runtime_core::compute::ExecutorHibernationPlan;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::Token;

use crate::attn::gdn::backend::GDN;
use crate::attn::gdn::state_table::GDNStateCapacity;
use crate::checkpoint::SafeTensorStore;
use crate::def::submission::MetalReplaySubmission;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedConfig;
use crate::model::page_arena::PageArena;
use crate::model::qwen::v3_5::main::output::Qwen35GatherUnembed;
use crate::model::qwen::v3_5::main::output::Qwen35GatherUnembedArgs;
use crate::model::qwen::v3_5::main::output::Qwen35GatherUnembedReplayKey;
use crate::model::qwen::v3_5::main::text_embed::Qwen35MainTextEmbed;
use crate::model::qwen::v3_5::main::text_embed::Qwen35MainTextEmbedArgs;
use crate::model::qwen::v3_5::main::text_embed::Qwen35MainTextEmbedReplayKey;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v4_x::main::Qwen4xMain;
use crate::model::qwen::v4_x::main::Qwen4xMainArgs;
use crate::model::qwen::v4_x::main::Qwen4xMainReplayKey;
use crate::model::qwen::v4_x::main::gdn_configs;
use crate::model::qwen::v4_x::main::uniform_layout;
use crate::model::qwen::v4_x::state::Qwen4xRequestState;
use crate::model::qwen::v4_x::state::Qwen4xState;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GDNStateSnapshotFiles;
use crate::model::state_snapshot::GQAStateSnapshotFiles;
use crate::model::state_snapshot::PageArenaStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotFile;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedConfig;
use crate::replay::Replay;
use crate::replay::ReplayComponent;
use crate::sampling::sampling_params::SamplingParamsStore;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_replay::SamplingInput;
use crate::sampling::top_k_replay::TopKSamplingReplayKey;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use crate::sampling::top_k_sampling::config::TopKSamplingBounds;

#[derive(Clone, Copy, Debug)]
pub struct Qwen4xExecutorConfig {
    pub max_requests: usize,
    pub max_tokens: usize,
    pub max_tokens_per_request: usize,
    pub num_cache_pages: usize,
    pub num_tokens_per_block: usize,
}
impl Qwen4xExecutorConfig {
    fn validate(self) {
        assert!(self.max_requests > 0 && self.max_requests <= self.max_tokens);
        assert!(i32::try_from(self.max_tokens).is_ok());
        assert!(self.max_tokens_per_request > 0 && self.max_tokens_per_request <= self.max_tokens);
        assert!(self.num_tokens_per_block > 0 && u32::try_from(self.num_tokens_per_block).is_ok());
        assert!(self.num_cache_pages > 0 && u32::try_from(self.num_cache_pages - 1).is_ok());
    }
}

pub struct Qwen4xExecutor {
    model_dir: PathBuf,
    model_config: Qwen4xModelConfig,
    config: Qwen4xExecutorConfig,
    bindings: Qwen4xModelWeightBindings,
    runtime: MetalRuntime,
    token_ids: Buffer,
    hidden_input: Rc<Buffer>,
    hidden_output: Rc<Buffer>,
    gather_indices: Buffer,
    gathered_hidden: Buffer,
    logits: Buffer,
    embed: Replay<Qwen35MainTextEmbed>,
    main: Replay<Qwen4xMain>,
    output: Replay<Qwen35GatherUnembed>,
    sampling: Replay<Sampling>,
    sampler: Rc<TopKSampling>,
    sampler_output: TopKSamplingOutputBuffers,
    request_sampling: RequestSamplingState,
    gdn: Qwen3xGDNState,
    state: Qwen4xState,
    pages: PageArena,
    transactions: Qwen35PendingTransactions,
    stop_sequences: Vec<Vec<Token>>,
    unloaded_embed: Option<Embed>,
    unloaded_unembed: Option<Unembed>,
}

pub fn init_qwen4x_model(
    model_dir: impl AsRef<Path>,
    config: Qwen4xExecutorConfig,
) -> Result<Qwen4xExecutor, ModelExecutorError> {
    let model = init_qwen4x_model_config(&model_dir)?;
    Qwen4xExecutor::new(model_dir, model, config)
}

fn validate_metal_model_config(model: &Qwen4xModelConfig) -> Result<(), ModelExecutorError> {
    let text = &model.text_config;
    let dimensions = [
        text.hidden_size,
        text.num_hidden_layers,
        text.num_attention_heads,
        text.num_key_value_heads,
        text.head_dim,
        text.linear_num_value_heads,
        text.linear_num_key_heads,
        text.linear_key_head_dim,
        text.linear_value_head_dim,
        text.linear_conv_kernel_dim,
        text.num_experts,
        text.num_experts_per_tok,
        text.moe_intermediate_size,
        text.vocab_size,
        text.max_position_embeddings,
        text.hc_count,
        text.hc_lowrank,
        text.indexer_n_heads,
        text.indexer_head_dim,
        text.indexer_budget,
    ];
    if dimensions.iter().any(|&v| i32::try_from(v).is_err())
        || text.num_experts > 512
        || u32::try_from(text.hc_count * text.hidden_size).is_err()
    {
        return Err(ModelExecutorError::custom(
            "Qwen4x Metal dimensions exceed their supported shader domains",
        ));
    }
    if (
        text.num_attention_heads,
        text.num_key_value_heads,
        text.head_dim,
        text.indexer_n_heads,
        text.indexer_head_dim,
        text.indexer_budget,
        text.indexer_compress_ratio,
    ) != (24, 2, 256, 4, 128, 2048, 4)
    {
        return Err(ModelExecutorError::custom(
            "Qwen4x Metal requires QSA 24/2/256 and indexer 4/1/128, budget 2048, ratio 4",
        ));
    }
    Ok(())
}

impl Qwen4xExecutor {
    pub fn new(
        model_dir: impl AsRef<Path>,
        mut model: Qwen4xModelConfig,
        config: Qwen4xExecutorConfig,
    ) -> Result<Self, ModelExecutorError> {
        config.validate();
        model.normalize()?;
        validate_metal_model_config(&model)?;
        let model_dir = model_dir.as_ref();
        let text = &model.text_config;
        let generation = HFGenerationConfig::load(model_dir)?;
        let stop_ids = if generation.eos_token_ids().is_empty() {
            model.eos_token_ids()?
        } else {
            generation.eos_token_ids().to_vec()
        };
        if stop_ids.iter().any(|&id| id as usize >= text.vocab_size) {
            return Err(ModelExecutorError::custom(
                "Qwen4x stop token IDs exceed the vocabulary",
            ));
        }
        let stop_sequences = stop_ids.into_iter().map(|id| vec![Token::new(id)]).collect();
        let mut store = SafeTensorStore::from_model_dir(model_dir)?;
        let bindings = resolve_qwen4x_model_weight_bindings(&model, store.index().tensor_names())?;
        let device = Device::system_default();
        let runtime = MetalRuntime::new(device.clone());
        let state = Qwen4xState::new(&device, &model, config)?;
        let cores: Vec<_> = bindings
            .layers
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                if let Qwen4xAttentionWeightBindings::GDN(g) = &b.attention {
                    Some(gdn_configs(&model, i, g))
                } else {
                    None
                }
            })
            .collect::<Result<_, _>>()?;
        let (_, metal) = cores
            .first()
            .ok_or_else(|| ModelExecutorError::custom("Qwen4x requires GDN layers"))?;
        if cores.iter().any(|(_, m)| m != metal) {
            return Err(ModelExecutorError::custom(
                "Qwen4x GDN layers must share one affine layout",
            ));
        }
        let gdn_cores: Vec<_> = cores.iter().map(|(c, _)| c.clone()).collect();
        let jobs = config.max_tokens_per_request.div_ceil(config.num_tokens_per_block);
        let gdn = Qwen3xGDNState::new(
            &device,
            runtime.stream().new_on_same_queue(),
            &gdn_cores,
            *metal,
            config.max_requests,
            GDNStateCapacity::new(jobs + 2, jobs + 1, jobs, GDN::max_replay_tokens_per_request(0)),
            config.max_tokens,
            config.num_tokens_per_block,
            config.num_cache_pages,
            32 * 1024,
        );
        let mut main = Qwen4xMain::new(&device, &model, &bindings, config.max_tokens as u32, &gdn, &state)?;
        let embed_layout = uniform_layout(&model, &[&bindings.embed.weight], "Qwen4x token embedding")?;
        let mut embed = Embed::new(
            &device,
            EmbedConfig {
                max_tokens: config.max_tokens as u32,
                vocab_size: text.vocab_size as u32,
                hidden_dim: text.hidden_size as u32,
                group_size: embed_layout.group_size,
                bits: embed_layout.bits,
                scale_bias_dtype: Dtype::Bfloat16,
                output_dtype: Dtype::Bfloat16,
            },
        );
        embed.load_weights(&device, &mut store, bindings.embed.clone())?;
        let unembed_layout = uniform_layout(&model, &[&bindings.unembed.weight], "Qwen4x unembedding")?;
        let mut unembed = Unembed::new(
            &device,
            UnembedConfig {
                max_tokens: config.max_requests as u32,
                vocab_size: text.vocab_size as u32,
                hidden_dim: text.hidden_size as u32,
                group_size: unembed_layout.group_size,
                bits: unembed_layout.bits,
                input_dtype: Dtype::Bfloat16,
                output_dtype: Dtype::Bfloat16,
                scale_bias_dtype: Dtype::Bfloat16,
            },
        );
        if model.tie_word_embeddings || text.tie_word_embeddings {
            unembed.load_tied_weights(embed.tied_weights());
        } else {
            unembed.load_weights(&device, &mut store, bindings.unembed.clone())?;
        }
        main.load_weights(&device, &mut store, &model, &bindings)?;
        let bounds = TopKSamplingBounds {
            max_sampling_inputs: config.max_requests as u32,
            vocab_size: text.vocab_size as u32,
            top_k: MAX_TOP_K as u32,
        };
        bounds.validate();
        let params = Rc::new(SamplingParamsStore::new(&device, bounds, config.max_requests as u32));
        let sampler = Rc::new(TopKSampling::new(&device, params));
        let bf16 = |count: usize, width: usize| {
            Buffer::new_zeroed_elements(
                &device,
                count
                    .checked_mul(width)
                    .expect("Qwen4x workspace elements must fit usize"),
                Dtype::Bfloat16,
            )
        };
        Ok(Self {
            model_dir: model_dir.to_owned(),
            config,
            bindings,
            runtime,
            token_ids: Buffer::new_zeroed_elements(&device, config.max_tokens, Dtype::Int32),
            hidden_input: Rc::new(bf16(config.max_tokens, text.hidden_size)),
            hidden_output: Rc::new(bf16(config.max_tokens, text.hidden_size)),
            gather_indices: Buffer::new_zeroed_elements(&device, config.max_requests, Dtype::Uint32),
            gathered_hidden: bf16(config.max_requests, text.hidden_size),
            logits: bf16(config.max_requests, text.vocab_size),
            embed: Replay::new("Qwen4x MainTextEmbed", Qwen35MainTextEmbed::new(Rc::new(embed))),
            main: Replay::new("Qwen4x Main", main),
            output: Replay::new(
                "Qwen4x GatherUnembed",
                Qwen35GatherUnembed::new(&device, text.hidden_size as u32, Rc::new(unembed)),
            ),
            sampling: Replay::new("Qwen4x Sampling", Sampling::new(Rc::clone(&sampler))),
            sampler,
            sampler_output: TopKSamplingOutputBuffers::new(&device, bounds),
            request_sampling: RequestSamplingState::new(config.max_requests),
            gdn,
            state,
            pages: PageArena::new(&device, config.num_cache_pages, 32 * 1024),
            transactions: Qwen35PendingTransactions::new(),
            stop_sequences,
            unloaded_embed: None,
            unloaded_unembed: None,
            model_config: model,
        })
    }
    pub fn model_config(&self) -> &Qwen4xModelConfig {
        &self.model_config
    }
    pub fn num_kv_page_ids_per_block(&self) -> usize {
        self.state.kv_pages_per_block()
    }
    pub fn num_state_page_ids_per_block(&self) -> usize {
        self.gdn.num_pages_per_state_slot() + self.state.aux_pages_per_block()
    }
}

pub struct Qwen4xModelOpsRecorder {
    embed_key: Qwen35MainTextEmbedReplayKey,
    embed_arguments: ReplayArguments,
    main_key: Qwen4xMainReplayKey,
    main_arguments: ReplayArguments,
    output: Option<(Qwen35GatherUnembedReplayKey, ReplayArguments)>,
    sampling: Option<(TopKSamplingReplayKey, ReplayArguments)>,
}
pub struct Qwen4xSampledOutput {
    decisions: Vec<Qwen35DecodeDecision>,
    timing: ModelOutputTiming,
}

impl ReplayableDecoderModel for Qwen4xExecutor {
    type ModelBatchRequest = Qwen35ModelBatchRequest;
    type ModelBatchHidden = Rc<Buffer>;
    type ModelBatchResponse = ();
    type SampledOutput = Qwen4xSampledOutput;
    type ModelOpsRecorder = Qwen4xModelOpsRecorder;
    type Submission = MetalReplaySubmission;
    fn model_name(&self) -> &str {
        &self.model_config.model_type
    }
    fn vocab_size(&self) -> usize {
        self.model_config.text_config.vocab_size
    }

    fn model_mode(&self) -> &'static str {
        "vanilla"
    }
    fn default_stop_sequences(&self) -> Vec<Vec<Token>> {
        self.stop_sequences.clone()
    }
    fn reset_req_slots(&mut self, slots: &[RawRequestSlot]) {
        self.state.finish_commit();
        self.gdn.finish_commit();
        self.request_sampling.reset(slots);
        self.gdn.reset_req_slots(slots);
        self.state.reset_req_slots(slots);
    }
    fn clear_replay_cache(&mut self) {
        self.state.finish_commit();
        self.gdn.finish_commit();
        self.embed.clear();
        self.main.clear();
        self.output.clear();
        self.sampling.clear();
        self.gdn.clear_replay_cache();
    }
    fn unload_weights(&mut self) {
        let mut unembed = Rc::try_unwrap(self.output.component_mut().unload_weights())
            .unwrap_or_else(|_| panic!("Qwen4x unembedding must have one owner during unload"));
        unembed.unload_weights();
        self.unloaded_unembed = Some(unembed);
        let mut embed = Rc::try_unwrap(self.embed.component_mut().unload_weights())
            .unwrap_or_else(|_| panic!("Qwen4x embedding must have one owner during unload"));
        embed.unload_weights();
        self.unloaded_embed = Some(embed);
        self.main.component_mut().unload_weights();
    }
    fn load_weights(&mut self) -> Result<(), ModelExecutorError> {
        let device = self.runtime.device();
        let mut store = SafeTensorStore::from_model_dir(&self.model_dir)?;
        let mut embed = self.unloaded_embed.take().expect("Qwen4x embedding shell must exist");
        embed.load_weights(device, &mut store, self.bindings.embed.clone())?;
        let mut unembed = self
            .unloaded_unembed
            .take()
            .expect("Qwen4x unembedding shell must exist");
        if self.model_config.tie_word_embeddings || self.model_config.text_config.tie_word_embeddings {
            unembed.load_tied_weights(embed.tied_weights());
        } else {
            unembed.load_weights(device, &mut store, self.bindings.unembed.clone())?;
        }
        self.main
            .component_mut()
            .load_weights(device, &mut store, &self.model_config, &self.bindings)?;
        self.embed.component_mut().load_weights(Rc::new(embed));
        self.output.component_mut().load_weights(Rc::new(unembed));
        Ok(())
    }
    fn unload_state(&mut self, path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        self.write_state(path, plan)?;
        self.main.component_mut().unload_state();
        self.gdn.release_resources();
        self.state.release_resources();
        self.pages.release_resources();
        Ok(())
    }
    fn load_state(&mut self, path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        self.read_state(path, plan)
    }
    fn prepare_batch(&mut self, request: &mut BatchDeviceRequest) -> Self::ModelBatchRequest {
        self.state.finish_commit();
        self.gdn.finish_commit();
        assert!(
            !request.dev_reqs.is_empty()
                && request.dev_reqs.len() <= self.config.max_requests
                && request.token_cost() <= self.config.max_tokens
        );
        for req in &request.dev_reqs {
            assert!((req.req_slot as usize) < self.config.max_requests && req.resource_placements.is_empty());
            let q = &req.decoder_query_tokens;
            assert_eq!(q.num_spec_tokens(), 0, "Qwen4x MTP is not wired");
            assert!(q.token_consumption() > 0 && q.token_consumption() <= self.config.max_tokens_per_request);
            assert!(
                q.token_index()
                    .checked_add(q.token_consumption())
                    .is_some_and(|end| end <= self.model_config.text_config.max_position_embeddings)
            );
        }
        request.dev_reqs.sort_by_key(|r| {
            if matches!(r.decoder_query_tokens, QueryTokens::Prefill { .. })
                || self.gdn.backend().uses_recurrent(r.token_cost(), 0)
            {
                0
            } else {
                1
            }
        });
        let sampling = request
            .dev_reqs
            .iter()
            .map(|r| {
                SamplerConfig::from_runtime(
                    &r.sampling_config,
                    self.request_sampling.resolve(r.req_slot, r.sampling_config.seed),
                )
            })
            .collect();
        let batch = Qwen35ModelBatchRequest::from_core_batch(request, 0, sampling);
        let micro = batch.microbatch();
        self.sampler.set_params(micro.req_slots(), micro.sampler_configs());
        self.token_ids.write_typed(0, micro.flat_token_ids());
        let num_total_tokens = self.main.component().capacity(micro.total_tokens() as u32);
        let runtime = self.runtime.stream();
        self.state.prepare(
            self.runtime.device(),
            runtime,
            self.pages.buffer(),
            request,
            micro,
            num_total_tokens,
            self.gdn.num_pages_per_state_slot(),
        );
        let num_active_recurrent_requests = (0..micro.num_reqs())
            .take_while(|&i| {
                !micro.is_decode_req(i)
                    || self
                        .gdn
                        .backend()
                        .uses_recurrent((micro.cu_tokens()[i + 1] - micro.cu_tokens()[i]) as usize, 0)
            })
            .count();
        let prepared = self.gdn.prepare_states(
            micro.req_slots(),
            micro.block_indices(),
            micro.token_indices(),
            micro.cu_tokens(),
            &vec![0; micro.num_reqs()],
            num_active_recurrent_requests,
            self.state.gdn_page_ids(),
        );
        self.gdn.prepare_metadata(
            micro.cu_tokens(),
            num_active_recurrent_requests as u32,
            &prepared,
            num_total_tokens,
        );
        self.gdn.restore(runtime, self.pages.buffer());
        batch
    }
    fn commit_batch(&mut self, request: BatchDeviceRequest, output: Self::SampledOutput) -> BatchDeviceResponse {
        let batch = self.transactions.pending_microbatch(request.seq).clone();
        let versions = self.transactions.commit(request.seq, &output.decisions);
        self.state.commit(
            self.runtime.device(),
            self.runtime.stream(),
            self.pages.buffer(),
            self.main.component(),
            &batch,
            &versions,
        );
        self.gdn.commit(self.pages.buffer(), &versions);
        to_core_batch_resp(request, output.decisions)
    }
    fn begin_ops_recording(&mut self, batch: &Self::ModelBatchRequest) -> Self::ModelOpsRecorder {
        let (embed_key, embed_arguments) = self
            .embed
            .component()
            .prepare_replay(batch.microbatch().total_tokens() as u32);
        let input = Qwen4xMainArgs {
            hidden_input: &self.hidden_input,
            hidden_output: &self.hidden_output,
            pages: self.pages.buffer(),
            state: &self.state,
            gdn: &self.gdn,
        };
        let main_key = self.main.component().replay_key(&input);
        let main_arguments = self.main.component().replay_arguments(&input);
        Qwen4xModelOpsRecorder {
            embed_key,
            embed_arguments,
            main_key,
            main_arguments,
            output: None,
            sampling: None,
        }
    }
    fn embed_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        batch: &Self::ModelBatchRequest,
    ) -> Self::ModelBatchHidden {
        let (key, _) = self.embed.record(
            self.runtime.stream(),
            &Qwen35MainTextEmbedArgs {
                num_tokens: batch.microbatch().total_tokens() as u32,
                token_ids: &self.token_ids,
                hidden_output: &self.hidden_input,
            },
        );
        debug_assert_eq!(key, recorder.embed_key);
        Rc::clone(&self.hidden_input)
    }
    fn forward_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        batch: &Self::ModelBatchRequest,
        hidden: Self::ModelBatchHidden,
    ) -> Self::ModelBatchHidden {
        debug_assert!(Rc::ptr_eq(&hidden, &self.hidden_input));
        let input = Qwen4xMainArgs {
            hidden_input: &self.hidden_input,
            hidden_output: &self.hidden_output,
            pages: self.pages.buffer(),
            state: &self.state,
            gdn: &self.gdn,
        };
        let (key, _) = self.main.record(self.runtime.stream(), &input);
        debug_assert_eq!(key, recorder.main_key);
        self.transactions.push(batch.compute_seq(), batch.microbatch().clone());
        Rc::clone(&self.hidden_output)
    }
    fn unembed_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        batch: &Self::ModelBatchRequest,
        hidden: &Self::ModelBatchHidden,
    ) {
        let indices = gather_flat_indices(batch.microbatch());
        if indices.is_empty() {
            return;
        }
        debug_assert!(Rc::ptr_eq(hidden, &self.hidden_output));
        self.gather_indices.write_typed(0, &indices);
        let count = indices.len() as u32;
        let (_, arguments) = self.output.component().prepare_replay(count);
        let (key, _) = self.output.record(
            self.runtime.stream(),
            &Qwen35GatherUnembedArgs {
                num_rows: count,
                hidden_input: &self.hidden_output,
                row_indices: &self.gather_indices,
                hidden_output: &self.gathered_hidden,
                logits: &self.logits,
            },
        );
        recorder.output = Some((key, arguments));
    }
    fn sample_main(&mut self, recorder: &mut Self::ModelOpsRecorder, batch: &Self::ModelBatchRequest, _: &()) {
        if recorder.output.is_none() {
            return;
        }
        let micro = batch.microbatch();
        let configs = sample_sampler_configs(micro);
        let shape = self.sampling.component().prepare_shape(&configs);
        let (key, _) = self.sampling.record(
            self.runtime.stream(),
            &SamplingInput {
                shape,
                logits: &self.logits,
                output: self.sampler_output.as_output(),
            },
        );
        self.sampler
            .prepare(&sample_req_slots(micro), &sample_token_positions(micro));
        let mut args = ReplayArguments::new();
        self.sampler.add_replay_arguments(shape, &mut args);
        recorder.sampling = Some((key, args));
    }
    fn submit_main(&mut self, r: &Self::ModelOpsRecorder) -> Self::Submission {
        let mut sequence = vec![
            ReplayExecution::new(self.embed.replay(&r.embed_key), &r.embed_arguments),
            ReplayExecution::new(self.main.replay(&r.main_key), &r.main_arguments),
        ];
        if let Some((key, args)) = &r.output {
            sequence.push(ReplayExecution::new(self.output.replay(key), args));
        }
        if let Some((key, args)) = &r.sampling {
            sequence.push(ReplayExecution::new(self.sampling.replay(key), args));
        }
        self.runtime.stream().submit_replay_sequence(&sequence).into()
    }
    fn read_main(
        &mut self,
        r: &Self::ModelOpsRecorder,
        batch: &Self::ModelBatchRequest,
        elapsed: Duration,
        _: Option<&[Duration]>,
    ) -> Self::SampledOutput {
        let start = Instant::now();
        let count = sample_req_slots(batch.microbatch()).len();
        let decisions = if r.sampling.is_some() {
            sample_decisions_from_sampled_tokens(&Qwen35SampledTokens::new(
                self.sampler_output.token_ids.read_typed(0, count),
                self.sampler_output.token_probs.read_typed(0, count),
            ))
        } else {
            Vec::new()
        };
        Qwen4xSampledOutput {
            decisions,
            timing: ModelOutputTiming {
                main_sample_replay_elapsed: if count > 0 { elapsed } else { Duration::ZERO },
                main_replay_elapsed: if count == 0 { elapsed } else { Duration::ZERO },
                sample_read_elapsed: start.elapsed(),
                ..Default::default()
            },
        }
    }
    fn empty_sampled_output(&self) -> Self::SampledOutput {
        Qwen4xSampledOutput {
            decisions: Vec::new(),
            timing: Default::default(),
        }
    }
    fn sampled_output_len(&self, o: &Self::SampledOutput) -> usize {
        o.decisions.len()
    }
    fn sampled_output_timing(&self, o: &Self::SampledOutput) -> Option<ModelOutputTiming> {
        Some(o.timing)
    }
}

include!("file_io.rs");

#[cfg(test)]
#[path = "executor_test.rs"]
mod tests;
