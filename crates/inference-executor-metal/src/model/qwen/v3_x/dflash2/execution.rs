use std::path::Path;
use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_backend_metal::metal::Stream;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2WeightBindings;
use inference_executor_core::model::qwen::v3_x::dflash2::resolve_qwen3x_dflash2_weight_bindings;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::bidi_block_gqa::state::BiDiBlockGQAState;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::request_page_table::GQAPageTableLayout;
use crate::checkpoint::SafeTensorStore;
use crate::model::embedding::Embed;
use crate::model::qwen::v3_x::SpecReplayStageEnds;
use crate::model::qwen::v3_x::dflash2::embed::Qwen3xDFlash2Embed;
use crate::model::qwen::v3_x::dflash2::embed::Qwen3xDFlash2EmbedArgs;
use crate::model::qwen::v3_x::dflash2::embed::Qwen3xDFlash2EmbedReplayKey;
use crate::model::qwen::v3_x::dflash2::load::Qwen3xDFlash2Loaded;
use crate::model::qwen::v3_x::dflash2::main_feature::Qwen3xDFlash2MainFeatureProjector;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2Body;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2BodyArgs;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2BodyReplayKey;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2Model;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2Prefill;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2PrefillArgs;
use crate::model::qwen::v3_x::dflash2::model::Qwen3xDFlash2PrefillReplayKey;
use crate::model::qwen::v3_x::dflash2::output::Qwen3xDFlash2Output;
use crate::model::qwen::v3_x::dflash2::output::Qwen3xDFlash2OutputArgs;
use crate::model::qwen::v3_x::dflash2::output::Qwen3xDFlash2OutputReplayKey;
use crate::model::qwen::v3_x::dflash2::output::Qwen3xDFlash2Proposal;
use crate::model::qwen::v3_x::spec_decode_input::SpecDecodeInput;
use crate::model::qwen::v3_x::spec_decode_input::SpecDecodeInputArgs;
use crate::model::qwen::v3_x::spec_decode_input::SpecDecodeInputConfig;
use crate::model::qwen::v3_x::spec_decode_input::SpecDecodeInputRecording;
use crate::model::unembedding::Unembed;
use crate::replay::Replay;
use crate::sampling::rejection_sampling::SparseRejectionSamplingOutput;
use crate::sampling::spec_probs::SpecProbsStore;

mod file_io;

pub struct Qwen3xDFlash2Execution {
    prefill: Replay<Qwen3xDFlash2Prefill>,
    gqa_state: BiDiBlockGQAState,
    embed: Replay<Qwen3xDFlash2Embed>,
    body: Replay<Qwen3xDFlash2Body>,
    output: Replay<Qwen3xDFlash2Output>,
    decode_input: Replay<SpecDecodeInput>,
    unloaded_model: Option<Qwen3xDFlash2Model>,
    hidden_input: Rc<Buffer>,
    hidden_output: Rc<Buffer>,
    page_table_layout: GQAPageTableLayout,
    num_spec_tokens: usize,
    page_bytes: usize,
}

impl Qwen3xDFlash2Execution {
    pub fn new(device: &Device, loaded: Qwen3xDFlash2Loaded) -> Self {
        assert!(loaded.num_spec_tokens > 0);
        let spec_block_size = loaded.num_spec_tokens + 1;
        let max_query_tokens = (loaded.page_table_layout.num_req_slots as usize)
            .checked_mul(spec_block_size)
            .expect("Qwen3x DFlash2 query token capacity must fit usize");
        let hidden_bytes = max_query_tokens
            .checked_mul(loaded.output.hidden_dim() as usize)
            .and_then(|elements| elements.checked_mul(Dtype::Bfloat16.item_size()))
            .expect("Qwen3x DFlash2 hidden byte capacity must fit usize");
        let sliding_window = loaded
            .sliding_window
            .try_into()
            .expect("Qwen3x DFlash2 sliding window must fit u32");
        let sdpa = loaded.gqa_state.metadata().variant().map.thread_block;
        let decode_input = SpecDecodeInput::new(
            device,
            SpecDecodeInputConfig::new(
                loaded.page_table_layout.num_req_slots,
                spec_block_size as u32,
                sdpa,
                sliding_window,
                loaded.max_anchor_position,
                loaded.gqa_state.max_sdpa_map_task_templates(),
                loaded.mask_token_id,
            ),
        );
        Self {
            prefill: Replay::new(
                "Qwen3x DFlash2 Prefill",
                Qwen3xDFlash2Prefill::new(Rc::clone(&loaded.model)),
            ),
            gqa_state: loaded.gqa_state,
            embed: Replay::new("Qwen3x DFlash2 Embed", Qwen3xDFlash2Embed::new(loaded.embed)),
            body: Replay::new("Qwen3x DFlash2 Body", Qwen3xDFlash2Body::new(Rc::clone(&loaded.model))),
            output: Replay::new("Qwen3x DFlash2 Output", loaded.output),
            decode_input: Replay::new("Qwen3x DFlash2 Spec Decode Prepare", decode_input),
            unloaded_model: None,
            hidden_input: Rc::new(Buffer::new_zeroed(device, hidden_bytes)),
            hidden_output: Rc::new(Buffer::new_zeroed(device, hidden_bytes)),
            page_table_layout: loaded.page_table_layout,
            num_spec_tokens: loaded.num_spec_tokens,
            page_bytes: loaded.page_bytes,
        }
    }

    pub fn num_spec_tokens(&self) -> usize {
        self.num_spec_tokens
    }

    pub fn num_gqa_page_ids_per_block(&self) -> usize {
        self.page_table_layout.num_gqa_layers as usize * self.page_table_layout.num_page_ids_per_block as usize
    }

    pub fn num_tokens_per_page(&self) -> usize {
        self.gqa_state.num_tokens_per_page()
    }

    pub fn num_physical_pages_per_request(&self) -> usize {
        self.page_table_layout.num_physical_pages_per_request()
    }

    pub fn reset_req_slots(&self, request_slots: &[RawRequestSlot]) {
        self.gqa_state.reset_req_slots(request_slots);
    }

    pub fn clear_replay_cache(&mut self) {
        self.prefill.clear();
        self.embed.clear();
        self.body.clear();
        self.output.clear();
        self.decode_input.clear();
    }

    pub fn unload_weights(&mut self) {
        self.unloaded_model
            .as_mut()
            .expect("Qwen3x DFlash2 state must be unloaded before weights")
            .unload_weights();
        drop(self.output.component_mut().unload_weights());
        drop(self.embed.component_mut().unload_weights());
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        model_dir: &Path,
        config: &Qwen3xDFlash2Config,
        main_text_embed: &Rc<Embed>,
        main_unembed: &Rc<Unembed>,
    ) -> Result<(), ModelExecutorError> {
        let mut store = SafeTensorStore::from_model_dir(model_dir)?;
        let Qwen3xDFlash2WeightBindings {
            main_feature,
            layers,
            final_norm_weight,
            selector,
        } = resolve_qwen3x_dflash2_weight_bindings(config, store.index().tensor_names())?;
        let spec_block_size = self.num_spec_tokens + 1;
        let max_requests = self.page_table_layout.num_req_slots as usize;
        let max_query_tokens = max_requests
            .checked_mul(spec_block_size)
            .expect("Qwen3x DFlash2 query token capacity must fit usize");
        let max_proposal_tokens = max_requests
            .checked_mul(self.num_spec_tokens)
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
        self.unloaded_model
            .as_mut()
            .expect("Qwen3x DFlash2 state must remain unloaded during weight loading")
            .load_weights(device, &mut store, config, &main_feature, layers, final_norm_weight)?;
        self.output.component_mut().load_unembed(unembed);
        self.output.component_mut().load_weights(device, &mut store, selector)?;
        self.embed.component_mut().load_weights(embed);
        Ok(())
    }

    pub fn main_feature_projector(&self) -> Rc<Qwen3xDFlash2MainFeatureProjector> {
        self.unloaded_model
            .as_ref()
            .expect("Qwen3x DFlash2 state must be unloaded while restoring Main shared weights")
            .main_feature_projector()
    }

    pub fn unload_state(&mut self) {
        assert!(
            self.unloaded_model.is_none(),
            "Qwen3x DFlash2 model state is already unloaded"
        );
        let prefill_model = self.prefill.component_mut().unload_model();
        let body_model = self.body.component_mut().unload_model();
        assert!(Rc::ptr_eq(&prefill_model, &body_model));
        drop(body_model);
        let mut model = Rc::try_unwrap(prefill_model)
            .unwrap_or_else(|_| panic!("Qwen3x DFlash2 model must be uniquely owned during state unloading"));
        model.unload_state();
        self.unloaded_model = Some(model);
        self.gqa_state.release_resources();
    }

    pub fn allocate_resources(&mut self, device: &Device) {
        self.gqa_state.allocate_resources(device);
    }

    pub fn release_resources(&mut self) {
        self.gqa_state.release_resources();
    }

    pub fn load_state(&mut self) {
        let mut model = self
            .unloaded_model
            .take()
            .expect("Qwen3x DFlash2 model state is not unloaded");
        model.load_state(&self.gqa_state);
        let model = Rc::new(model);
        self.prefill.component_mut().load_model(Rc::clone(&model));
        self.body.component_mut().load_model(model);
    }

    pub fn write_page_ids(&self, req_slot: u32, block_index: usize, page_ids: &[u32]) {
        self.gqa_state.write_page_ids(req_slot, block_index, page_ids);
    }

    pub fn record_spec_prefill(
        &mut self,
        runtime: &Stream,
        spec_prefill: &GQAMetadataBuffers,
        pages: &Buffer,
    ) -> Qwen3xDFlash2PrefillRecording {
        let num_tokens = spec_prefill.replay_shape().num_tokens;
        let input = Qwen3xDFlash2PrefillArgs {
            num_tokens,
            req_slots: spec_prefill.req_slots(),
            flat_token_indices: spec_prefill.flat_token_indices(),
            pages,
        };
        let (key, _) = self.prefill.record(runtime, &input);
        let arguments = self.prefill.component().replay_arguments(&key);
        Qwen3xDFlash2PrefillRecording { key, arguments }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_decode_prepare(
        &mut self,
        runtime: &Stream,
        token_ids: &Buffer,
        rejection_sampling: SparseRejectionSamplingOutput<'_>,
        req_slots: &[u32],
        anchor_indices: &[u32],
        num_spec_tokens: &[u32],
        distribution_store: &SpecProbsStore,
    ) -> (SpecDecodeInputRecording, u32) {
        let block = self
            .decode_input
            .component()
            .prepare(req_slots, anchor_indices, num_spec_tokens);
        let num_active_requests = req_slots.len();
        let num_total_requests = block.num_requests();
        self.gqa_state
            .prepare_bidi_block_with_active_requests(&block, num_active_requests);
        let num_requests = self.output.component().prepare_static(req_slots, distribution_store);
        let input = SpecDecodeInputArgs {
            num_active_requests: num_active_requests as u32,
            num_total_requests: num_total_requests as u32,
            rejection_sampling,
            metadata: self.gqa_state.metadata(),
            block_token_ids: token_ids,
            anchor_token_ids: self.output.component().anchor_token_ids(),
        };
        let (key, _) = self.decode_input.record(runtime, &input);
        let arguments = self
            .decode_input
            .component()
            .replay_arguments(&key, input.num_active_requests);
        (SpecDecodeInputRecording { key, arguments }, num_requests)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_spec_decode(
        &mut self,
        runtime: &Stream,
        token_ids: &Buffer,
        decode_prepare: SpecDecodeInputRecording,
        num_requests: u32,
        req_slots: Vec<u32>,
        pages: &Buffer,
        distribution_store: &SpecProbsStore,
    ) -> Qwen3xDFlash2DecodeRecording {
        let metadata = self.gqa_state.metadata();
        let embed_input = Qwen3xDFlash2EmbedArgs {
            num_tokens: metadata.replay_shape().num_tokens,
            token_ids,
            hidden_output: &self.hidden_input,
        };
        let (embed_key, _) = self.embed.record(runtime, &embed_input);
        let embed_arguments = self.embed.component().replay_arguments(&embed_key);
        let body_input = Qwen3xDFlash2BodyArgs {
            metadata,
            hidden_input: &self.hidden_input,
            hidden_output: &self.hidden_output,
            pages,
        };
        let (body_key, _) = self.body.record(runtime, &body_input);
        let mut body_arguments = ReplayArguments::new();
        self.gqa_state.add_replay_arguments(&mut body_arguments);
        self.body
            .component()
            .add_replay_arguments(metadata.replay_shape(), &mut body_arguments);
        let output_input = Qwen3xDFlash2OutputArgs {
            num_requests,
            hidden: &self.hidden_output,
            sample_positions: self.decode_input.component().sample_positions(),
            distribution_store,
        };
        let (output_key, _) = self.output.record(runtime, &output_input);
        let mut output_arguments = ReplayArguments::new();
        self.output
            .component()
            .add_replay_arguments(num_requests, &mut output_arguments);
        Qwen3xDFlash2DecodeRecording {
            decode_prepare,
            embed_key,
            embed_arguments,
            body_key,
            body_arguments,
            output_key,
            output_arguments,
            req_slots,
        }
    }

    pub fn append_spec_replays<'a>(
        &'a self,
        sequence: &mut Vec<ReplayExecution<'a>>,
        prefill: &'a Qwen3xDFlash2PrefillRecording,
        decode: Option<&'a Qwen3xDFlash2DecodeRecording>,
    ) -> SpecReplayStageEnds {
        let mut decode_prepare_end = None;
        if let Some(decode) = decode {
            sequence.push(ReplayExecution::new(
                self.decode_input.replay(&decode.decode_prepare.key),
                &decode.decode_prepare.arguments,
            ));
            decode_prepare_end = Some(sequence.len());
        }
        sequence.push(ReplayExecution::new(
            self.prefill.replay(&prefill.key),
            &prefill.arguments,
        ));
        let prefill_end = sequence.len();
        let mut decode_end = None;
        if let Some(decode) = decode {
            sequence.push(ReplayExecution::new(
                self.embed.replay(&decode.embed_key),
                &decode.embed_arguments,
            ));
            sequence.push(ReplayExecution::new(
                self.body.replay(&decode.body_key),
                &decode.body_arguments,
            ));
            sequence.push(ReplayExecution::new(
                self.output.replay(&decode.output_key),
                &decode.output_arguments,
            ));
            decode_end = Some(sequence.len());
        }
        SpecReplayStageEnds {
            decode_prepare: decode_prepare_end,
            prefill: prefill_end,
            decode: decode_end,
        }
    }

    pub fn read_proposal(
        &self,
        recording: &Qwen3xDFlash2DecodeRecording,
        distribution_store: &mut SpecProbsStore,
    ) -> Qwen3xDFlash2Proposal {
        self.output
            .component()
            .read_proposal(&recording.req_slots, distribution_store)
    }
}

pub struct Qwen3xDFlash2PrefillRecording {
    key: Qwen3xDFlash2PrefillReplayKey,
    arguments: ReplayArguments,
}

pub struct Qwen3xDFlash2DecodeRecording {
    decode_prepare: SpecDecodeInputRecording,
    embed_key: Qwen3xDFlash2EmbedReplayKey,
    embed_arguments: ReplayArguments,
    body_key: Qwen3xDFlash2BodyReplayKey,
    body_arguments: ReplayArguments,
    output_key: Qwen3xDFlash2OutputReplayKey,
    output_arguments: ReplayArguments,
    req_slots: Vec<u32>,
}
