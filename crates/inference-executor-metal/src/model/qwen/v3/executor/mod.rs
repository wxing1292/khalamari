use std::collections::VecDeque;
use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::ModelOutputTiming;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_core::model::qwen::v3::Qwen3DecodeDecision;
use inference_executor_core::model::qwen::v3::Qwen3Microbatch;
use inference_executor_core::model::qwen::v3::Qwen3ModelBatchRequest;
use inference_executor_core::model::qwen::v3::Qwen3ModelConfig;
use inference_executor_core::model::qwen::v3::Qwen3SampledTokens;
use inference_executor_core::model::qwen::v3::gather_flat_indices;
use inference_executor_core::model::qwen::v3::num_main_output_rows;
use inference_executor_core::model::qwen::v3::sample_decisions_from_sampled_tokens;
use inference_executor_core::model::qwen::v3::sample_req_slots;
use inference_executor_core::model::qwen::v3::sample_sampler_configs;
use inference_executor_core::model::qwen::v3::sample_token_positions;
use inference_executor_core::model::qwen::v3::to_core_batch_resp;
use inference_executor_core::model::qwen::v3::weight_layout::Qwen3ModelWeightBindings;
use inference_executor_core::model::qwen::v3::weight_layout::resolve_qwen3_model_weight_bindings;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRModelConfig;
use inference_executor_core::model::qwen::v3_asr::weight_layout::Qwen3ASRWeightBindings;
use inference_executor_core::model::qwen::v3_asr::weight_layout::resolve_qwen3_asr_weight_bindings;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::SamplerConfig;
use inference_runtime_core::compute::BatchDevReq;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::ExecutorHibernationPlan;
use inference_runtime_core::runtime::RawComputeSlotSeq;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::Token;

use crate::attn::gqa::request_page_table::GQAPageTableLayout;
use crate::checkpoint::SafeTensorStore;
use crate::def::submission::MetalReplaySubmission;
use crate::model::embedding::Embed;
use crate::model::main_resource_embed::MainResourceEmbed;
use crate::model::page_arena::PageArena;
use crate::model::qwen::apply_main_gpu_timing;
use crate::model::qwen::v3::main::Qwen3Main;
use crate::model::qwen::v3::main::Qwen3MainArgs;
use crate::model::qwen::v3::main::Qwen3MainReplayKey;
use crate::model::qwen::v3::main::gqa::Qwen3MainGQAState;
use crate::model::qwen::v3::main::output::Qwen3GatherUnembed;
use crate::model::qwen::v3::main::output::Qwen3GatherUnembedArgs;
use crate::model::qwen::v3::main::output::Qwen3GatherUnembedReplayKey;
use crate::model::qwen::v3::main::text_embed::Qwen3MainTextEmbed;
use crate::model::qwen::v3::main::text_embed::Qwen3MainTextEmbedArgs;
use crate::model::qwen::v3::main::text_embed::Qwen3MainTextEmbedReplayKey;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GQAStateSnapshotFiles;
use crate::model::state_snapshot::PageArenaStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotFile;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;
use crate::model::unembedding::Unembed;
use crate::replay::Replay;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_replay::SamplingInput;
use crate::sampling::top_k_replay::TopKSamplingReplayKey;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingInputs;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;

mod load;

pub use load::Qwen3ASRLoaded;
pub use load::Qwen3ExecutorConfig;
use load::Qwen3ModelLayout;
pub use load::init_qwen_3_model;
pub use load::init_qwen3_asr_model;

include!("batch.rs");
include!("main.rs");
include!("recording.rs");
include!("sampling.rs");

const PAGE_ARENA_STATE_FILES: PageArenaStateSnapshotFiles =
    PageArenaStateSnapshotFiles::new(StateSnapshotFile::PageArena);
const MAIN_GQA_STATE_FILES: GQAStateSnapshotFiles =
    GQAStateSnapshotFiles::new(StateSnapshotFile::MainGQARequestPageTable);
const VANILLA_STATE_SNAPSHOT_FILES: &[StateSnapshotFile] = &[
    PAGE_ARENA_STATE_FILES.pages(),
    MAIN_GQA_STATE_FILES.request_page_table(),
];

enum Qwen3Checkpoint {
    Text(Qwen3ModelConfig),
    Asr(Box<Qwen3ASRModelConfig>),
}

impl Qwen3Checkpoint {
    fn text(&self) -> &inference_executor_core::model::qwen::v3::Qwen3TextConfig {
        match self {
            Self::Text(config) => &config.text_config,
            Self::Asr(config) => &config.text,
        }
    }
}

pub struct Qwen3Executor {
    model_name: String,
    model_dir: PathBuf,
    checkpoint: Qwen3Checkpoint,
    default_stop_sequences: Vec<Vec<Token>>,
    config: Qwen3ExecutorConfig,
    runtime: MetalRuntime,
    layout: Qwen3ModelLayout,
    token_ids: Buffer,
    token_hidden_input: Rc<Buffer>,
    hidden_output: Rc<Buffer>,
    gather_flat_indices: Buffer,
    unembed_hidden: Buffer,
    unembed_logits: Buffer,
    main_text_embed: Replay<Qwen3MainTextEmbed>,
    main_resource_embed: Option<MainResourceEmbed>,
    main: Replay<Qwen3Main>,
    gather_unembed: Replay<Qwen3GatherUnembed>,
    sampling: Replay<Sampling>,
    sampler: Rc<TopKSampling>,
    sampler_output: TopKSamplingOutputBuffers,
    request_sampling: RequestSamplingState,
    main_gqa_state: Qwen3MainGQAState,
    pages: PageArena,
    pending_transactions: Qwen3PendingTransactions,
    gqa_page_table_layout: GQAPageTableLayout,
    unloaded_embed: Option<Embed>,
    unloaded_unembed: Option<Unembed>,
}

impl Qwen3Executor {
    fn state_snapshot_files(&self) -> &'static [StateSnapshotFile] {
        VANILLA_STATE_SNAPSHOT_FILES
    }

    pub fn clear_replay_cache(&mut self) {
        self.main_text_embed.clear();
        if let Some(main_resource_embed) = &mut self.main_resource_embed {
            main_resource_embed.clear();
        }
        self.main.clear();
        self.gather_unembed.clear();
        self.sampling.clear();
    }

    pub fn unload_weights(&mut self) {
        let embed = self.main_text_embed.component_mut().unload_weights();
        let unembed = self.gather_unembed.component_mut().unload_weights();

        self.main.component_mut().unload_weights();

        let mut unembed = Rc::try_unwrap(unembed)
            .unwrap_or_else(|_| panic!("qwen3 Main unembed must be uniquely owned during weight unloading"));
        unembed.unload_weights();
        self.unloaded_unembed = Some(unembed);

        let mut embed = Rc::try_unwrap(embed)
            .unwrap_or_else(|_| panic!("qwen3 Main embed must be uniquely owned during weight unloading"));
        embed.unload_weights();
        self.unloaded_embed = Some(embed);
    }

    pub fn load_weights(&mut self) -> Result<(), ModelExecutorError> {
        let device = self.runtime.device().clone();
        let mut store = SafeTensorStore::from_model_dir(&self.model_dir)?;
        let (embed, main, unembed) = match &self.checkpoint {
            Qwen3Checkpoint::Text(config) => {
                let Qwen3ModelWeightBindings { embed, main, unembed } =
                    resolve_qwen3_model_weight_bindings(config, store.index().tensor_names())?;
                (embed, main, Some(unembed))
            },
            Qwen3Checkpoint::Asr(config) => {
                let Qwen3ASRWeightBindings { embed, text, .. } =
                    resolve_qwen3_asr_weight_bindings(config, store.index().tensor_names())?;
                (embed, text, None)
            },
        };

        let mut loaded_embed = self
            .unloaded_embed
            .take()
            .expect("qwen3 Main embed shell must exist during weight loading");
        loaded_embed.load_weights(&device, &mut store, embed)?;
        let loaded_embed = Rc::new(loaded_embed);
        let mut loaded_unembed = self
            .unloaded_unembed
            .take()
            .expect("qwen3 Main unembed shell must exist during weight loading");
        match unembed {
            Some(unembed) => loaded_unembed.load_weights(&device, &mut store, unembed)?,
            None => loaded_unembed.load_tied_weights(loaded_embed.tied_weights()),
        }
        let loaded_unembed = Rc::new(loaded_unembed);
        self.main
            .component_mut()
            .load_weights(&device, &mut store, self.checkpoint.text(), main)?;

        self.main_text_embed.component_mut().load_weights(loaded_embed);
        self.gather_unembed.component_mut().load_weights(loaded_unembed);
        Ok(())
    }

    fn write_state(&self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        assert!(
            self.pending_transactions.transactions.is_empty(),
            "qwen3 state snapshots require all pending model transactions to complete"
        );
        let mut writer = StateSnapshotWriter::new(
            snapshot_path,
            self.state_snapshot_files(),
            plan,
            self.runtime.buffer_io(),
        )?;
        match plan {
            ExecutorHibernationPlan::All => {
                self.main_gqa_state
                    .write_full_state(&mut writer, MAIN_GQA_STATE_FILES)?;
                self.pages.write_full_state(&mut writer, PAGE_ARENA_STATE_FILES)?;
            },
            ExecutorHibernationPlan::Selected {
                request_slot_ranges,
                page_id_ranges,
            } => {
                self.main_gqa_state
                    .write_selected_state(&mut writer, MAIN_GQA_STATE_FILES, request_slot_ranges)?;
                self.pages
                    .write_selected_state(&mut writer, PAGE_ARENA_STATE_FILES, page_id_ranges)?;
            },
        }
        writer.commit()
    }

    pub fn unload_state(
        &mut self,
        snapshot_path: &Path,
        plan: &ExecutorHibernationPlan,
    ) -> Result<(), ModelExecutorError> {
        self.write_state(snapshot_path, plan)?;

        self.main.component_mut().unload_state();
        self.main_gqa_state.release_resources();
        self.pages.release_resources();
        Ok(())
    }

    pub fn load_state(
        &mut self,
        snapshot_path: &Path,
        plan: &ExecutorHibernationPlan,
    ) -> Result<(), ModelExecutorError> {
        let mut reader = StateSnapshotReader::open(
            snapshot_path,
            self.state_snapshot_files(),
            plan,
            self.runtime.buffer_io(),
        )?;
        let device = self.runtime.device().clone();
        self.main_gqa_state.allocate_resources(&device);
        self.pages.allocate_resources(&device);

        let result = (|| {
            match plan {
                ExecutorHibernationPlan::All => {
                    self.main_gqa_state.read_full_state(&mut reader, MAIN_GQA_STATE_FILES)?;
                    self.pages.read_full_state(&mut reader, PAGE_ARENA_STATE_FILES)?;
                },
                ExecutorHibernationPlan::Selected {
                    request_slot_ranges,
                    page_id_ranges,
                } => {
                    self.main_gqa_state
                        .read_selected_state(&mut reader, MAIN_GQA_STATE_FILES, request_slot_ranges)?;
                    self.pages
                        .read_selected_state(&mut reader, PAGE_ARENA_STATE_FILES, page_id_ranges)?;
                },
            }
            reader.finish()
        })();
        if let Err(error) = result {
            self.pages.release_resources();
            self.main_gqa_state.release_resources();
            return Err(error);
        }

        self.main.component_mut().load_state(&self.main_gqa_state);
        Ok(())
    }
}

pub struct Qwen3ModelOpsRecorder {
    main_text_embed_key: Qwen3MainTextEmbedReplayKey,
    main_text_embed_arguments: ReplayArguments,
    resource_embed_key: Option<inference_backend_metal::components::resource_embed::Shape>,
    resource_embed_arguments: ReplayArguments,
    main_key: Qwen3MainReplayKey,
    main_arguments: ReplayArguments,
    gather_unembed_key: Option<Qwen3GatherUnembedReplayKey>,
    gather_unembed_arguments: ReplayArguments,
    sampling_key: Option<TopKSamplingReplayKey>,
    sampling_arguments: ReplayArguments,
    num_main_sample_rows: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3ModelBatchResponse;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen3SampledOutput {
    decisions: Vec<Qwen3DecodeDecision>,
    timing: ModelOutputTiming,
}

struct Qwen3PendingTransactions {
    transactions: VecDeque<Qwen3PendingTransaction>,
}

struct Qwen3PendingTransaction {
    compute_seq: RawComputeSlotSeq,
}

impl Qwen3PendingTransactions {
    fn new() -> Self {
        Self {
            transactions: VecDeque::new(),
        }
    }

    fn push(&mut self, compute_seq: RawComputeSlotSeq) {
        if let Some(last) = self.transactions.back() {
            assert!(
                last.compute_seq < compute_seq,
                "qwen3 pending transaction sequences must increase"
            );
        }
        self.transactions.push_back(Qwen3PendingTransaction { compute_seq });
    }

    fn commit(&mut self, compute_seq: RawComputeSlotSeq) {
        let transaction = self
            .transactions
            .pop_front()
            .expect("qwen3 commit requires a pending batch");
        assert_eq!(
            transaction.compute_seq, compute_seq,
            "qwen3 commit sequence must match the oldest pending transaction"
        );
    }
}

impl ReplayableDecoderModel for Qwen3Executor {
    type ModelBatchRequest = Qwen3ModelBatchRequest;
    type ModelBatchHidden = Rc<Buffer>;
    type ModelBatchResponse = Qwen3ModelBatchResponse;
    type SampledOutput = Qwen3SampledOutput;
    type ModelOpsRecorder = Qwen3ModelOpsRecorder;
    type Submission = MetalReplaySubmission;

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn vocab_size(&self) -> usize {
        self.layout.vocab_size as usize
    }

    fn model_mode(&self) -> &'static str {
        match &self.checkpoint {
            Qwen3Checkpoint::Text(_) => "vanilla",
            Qwen3Checkpoint::Asr(_) => "asr",
        }
    }

    fn num_spec_tokens(&self) -> usize {
        Qwen3Executor::num_spec_tokens(self)
    }

    fn default_stop_sequences(&self) -> Vec<Vec<Token>> {
        self.default_stop_sequences.clone()
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        self.request_sampling.reset(request_slots);
        self.main_gqa_state.reset_req_slots(request_slots);
    }

    fn clear_replay_cache(&mut self) {
        Qwen3Executor::clear_replay_cache(self);
    }

    fn unload_state(&mut self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        Qwen3Executor::unload_state(self, snapshot_path, plan)
    }

    fn unload_weights(&mut self) {
        Qwen3Executor::unload_weights(self);
    }

    fn load_weights(&mut self) -> Result<(), ModelExecutorError> {
        Qwen3Executor::load_weights(self)
    }

    fn load_state(&mut self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        Qwen3Executor::load_state(self, snapshot_path, plan)
    }

    fn prepare_batch(&mut self, core_batch_req: &mut BatchDeviceRequest) -> Self::ModelBatchRequest {
        self.validate_input(core_batch_req);
        let sampler_configs = core_batch_req
            .dev_reqs
            .iter()
            .map(|request| {
                let seed = self
                    .request_sampling
                    .resolve(request.req_slot, request.sampling_config.seed);
                SamplerConfig::from_runtime(&request.sampling_config, seed)
            })
            .collect();
        let model_batch_request = Qwen3ModelBatchRequest::from_core_batch(core_batch_req, sampler_configs);
        let microbatch = model_batch_request.microbatch();
        self.sampler
            .set_params(microbatch.req_slots(), microbatch.sampler_configs());
        self.write_token_ids(microbatch.flat_token_ids());
        let num_main_active_tokens = microbatch
            .total_tokens()
            .try_into()
            .expect("qwen3 Main token count must fit u32");
        let num_main_total_tokens = num_main_active_tokens;
        self.prepare_gqa_page_ids(core_batch_req);
        let gqa_shape = self.main_gqa_state.prepare_metadata(
            microbatch.req_slots(),
            microbatch.token_indices(),
            microbatch.cu_tokens(),
            num_main_total_tokens,
        );
        debug_assert_eq!(gqa_shape.num_tokens as usize, microbatch.total_tokens());
        if let Some(main_resource_embed) = &mut self.main_resource_embed {
            main_resource_embed.prepare(&core_batch_req.dev_reqs);
        } else {
            debug_assert!(
                core_batch_req
                    .dev_reqs
                    .iter()
                    .all(|request| request.resource_placements.is_empty()),
                "text-only Qwen3 does not accept resource placements"
            );
        }
        model_batch_request
    }

    fn commit_batch(
        &mut self,
        core_batch_req: BatchDeviceRequest,
        sampled_output: Self::SampledOutput,
    ) -> BatchDeviceResponse {
        self.pending_transactions.commit(core_batch_req.seq);
        to_core_batch_resp(core_batch_req, sampled_output.decisions)
    }

    fn begin_ops_recording(&mut self, model_batch_request: &Self::ModelBatchRequest) -> Self::ModelOpsRecorder {
        let num_main_active_tokens = model_batch_request
            .microbatch()
            .total_tokens()
            .try_into()
            .expect("qwen3 Main token count must fit u32");
        let (main_text_embed_key, main_text_embed_arguments) =
            self.main_text_embed.component().prepare_replay(num_main_active_tokens);
        let (resource_embed_key, resource_embed_arguments) = self
            .main_resource_embed
            .as_ref()
            .and_then(MainResourceEmbed::prepare_replay)
            .map_or((None, ReplayArguments::new()), |(key, arguments)| {
                (Some(key), arguments)
            });
        let (main_key, mut main_arguments) = self.main.component().prepare_replay(
            num_main_active_tokens,
            self.main_gqa_state.metadata().replay_shape(),
            self.main_gqa_state.replay_topology(),
        );
        self.main_gqa_state.add_private_replay_arguments(&mut main_arguments);
        Qwen3ModelOpsRecorder {
            main_text_embed_key,
            main_text_embed_arguments,
            resource_embed_key,
            resource_embed_arguments,
            main_key,
            main_arguments,
            gather_unembed_key: None,
            gather_unembed_arguments: ReplayArguments::new(),
            sampling_key: None,
            sampling_arguments: ReplayArguments::new(),
            num_main_sample_rows: num_main_output_rows(model_batch_request.microbatch()),
        }
    }

    fn embed_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_request: &Self::ModelBatchRequest,
    ) -> Self::ModelBatchHidden {
        let input = Qwen3MainTextEmbedArgs {
            num_tokens: model_batch_request
                .microbatch()
                .total_tokens()
                .try_into()
                .expect("qwen3 MainTextEmbed token count must fit u32"),
            token_ids: &self.token_ids,
            hidden_output: &self.token_hidden_input,
        };
        let runtime = self.runtime.stream();
        let (recorded_key, _) = self.main_text_embed.record(runtime, &input);
        debug_assert_eq!(
            recorded_key, recorder.main_text_embed_key,
            "qwen3 MainTextEmbed replay input must match the prepared replay key"
        );
        let resource_embed_key = self
            .main_resource_embed
            .as_mut()
            .and_then(|main_resource_embed| main_resource_embed.record(runtime, &self.token_hidden_input));
        debug_assert_eq!(
            resource_embed_key, recorder.resource_embed_key,
            "Qwen3 ResourceEmbed replay input must match the prepared replay key"
        );
        Rc::clone(&self.token_hidden_input)
    }

    fn forward_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: Self::ModelBatchHidden,
    ) -> Self::ModelBatchHidden {
        let microbatch = model_batch_req.microbatch();
        debug_assert!(
            Rc::ptr_eq(&model_batch_hidden, &self.token_hidden_input),
            "qwen3 Main must consume the MainTextEmbed hidden workspace"
        );
        let input = Qwen3MainArgs {
            num_tokens: microbatch
                .total_tokens()
                .try_into()
                .expect("qwen3 Main token count must fit u32"),
            hidden_input: &model_batch_hidden,
            hidden_output: &self.hidden_output,
            gqa: self.main_gqa_state.metadata(),
            gqa_replay_topology: self.main_gqa_state.replay_topology(),
            pages: self.pages.buffer(),
        };
        let runtime = self.runtime.stream();
        let (recorded_key, _) = self.main.record(runtime, &input);
        debug_assert_eq!(
            recorded_key, recorder.main_key,
            "qwen3 Main replay input must match the prepared replay key"
        );
        self.pending_transactions.push(model_batch_req.compute_seq());
        Rc::clone(&self.hidden_output)
    }

    fn unembed_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: &Self::ModelBatchHidden,
    ) -> Self::ModelBatchResponse {
        debug_assert!(
            Rc::ptr_eq(model_batch_hidden, &self.hidden_output),
            "qwen3 Output must consume the executor final-norm hidden workspace"
        );
        if num_main_output_rows(model_batch_req.microbatch()) == 0 {
            return Qwen3ModelBatchResponse;
        }
        let (key, arguments) = self.prepare_gather_unembed_replay(model_batch_req.microbatch(), model_batch_hidden);
        recorder.gather_unembed_key = Some(key);
        recorder.gather_unembed_arguments = arguments;
        Qwen3ModelBatchResponse
    }

    fn sample_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        _model_batch_resp: &Self::ModelBatchResponse,
    ) {
        let microbatch = model_batch_req.microbatch();
        let num_main_sample_rows = num_main_output_rows(microbatch);
        assert_eq!(
            num_main_sample_rows, recorder.num_main_sample_rows,
            "qwen3 sampling rows must match the recording"
        );
        if num_main_sample_rows == 0 {
            return;
        }
        let (sampling_key, sampling_arguments) = self.record_sampling(microbatch);
        recorder.sampling_key = Some(sampling_key);
        recorder.sampling_arguments = sampling_arguments;
    }

    fn submit_main(&mut self, recorder: &Self::ModelOpsRecorder) -> Self::Submission {
        self.submit_main_recording(recorder)
    }

    fn read_main(
        &mut self,
        recorder: &Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        replay_elapsed: Duration,
        gpu_timestamp_durations: Option<&[Duration]>,
    ) -> Self::SampledOutput {
        let (decisions, mut timing) = if recorder.num_main_sample_rows == 0 {
            (
                Vec::new(),
                ModelOutputTiming {
                    main_replay_elapsed: replay_elapsed,
                    ..ModelOutputTiming::default()
                },
            )
        } else {
            let read_start = Instant::now();
            let decisions = self.read_sample_decisions(recorder.num_main_sample_rows);
            (
                decisions,
                ModelOutputTiming {
                    main_sample_replay_elapsed: replay_elapsed,
                    sample_read_elapsed: read_start.elapsed(),
                    ..ModelOutputTiming::default()
                },
            )
        };
        apply_main_gpu_timing(&mut timing, gpu_timestamp_durations, false, false, false);
        Qwen3SampledOutput { decisions, timing }
    }

    fn run_spec_prefill(&self, _model_batch_req: &Self::ModelBatchRequest) -> bool {
        false
    }

    fn prefill_spec(
        &mut self,
        _recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _sampled_output: &Self::SampledOutput,
    ) {
        panic!("Qwen3 does not support speculative execution")
    }

    fn run_spec_decode(
        &self,
        _model_batch_req: &Self::ModelBatchRequest,
        _sampled_output: &Self::SampledOutput,
    ) -> bool {
        false
    }

    fn decode_spec(
        &mut self,
        _recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _sampled_output: &Self::SampledOutput,
    ) {
        panic!("Qwen3 does not support speculative execution")
    }

    fn submit_spec(&mut self, _recorder: &Self::ModelOpsRecorder) -> Self::Submission {
        panic!("Qwen3 does not support speculative execution")
    }

    fn read_spec(
        &mut self,
        _recorder: &Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _sampled_output: Self::SampledOutput,
        _replay_elapsed: Duration,
        _gpu_timestamp_durations: Option<&[Duration]>,
    ) -> Self::SampledOutput {
        panic!("Qwen3 does not support speculative execution")
    }

    fn empty_sampled_output(&self) -> Self::SampledOutput {
        Qwen3SampledOutput::default()
    }

    fn sampled_output_len(&self, sampled_output: &Self::SampledOutput) -> usize {
        sampled_output.decisions.len()
    }

    fn sampled_output_timing(&self, sampled_output: &Self::SampledOutput) -> Option<ModelOutputTiming> {
        (!sampled_output.timing.is_zero()).then_some(sampled_output.timing)
    }
}

fn num_page_ids_per_block(num_tokens_per_block: usize, num_tokens_per_page: usize) -> usize {
    assert!(num_tokens_per_block > 0, "qwen3 GQA requires positive tokens per block");
    assert!(num_tokens_per_page > 0, "qwen3 GQA requires positive tokens per page");
    assert!(
        num_tokens_per_block.is_multiple_of(num_tokens_per_page),
        "qwen3 GQA tokens per block must be divisible by tokens per page"
    );
    num_tokens_per_block / num_tokens_per_page
}

#[cfg(test)]
mod tests {
    use inference_executor_core::model::ExecutionSubmission;
    use inference_executor_core::model::ReplayableDecoderModel;
    use inference_executor_core::model::qwen::v3::Qwen3ModelBatchRequest;
    use inference_executor_core::model::qwen::v3_asr::audio_output_rows;
    use inference_runtime_core::compute::BatchDeviceRequest;
    use inference_runtime_core::compute::DecoderSyncBlocks;
    use inference_runtime_core::compute::DeviceRequest;
    use inference_runtime_core::compute::DeviceResourcePlacement;
    use inference_runtime_core::compute::QueryTokens;
    use inference_runtime_core::config::SamplingConfig;
    use inference_runtime_core::memory::BlockAllocator;
    use inference_runtime_core::runtime::Token;

    use super::Qwen3Executor;
    use super::Qwen3ExecutorConfig;
    use super::init_qwen3_asr_model;

    #[test]
    fn test_executor_config_supports_qwen3_batch_contract() {
        Qwen3ExecutorConfig {
            max_requests: 1,
            max_tokens: 4,
            max_tokens_per_request: 4,
            num_cache_pages: 1,
            num_tokens_per_block: 1024,
        }
        .validate();
        fn assert_compact_qwen3_batch<T: ReplayableDecoderModel<ModelBatchRequest = Qwen3ModelBatchRequest>>() {}
        assert_compact_qwen3_batch::<Qwen3Executor>();
    }

    #[test]
    #[ignore = "requires KHALAMARI_QWEN3_ASR_MODEL_DIR and the pinned local checkpoint"]
    fn test_qwen3_asr_replaces_placeholder_embedding_before_text_decoder() {
        let model_dir = std::env::var("KHALAMARI_QWEN3_ASR_MODEL_DIR").unwrap();
        let mut loaded = init_qwen3_asr_model(
            model_dir,
            Qwen3ExecutorConfig {
                max_requests: 1,
                max_tokens: 16,
                max_tokens_per_request: 16,
                num_cache_pages: 64,
                num_tokens_per_block: 1024,
            },
        )
        .unwrap();
        let config = loaded.executor.asr_model_config();

        assert_eq!(loaded.executor.model_name(), "qwen3_asr");
        assert_eq!(loaded.executor.model_mode(), "asr");
        assert_eq!(config.text.hidden_size, 2048);
        assert_eq!(
            loaded.resource_arena.storage().buffer().len_bytes(),
            audio_output_rows(config.preprocessor.max_frames)
                * config.text.hidden_size
                * std::mem::size_of::<half::bf16>()
        );

        let hidden_dim = config.text.hidden_size;
        let replacement = half::bf16::from_f32(0.5).to_bits();
        let mut allocation = loaded
            .resource_arena
            .alloc_segment(hidden_dim * std::mem::size_of::<half::bf16>())
            .unwrap();
        for bytes in allocation.slice_mut().as_chunks_mut::<2>().0 {
            bytes.copy_from_slice(&replacement.to_ne_bytes());
        }
        let page_ids = vec![0; loaded.executor.num_kv_page_ids_per_block()];
        let mut core_batch = BatchDeviceRequest::new(
            0,
            [DeviceRequest::new(
                0,
                0,
                QueryTokens::Prefill {
                    epoch: 0,
                    token_index: 0,
                    tokens: vec![Token::new(1), Token::new(config.audio_token_id)],
                    window: 2,
                },
                DecoderSyncBlocks::new(0, vec![vec![page_ids]], vec![]),
                vec![DeviceResourcePlacement::new(
                    allocation.offset_bytes(),
                    allocation.len_bytes(),
                    vec![(1, 0, 1)],
                )],
                SamplingConfig::default(),
            )],
        );
        let model_batch = loaded.executor.prepare_batch(&mut core_batch);
        let mut recorder = loaded.executor.begin_ops_recording(&model_batch);
        let hidden = loaded.executor.embed_main(&mut recorder, &model_batch);
        loaded.executor.forward_main(&mut recorder, &model_batch, hidden);
        loaded.executor.submit_main(&recorder).wait();

        assert_eq!(
            loaded
                .executor
                .token_hidden_input
                .read_typed::<u16>(hidden_dim, hidden_dim),
            vec![replacement; hidden_dim]
        );
    }
}
