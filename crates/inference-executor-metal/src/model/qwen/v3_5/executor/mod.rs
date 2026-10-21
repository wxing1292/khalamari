//! Qwen3.5-family outer executor.
//!
//! The outer executor owns one closed model composition. Main always runs first. Vanilla returns the Main decision.
//! MTP uses its existing combined Spec invocation. DFlash2 use independent Spec Prefill and Spec Decode
//! recordings. The Spec checkpoint is never part of the Main owner.
//!
//! ```text
//! previous Spec proposal
//! {draft tokens, draft probabilities}
//!                    |
//!                    v
//! request microbatch
//! {committed tokens + previous speculative suffix}
//!                    |
//!                    v
//! +----------------------------- Main module -----------------------------+
//! | token IDs -> Main Embed -> Main transformer                            |
//! |                              |                                         |
//! |                              +-> selected residual capture             |
//! |                              |                                         |
//! |                              v                                         |
//! |                    Gather + Main Unembed                               |
//! |                              |                                         |
//! |                              v                                         |
//! |              normal sampling or rejection sampling                    |
//! |                              |                                         |
//! |                              v                                         |
//! |        {validated draft prefix, newly sampled anchor token}            |
//! +------------------------------+-----------------------------------------+
//!                                |
//!             +------------------+------------------+
//!             |                  |                  |
//!             v                  v                  v
//!          Vanilla              MTP        DFlash2
//!           return       combined Spec       Spec Prefill
//!                         invocation          Spec Decode
//! ```

use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use inference_backend_metal::MetalRuntime;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayExecution;
use inference_backend_metal::metal::ReplayProgram;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::ExecutionSubmission;
use inference_executor_core::model::ModelOutputTiming;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_core::model::qwen::v3_5::Qwen35DecodeDecision;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelBatchRequest;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::Qwen35PendingTransactions;
use inference_executor_core::model::qwen::v3_5::Qwen35SampledTokens;
use inference_executor_core::model::qwen::v3_5::gather_flat_indices;
use inference_executor_core::model::qwen::v3_5::num_main_output_rows;
use inference_executor_core::model::qwen::v3_5::sample_decisions_from_sampled_tokens;
use inference_executor_core::model::qwen::v3_5::sample_req_slots;
use inference_executor_core::model::qwen::v3_5::sample_sampler_configs;
use inference_executor_core::model::qwen::v3_5::sample_token_positions;
use inference_executor_core::model::qwen::v3_5::to_core_batch_resp;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MTPWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35ModelWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::resolve_qwen35_model_weight_bindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::resolve_qwen35_mtp_weight_bindings;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::sampling::RequestSamplingState;
use inference_executor_core::sampling::SamplerConfig;
use inference_runtime_core::compute::BatchDevReq;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::ExecutorHibernationPlan;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::runtime::RawComputeSlotSeq;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::Token;

use crate::attn::gqa::batch_metadata::GQAReplayShape;
use crate::attn::gqa::request_page_table::GQAPageTableLayout;
use crate::checkpoint::SafeTensorStore;
use crate::def::submission::MetalReplaySubmission;
use crate::model::embedding::Embed;
use crate::model::main_residual_capture::MainResidualCapture;
use crate::model::main_resource_embed::MainResourceEmbed;
use crate::model::page_arena::PageArena;
use crate::model::qwen::apply_main_gpu_timing;
use crate::model::qwen::split_main_lane_page_ids;
use crate::model::qwen::v3_5::main::Qwen35Main;
use crate::model::qwen::v3_5::main::Qwen35MainArgs;
use crate::model::qwen::v3_5::main::Qwen35MainReplayKey;
use crate::model::qwen::v3_5::main::output::Qwen35GatherUnembed;
use crate::model::qwen::v3_5::main::output::Qwen35GatherUnembedArgs;
use crate::model::qwen::v3_5::main::output::Qwen35GatherUnembedReplayKey;
use crate::model::qwen::v3_5::main::text_embed::Qwen35MainTextEmbed;
use crate::model::qwen::v3_5::main::text_embed::Qwen35MainTextEmbedArgs;
use crate::model::qwen::v3_5::main::text_embed::Qwen35MainTextEmbedReplayKey;
use crate::model::qwen::v3_5::mtp::Qwen35MTP;
use crate::model::qwen::v3_5::mtp::Qwen35MTPArgs;
use crate::model::qwen::v3_5::mtp::Qwen35MTPReplayKey;
use crate::model::qwen::v3_5::mtp::decode_plan::Qwen35MTPDecodePlan;
use crate::model::qwen::v3_5::mtp::decode_plan::Qwen35MTPDecodeTokenSource;
use crate::model::qwen::v3_5::mtp::embed::Qwen35MTPEmbed;
use crate::model::qwen::v3_5::mtp::embed::Qwen35MTPEmbedArgs;
use crate::model::qwen::v3_5::mtp::embed::Qwen35MTPEmbedReplayKey;
use crate::model::qwen::v3_5::mtp::hidden_state_cache::Qwen35MTPCacheState;
use crate::model::qwen::v3_5::mtp::hidden_state_cache::Qwen35MTPHiddenStateCache;
use crate::model::qwen::v3_5::mtp::hidden_state_transfer::Qwen35MTPHiddenStateTransfer;
use crate::model::qwen::v3_5::mtp::hidden_state_transfer::Qwen35MTPHiddenStateTransferArgs;
use crate::model::qwen::v3_5::mtp::hidden_state_transfer::Qwen35MTPHiddenStateTransferPlan;
use crate::model::qwen::v3_5::mtp::hidden_state_transfer::Qwen35MTPHiddenStateTransferReplayKey;
use crate::model::qwen::v3_5::mtp::hidden_state_transfer::append_mtp_hidden_state_cache_write_routes;
use crate::model::qwen::v3_x::dflash2::execution::Qwen3xDFlash2DecodeRecording;
use crate::model::qwen::v3_x::dflash2::execution::Qwen3xDFlash2Execution;
use crate::model::qwen::v3_x::dflash2::execution::Qwen3xDFlash2PrefillRecording;
use crate::model::qwen::v3_x::state::Qwen3xGDNState;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::state_snapshot::FullStateIO;
use crate::model::state_snapshot::GDNStateSnapshotFiles;
use crate::model::state_snapshot::GQAStateSnapshotFiles;
use crate::model::state_snapshot::PageArenaStateSnapshotFiles;
use crate::model::state_snapshot::SelectedStateIO;
use crate::model::state_snapshot::StateSnapshotFile;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;
use crate::model::unembedding::Unembed;
use crate::replay::Replay;
use crate::sampling::rejection_replay::PreparedRejection;
use crate::sampling::rejection_replay::RejectionReplayKey;
use crate::sampling::rejection_replay::RejectionSamplerInput;
use crate::sampling::rejection_replay::RejectionSampling;
use crate::sampling::rejection_replay::RejectionSamplingInput;
use crate::sampling::rejection_sampling::SparseRejectionSamplingReqParams;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_replay::DraftSampling;
use crate::sampling::top_k_replay::DraftSamplingInput;
use crate::sampling::top_k_replay::Sampling;
use crate::sampling::top_k_replay::SamplingInput;
use crate::sampling::top_k_replay::TopKSamplingReplayKey;
use crate::sampling::top_k_sampling::TopKSampling;
use crate::sampling::top_k_sampling::TopKSamplingOutputBuffers;
use crate::sampling::top_k_sampling::TopKSamplingWriteDistributionOutput;
use crate::sampling::top_k_sampling::config::TopKSamplingBounds;
use crate::trace;

mod load;

pub use load::Qwen35ExecutorConfig;
use load::Qwen35ModelLayout;
pub use load::init_qwen_3_5_model;
pub use load::init_qwen_3_5_model_with_dflash2;
pub use load::init_qwen_3_5_model_with_mtp;

include!("batch.rs");
include!("dflash2.rs");
include!("main.rs");
include!("mtp.rs");
include!("recording.rs");
include!("sampling.rs");

const PAGE_ARENA_STATE_FILES: PageArenaStateSnapshotFiles =
    PageArenaStateSnapshotFiles::new(StateSnapshotFile::PageArena);
const MAIN_GQA_STATE_FILES: GQAStateSnapshotFiles =
    GQAStateSnapshotFiles::new(StateSnapshotFile::MainGQARequestPageTable);
const MAIN_GDN_STATE_FILES: GDNStateSnapshotFiles = GDNStateSnapshotFiles::new(
    StateSnapshotFile::MainGDNRequestStateTable,
    StateSnapshotFile::MainGDNRecurrentState,
    StateSnapshotFile::MainGDNConvState,
);
const MTP_GQA_STATE_FILES: GQAStateSnapshotFiles =
    GQAStateSnapshotFiles::new(StateSnapshotFile::MTPGQARequestPageTable);
const DFLASH2_GQA_STATE_FILES: GQAStateSnapshotFiles =
    GQAStateSnapshotFiles::new(StateSnapshotFile::DFlash2GQARequestPageTable);
const VANILLA_STATE_SNAPSHOT_FILES: &[StateSnapshotFile] = &[
    PAGE_ARENA_STATE_FILES.pages(),
    MAIN_GQA_STATE_FILES.request_page_table(),
    MAIN_GDN_STATE_FILES.request_state_table(),
    MAIN_GDN_STATE_FILES.recurrent_state(),
    MAIN_GDN_STATE_FILES.conv_state(),
];
const MTP_STATE_SNAPSHOT_FILES: &[StateSnapshotFile] = &[
    PAGE_ARENA_STATE_FILES.pages(),
    MAIN_GQA_STATE_FILES.request_page_table(),
    MAIN_GDN_STATE_FILES.request_state_table(),
    MAIN_GDN_STATE_FILES.recurrent_state(),
    MAIN_GDN_STATE_FILES.conv_state(),
    MTP_GQA_STATE_FILES.request_page_table(),
];
const DFLASH2_STATE_SNAPSHOT_FILES: &[StateSnapshotFile] = &[
    PAGE_ARENA_STATE_FILES.pages(),
    MAIN_GQA_STATE_FILES.request_page_table(),
    MAIN_GDN_STATE_FILES.request_state_table(),
    MAIN_GDN_STATE_FILES.recurrent_state(),
    MAIN_GDN_STATE_FILES.conv_state(),
    DFLASH2_GQA_STATE_FILES.request_page_table(),
];

#[allow(clippy::upper_case_acronyms)]
enum Qwen35Speculator {
    Vanilla,
    MTP(Box<Qwen35MTPSpeculator>),
    DFlash2(Box<Qwen35DFlash2Speculator>),
}

#[allow(clippy::upper_case_acronyms)]
enum Qwen35WeightSource {
    Vanilla,
    MTP {
        model_dir: PathBuf,
        config: Box<Qwen35ModelConfig>,
    },
    DFlash2 {
        model_dir: PathBuf,
        config: Box<Qwen3xDFlash2Config>,
    },
}

struct Qwen35SpeculativeResources {
    rejection_sampling: Replay<RejectionSampling>,
    spec_probs: SpecProbsStore,
    target_distribution_indices: Buffer,
}

struct Qwen35MTPSpeculator {
    common: Qwen35SpeculativeResources,
    num_spec_tokens: usize,
    hidden_input: Rc<Buffer>,
    previous_hidden_routes: Buffer,
    hidden_state_cache_write_routes: Buffer,
    draft_distribution_indices: Buffer,
    previous_hidden: Buffer,
    hidden_state_cache: Qwen35MTPHiddenStateCache,
    hidden_state_transfers: Vec<Replay<Qwen35MTPHiddenStateTransfer>>,
    embed: Replay<Qwen35MTPEmbed>,
    body: Replay<Qwen35MTP>,
    sampling: Replay<DraftSampling>,
    gqa_state: Qwen3xGQAState,
    execution: Qwen35MTPExecution,
}

struct Qwen35MTPExecution {
    requests: Vec<Qwen35MTPRequest>,
    draft_token_ids: Vec<i32>,
    draft_probs: Vec<f32>,
    read_elapsed: Duration,
    build_elapsed: Duration,
    completed_steps: usize,
    gpu_elapsed: Option<Duration>,
    pending_hidden_state_cache_write_routes: Vec<u32>,
}

impl Qwen35MTPExecution {
    fn new(max_requests: usize, num_spec_tokens: usize) -> Self {
        let proposal_capacity = max_requests
            .checked_mul(num_spec_tokens)
            .expect("qwen3.5 MTP proposal capacity must fit usize");
        Self {
            requests: Vec::with_capacity(max_requests),
            draft_token_ids: Vec::with_capacity(proposal_capacity),
            draft_probs: Vec::with_capacity(proposal_capacity),
            read_elapsed: Duration::ZERO,
            build_elapsed: Duration::ZERO,
            completed_steps: 0,
            gpu_elapsed: Some(Duration::ZERO),
            pending_hidden_state_cache_write_routes: Vec::with_capacity(num_spec_tokens * max_requests * 2),
        }
    }

    fn begin(&mut self, requests: Vec<Qwen35MTPRequest>) {
        self.requests = requests;
        self.draft_token_ids.clear();
        self.draft_probs.clear();
        self.read_elapsed = Duration::ZERO;
        self.build_elapsed = Duration::ZERO;
        self.completed_steps = 0;
        self.gpu_elapsed = Some(Duration::ZERO);
        self.pending_hidden_state_cache_write_routes.clear();
    }

    fn push_step(
        &mut self,
        draft_token_ids: &[i32],
        draft_probs: &[f32],
        read_elapsed: Duration,
        gpu_elapsed: Option<Duration>,
    ) {
        assert_eq!(draft_token_ids.len(), draft_probs.len());
        self.draft_token_ids.extend_from_slice(draft_token_ids);
        self.draft_probs.extend_from_slice(draft_probs);
        self.read_elapsed += read_elapsed;
        self.gpu_elapsed = self
            .gpu_elapsed
            .zip(gpu_elapsed)
            .map(|(total, elapsed)| total + elapsed);
        self.completed_steps += 1;
    }
}

struct Qwen35DFlash2Speculator {
    common: Qwen35SpeculativeResources,
    execution: Qwen3xDFlash2Execution,
}

impl Qwen35Speculator {
    fn is_enabled(&self) -> bool {
        !matches!(self, Self::Vanilla)
    }

    fn is_mtp(&self) -> bool {
        matches!(self, Self::MTP(_))
    }

    fn is_dflash2(&self) -> bool {
        matches!(self, Self::DFlash2(_))
    }

    fn num_spec_tokens(&self) -> usize {
        match self {
            Self::Vanilla => 0,
            Self::MTP(mtp) => mtp.num_spec_tokens,
            Self::DFlash2(dflash2) => dflash2.execution.num_spec_tokens(),
        }
    }

    fn num_gqa_page_ids_per_main_lane_block(&self) -> usize {
        match self {
            Self::Vanilla | Self::MTP(_) => 0,
            Self::DFlash2(dflash2) => dflash2.execution.num_gqa_page_ids_per_block(),
        }
    }

    fn write_page_ids(&self, req_slot: u32, block_index: usize, page_ids: &[u32]) {
        match self {
            Self::Vanilla | Self::MTP(_) => {
                assert!(
                    page_ids.is_empty(),
                    "qwen3.5 Main cache block must not contain page IDs for a speculator without persistent KV"
                )
            },
            Self::DFlash2(dflash2) => dflash2.execution.write_page_ids(req_slot, block_index, page_ids),
        }
    }

    fn mtp(&self) -> &Qwen35MTPSpeculator {
        match self {
            Self::MTP(mtp) => mtp,
            Self::Vanilla | Self::DFlash2(_) => panic!("qwen3.5 executor has no MTP resources"),
        }
    }

    fn mtp_mut(&mut self) -> &mut Qwen35MTPSpeculator {
        match self {
            Self::MTP(mtp) => mtp,
            Self::Vanilla | Self::DFlash2(_) => panic!("qwen3.5 executor has no MTP resources"),
        }
    }

    fn dflash2(&self) -> &Qwen35DFlash2Speculator {
        match self {
            Self::DFlash2(dflash2) => dflash2,
            Self::Vanilla | Self::MTP(_) => panic!("qwen3.5 executor has no DFlash2 resources"),
        }
    }

    fn dflash2_mut(&mut self) -> &mut Qwen35DFlash2Speculator {
        match self {
            Self::DFlash2(dflash2) => dflash2,
            Self::Vanilla | Self::MTP(_) => panic!("qwen3.5 executor has no DFlash2 resources"),
        }
    }

    fn common(&self) -> &Qwen35SpeculativeResources {
        match self {
            Self::Vanilla => panic!("qwen3.5 Vanilla executor has no speculative resources"),
            Self::MTP(mtp) => &mtp.common,
            Self::DFlash2(dflash2) => &dflash2.common,
        }
    }

    fn common_mut(&mut self) -> &mut Qwen35SpeculativeResources {
        match self {
            Self::Vanilla => panic!("qwen3.5 Vanilla executor has no speculative resources"),
            Self::MTP(mtp) => &mut mtp.common,
            Self::DFlash2(dflash2) => &mut dflash2.common,
        }
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        match self {
            Self::Vanilla => {},
            Self::MTP(mtp) => {
                mtp.gqa_state.reset_req_slots(request_slots);
                mtp.hidden_state_cache.reset_req_slots(request_slots);
                mtp.common.spec_probs.reset_req_slots(request_slots);
            },
            Self::DFlash2(dflash2) => {
                dflash2.execution.reset_req_slots(request_slots);
                dflash2.common.spec_probs.reset_req_slots(request_slots);
            },
        }
    }

    fn clear_replay_cache(&mut self) {
        match self {
            Self::Vanilla => {},
            Self::MTP(mtp) => {
                mtp.embed.clear();
                mtp.body.clear();
                for transfer in &mut mtp.hidden_state_transfers {
                    transfer.clear();
                }
                mtp.sampling.clear();
                mtp.common.rejection_sampling.clear();
            },
            Self::DFlash2(dflash2) => {
                dflash2.execution.clear_replay_cache();
                dflash2.common.rejection_sampling.clear();
            },
        }
    }

    fn unload_weights(&mut self) {
        match self {
            Self::Vanilla => {},
            Self::MTP(mtp) => {
                mtp.body.component_mut().unload_weights();
                drop(mtp.embed.component_mut().unload_weights());
            },
            Self::DFlash2(dflash2) => dflash2.execution.unload_weights(),
        }
    }

    fn load_weights(
        &mut self,
        device: &inference_backend_metal::metal::Device,
        source: &Qwen35WeightSource,
        main_text_embed: &Rc<Embed>,
        main_unembed: &Rc<Unembed>,
    ) -> Result<Option<Rc<dyn MainResidualCapture>>, ModelExecutorError> {
        match (self, source) {
            (Self::Vanilla, Qwen35WeightSource::Vanilla) => Ok(None),
            (Self::MTP(mtp), Qwen35WeightSource::MTP { model_dir, config }) => {
                let mut store = SafeTensorStore::from_model_dir(model_dir)?;
                let Qwen35MTPWeightBindings {
                    embed,
                    body,
                    final_norm_weight,
                } = resolve_qwen35_mtp_weight_bindings(config, store.index().tensor_names())?;
                mtp.embed
                    .component_mut()
                    .load_weights(device, &mut store, config, embed)?;
                mtp.embed
                    .component_mut()
                    .load_shared_weights(Rc::clone(main_text_embed));
                mtp.body
                    .component_mut()
                    .load_weights(device, &mut store, config, body, final_norm_weight)?;
                Ok(None)
            },
            (Self::DFlash2(dflash2), Qwen35WeightSource::DFlash2 { model_dir, config }) => {
                dflash2
                    .execution
                    .load_weights(device, model_dir, config, main_text_embed, main_unembed)?;
                let capture: Rc<dyn MainResidualCapture> = dflash2.execution.main_feature_projector();
                Ok(Some(capture))
            },
            _ => panic!("qwen3.5 speculator and weight source must have matching variants"),
        }
    }

    fn write_full_state(&self, writer: &mut StateSnapshotWriter) -> Result<(), ModelExecutorError> {
        match self {
            Self::Vanilla => Ok(()),
            Self::MTP(mtp) => mtp.gqa_state.write_full_state(writer, MTP_GQA_STATE_FILES),
            Self::DFlash2(dflash2) => dflash2.execution.write_full_state(writer, DFLASH2_GQA_STATE_FILES),
        }
    }

    fn write_selected_state(
        &self,
        writer: &mut StateSnapshotWriter,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        match self {
            Self::Vanilla => Ok(()),
            Self::MTP(mtp) => {
                mtp.gqa_state
                    .write_selected_state(writer, MTP_GQA_STATE_FILES, request_slot_ranges)
            },
            Self::DFlash2(dflash2) => {
                dflash2
                    .execution
                    .write_selected_state(writer, DFLASH2_GQA_STATE_FILES, request_slot_ranges)
            },
        }
    }

    fn unload_state(&mut self) {
        match self {
            Self::Vanilla => {},
            Self::MTP(mtp) => {
                mtp.body.component_mut().unload_state();
                mtp.gqa_state.release_resources();
            },
            Self::DFlash2(dflash2) => dflash2.execution.unload_state(),
        }
    }

    fn allocate_resources(&mut self, device: &inference_backend_metal::metal::Device) {
        match self {
            Self::Vanilla => {},
            Self::MTP(mtp) => {
                mtp.gqa_state.allocate_resources(device);
            },
            Self::DFlash2(dflash2) => dflash2.execution.allocate_resources(device),
        }
    }

    fn release_resources(&mut self) {
        match self {
            Self::Vanilla => {},
            Self::MTP(mtp) => mtp.gqa_state.release_resources(),
            Self::DFlash2(dflash2) => dflash2.execution.release_resources(),
        }
    }

    fn load_state(&mut self) {
        match self {
            Self::Vanilla => {},
            Self::MTP(mtp) => mtp.body.component_mut().load_state(&mtp.gqa_state),
            Self::DFlash2(dflash2) => dflash2.execution.load_state(),
        }
    }

    fn read_full_state(&mut self, reader: &mut StateSnapshotReader) -> Result<(), ModelExecutorError> {
        match self {
            Self::Vanilla => Ok(()),
            Self::MTP(mtp) => mtp.gqa_state.read_full_state(reader, MTP_GQA_STATE_FILES),
            Self::DFlash2(dflash2) => dflash2.execution.read_full_state(reader, DFLASH2_GQA_STATE_FILES),
        }
    }

    fn read_selected_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        request_slot_ranges: &[Range<RawRequestSlot>],
    ) -> Result<(), ModelExecutorError> {
        match self {
            Self::Vanilla => Ok(()),
            Self::MTP(mtp) => {
                mtp.gqa_state
                    .read_selected_state(reader, MTP_GQA_STATE_FILES, request_slot_ranges)
            },
            Self::DFlash2(dflash2) => {
                dflash2
                    .execution
                    .read_selected_state(reader, DFLASH2_GQA_STATE_FILES, request_slot_ranges)
            },
        }
    }
}

pub struct Qwen35Executor {
    model_name: String,
    model_dir: PathBuf,
    model_config: Qwen35ModelConfig,
    weight_source: Qwen35WeightSource,
    default_stop_sequences: Vec<Vec<Token>>,
    config: Qwen35ExecutorConfig,
    runtime: MetalRuntime,
    layout: Qwen35ModelLayout,
    token_ids: Buffer,
    token_hidden_input: Rc<Buffer>,
    hidden_output: Rc<Buffer>,
    gather_flat_indices: Buffer,
    unembed_hidden: Buffer,
    unembed_logits: Buffer,
    main_text_embed: Replay<Qwen35MainTextEmbed>,
    main_resource_embed: Option<MainResourceEmbed>,
    main: Replay<Qwen35Main>,
    gather_unembed: Replay<Qwen35GatherUnembed>,
    sampling: Replay<Sampling>,
    sampler: Rc<TopKSampling>,
    sampler_bounds: TopKSamplingBounds,
    sampler_output: TopKSamplingOutputBuffers,
    request_sampling: RequestSamplingState,
    main_gqa_state: Qwen3xGQAState,
    main_gdn_state: Qwen3xGDNState,
    speculator: Qwen35Speculator,
    pages: PageArena,
    pending_transactions: Qwen35PendingTransactions,
    gqa_page_table_layout: GQAPageTableLayout,
    num_gqa_page_ids_per_main_lane_block: usize,
    unloaded_embed: Option<Embed>,
    unloaded_unembed: Option<Unembed>,
}

impl Qwen35Executor {
    fn record_spec(&mut self, recorder: &mut Qwen35ModelOpsRecorder, microbatch: &Qwen35Microbatch) {
        let runtime = self.runtime.stream();
        let pages = self.pages.buffer();
        // Spec Prefill consumes every Main row. Main GQA metadata already owns
        // the expanded per-row request slots and token indices.
        let spec_prefill = self.main_gqa_state.metadata();
        let decode_req_indices = recorder
            .rejection_prepared
            .as_ref()
            .map_or(&[][..], |prepared| prepared.decode_req_indices.as_slice());

        let mut req_slots = Vec::with_capacity(decode_req_indices.len());
        let mut anchor_indices = Vec::with_capacity(decode_req_indices.len());
        let mut num_spec_tokens_by_request = Vec::with_capacity(decode_req_indices.len());
        let mut sampler_configs = Vec::with_capacity(decode_req_indices.len());
        for &req_index in decode_req_indices {
            let num_spec_tokens = microbatch.num_spec_tokens(req_index);
            let anchor_index =
                microbatch.token_indices()[req_index] + microbatch.num_total_tokens(req_index) - num_spec_tokens;
            req_slots.push(microbatch.req_slots()[req_index]);
            anchor_indices.push(anchor_index);
            num_spec_tokens_by_request.push(num_spec_tokens);
            sampler_configs.push(microbatch.sampler_configs()[req_index]);
        }

        let token_ids = &self.token_ids;
        match &mut self.speculator {
            Qwen35Speculator::DFlash2(dflash2) => {
                assert!(recorder.dflash2_spec_prefill.is_none() && recorder.dflash2_spec_decode.is_none());
                recorder.dflash2_spec_prefill =
                    Some(dflash2.execution.record_spec_prefill(runtime, spec_prefill, pages));
                if recorder.rejection_prepared.is_none() {
                    return;
                }
                let rejection_sampling = dflash2.common.rejection_sampling.component().rejector().output();
                let (decode_prepare, num_requests) = dflash2.execution.record_decode_prepare(
                    runtime,
                    token_ids,
                    rejection_sampling,
                    &req_slots,
                    &anchor_indices,
                    &num_spec_tokens_by_request,
                    &dflash2.common.spec_probs,
                );
                recorder.dflash2_spec_decode = Some(dflash2.execution.record_spec_decode(
                    runtime,
                    token_ids,
                    decode_prepare,
                    num_requests,
                    req_slots,
                    pages,
                    &dflash2.common.spec_probs,
                ));
            },
            Qwen35Speculator::Vanilla | Qwen35Speculator::MTP(_) => {
                panic!("qwen3.5 combined Spec recording requires DFlash2")
            },
        }
    }

    fn state_snapshot_files(&self) -> &'static [StateSnapshotFile] {
        match &self.speculator {
            Qwen35Speculator::Vanilla => VANILLA_STATE_SNAPSHOT_FILES,
            Qwen35Speculator::MTP(_) => MTP_STATE_SNAPSHOT_FILES,
            Qwen35Speculator::DFlash2(_) => DFLASH2_STATE_SNAPSHOT_FILES,
        }
    }

    pub fn clear_replay_cache(&mut self) {
        self.finish_gdn_commit();
        self.main_text_embed.clear();
        if let Some(main_resource_embed) = &mut self.main_resource_embed {
            main_resource_embed.clear();
        }
        self.main.clear();
        self.gather_unembed.clear();
        self.sampling.clear();
        self.main_gdn_state.clear_replay_cache();
        self.speculator.clear_replay_cache();
    }

    pub fn unload_weights(&mut self) {
        let embed = self.main_text_embed.component_mut().unload_weights();
        let unembed = self.gather_unembed.component_mut().unload_weights();

        self.main.component_mut().unset_residual_capture();
        self.speculator.unload_weights();
        self.main.component_mut().unload_weights();

        let mut unembed = Rc::try_unwrap(unembed)
            .unwrap_or_else(|_| panic!("qwen3.5 Main unembed must be uniquely owned during weight unloading"));
        unembed.unload_weights();
        self.unloaded_unembed = Some(unembed);

        let mut embed = Rc::try_unwrap(embed)
            .unwrap_or_else(|_| panic!("qwen3.5 Main embed must be uniquely owned during weight unloading"));
        embed.unload_weights();
        self.unloaded_embed = Some(embed);
    }

    pub fn load_weights(&mut self) -> Result<(), ModelExecutorError> {
        let device = self.runtime.device().clone();
        let mut store = SafeTensorStore::from_model_dir(&self.model_dir)?;
        let Qwen35ModelWeightBindings { embed, main, unembed } =
            resolve_qwen35_model_weight_bindings(&self.model_config, store.index().tensor_names())?;
        let mut loaded_embed = self
            .unloaded_embed
            .take()
            .expect("qwen3.5 Main embed shell must exist during weight loading");
        loaded_embed.load_weights(&device, &mut store, embed)?;
        let loaded_embed = Rc::new(loaded_embed);
        let mut loaded_unembed = self
            .unloaded_unembed
            .take()
            .expect("qwen3.5 Main unembed shell must exist during weight loading");
        loaded_unembed.load_weights(&device, &mut store, unembed)?;
        let loaded_unembed = Rc::new(loaded_unembed);
        self.main
            .component_mut()
            .load_weights(&device, &mut store, &self.model_config, main)?;
        let residual_capture =
            self.speculator
                .load_weights(&device, &self.weight_source, &loaded_embed, &loaded_unembed)?;
        self.main.component_mut().set_residual_capture(residual_capture);
        self.main_text_embed.component_mut().load_weights(loaded_embed);
        self.gather_unembed.component_mut().load_weights(loaded_unembed);
        Ok(())
    }

    fn write_state(&mut self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        self.finish_gdn_commit();
        assert!(
            self.pending_transactions.is_empty(),
            "qwen3.5 state snapshots require all pending model transactions to complete"
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
                self.main_gdn_state
                    .write_full_state(&mut writer, MAIN_GDN_STATE_FILES)?;
                self.speculator.write_full_state(&mut writer)?;
                self.pages.write_full_state(&mut writer, PAGE_ARENA_STATE_FILES)?;
            },
            ExecutorHibernationPlan::Selected {
                request_slot_ranges,
                page_id_ranges,
            } => {
                self.main_gqa_state
                    .write_selected_state(&mut writer, MAIN_GQA_STATE_FILES, request_slot_ranges)?;
                self.main_gdn_state
                    .write_selected_state(&mut writer, MAIN_GDN_STATE_FILES, request_slot_ranges)?;
                self.speculator.write_selected_state(&mut writer, request_slot_ranges)?;
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

        self.speculator.unload_state();
        self.main.component_mut().unload_state();
        self.main_gqa_state.release_resources();
        self.main_gdn_state.release_resources();
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
        self.main_gdn_state.allocate_resources(&device);
        self.speculator.allocate_resources(&device);
        self.pages.allocate_resources(&device);

        let result = (|| {
            match plan {
                ExecutorHibernationPlan::All => {
                    self.main_gqa_state.read_full_state(&mut reader, MAIN_GQA_STATE_FILES)?;
                    self.main_gdn_state.read_full_state(&mut reader, MAIN_GDN_STATE_FILES)?;
                    self.speculator.read_full_state(&mut reader)?;
                    self.pages.read_full_state(&mut reader, PAGE_ARENA_STATE_FILES)?;
                },
                ExecutorHibernationPlan::Selected {
                    request_slot_ranges,
                    page_id_ranges,
                } => {
                    self.main_gqa_state
                        .read_selected_state(&mut reader, MAIN_GQA_STATE_FILES, request_slot_ranges)?;
                    self.main_gdn_state
                        .read_selected_state(&mut reader, MAIN_GDN_STATE_FILES, request_slot_ranges)?;
                    self.speculator.read_selected_state(&mut reader, request_slot_ranges)?;
                    self.pages
                        .read_selected_state(&mut reader, PAGE_ARENA_STATE_FILES, page_id_ranges)?;
                },
            }
            reader.finish()
        })();
        if let Err(error) = result {
            self.pages.release_resources();
            self.speculator.release_resources();
            self.main_gdn_state.release_resources();
            self.main_gqa_state.release_resources();
            return Err(error);
        }

        self.main
            .component_mut()
            .load_state(&self.main_gqa_state, &self.main_gdn_state);
        self.speculator.load_state();
        Ok(())
    }
}

pub struct Qwen35ModelOpsRecorder {
    main_text_embed_key: Qwen35MainTextEmbedReplayKey,
    main_text_embed_arguments: ReplayArguments,
    resource_embed_key: Option<inference_backend_metal::components::resource_embed::Shape>,
    resource_embed_arguments: ReplayArguments,
    main_key: Qwen35MainReplayKey,
    main_arguments: ReplayArguments,
    main_text_embed_cache_hit: bool,
    main_cache_hit: bool,
    main_gather_unembed_key: Option<Qwen35GatherUnembedReplayKey>,
    main_gather_unembed_arguments: ReplayArguments,
    sampling_key: Option<TopKSamplingReplayKey>,
    sampling_arguments: ReplayArguments,
    rejection_key: Option<RejectionReplayKey>,
    rejection_arguments: ReplayArguments,
    rejection_prepared: Option<PreparedRejection>,
    rejection_build_elapsed: Duration,
    num_main_sample_rows: usize,
    mtp_hidden_state_transfer_key: Option<Qwen35MTPHiddenStateTransferReplayKey>,
    mtp_hidden_state_transfer_arguments: ReplayArguments,
    mtp_embed_key: Option<Qwen35MTPEmbedReplayKey>,
    mtp_embed_arguments: ReplayArguments,
    mtp_key: Option<Qwen35MTPReplayKey>,
    mtp_gqa_shape: Option<GQAReplayShape>,
    mtp_gqa_topology: Option<crate::attn::gqa::backend::GQAReplayTopology>,
    mtp_microbatch: Option<Qwen35Microbatch>,
    mtp_sampler_configs: Vec<SamplerConfig>,
    mtp_sample_positions: Vec<u32>,
    mtp_embed_cache_hit: bool,
    mtp_gather_unembed_key: Option<Qwen35GatherUnembedReplayKey>,
    mtp_gather_unembed_arguments: ReplayArguments,
    mtp_sampling_key: Option<TopKSamplingReplayKey>,
    mtp_sampling_arguments: ReplayArguments,
    mtp_sample_req_slots: Vec<u32>,
    mtp_sample_decision_indices: Vec<usize>,
    mtp_build_elapsed: Duration,
    dflash2_spec_prefill: Option<Qwen3xDFlash2PrefillRecording>,
    dflash2_spec_decode: Option<Qwen3xDFlash2DecodeRecording>,
}

impl Qwen35ModelOpsRecorder {
    fn main_replay_cache_hit(&self) -> bool {
        self.main_text_embed_cache_hit && self.main_cache_hit
    }

    fn num_mtp_sample_rows(&self) -> usize {
        let num_rows = self.mtp_sample_req_slots.len();
        debug_assert_eq!(self.mtp_sample_decision_indices.len(), num_rows);
        debug_assert_eq!(self.mtp_sampler_configs.len(), num_rows);
        debug_assert_eq!(self.mtp_sample_positions.len(), num_rows);
        num_rows
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Qwen35MTPRequest {
    req_slot: RawRequestSlot,
    block_index: usize,
    main_hidden_flat_start: usize,
    pending_token_index: u32,
    next_pending_token_index: u32,
    repair_tail: bool,
    sampler_config: SamplerConfig,
    kind: Qwen35MTPRequestKind,
}

#[derive(Clone, Debug, PartialEq)]
enum Qwen35MTPRequestKind {
    Prefill {
        token_ids_by_module: Vec<Vec<i32>>,
    },
    Decode {
        continuation_token_ids: Vec<i32>,
        sampled_token_id: i32,
        proposal_sample_position: u32,
        decision_index: usize,
    },
}

struct Qwen35MTPBatch {
    microbatch: Qwen35Microbatch,
    previous_hidden_routes: Vec<u32>,
    hidden_state_cache_write_routes: Vec<u32>,
    draft_distribution_indices: Vec<u32>,
    sampler_configs: Vec<SamplerConfig>,
    sample_positions: Vec<u32>,
}

struct Qwen35MTPRecordedStep {
    hidden_state_transfer_key: Qwen35MTPHiddenStateTransferReplayKey,
    hidden_state_transfer_arguments: ReplayArguments,
    embed_key: Qwen35MTPEmbedReplayKey,
    embed_arguments: ReplayArguments,
    body_key: Qwen35MTPReplayKey,
    gqa_shape: GQAReplayShape,
    gqa_topology: crate::attn::gqa::backend::GQAReplayTopology,
    gather_unembed: Option<(Qwen35GatherUnembedReplayKey, ReplayArguments)>,
    sampling: Option<(TopKSamplingReplayKey, ReplayArguments)>,
    num_sample_rows: usize,
}

fn mtp_proposal_sample_position(token_index: u32, num_tokens: u32, step_index: u32) -> u32 {
    token_index
        .checked_add(num_tokens)
        .and_then(|position| position.checked_add(step_index))
        .and_then(|position| position.checked_add(1))
        .expect("qwen3.5 MTP proposal sample position overflow")
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35ModelBatchResponse;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Qwen35SampledOutput {
    decisions: Vec<Qwen35DecodeDecision>,
    timing: ModelOutputTiming,
}

impl ReplayableDecoderModel for Qwen35Executor {
    type ModelBatchRequest = Qwen35ModelBatchRequest;
    type ModelBatchHidden = Rc<Buffer>;
    type ModelBatchResponse = Qwen35ModelBatchResponse;
    type SampledOutput = Qwen35SampledOutput;
    type ModelOpsRecorder = Qwen35ModelOpsRecorder;
    type Submission = MetalReplaySubmission;

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn vocab_size(&self) -> usize {
        self.layout.vocab_size as usize
    }

    fn model_mode(&self) -> &'static str {
        match &self.speculator {
            Qwen35Speculator::Vanilla => "vanilla",
            Qwen35Speculator::MTP(_) => "mtp",
            Qwen35Speculator::DFlash2(_) => "dflash2",
        }
    }

    fn num_spec_tokens(&self) -> usize {
        Qwen35Executor::num_spec_tokens(self)
    }

    fn default_stop_sequences(&self) -> Vec<Vec<Token>> {
        self.default_stop_sequences.clone()
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
        self.finish_gdn_commit();
        self.request_sampling.reset(request_slots);
        self.main_gqa_state.reset_req_slots(request_slots);
        self.speculator.reset_req_slots(request_slots);
        self.main_gdn_state.reset_req_slots(request_slots);
    }

    fn clear_replay_cache(&mut self) {
        Qwen35Executor::clear_replay_cache(self);
    }

    fn unload_state(&mut self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        Qwen35Executor::unload_state(self, snapshot_path, plan)
    }

    fn unload_weights(&mut self) {
        Qwen35Executor::unload_weights(self);
    }

    fn load_weights(&mut self) -> Result<(), ModelExecutorError> {
        Qwen35Executor::load_weights(self)
    }

    fn load_state(&mut self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        Qwen35Executor::load_state(self, snapshot_path, plan)
    }

    fn prepare_batch(&mut self, core_batch_req: &mut BatchDeviceRequest) -> Self::ModelBatchRequest {
        self.finish_gdn_commit();
        let batch_seq = core_batch_req.seq;
        self.validate_input(core_batch_req);
        let gdn = self.main_gdn_state.backend();
        core_batch_req.dev_reqs.sort_by_key(|request| {
            match &request.decoder_query_tokens {
                QueryTokens::Prefill { .. } => 0,
                query => {
                    if gdn.uses_recurrent(query.token_consumption(), query.num_spec_tokens()) {
                        1
                    } else {
                        2
                    }
                },
            }
        });
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
        let num_spec_tokens = match &self.speculator {
            Qwen35Speculator::MTP(mtp) => mtp.num_spec_tokens,
            Qwen35Speculator::Vanilla | Qwen35Speculator::DFlash2(_) => 0,
        };
        let model_batch_request =
            Qwen35ModelBatchRequest::from_core_batch(core_batch_req, num_spec_tokens, sampler_configs);
        let microbatch = model_batch_request.microbatch();
        self.sampler
            .set_params(microbatch.req_slots(), microbatch.sampler_configs());
        trace::qwen35_state(|| {
            format!(
                "event=batch_from_core seq={} req_slots={:?} token_indices={:?} num_spec_tokens={:?} seeds={:?} \
                 total_tokens={} num_reqs={}",
                batch_seq,
                microbatch.req_slots(),
                microbatch.token_indices(),
                microbatch
                    .req_slots()
                    .iter()
                    .enumerate()
                    .map(|(req_index, _)| microbatch.num_spec_tokens(req_index))
                    .collect::<Vec<_>>(),
                microbatch
                    .sampler_configs()
                    .iter()
                    .map(SamplerConfig::seed)
                    .collect::<Vec<_>>(),
                microbatch.total_tokens(),
                microbatch.num_reqs()
            )
        });
        self.write_token_ids(microbatch.flat_token_ids());
        let num_main_active_tokens = microbatch
            .total_tokens()
            .try_into()
            .expect("qwen3.5 Main token count must fit u32");
        let num_main_total_tokens = self.main.component().capacity(num_main_active_tokens);
        let prepare_start = Instant::now();
        let gqa_start = Instant::now();
        self.prepare_gqa_page_ids(core_batch_req);
        let gqa_shape = self.main_gqa_state.prepare_metadata(
            microbatch.req_slots(),
            microbatch.token_indices(),
            microbatch.cu_tokens(),
            num_main_total_tokens,
        );
        let gqa_elapsed = gqa_start.elapsed();
        debug_assert_eq!(gqa_shape.num_tokens as usize, microbatch.total_tokens());
        if let Some(main_resource_embed) = &mut self.main_resource_embed {
            main_resource_embed.prepare(&core_batch_req.dev_reqs);
        } else {
            debug_assert!(
                core_batch_req
                    .dev_reqs
                    .iter()
                    .all(|request| request.resource_placements.is_empty()),
                "text-only Qwen3.5 does not accept resource placements"
            );
        }
        let num_active_recurrent_requests = (0..microbatch.num_reqs())
            .take_while(|&req_index| {
                !microbatch.is_decode_req(req_index)
                    || self.main_gdn_state.backend().uses_recurrent(
                        (microbatch.cu_tokens()[req_index + 1] - microbatch.cu_tokens()[req_index]) as usize,
                        microbatch.num_spec_tokens(req_index) as usize,
                    )
            })
            .count();
        debug_assert!(
            (num_active_recurrent_requests..microbatch.num_reqs()).all(|req_index| {
                microbatch.is_decode_req(req_index)
                    && !self.main_gdn_state.backend().uses_recurrent(
                        (microbatch.cu_tokens()[req_index + 1] - microbatch.cu_tokens()[req_index]) as usize,
                        microbatch.num_spec_tokens(req_index) as usize,
                    )
            }),
            "Qwen3.5 batch requests must have a recurrent prefix"
        );
        let gdn_states_start = Instant::now();
        let gdn_prepared = self.main_gdn_state.prepare_states(
            microbatch.req_slots(),
            microbatch.block_indices(),
            microbatch.token_indices(),
            microbatch.cu_tokens(),
            &(0..microbatch.num_reqs())
                .map(|req_index| microbatch.num_spec_tokens(req_index))
                .collect::<Vec<_>>(),
            num_active_recurrent_requests,
            microbatch.gdn_state_page_ids_by_req(),
        );
        let gdn_states_elapsed = gdn_states_start.elapsed();
        let gdn_metadata_start = Instant::now();
        let gdn_shape = self.main_gdn_state.prepare_metadata(
            microbatch.cu_tokens(),
            num_active_recurrent_requests as u32,
            &gdn_prepared,
            num_main_total_tokens,
        );
        let gdn_metadata_elapsed = gdn_metadata_start.elapsed();
        debug_assert_eq!(gdn_shape.num_tokens as usize, microbatch.total_tokens());
        debug_assert_eq!(gdn_shape.num_reqs as usize, microbatch.num_reqs());
        if self.speculator.is_mtp() {
            self.speculator.mtp().body.component().prepare_pages(core_batch_req);
        }
        let prepare_elapsed = prepare_start.elapsed();
        trace::qwen35_state(|| {
            format!(
                "event=prepare_sync seq={} gqa_us={} gdn_states_us={} gdn_metadata_us={} wall_us={}",
                batch_seq,
                gqa_elapsed.as_micros(),
                gdn_states_elapsed.as_micros(),
                gdn_metadata_elapsed.as_micros(),
                prepare_elapsed.as_micros()
            )
        });
        let restore_elapsed = self.submit_gdn_state_restore();
        trace::qwen35_state(|| {
            format!(
                "event=prepare_batch_done seq={} gdn_restore_us={}",
                batch_seq,
                restore_elapsed.as_micros()
            )
        });
        model_batch_request
    }

    fn commit_batch(
        &mut self,
        core_batch_req: BatchDeviceRequest,
        sampled_output: Self::SampledOutput,
    ) -> BatchDeviceResponse {
        if !self.speculator.is_mtp() {
            self.commit(core_batch_req.seq, &sampled_output.decisions);
        }
        to_core_batch_resp(core_batch_req, sampled_output.decisions)
    }

    fn begin_ops_recording(&mut self, model_batch_request: &Self::ModelBatchRequest) -> Self::ModelOpsRecorder {
        let num_main_active_tokens = model_batch_request
            .microbatch()
            .total_tokens()
            .try_into()
            .expect("qwen3.5 Main token count must fit u32");
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
            self.main_gdn_state.metadata().replay_shape(),
            self.main_gdn_state.replay_topology(),
        );
        self.main_gqa_state.add_private_replay_arguments(&mut main_arguments);
        self.main_gdn_state.add_private_replay_arguments(&mut main_arguments);
        trace::qwen35_state(|| {
            format!(
                "event=begin_ops_recording main_text_embed_key={:?} main_key={:?}",
                main_text_embed_key, main_key
            )
        });
        Qwen35ModelOpsRecorder {
            main_text_embed_key,
            main_text_embed_arguments,
            resource_embed_key,
            resource_embed_arguments,
            main_key,
            main_arguments,
            main_text_embed_cache_hit: false,
            main_cache_hit: false,
            dflash2_spec_prefill: None,
            dflash2_spec_decode: None,
            main_gather_unembed_key: None,
            main_gather_unembed_arguments: ReplayArguments::new(),
            sampling_key: None,
            sampling_arguments: ReplayArguments::new(),
            rejection_key: None,
            rejection_arguments: ReplayArguments::new(),
            rejection_prepared: None,
            rejection_build_elapsed: Duration::ZERO,
            num_main_sample_rows: num_main_output_rows(model_batch_request.microbatch()),
            mtp_hidden_state_transfer_key: None,
            mtp_hidden_state_transfer_arguments: ReplayArguments::new(),
            mtp_embed_key: None,
            mtp_embed_arguments: ReplayArguments::new(),
            mtp_key: None,
            mtp_gqa_shape: None,
            mtp_gqa_topology: None,
            mtp_microbatch: None,
            mtp_sampler_configs: Vec::new(),
            mtp_sample_positions: Vec::new(),
            mtp_embed_cache_hit: false,
            mtp_gather_unembed_key: None,
            mtp_gather_unembed_arguments: ReplayArguments::new(),
            mtp_sampling_key: None,
            mtp_sampling_arguments: ReplayArguments::new(),
            mtp_sample_req_slots: Vec::new(),
            mtp_sample_decision_indices: Vec::new(),
            mtp_build_elapsed: Duration::ZERO,
        }
    }

    fn embed_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_request: &Self::ModelBatchRequest,
    ) -> Self::ModelBatchHidden {
        let input = Qwen35MainTextEmbedArgs {
            num_tokens: model_batch_request
                .microbatch()
                .total_tokens()
                .try_into()
                .expect("qwen3.5 MainTextEmbed token count must fit u32"),
            token_ids: &self.token_ids,
            hidden_output: &self.token_hidden_input,
        };
        let runtime = self.runtime.stream();
        let (recorded_key, cache_hit) = self.main_text_embed.record(runtime, &input);
        debug_assert_eq!(
            recorded_key, recorder.main_text_embed_key,
            "qwen3.5 MainTextEmbed replay input must match the prepared replay key"
        );
        recorder.main_text_embed_cache_hit = cache_hit;
        let resource_embed_key = self
            .main_resource_embed
            .as_mut()
            .and_then(|main_resource_embed| main_resource_embed.record(runtime, &self.token_hidden_input));
        debug_assert_eq!(
            resource_embed_key, recorder.resource_embed_key,
            "Qwen3.8 ResourceEmbed replay input must match the prepared replay key"
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
            "qwen3.5 Main must consume the MainTextEmbed hidden workspace"
        );
        let input = Qwen35MainArgs {
            num_tokens: microbatch
                .total_tokens()
                .try_into()
                .expect("qwen3.5 Main token count must fit u32"),
            hidden_input: &model_batch_hidden,
            hidden_output: &self.hidden_output,
            gqa: self.main_gqa_state.metadata(),
            gqa_replay_topology: self.main_gqa_state.replay_topology(),
            gdn: self.main_gdn_state.metadata(),
            gdn_replay_topology: self.main_gdn_state.replay_topology(),
            pages: self.pages.buffer(),
        };
        let runtime = self.runtime.stream();
        let (recorded_key, cache_hit) = self.main.record(runtime, &input);
        debug_assert_eq!(
            recorded_key, recorder.main_key,
            "qwen3.5 Main replay input must match the prepared replay key"
        );
        recorder.main_cache_hit = cache_hit;
        trace::qwen35_state(|| {
            format!(
                "event=main_replays main_text_embed_key={:?} main_key={:?} main_text_embed_cache_hit={} \
                 main_cache_hit={} cache_hit={}",
                recorder.main_text_embed_key,
                recorder.main_key,
                recorder.main_text_embed_cache_hit,
                recorder.main_cache_hit,
                recorder.main_replay_cache_hit(),
            )
        });
        self.pending_transactions
            .push(model_batch_req.compute_seq(), microbatch.clone());
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
            "qwen3.5 Output must consume the executor final-norm hidden workspace"
        );
        if num_main_output_rows(model_batch_req.microbatch()) == 0 {
            return Qwen35ModelBatchResponse;
        }
        let (gather_unembed_key, gather_unembed_arguments) =
            self.prepare_gather_unembed_replay(model_batch_req.microbatch(), model_batch_hidden);
        recorder.main_gather_unembed_key = Some(gather_unembed_key);
        recorder.main_gather_unembed_arguments = gather_unembed_arguments;
        Qwen35ModelBatchResponse
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
            "qwen3.5 sampling rows must match the recording"
        );
        if num_main_sample_rows == 0 {
            if self.speculator.is_dflash2() {
                self.record_spec(recorder, microbatch);
            }
            return;
        }
        if self.speculator.is_enabled() {
            self.record_rejection_sampling(recorder, microbatch);
            if self.speculator.is_dflash2() {
                self.record_spec(recorder, microbatch);
            }
        } else {
            let (sampling_key, sampling_arguments) = self.record_sampling(microbatch);
            recorder.sampling_key = Some(sampling_key);
            recorder.sampling_arguments = sampling_arguments;
        }
    }

    fn submit_main(&mut self, recorder: &Self::ModelOpsRecorder) -> Self::Submission {
        self.submit_main_recording(recorder)
    }

    fn read_main(
        &mut self,
        recorder: &Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        replay_elapsed: Duration,
        gpu_timestamp_durations: Option<&[Duration]>,
    ) -> Self::SampledOutput {
        if recorder.num_main_sample_rows == 0 {
            let mut timing = if self.speculator.is_dflash2() {
                ModelOutputTiming {
                    main_spec_replay_elapsed: replay_elapsed,
                    ..ModelOutputTiming::default()
                }
            } else {
                ModelOutputTiming {
                    main_replay_elapsed: replay_elapsed,
                    ..ModelOutputTiming::default()
                }
            };
            apply_main_gpu_timing(
                &mut timing,
                gpu_timestamp_durations,
                self.speculator.is_dflash2(),
                recorder.rejection_key.is_some(),
                recorder.dflash2_spec_decode.is_some(),
            );
            return Qwen35SampledOutput {
                decisions: Vec::new(),
                timing,
            };
        }
        let (mut decisions, mut timing) = if self.speculator.is_enabled() {
            self.read_rejection_sampling(recorder, model_batch_req.microbatch(), replay_elapsed)
        } else {
            self.read_sampling(recorder.num_main_sample_rows, replay_elapsed)
        };
        if self.speculator.is_dflash2() {
            timing.main_spec_replay_elapsed = timing.main_sample_replay_elapsed;
            timing.main_sample_replay_elapsed = Duration::ZERO;
            timing.spec_passes = 1;
            let read_start = Instant::now();
            decisions = self.read_dflash2_proposal(recorder, decisions);
            timing.spec_read_elapsed = read_start.elapsed();
        }
        apply_main_gpu_timing(
            &mut timing,
            gpu_timestamp_durations,
            self.speculator.is_dflash2(),
            recorder.rejection_key.is_some(),
            recorder.dflash2_spec_decode.is_some(),
        );
        Qwen35SampledOutput { decisions, timing }
    }

    fn run_spec(&self, _model_batch_req: &Self::ModelBatchRequest, _sampled_output: &Self::SampledOutput) -> bool {
        self.speculator.is_mtp()
    }

    fn embed_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: &Self::ModelBatchHidden,
        sampled_output: &Self::SampledOutput,
    ) -> Self::ModelBatchHidden {
        debug_assert!(
            Rc::ptr_eq(model_batch_hidden, &self.hidden_output),
            "qwen3.5 MTP must follow the Main hidden workspace"
        );
        let microbatch = model_batch_req.microbatch();
        let num_decode_reqs = (0..microbatch.num_reqs())
            .filter(|&req_index| microbatch.is_decode_req(req_index))
            .count();
        assert_eq!(
            sampled_output.decisions.len(),
            num_decode_reqs,
            "qwen3.5 MTP requires one decision per decode request"
        );
        // Main has completed. Its accepted state no longer depends on the next proposal.
        // The GDN commit stream can reconstruct it while MTP uses the Main hidden output.
        self.commit(model_batch_req.compute_seq(), &sampled_output.decisions);
        self.record_mtp_embed(
            recorder,
            microbatch,
            Rc::clone(model_batch_hidden),
            &sampled_output.decisions,
        )
    }

    fn forward_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: Self::ModelBatchHidden,
    ) -> Self::ModelBatchHidden {
        self.record_mtp(recorder, model_batch_hidden)
    }

    fn unembed_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: &Self::ModelBatchHidden,
    ) -> Self::ModelBatchResponse {
        self.record_mtp_gather_unembed(recorder, model_batch_hidden);
        Qwen35ModelBatchResponse
    }

    fn sample_spec(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _model_batch_resp: &Self::ModelBatchResponse,
    ) {
        self.record_mtp_sampling(recorder);
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
        panic!("qwen3.5 DFlash2 Spec Prefill is recorded with Main")
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
        panic!("qwen3.5 DFlash2 Spec Decode is recorded with Main")
    }

    fn submit_spec(&mut self, recorder: &Self::ModelOpsRecorder) -> Self::Submission {
        assert!(self.speculator.is_mtp(), "qwen3.5 DFlash2 Spec is submitted with Main");
        self.submit_mtp_recording(recorder)
    }

    fn read_spec(
        &mut self,
        recorder: &Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        mut sampled_output: Self::SampledOutput,
        replay_elapsed: Duration,
        gpu_timestamp_durations: Option<&[Duration]>,
    ) -> Self::SampledOutput {
        assert!(self.speculator.is_mtp(), "qwen3.5 DFlash2 Spec is read with Main");
        let timing = self.read_mtp_proposal(
            recorder,
            &mut sampled_output.decisions,
            replay_elapsed,
            gpu_timestamp_durations,
        );
        sampled_output.timing.add_assign(timing);
        sampled_output
    }

    fn empty_sampled_output(&self) -> Self::SampledOutput {
        Qwen35SampledOutput::default()
    }

    fn sampled_output_len(&self, sampled_output: &Self::SampledOutput) -> usize {
        sampled_output.decisions.len()
    }

    fn sampled_output_timing(&self, sampled_output: &Self::SampledOutput) -> Option<ModelOutputTiming> {
        (!sampled_output.timing.is_zero()).then_some(sampled_output.timing)
    }
}

fn num_page_ids_per_block(num_tokens_per_block: usize, num_tokens_per_page: usize) -> usize {
    assert!(
        num_tokens_per_block > 0,
        "qwen3.5 GQA requires positive tokens per block"
    );
    assert!(num_tokens_per_page > 0, "qwen3.5 GQA requires positive tokens per page");
    assert!(
        num_tokens_per_block.is_multiple_of(num_tokens_per_page),
        "qwen3.5 GQA tokens per block must be divisible by tokens per page"
    );
    num_tokens_per_block / num_tokens_per_page
}

fn trace_decisions(event: &str, decisions: &[Qwen35DecodeDecision]) {
    trace::qwen35_state(|| {
        let decisions = decisions
            .iter()
            .map(|decision| {
                (
                    decision.validated_tokens.as_slice(),
                    decision.sampled_token,
                    decision.spec_tokens.as_slice(),
                    decision.validated_probs.len(),
                    decision.spec_probs.len(),
                )
            })
            .collect::<Vec<_>>();
        format!("event={} decisions={:?}", event, decisions)
    });
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::Qwen35MTPExecution;
    use super::mtp_proposal_sample_position;

    #[test]
    fn test_mtp_gpu_timing_reset_and_missing_sample() {
        let mut execution = Qwen35MTPExecution::new(1, 3);
        for (samples, expected) in [
            ([Some(2), Some(3), Some(4)], Some(9)),
            ([Some(2), None, Some(4)], None),
            ([Some(0), Some(0), Some(0)], Some(0)),
        ] {
            execution.begin(Vec::new());
            for elapsed in samples {
                execution.push_step(&[], &[], Duration::ZERO, elapsed.map(Duration::from_micros));
            }
            assert_eq!(execution.gpu_elapsed, expected.map(Duration::from_micros));
            assert_eq!(execution.completed_steps, 3);
        }
    }

    #[test]
    fn test_mtp_proposal_sample_position_advances_per_step() {
        assert_eq!(mtp_proposal_sample_position(17, 3, 0), 21);
        assert_eq!(mtp_proposal_sample_position(17, 3, 3), 24);
    }
}
