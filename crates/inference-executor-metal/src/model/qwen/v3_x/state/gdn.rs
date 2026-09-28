use std::rc::Rc;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::GDNReplayShape;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gdn::backend::GDN;
use crate::attn::gdn::backend::GDNMetalConfig;
use crate::attn::gdn::backend::GDNReplayTopology;
use crate::attn::gdn::backend::add_gdn_private_replay_arguments;
use crate::attn::gdn::backend::add_gdn_replay_arguments;
use crate::attn::gdn::batch_metadata::GDNMetadataBuffers;
use crate::attn::gdn::batch_metadata::GDNReplayBucketPolicy;
use crate::attn::gdn::scratch::GDNScratch;
use crate::attn::gdn::state_table::GDNPreparedRequestState;
use crate::attn::gdn::state_table::GDNRequestStateResources;
use crate::attn::gdn::state_table::GDNRequestStateTable;
use crate::attn::gdn::state_table::GDNStateCapacity;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::MetalReplaySubmission;
use crate::def::replay_op::ReplayRecorder;
use crate::replay::Replay;
use crate::replay::ReplayComponent;
use crate::trace;

mod file_io;

const GDN_STATE_RESTORE_NUM_ACTIVE_REQUESTS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3x.gdn_state_restore.num_active_requests");
const GDN_STATE_COMMIT_NUM_ACTIVE_JOBS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3x.gdn_state_commit.num_active_jobs");
const GDN_STATE_COMMIT_NUM_ACTIVE_PUBLISHES: ReplayParameterKey =
    ReplayParameterKey::new("qwen3x.gdn_state_commit.num_active_publishes");

pub struct Qwen3xGDNState {
    backend: Option<Rc<GDN>>,
    scratch: Option<Rc<GDNScratch>>,
    metadata: Option<GDNMetadataBuffers>,
    representative_core: GDNCore,
    metal: GDNMetalConfig,
    num_req_slots: usize,
    max_tokens: usize,
    replay_bucket_policy: GDNReplayBucketPolicy,
    request_state_table: GDNRequestStateTable,
    state_restore: Replay<GDNStateRestore>,
    state_commit: Replay<GDNStateCommit>,
    commit_stream: Stream,
    pending_commit: Option<MetalReplaySubmission>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GDNStateCommitKey {
    num_total_replay_jobs: u32,
    num_total_state_io_requests: u32,
}

struct GDNStateCommitInput<'a> {
    request_state_table: &'a GDNRequestStateTable,
    pages: &'a Buffer,
    key: GDNStateCommitKey,
}

struct GDNStateCommit;

impl ReplayComponent for GDNStateCommit {
    type Key = GDNStateCommitKey;
    type Input<'a> = GDNStateCommitInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        input.key
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        if input.key.num_total_replay_jobs > 0 {
            input.request_state_table.record_commit(
                recorder,
                input.key.num_total_replay_jobs,
                ReplayU32::Parameter(GDN_STATE_COMMIT_NUM_ACTIVE_JOBS),
            );
        }
        if input.key.num_total_state_io_requests > 0 {
            input.request_state_table.record_publish(
                recorder,
                input.pages,
                input.key.num_total_state_io_requests,
                ReplayU32::Parameter(GDN_STATE_COMMIT_NUM_ACTIVE_PUBLISHES),
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GDNStateRestoreKey {
    num_total_state_io_requests: u32,
}

#[derive(Clone, Copy)]
struct GDNStateRestoreInput<'a> {
    request_state_table: &'a GDNRequestStateTable,
    pages: &'a Buffer,
    key: GDNStateRestoreKey,
}

struct GDNStateRestore;

impl ReplayComponent for GDNStateRestore {
    type Key = GDNStateRestoreKey;
    type Input<'a> = GDNStateRestoreInput<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        input.key
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        input.request_state_table.record_restore(
            recorder,
            input.pages,
            input.key.num_total_state_io_requests,
            ReplayU32::Parameter(GDN_STATE_RESTORE_NUM_ACTIVE_REQUESTS),
        );
    }
}

impl Qwen3xGDNState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        commit_stream: Stream,
        cores: &[GDNCore],
        metal: GDNMetalConfig,
        num_req_slots: usize,
        state_capacity: GDNStateCapacity,
        max_tokens: usize,
        num_tokens_per_block: usize,
        num_cache_pages: usize,
        page_bytes: usize,
    ) -> Self {
        let representative = cores
            .first()
            .expect("qwen3.x GDN state requires at least one GDN layer");
        for core in cores {
            core.validate();
            let mut shared_core = core.clone();
            shared_core.model_layer_index = representative.model_layer_index;
            assert_eq!(
                &shared_core, representative,
                "qwen3.x GDN layers that share one backend must have compatible compute and affine layouts"
            );
        }
        let request_state_table = GDNRequestStateTable::new(
            device,
            cores,
            num_req_slots,
            state_capacity,
            num_tokens_per_block,
            num_cache_pages,
            page_bytes,
            max_tokens,
        );
        let backend = Rc::new(GDN::new(device, representative.clone(), metal));
        let max_requests = num_req_slots
            .try_into()
            .expect("qwen3.x GDN request capacity must fit u32");
        let max_tokens_u32 = max_tokens.try_into().expect("qwen3.x GDN token capacity must fit u32");
        let replay_bucket_policy = backend.replay_bucket_policy(max_requests, max_tokens_u32);
        Self {
            scratch: Some(Rc::new(backend.new_scratch(max_tokens))),
            backend: Some(backend),
            metadata: Some(GDNMetadataBuffers::new(device, num_req_slots, max_tokens)),
            representative_core: representative.clone(),
            metal,
            num_req_slots,
            max_tokens,
            replay_bucket_policy,
            request_state_table,
            state_restore: Replay::new("qwen3.x GDN state restore", GDNStateRestore),
            state_commit: Replay::new("qwen3.x GDN state commit", GDNStateCommit),
            commit_stream,
            pending_commit: None,
        }
    }

    pub fn backend(&self) -> &Rc<GDN> {
        self.backend.as_ref().expect("Qwen3.x GDN backend state must be loaded")
    }

    pub fn scratch(&self) -> &Rc<GDNScratch> {
        self.scratch.as_ref().expect("Qwen3.x GDN scratch state must be loaded")
    }

    pub fn request_state_resources(&self) -> &Rc<GDNRequestStateResources> {
        self.request_state_table.resources()
    }

    pub fn metadata(&self) -> &GDNMetadataBuffers {
        self.metadata
            .as_ref()
            .expect("Qwen3.x GDN metadata state must be loaded")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_states(
        &self,
        req_slots: &[u32],
        block_indices: &[usize],
        token_indices: &[u32],
        cu_tokens: &[u32],
        num_spec_tokens: &[u32],
        num_recurrent_requests: usize,
        state_page_ids_by_req: &[Vec<Vec<u32>>],
    ) -> GDNPreparedRequestState {
        self.request_state_table.prepare(
            req_slots,
            block_indices,
            token_indices,
            cu_tokens,
            num_spec_tokens,
            num_recurrent_requests,
            state_page_ids_by_req,
        )
    }

    pub fn prepare_metadata(
        &self,
        cu_tokens: &[u32],
        num_active_recurrent_requests: u32,
        prepared: &GDNPreparedRequestState,
        num_total_tokens: u32,
    ) -> GDNReplayShape {
        self.backend().prepare(
            self.metadata(),
            cu_tokens,
            num_active_recurrent_requests,
            prepared,
            &self.replay_bucket_policy,
            num_total_tokens,
        )
    }

    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        self.backend().replay_token_topology_boundaries()
    }

    pub fn replay_topology(&self) -> GDNReplayTopology {
        self.backend().replay_topology(self.metadata())
    }

    pub fn add_replay_arguments(&self, arguments: &mut ReplayArguments) {
        add_gdn_replay_arguments(
            self.metadata().replay_shape(),
            self.metadata().num_active_recurrent_requests(),
            arguments,
        );
    }

    pub fn add_private_replay_arguments(&self, arguments: &mut ReplayArguments) {
        add_gdn_private_replay_arguments(
            self.metadata().replay_shape(),
            self.metadata().num_active_recurrent_requests(),
            arguments,
        );
    }

    pub fn restore(&mut self, runtime: &MetalReplayRuntime<'_>, pages: &Buffer) {
        let num_active_state_io_requests = self.request_state_table.prepare_restore(pages);
        if num_active_state_io_requests == 0 {
            trace::gdn_state(|| "event=gdn_restore skipped=true".to_string());
            return;
        }
        let input = GDNStateRestoreInput {
            request_state_table: &self.request_state_table,
            pages,
            key: GDNStateRestoreKey {
                num_total_state_io_requests: num_active_state_io_requests,
            },
        };
        let (key, cache_hit) = self.state_restore.record(runtime, &input);
        trace::gdn_state(|| format!("event=gdn_restore key={key:?} cache_hit={cache_hit}"));
        let arguments =
            ReplayArguments::new().with_u32(GDN_STATE_RESTORE_NUM_ACTIVE_REQUESTS, num_active_state_io_requests);
        runtime
            .submit_replay_with_arguments(self.state_restore.replay(&key), &arguments)
            .wait();
        self.request_state_table.finish_restore();
    }

    pub fn commit(&mut self, pages: &Buffer, state_versions: &[u32]) {
        assert!(
            self.pending_commit.is_none(),
            "GDN commit cannot overlap a previous commit"
        );
        let jobs = self.request_state_table.commit(state_versions);
        let num_active_replay_jobs = jobs.len() as u32;
        let num_active_state_io_requests = self.request_state_table.prepare_publish(pages);
        if num_active_replay_jobs == 0 && num_active_state_io_requests == 0 {
            return;
        }
        let input = GDNStateCommitInput {
            request_state_table: &self.request_state_table,
            pages,
            key: GDNStateCommitKey {
                num_total_replay_jobs: num_active_replay_jobs,
                num_total_state_io_requests: num_active_state_io_requests,
            },
        };
        let runtime = MetalReplayRuntime::new(&self.commit_stream);
        let (key, cache_hit) = self.state_commit.record(&runtime, &input);
        trace::gdn_state(|| format!("event=gdn_commit key={key:?} cache_hit={cache_hit}"));
        let mut arguments = ReplayArguments::new();
        if num_active_replay_jobs > 0 {
            arguments.set_u32(GDN_STATE_COMMIT_NUM_ACTIVE_JOBS, num_active_replay_jobs);
        }
        if num_active_state_io_requests > 0 {
            arguments.set_u32(GDN_STATE_COMMIT_NUM_ACTIVE_PUBLISHES, num_active_state_io_requests);
        }
        self.pending_commit = Some(runtime.submit_replay_with_arguments(self.state_commit.replay(&key), &arguments));
    }

    pub fn finish_commit(&mut self) {
        if let Some(submission) = self.pending_commit.take() {
            submission.wait();
        }
        self.request_state_table.finish_commit();
    }

    pub fn clear_replay_cache(&mut self) {
        assert!(
            self.pending_commit.is_none(),
            "GDN replay cache cannot be cleared while a state commit is pending"
        );
        self.state_restore.clear();
        self.state_commit.clear();
    }

    pub fn release_resources(&mut self) {
        assert!(
            self.backend.is_some() && self.scratch.is_some() && self.metadata.is_some(),
            "Qwen3.x GDN state resources are not loaded"
        );
        self.request_state_table.release_resources();
        self.metadata.take();
        self.scratch.take();
        self.backend.take();
    }

    pub fn allocate_resources(&mut self, device: &Device) {
        assert!(
            self.backend.is_none() && self.scratch.is_none() && self.metadata.is_none(),
            "Qwen3.x GDN state resources are already loaded"
        );
        let backend = Rc::new(GDN::new(device, self.representative_core.clone(), self.metal));
        self.scratch = Some(Rc::new(backend.new_scratch(self.max_tokens)));
        self.metadata = Some(GDNMetadataBuffers::new(device, self.num_req_slots, self.max_tokens));
        self.backend = Some(backend);
        self.request_state_table.allocate_resources(device);
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_state_table.reset_req_slots(req_slots);
    }

    pub fn num_pages_per_state_slot(&self) -> usize {
        self.request_state_table.num_pages_per_state_slot()
    }
}
