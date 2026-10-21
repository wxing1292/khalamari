use std::cell::RefCell;
use std::mem::size_of;
use std::rc::Rc;

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gdn::compute::ReplayBuffers;
use inference_backend_metal::components::gdn::state_pages as backend_state_pages;
use inference_backend_metal::components::gdn::state_replay as backend_state_replay;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::gdn::state::GDNStateTxn;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gdn::request_slots::GDNRequestSlots;
use crate::attn::gdn::request_slots::GDNStatePages;
use crate::attn::gdn::request_slots::GDNStatePublish;
use crate::attn::gdn::request_slots::GDNStateRestore;
use crate::trace;

mod file_io;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GDNStateLayout {
    max_tokens: usize,
    num_gdn_layers: usize,
    num_state_slots: usize,
    max_state_io_requests: usize,
    recurrent_state_bytes: usize,
    conv_state_bytes: usize,
    page_bytes: usize,
    replay: backend_state_replay::Config,
    max_replay_jobs: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDNStateCapacity {
    max_replay_tokens_per_req: usize,
    num_state_slots_per_req: usize,
    max_materialized_states_per_req: usize,
    max_publish_jobs_per_req: usize,
}

impl GDNStateCapacity {
    pub fn new(
        num_state_slots_per_req: usize,
        max_materialized_states_per_req: usize,
        max_publish_jobs_per_req: usize,
        max_replay_tokens_per_req: usize,
    ) -> Self {
        let capacity = Self {
            max_replay_tokens_per_req,
            num_state_slots_per_req,
            max_materialized_states_per_req,
            max_publish_jobs_per_req,
        };
        capacity.validate();
        capacity
    }

    fn validate(self) {
        assert!(self.max_replay_tokens_per_req > 0, "GDN replay requires request tokens");
        assert!(
            self.max_materialized_states_per_req > 0,
            "GDN state requires materialized-state capacity"
        );
        assert!(
            self.num_state_slots_per_req > self.max_materialized_states_per_req,
            "GDN state slots must include current state and all materialized states"
        );
        assert!(self.max_publish_jobs_per_req > 0, "GDN state requires publish capacity");
    }
}

pub struct GDNRequestStateTable {
    layout: GDNStateLayout,
    num_req_slots: usize,
    num_state_slots_per_req: usize,
    num_tokens_per_block: usize,
    num_cache_pages: usize,
    max_materialized_states_per_req: usize,
    max_publish_jobs_per_req: usize,
    resources: Option<Rc<GDNRequestStateResources>>,
    request_table: Option<RefCell<GDNRequestSlots>>,
    restores: RefCell<Vec<GDNStateRestore>>,
    publishes: RefCell<Vec<GDNStatePublish>>,
    pending_commits: RefCell<Vec<GDNRequestCommit>>,
}

pub struct GDNRequestStateResources {
    layout: GDNStateLayout,
    recurrent_states: Buffer,
    conv_states: Buffer,
    page_io: GDNStatePageIO,
    replay_alpha: Buffer,
    replay_k: Buffer,
    replay_u: Buffer,
    replay_qkv: Buffer,
    replay_jobs: Buffer,
    replay_commit: backend_state_replay::Commit,
}

pub struct GDNStatePageIO {
    page_ids: Buffer,
    recurrent_state_slots: Buffer,
    conv_state_slots: Buffer,
    read: backend_state_pages::Read,
    write: backend_state_pages::Write,
}

pub struct GDNPreparedRequestState {
    pub src_recurrent_state_slots: Vec<u32>,
    pub src_conv_state_slots: Vec<u32>,
    /// Persistent recurrent state slot for each forward row.
    ///
    /// `u32::MAX` means that the row produces its normal output, but its recurrent state
    /// must not be written to the persistent recurrent state arena.
    pub flat_recurrent_state_write_slots: Vec<u32>,
    /// Persistent convolution state slot for each forward row.
    ///
    /// `u32::MAX` means that the row produces its normal output, but its convolution state
    /// must not be written to the persistent convolution state arena.
    pub flat_conv_state_write_slots: Vec<u32>,
}

#[derive(Clone, Copy)]
struct GDNRequestCommit {
    req_slot: u32,
    commit: GDNStateCommit,
}

#[derive(Clone, Copy)]
enum GDNStateCommit {
    Recurrent { final_state_version: u32 },
    Replay { txn: GDNStateTxn, replay_token_begin: u32 },
}

#[derive(Clone, Copy)]
pub struct GDNStateArenaBindings<'a> {
    pub recurrent_states: &'a Buffer,
    pub recurrent_layer_offset_bytes: u64,
    pub conv_states: &'a Buffer,
    pub conv_layer_offset_bytes: u64,
    pub replay: ReplayBuffers<'a>,
}

impl GDNRequestStateTable {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        cores: &[GDNCore],
        num_req_slots: usize,
        capacity: GDNStateCapacity,
        num_tokens_per_block: usize,
        num_cache_pages: usize,
        page_bytes: usize,
        max_tokens: usize,
    ) -> Self {
        assert!(!cores.is_empty(), "GDN state requires layers");
        assert!(num_tokens_per_block > 0, "GDN state requires tokens per block");
        assert!(num_cache_pages > 0, "GDN state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "GDN cache page IDs must fit u32"
        );
        assert!(page_bytes.is_power_of_two(), "GDN page size must be a power of two");
        assert!(
            page_bytes.is_multiple_of(16),
            "GDN page size must contain an integral number of 16-byte vectors"
        );
        let request_table = GDNRequestSlots::new(num_req_slots, capacity.num_state_slots_per_req);
        let num_state_slots = num_req_slots
            .checked_mul(capacity.num_state_slots_per_req)
            .expect("GDN state slot count overflow");

        for core in cores {
            core.validate();
        }
        let first_core = &cores[0];
        let recurrent_state_bytes = [
            first_core.num_v_heads,
            first_core.v_head_dim,
            first_core.qk_head_dim,
            size_of::<u16>(),
        ]
        .into_iter()
        .try_fold(1usize, |product, factor| product.checked_mul(factor))
        .expect("GDN recurrent state slot size must fit usize");
        let conv_state_bytes = [first_core.qkv_dim(), first_core.conv_state_len(), size_of::<u16>()]
            .into_iter()
            .try_fold(1usize, |product, factor| product.checked_mul(factor))
            .expect("GDN convolution state slot size must fit usize");
        u32::try_from(recurrent_state_bytes).expect("GDN recurrent state slot bytes must fit shader u32");
        u32::try_from(conv_state_bytes).expect("GDN convolution state slot bytes must fit shader u32");
        assert!(
            recurrent_state_bytes.is_multiple_of(16),
            "GDN recurrent state slot must contain an integral number of 16-byte vectors"
        );
        assert!(
            conv_state_bytes.is_multiple_of(16),
            "GDN convolution state slot must contain an integral number of 16-byte vectors"
        );
        assert!(
            cores.iter().all(|core| {
                core.num_v_heads == first_core.num_v_heads
                    && core.v_head_dim == first_core.v_head_dim
                    && core.qk_head_dim == first_core.qk_head_dim
                    && core.qkv_dim() == first_core.qkv_dim()
                    && core.conv_state_len() == first_core.conv_state_len()
            }),
            "GDN all-layer state IO requires one shared layer layout"
        );
        let num_gdn_layers = cores.len();
        let max_state_io_requests = num_req_slots
            .checked_mul(capacity.max_publish_jobs_per_req)
            .expect("GDN publish job count overflow");
        let max_replay_tokens = num_req_slots
            .checked_mul(capacity.max_replay_tokens_per_req)
            .expect("GDN replay token capacity must fit usize")
            .min(max_tokens);
        let replay = backend_state_replay::Config {
            num_gdn_layers: num_gdn_layers.try_into().expect("GDN replay layer count must fit u32"),
            num_state_slots: num_state_slots.try_into().expect("GDN replay state slots must fit u32"),
            max_replay_tokens: max_replay_tokens
                .try_into()
                .expect("GDN replay token capacity must fit u32"),
            num_qk_heads: first_core.num_qk_heads as u32,
            qk_head_dim: first_core.qk_head_dim as u32,
            num_v_heads: first_core.num_v_heads as u32,
            v_head_dim: first_core.v_head_dim as u32,
            conv_state_len: first_core.conv_state_len() as u32,
        };
        replay.validate();
        let layout = GDNStateLayout {
            max_tokens,
            num_gdn_layers,
            num_state_slots,
            max_state_io_requests,
            recurrent_state_bytes,
            conv_state_bytes,
            page_bytes,
            replay,
            max_replay_jobs: num_req_slots
                .checked_mul(capacity.max_materialized_states_per_req)
                .expect("GDN replay job capacity must fit usize"),
        };
        Self {
            layout,
            num_req_slots,
            num_state_slots_per_req: capacity.num_state_slots_per_req,
            num_tokens_per_block,
            num_cache_pages,
            max_materialized_states_per_req: capacity.max_materialized_states_per_req,
            max_publish_jobs_per_req: capacity.max_publish_jobs_per_req,
            resources: Some(Rc::new(GDNRequestStateResources::new(device, layout))),
            request_table: Some(RefCell::new(request_table)),
            restores: RefCell::new(Vec::with_capacity(num_req_slots)),
            publishes: RefCell::new(Vec::with_capacity(layout.max_state_io_requests)),
            pending_commits: RefCell::new(Vec::with_capacity(num_req_slots)),
        }
    }

    pub fn num_pages_per_state_slot(&self) -> usize {
        (self.recurrent_state_bytes().div_ceil(self.layout.page_bytes)
            + self.conv_state_bytes().div_ceil(self.layout.page_bytes))
            * self.layout.num_gdn_layers
    }

    pub fn num_req_slots(&self) -> usize {
        self.request_table().borrow().num_req_slots()
    }

    pub fn num_layers(&self) -> usize {
        self.layout.num_gdn_layers
    }

    pub fn release_resources(&mut self) {
        self.assert_snapshot_ready();
        self.request_table
            .take()
            .expect("GDN request-state table must be loaded");
        self.resources
            .take()
            .expect("GDN request-state resources must be loaded");
    }

    pub fn allocate_resources(&mut self, device: &Device) {
        assert!(
            self.request_table.is_none() && self.resources.is_none(),
            "GDN request-state resources are already loaded"
        );
        self.request_table = Some(RefCell::new(GDNRequestSlots::new(
            self.num_req_slots,
            self.num_state_slots_per_req,
        )));
        self.resources = Some(Rc::new(GDNRequestStateResources::new(device, self.layout)));
    }

    fn recurrent_state_bytes(&self) -> usize {
        self.layout.recurrent_state_bytes
    }

    fn conv_state_bytes(&self) -> usize {
        self.layout.conv_state_bytes
    }

    pub fn resources(&self) -> &Rc<GDNRequestStateResources> {
        self.resources
            .as_ref()
            .expect("GDN request-state resources must be loaded before execution")
    }

    fn request_table(&self) -> &RefCell<GDNRequestSlots> {
        self.request_table
            .as_ref()
            .expect("GDN request-state table must be loaded before execution")
    }

    fn assert_snapshot_ready(&self) {
        assert!(
            self.restores.borrow().is_empty(),
            "GDN state snapshot requires no restore jobs"
        );
        assert!(
            self.publishes.borrow().is_empty(),
            "GDN state snapshot requires no publish jobs"
        );
        assert!(
            self.pending_commits.borrow().is_empty(),
            "GDN state snapshot requires no pending commits"
        );
    }

    pub fn layer_bindings(&self, gdn_layer_index: usize) -> GDNStateArenaBindings<'_> {
        self.resources().layer_bindings(gdn_layer_index)
    }

    pub fn restores(&self) -> Vec<GDNStateRestore> {
        self.restores.borrow().clone()
    }

    pub fn publishes(&self) -> Vec<GDNStatePublish> {
        self.publishes.borrow().clone()
    }

    pub fn prepare_restore(&self, pages: &Buffer) -> u32 {
        let restores = self.restores.borrow();
        if restores.is_empty() {
            return 0;
        }
        assert!(
            restores.len() <= self.request_table().borrow().num_req_slots(),
            "GDN restore I/O requests exceed request-slot capacity"
        );
        let resources = self.resources();
        self.assert_page_buffer(pages);
        resources
            .page_io
            .prepare_restore(&restores, self.num_pages_per_state_slot());
        restores.len() as u32
    }

    pub fn record_restore<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        pages: &'a Buffer,
        num_total_state_io_requests: u32,
        num_active_state_io_requests: ReplayU32,
    ) {
        let resources = self.resources();
        resources.page_io.record_restore(
            recorder,
            pages,
            &resources.recurrent_states,
            &resources.conv_states,
            num_total_state_io_requests,
            num_active_state_io_requests,
        );
    }

    pub fn finish_restore(&self) {
        self.restores.borrow_mut().clear();
    }

    pub fn prepare_publish(&self, pages: &Buffer) -> u32 {
        let publishes = self.publishes.borrow();
        if publishes.is_empty() {
            return 0;
        }
        assert!(
            publishes.len() <= self.layout.max_state_io_requests,
            "GDN publish I/O requests exceed state-I/O request capacity"
        );
        let resources = self.resources();
        self.assert_page_buffer(pages);
        resources
            .page_io
            .prepare_publish(&publishes, self.num_pages_per_state_slot());
        publishes.len() as u32
    }

    pub fn record_publish<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        pages: &'a Buffer,
        num_total_state_io_requests: u32,
        num_active_state_io_requests: ReplayU32,
    ) {
        let resources = self.resources();
        resources.page_io.record_publish(
            recorder,
            pages,
            &resources.recurrent_states,
            &resources.conv_states,
            num_total_state_io_requests,
            num_active_state_io_requests,
        );
    }

    pub fn finish_commit(&self) {
        self.publishes.borrow_mut().clear();
    }

    fn assert_page_buffer(&self, pages: &Buffer) {
        assert_eq!(
            pages.len_bytes() % self.layout.page_bytes,
            0,
            "GDN page buffer must contain whole pages"
        );
        // validate_batch already bounds every page ID by the configured arena.
        assert!(
            pages.len_bytes() / self.layout.page_bytes >= self.num_cache_pages,
            "GDN page buffer must cover the configured cache-page capacity"
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        req_slots: &[u32],
        block_indices: &[usize],
        token_indices: &[u32],
        cu_tokens: &[u32],
        num_spec_tokens: &[u32],
        num_recurrent_requests: usize,
        state_page_ids_by_req: &[Vec<Vec<u32>>],
    ) -> GDNPreparedRequestState {
        self.validate_batch(
            req_slots,
            block_indices,
            token_indices,
            cu_tokens,
            num_spec_tokens,
            num_recurrent_requests,
            state_page_ids_by_req,
        );
        let mut request_table = self.request_table().borrow_mut();
        let mut restores = self.restores.borrow_mut();
        let mut pending_commits = self.pending_commits.borrow_mut();
        restores.clear();
        pending_commits.clear();
        self.publishes.borrow_mut().clear();
        let num_tokens = cu_tokens[req_slots.len()] as usize;
        let mut prepared = GDNPreparedRequestState {
            src_recurrent_state_slots: Vec::with_capacity(req_slots.len()),
            src_conv_state_slots: Vec::with_capacity(req_slots.len()),
            flat_recurrent_state_write_slots: vec![u32::MAX; num_tokens],
            flat_conv_state_write_slots: vec![u32::MAX; num_tokens],
        };
        for (req_index, &req_slot) in req_slots.iter().enumerate() {
            let token_index = token_indices[req_index];
            let current_state_version = request_table.current_state_version(req_slot);
            assert!(
                current_state_version <= token_index,
                "GDN current state version exceeds the runtime input token index"
            );
            let mut restore_target = None;
            let mut publish_pages = Vec::new();
            for (block_offset, page_ids) in state_page_ids_by_req[req_index].iter().enumerate() {
                // validate_batch proved this runtime block range fits u32.
                let state_version = ((block_indices[req_index] + block_offset + 1) * self.num_tokens_per_block) as u32;
                if state_version <= current_state_version {
                    continue;
                }
                if state_version <= token_index {
                    restore_target = Some((state_version, page_ids));
                } else {
                    publish_pages.push(GDNStatePages {
                        state_version,
                        page_ids: page_ids.clone(),
                    });
                }
            }
            if let Some((state_version, page_ids)) = restore_target {
                restores.push(request_table.restore(req_slot, state_version, page_ids.clone()));
            }
            assert_eq!(
                request_table.current_state_version(req_slot),
                token_index,
                "GDN current state version must match the runtime input token index"
            );
            request_table.set_publish_pages(req_slot, publish_pages);
            prepared
                .src_recurrent_state_slots
                .push(request_table.current_recurrent_state_slot(req_slot));
            prepared
                .src_conv_state_slots
                .push(request_table.current_conv_state_slot(req_slot));
            let flat_token_begin = cu_tokens[req_index];
            let num_tokens = cu_tokens[req_index + 1] - flat_token_begin;
            let commit = if req_index < num_recurrent_requests {
                let final_state_version = token_index + num_tokens;
                let mut versions = request_table
                    .pending_publish_state_versions(req_slot)
                    .take_while(|&version| version < final_state_version)
                    .collect::<Vec<_>>();
                versions.push(final_state_version);
                debug_assert!(versions.len() <= self.max_materialized_states_per_req);
                request_table.prepare_states(req_slot, &versions, &versions);
                for version in versions {
                    let flat_index = (flat_token_begin + (version - token_index - 1)) as usize;
                    prepared.flat_recurrent_state_write_slots[flat_index] =
                        request_table.materialized_recurrent_state_slot(req_slot, version);
                    prepared.flat_conv_state_write_slots[flat_index] =
                        request_table.materialized_conv_state_slot(req_slot, version);
                }
                GDNStateCommit::Recurrent { final_state_version }
            } else {
                GDNStateCommit::Replay {
                    txn: GDNStateTxn::from_state_versions(
                        token_index + num_tokens - num_spec_tokens[req_index],
                        token_index + num_tokens + 1,
                    ),
                    replay_token_begin: flat_token_begin - cu_tokens[num_recurrent_requests],
                }
            };
            pending_commits.push(GDNRequestCommit { req_slot, commit });
        }
        prepared
    }

    pub fn commit(&self, dst_state_versions: &[u32]) -> Vec<backend_state_replay::Job> {
        let mut pending_commits = self.pending_commits.borrow_mut();
        let mut publishes_out = self.publishes.borrow_mut();
        let mut request_table = self.request_table().borrow_mut();
        assert_eq!(pending_commits.len(), dst_state_versions.len());
        let mut replay_jobs = Vec::new();
        // Allocate every destination before promotion frees any source slot.
        // GPU commit jobs still read those sources after this CPU method returns.
        for (request, &dst_state_version) in pending_commits.iter().zip(dst_state_versions) {
            match request.commit {
                GDNStateCommit::Recurrent { final_state_version } => {
                    assert_eq!(
                        dst_state_version, final_state_version,
                        "GDN recurrent commits its complete input"
                    );
                },
                GDNStateCommit::Replay {
                    txn,
                    replay_token_begin,
                } => {
                    assert!(
                        txn.contains_dst_state_version(dst_state_version),
                        "GDN commit must select an input prefix"
                    );
                    let req_slot = request.req_slot;
                    let src_state_version = request_table.current_state_version(req_slot);
                    let src_recurrent_state_slot = request_table.current_recurrent_state_slot(req_slot);
                    let src_conv_state_slot = request_table.current_conv_state_slot(req_slot);
                    let mut versions = request_table
                        .pending_publish_state_versions(req_slot)
                        .take_while(|&version| version < dst_state_version)
                        .collect::<Vec<_>>();
                    if dst_state_version > src_state_version {
                        versions.push(dst_state_version);
                    }
                    debug_assert!(versions.len() <= self.max_materialized_states_per_req);
                    request_table.prepare_states(req_slot, &versions, &versions);
                    for version in versions {
                        replay_jobs.push(backend_state_replay::Job {
                            src_recurrent_state_slot,
                            src_conv_state_slot,
                            dst_recurrent_state_slot: request_table
                                .materialized_recurrent_state_slot(req_slot, version),
                            dst_conv_state_slot: request_table.materialized_conv_state_slot(req_slot, version),
                            replay_token_begin,
                            num_tokens: version - src_state_version,
                        });
                    }
                },
            }
        }
        publishes_out.clear();
        for (request, &dst_state_version) in pending_commits.drain(..).zip(dst_state_versions) {
            let publishes = request_table.commit(request.req_slot, dst_state_version);
            debug_assert!(publishes.len() <= self.max_publish_jobs_per_req);
            publishes_out.extend(publishes);
        }
        backend_state_replay::write_jobs(&self.resources().replay_jobs, &replay_jobs);
        replay_jobs
    }

    pub fn record_commit(
        &self,
        recorder: &mut ReplayRecorder,
        num_total_replay_jobs: u32,
        num_active_replay_jobs: ReplayU32,
    ) {
        let resources = self.resources();
        recorder.record(ReplayOp::opaque(resources.replay_commit.invoke(
            num_total_replay_jobs,
            num_active_replay_jobs,
            backend_state_replay::Buffers {
                recurrent_states: &resources.recurrent_states,
                conv_states: &resources.conv_states,
                alpha: &resources.replay_alpha,
                k: &resources.replay_k,
                u: &resources.replay_u,
                qkv: &resources.replay_qkv,
                jobs: &resources.replay_jobs,
            },
        )));
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        let mut request_table = self.request_table().borrow_mut();
        request_table.reset_req_slots(req_slots);
        for &req_slot in req_slots {
            self.zero_state_slots(
                request_table.current_recurrent_state_slot(req_slot),
                request_table.current_conv_state_slot(req_slot),
            );
        }
    }

    pub fn reset_req_slot(&self, req_slot: RawRequestSlot) {
        self.reset_req_slots(&[req_slot]);
    }

    fn zero_state_slots(&self, recurrent_state_slot: u32, conv_state_slot: u32) {
        let recurrent_state_slot_index = recurrent_state_slot as usize;
        let conv_state_slot_index = conv_state_slot as usize;
        assert!(recurrent_state_slot_index < self.layout.num_state_slots);
        assert!(conv_state_slot_index < self.layout.num_state_slots);
        let resources = self.resources();
        for gdn_layer_index in 0..self.layout.num_gdn_layers {
            let recurrent_state_slot_offset_bytes = (gdn_layer_index * self.layout.num_state_slots
                + recurrent_state_slot_index)
                * self.recurrent_state_bytes();
            debug_assert!(
                recurrent_state_slot_offset_bytes + self.recurrent_state_bytes()
                    <= resources.recurrent_states.len_bytes()
            );
            resources
                .recurrent_states
                .zero_bytes(recurrent_state_slot_offset_bytes, self.recurrent_state_bytes());
            let conv_state_slot_offset_bytes =
                (gdn_layer_index * self.layout.num_state_slots + conv_state_slot_index) * self.conv_state_bytes();
            debug_assert!(conv_state_slot_offset_bytes + self.conv_state_bytes() <= resources.conv_states.len_bytes());
            resources
                .conv_states
                .zero_bytes(conv_state_slot_offset_bytes, self.conv_state_bytes());
        }
        trace::gdn_state(|| {
            format!("event=gdn_state_zero recurrent_slot={recurrent_state_slot} conv_slot={conv_state_slot}")
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_batch(
        &self,
        req_slots: &[u32],
        block_indices: &[usize],
        token_indices: &[u32],
        cu_tokens: &[u32],
        num_spec_tokens: &[u32],
        num_recurrent_requests: usize,
        state_page_ids_by_req: &[Vec<Vec<u32>>],
    ) {
        assert!(!req_slots.is_empty(), "GDN state batch requires requests");
        assert_eq!(block_indices.len(), req_slots.len());
        assert_eq!(token_indices.len(), req_slots.len());
        assert_eq!(num_spec_tokens.len(), req_slots.len());
        assert!(num_recurrent_requests <= req_slots.len());
        assert_eq!(cu_tokens.len(), req_slots.len() + 1);
        assert!(cu_tokens[req_slots.len()] as usize <= self.layout.max_tokens);
        assert_eq!(state_page_ids_by_req.len(), req_slots.len());
        assert_eq!(cu_tokens[0], 0, "GDN state batch cu_tokens must start at zero");
        let num_req_slots = self.request_table().borrow().num_req_slots();
        assert!(
            req_slots.len() <= num_req_slots,
            "GDN request count exceeds state-table capacity"
        );
        assert!(
            req_slots.iter().all(|&req_slot| (req_slot as usize) < num_req_slots),
            "GDN request slot exceeds state-table capacity"
        );
        let num_pages_per_state_slot = self.num_pages_per_state_slot();
        for req_index in 0..req_slots.len() {
            let num_tokens = cu_tokens[req_index + 1]
                .checked_sub(cu_tokens[req_index])
                .expect("GDN state batch cu_tokens must be nondecreasing");
            assert!(num_tokens > 0, "GDN state batch requires tokens for every request");
            assert!(num_spec_tokens[req_index] <= num_tokens);
            if req_index < num_recurrent_requests {
                assert_eq!(num_spec_tokens[req_index], 0, "GDN recurrent inputs must be committed");
            }
            token_indices[req_index]
                .checked_add(num_tokens)
                .and_then(|version| version.checked_add(1))
                .expect("GDN forward state range must fit u32");
            let state_blocks = &state_page_ids_by_req[req_index];
            if !state_blocks.is_empty() {
                let block_count = block_indices[req_index]
                    .checked_add(state_blocks.len())
                    .expect("GDN cache block range must fit usize");
                let block_end = block_count
                    .checked_mul(self.num_tokens_per_block)
                    .expect("GDN cache block end must fit usize");
                u32::try_from(block_end).expect("GDN cache block state version must fit u32");
            }
            for page_ids in state_blocks {
                assert_eq!(
                    page_ids.len(),
                    num_pages_per_state_slot,
                    "GDN block state page count must cover every GDN layer"
                );
                assert!(
                    page_ids
                        .iter()
                        .all(|&page_id| (page_id as usize) < self.num_cache_pages),
                    "GDN runtime supplied a page ID outside the cache-page buffer"
                );
            }
        }
        // The prefix writes full states. Only suffix rows occupy the replay log.
        let num_replay_tokens = cu_tokens[req_slots.len()] - cu_tokens[num_recurrent_requests];
        assert!(
            num_replay_tokens <= self.layout.replay.max_replay_tokens,
            "GDN replay suffix exceeds log capacity"
        );
    }
}

impl GDNRequestStateResources {
    fn new(device: &Device, layout: GDNStateLayout) -> Self {
        let recurrent_layer_bytes = (layout.num_state_slots as u64)
            .checked_mul(layout.recurrent_state_bytes as u64)
            .expect("GDN recurrent layer byte length must fit u64");
        let conv_layer_bytes = (layout.num_state_slots as u64)
            .checked_mul(layout.conv_state_bytes as u64)
            .expect("GDN convolution layer byte length must fit u64");
        // Kernels bind the aggregate arenas at offset zero and add these layer
        // bases with Metal `ulong`. Their layer-local element indices remain u32.
        assert_u32_element_index_domain(recurrent_layer_bytes, size_of::<u16>(), "GDN recurrent layer state");
        assert_u32_element_index_domain(conv_layer_bytes, size_of::<u16>(), "GDN convolution layer state");
        let recurrent_states_bytes = (layout.num_gdn_layers as u64)
            .checked_mul(recurrent_layer_bytes)
            .expect("GDN recurrent state arena byte length must fit u64");
        let conv_states_bytes = (layout.num_gdn_layers as u64)
            .checked_mul(conv_layer_bytes)
            .expect("GDN convolution state arena byte length must fit u64");
        let pages_per_state_slot = layout
            .recurrent_state_bytes
            .div_ceil(layout.page_bytes)
            .checked_add(layout.conv_state_bytes.div_ceil(layout.page_bytes))
            .and_then(|pages| pages.checked_mul(layout.num_gdn_layers))
            .expect("GDN all-layer pages per state slot must fit usize");
        let num_page_ids = layout
            .max_state_io_requests
            .checked_mul(pages_per_state_slot)
            .expect("GDN page-ID size overflow");
        u32::try_from(layout.max_state_io_requests).expect("GDN state I/O request capacity must fit u32");
        u32::try_from(layout.max_replay_jobs).expect("GDN replay job capacity must fit u32");
        let replay_job_bytes = layout
            .max_replay_jobs
            .checked_mul(size_of::<backend_state_replay::Job>())
            .expect("GDN replay job bytes must fit usize");
        Self {
            layout,
            recurrent_states: Buffer::new_zeroed(device, recurrent_states_bytes),
            conv_states: Buffer::new_zeroed(device, conv_states_bytes),
            page_io: GDNStatePageIO::new(device, num_page_ids, layout),
            replay_alpha: Buffer::new_zeroed(device, layout.replay.alpha_bytes()),
            replay_k: Buffer::new_zeroed(device, layout.replay.k_bytes()),
            replay_u: Buffer::new_zeroed(device, layout.replay.u_bytes()),
            replay_qkv: Buffer::new_zeroed(device, layout.replay.qkv_bytes()),
            replay_jobs: Buffer::new_zeroed(device, replay_job_bytes),
            replay_commit: backend_state_replay::Commit::new(device, layout.replay),
        }
    }

    pub fn layer_bindings(&self, gdn_layer_index: usize) -> GDNStateArenaBindings<'_> {
        assert!(gdn_layer_index < self.layout.num_gdn_layers);
        let recurrent_layer_bytes = self.layout.num_state_slots * self.layout.recurrent_state_bytes;
        let conv_layer_bytes = self.layout.num_state_slots * self.layout.conv_state_bytes;
        let recurrent_layer_offset_bytes = gdn_layer_index * recurrent_layer_bytes;
        let conv_layer_offset_bytes = gdn_layer_index * conv_layer_bytes;
        debug_assert!(recurrent_layer_offset_bytes + recurrent_layer_bytes <= self.recurrent_states.len_bytes());
        debug_assert!(conv_layer_offset_bytes + conv_layer_bytes <= self.conv_states.len_bytes());
        GDNStateArenaBindings {
            recurrent_states: &self.recurrent_states,
            recurrent_layer_offset_bytes: recurrent_layer_offset_bytes as u64,
            conv_states: &self.conv_states,
            conv_layer_offset_bytes: conv_layer_offset_bytes as u64,
            replay: ReplayBuffers {
                alpha: &self.replay_alpha,
                k: &self.replay_k,
                u: &self.replay_u,
                qkv: &self.replay_qkv,
                token_offset: gdn_layer_index as u64 * self.layout.replay.max_replay_tokens as u64,
                num_total_tokens: self.layout.replay.max_replay_tokens,
            },
        }
    }
}

impl GDNStatePageIO {
    fn new(device: &Device, num_page_ids: usize, layout: GDNStateLayout) -> Self {
        let config = Self::config(layout);
        Self {
            page_ids: Buffer::new_zeroed_elements(device, num_page_ids, inference_backend_metal::metal::Dtype::Uint32),
            recurrent_state_slots: Buffer::new_zeroed_elements(
                device,
                layout.max_state_io_requests,
                inference_backend_metal::metal::Dtype::Uint32,
            ),
            conv_state_slots: Buffer::new_zeroed_elements(
                device,
                layout.max_state_io_requests,
                inference_backend_metal::metal::Dtype::Uint32,
            ),
            read: backend_state_pages::Read::new(device, config),
            write: backend_state_pages::Write::new(device, config),
        }
    }

    fn prepare_restore(&self, restores: &[GDNStateRestore], num_pages_per_state_slot: usize) {
        self.recurrent_state_slots.write_typed(
            0,
            &restores
                .iter()
                .map(|restore| restore.dst_recurrent_state_slot)
                .collect::<Vec<_>>(),
        );
        self.conv_state_slots.write_typed(
            0,
            &restores
                .iter()
                .map(|restore| restore.dst_conv_state_slot)
                .collect::<Vec<_>>(),
        );
        for (state_io_request_index, restore) in restores.iter().enumerate() {
            self.write_page_ids(state_io_request_index, &restore.page_ids, num_pages_per_state_slot);
        }
    }

    fn prepare_publish(&self, publishes: &[GDNStatePublish], num_pages_per_state_slot: usize) {
        self.recurrent_state_slots.write_typed(
            0,
            &publishes
                .iter()
                .map(|publish| publish.src_recurrent_state_slot)
                .collect::<Vec<_>>(),
        );
        self.conv_state_slots.write_typed(
            0,
            &publishes
                .iter()
                .map(|publish| publish.src_conv_state_slot)
                .collect::<Vec<_>>(),
        );
        for (state_io_request_index, publish) in publishes.iter().enumerate() {
            self.write_page_ids(state_io_request_index, &publish.page_ids, num_pages_per_state_slot);
        }
    }

    fn record_restore<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        pages: &'a Buffer,
        recurrent_states: &'a Buffer,
        conv_states: &'a Buffer,
        num_total_state_io_requests: u32,
        num_active_state_io_requests: ReplayU32,
    ) {
        recorder.record(ReplayOp::opaque(self.read.invoke(
            Self::shape(num_total_state_io_requests),
            num_active_state_io_requests,
            backend_state_pages::ReadBuffers {
                pages,
                recurrent_states,
                conv_states,
                page_ids: &self.page_ids,
                recurrent_state_slots: &self.recurrent_state_slots,
                conv_state_slots: &self.conv_state_slots,
            },
        )));
    }

    fn record_publish<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        pages: &'a Buffer,
        recurrent_states: &'a Buffer,
        conv_states: &'a Buffer,
        num_total_state_io_requests: u32,
        num_active_state_io_requests: ReplayU32,
    ) {
        recorder.record(ReplayOp::opaque(self.write.invoke(
            Self::shape(num_total_state_io_requests),
            num_active_state_io_requests,
            backend_state_pages::WriteBuffers {
                pages,
                recurrent_states,
                conv_states,
                page_ids: &self.page_ids,
                recurrent_state_slots: &self.recurrent_state_slots,
                conv_state_slots: &self.conv_state_slots,
            },
        )));
    }

    fn shape(num_state_io_requests: u32) -> backend_state_pages::Shape {
        backend_state_pages::Shape {
            num_total_state_io_requests: num_state_io_requests,
        }
    }

    fn config(layout: GDNStateLayout) -> backend_state_pages::Config {
        backend_state_pages::Config {
            num_gdn_layers: layout.num_gdn_layers.try_into().expect("GDN layer count must fit u32"),
            num_state_slots: layout
                .num_state_slots
                .try_into()
                .expect("GDN state slot count must fit u32"),
            recurrent_state_bytes: layout
                .recurrent_state_bytes
                .try_into()
                .expect("GDN recurrent state bytes must fit u32"),
            conv_state_bytes: layout
                .conv_state_bytes
                .try_into()
                .expect("GDN convolution state bytes must fit u32"),
            page_bytes: layout.page_bytes.try_into().expect("GDN page bytes must fit u32"),
        }
    }

    fn write_page_ids(&self, state_io_request_index: usize, page_ids: &[u32], pages_per_state_slot: usize) {
        assert_eq!(page_ids.len(), pages_per_state_slot);
        let start = state_io_request_index * pages_per_state_slot;
        debug_assert!(start + page_ids.len() <= self.page_ids.len_bytes() / size_of::<u32>());
        self.page_ids.write_typed(start, page_ids);
    }
}

fn assert_u32_element_index_domain(len_bytes: u64, item_size: usize, name: &str) {
    let item_size_u64 = item_size as u64;
    assert_eq!(
        len_bytes % item_size_u64,
        0,
        "{name} buffer must contain whole elements"
    );
    let num_elements = len_bytes / item_size_u64;
    assert!(num_elements > 0, "{name} buffer must not be empty");
    assert!(
        u32::try_from(num_elements - 1).is_ok(),
        "{name} buffer exceeds the shader u32 element-index domain: num_elements={num_elements}"
    );
}

#[cfg(test)]
#[path = "state_table_test.rs"]
mod tests;
