use std::rc::Rc;

use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::attn::UngatedGQACore;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::weight_layout::Qwen3xGQAWeightBindings;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gqa::backend::GQAKVCacheBindings;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;
use crate::attn::gqa::batch_metadata::GQAReplayShape;
use crate::attn::gqa::request_page_table::GQAPageTableLayout;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::attn::gqa::ungated_backend::UngatedGQA;
use crate::attn::gqa::ungated_backend::UngatedGQAInput;
use crate::attn::gqa::ungated_backend::UngatedGQAReplayTopology;
use crate::attn::gqa::ungated_backend::add_ungated_gqa_private_replay_arguments;
use crate::attn::gqa::ungated_scratch::UngatedGQAScratch;
use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::model::qwen::v3_x::layer::Qwen3xUngatedGQAWeightBuffers;

mod file_io;

pub struct Qwen3MainGQA {
    model_layer_index: usize,
    core: UngatedGQACore,
    metal: GQAMetalConfig,
    weights: Option<Qwen3xUngatedGQAWeightBuffers>,
    backend: Option<Rc<UngatedGQA>>,
    scratch: Option<Rc<UngatedGQAScratch>>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
}

pub struct Qwen3MainGQAState {
    backend: Option<Rc<UngatedGQA>>,
    scratch: Option<Rc<UngatedGQAScratch>>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
    metadata: Option<GQAMetadataBuffers>,
    replay_bucket_policy: GQAReplayBucketPolicy,
    core: UngatedGQACore,
    metal: GQAMetalConfig,
    page_table_layout: GQAPageTableLayout,
    max_tokens: usize,
    num_cache_pages: usize,
}

impl Qwen3MainGQA {
    pub fn new(core: UngatedGQACore, metal: GQAMetalConfig, state: &Qwen3MainGQAState) -> Self {
        Self {
            model_layer_index: core.model_layer_index,
            core,
            metal,
            weights: None,
            backend: Some(Rc::clone(state.backend())),
            scratch: Some(Rc::clone(state.scratch())),
            request_page_table: Some(Rc::clone(state.request_page_table())),
        }
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen3xGQAWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(self.weights.is_none(), "Qwen3 Main GQA weights are already loaded");
        self.weights = Some(Qwen3xUngatedGQAWeightBuffers::load(
            device, store, &bindings, &self.core, self.metal,
        )?);
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        assert!(self.weights.is_some(), "Qwen3 Main GQA weights are not loaded");
        self.weights.take();
    }

    pub fn unload_state(&mut self) {
        assert!(
            self.backend.is_some() && self.scratch.is_some() && self.request_page_table.is_some(),
            "qwen3 Main GQA state is not loaded"
        );
        self.request_page_table.take();
        self.scratch.take();
        self.backend.take();
    }

    pub fn load_state(&mut self, state: &Qwen3MainGQAState) {
        assert!(
            self.backend.is_none() && self.scratch.is_none() && self.request_page_table.is_none(),
            "qwen3 Main GQA state is already loaded"
        );
        self.backend = Some(Rc::clone(state.backend()));
        self.scratch = Some(Rc::clone(state.scratch()));
        self.request_page_table = Some(Rc::clone(state.request_page_table()));
    }

    fn weights(&self) -> &Qwen3xUngatedGQAWeightBuffers {
        self.weights
            .as_ref()
            .expect("Qwen3 Main GQA weights must be loaded before execution")
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        input: &'a Buffer,
        output: &'a Buffer,
        pages: &'a Buffer,
        metadata: &'a GQAMetadataBuffers,
        num_active_tokens: ReplayU32,
    ) {
        let _ = <UngatedGQA as ReplayLayer>::record(
            self.backend(),
            recorder,
            UngatedGQAInput {
                page_table_layout: self.request_page_table().layout(),
                gqa_layer_index: ReplayU32::Fixed(
                    self.model_layer_index
                        .try_into()
                        .expect("qwen3 Main GQA layer index must fit u32"),
                ),
                batch_metadata: metadata,
                hidden_state: input,
                next_hidden_state: output,
                kv_cache: GQAKVCacheBindings {
                    kv_pages: pages,
                    page_ids: self.request_page_table().page_ids_buffer(),
                },
                weights: self.weights().as_borrowed(),
                scratch: self.scratch().bindings(),
                num_active_tokens,
            },
        );
    }

    fn backend(&self) -> &UngatedGQA {
        self.backend
            .as_deref()
            .expect("qwen3 Main GQA state must be loaded before execution")
    }

    fn scratch(&self) -> &UngatedGQAScratch {
        self.scratch
            .as_deref()
            .expect("qwen3 Main GQA state must be loaded before execution")
    }

    fn request_page_table(&self) -> &GQARequestPageTable {
        self.request_page_table
            .as_deref()
            .expect("qwen3 Main GQA state must be loaded before execution")
    }

    pub fn replay_topology(&self, metadata: &GQAMetadataBuffers) -> UngatedGQAReplayTopology {
        self.backend().replay_topology(metadata)
    }
}

impl Qwen3MainGQAState {
    pub fn new(
        device: &Device,
        core: UngatedGQACore,
        metal: GQAMetalConfig,
        page_table_layout: GQAPageTableLayout,
        max_tokens: usize,
        num_cache_pages: usize,
    ) -> Self {
        assert!(num_cache_pages > 0, "qwen3 Main GQA state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "qwen3 Main cache page IDs must fit u32"
        );
        page_table_layout.validate();
        let backend = Rc::new(UngatedGQA::new(
            device,
            core.clone(),
            metal,
            max_tokens,
            page_table_layout,
        ));
        let scratch = Rc::new(backend.new_scratch());
        let metadata = backend.new_metadata();
        let replay_bucket_policy = backend.replay_bucket_policy();
        Self {
            backend: Some(backend),
            scratch: Some(scratch),
            request_page_table: Some(Rc::new(GQARequestPageTable::new(device, page_table_layout))),
            metadata: Some(metadata),
            replay_bucket_policy,
            core,
            metal,
            page_table_layout,
            max_tokens,
            num_cache_pages,
        }
    }

    pub fn num_tokens_per_page(&self) -> usize {
        self.backend().num_tokens_per_page() as usize
    }

    pub fn metadata(&self) -> &GQAMetadataBuffers {
        self.metadata
            .as_ref()
            .expect("qwen3 Main GQA metadata state must be loaded")
    }

    fn backend(&self) -> &Rc<UngatedGQA> {
        self.backend
            .as_ref()
            .expect("qwen3 Main GQA backend state must be loaded")
    }

    fn scratch(&self) -> &Rc<UngatedGQAScratch> {
        self.scratch
            .as_ref()
            .expect("qwen3 Main GQA scratch state must be loaded")
    }

    fn request_page_table(&self) -> &Rc<GQARequestPageTable> {
        self.request_page_table
            .as_ref()
            .expect("qwen3 Main GQA request page-table state must be loaded")
    }

    pub fn write_page_ids(&self, req_slot: u32, block_index: usize, page_ids: &[u32]) {
        let request_page_table = self.request_page_table();
        let num_page_ids_per_layer = request_page_table.num_page_ids_per_block();
        let expected_page_ids = request_page_table.num_layers() * num_page_ids_per_layer;
        assert_eq!(
            page_ids.len(),
            expected_page_ids,
            "qwen3 Main GQA cache block must contain all layer page IDs"
        );
        assert!(
            page_ids
                .iter()
                .all(|&page_id| (page_id as usize) < self.num_cache_pages),
            "runtime supplied a qwen3 Main GQA page ID outside the cache-page buffer"
        );
        for (layer_index, layer_page_ids) in page_ids.chunks_exact(num_page_ids_per_layer).enumerate() {
            request_page_table.write_page_ids(req_slot, layer_index, block_index, layer_page_ids);
        }
    }

    pub fn read_page_ids(&self, req_slot: u32, block_index: usize) -> Vec<u32> {
        let request_page_table = self.request_page_table();
        let mut page_ids =
            Vec::with_capacity(request_page_table.num_layers() * request_page_table.num_page_ids_per_block());
        for layer_index in 0..request_page_table.num_layers() {
            page_ids.extend(request_page_table.read_page_ids(req_slot, layer_index, block_index));
        }
        page_ids
    }

    pub fn prepare_metadata(
        &self,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
        num_total_tokens: u32,
    ) -> GQAReplayShape {
        self.backend().prepare(
            self.metadata(),
            req_slots,
            token_indices,
            cu_tokens,
            &self.replay_bucket_policy,
            num_total_tokens,
        )
    }

    pub fn replay_topology(&self) -> UngatedGQAReplayTopology {
        self.backend().replay_topology(self.metadata())
    }

    pub fn add_private_replay_arguments(&self, arguments: &mut ReplayArguments) {
        add_ungated_gqa_private_replay_arguments(self.metadata().replay_shape(), self.replay_topology(), arguments);
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_page_table().reset_req_slots(req_slots);
    }

    pub fn release_resources(&mut self) {
        assert!(
            self.backend.is_some()
                && self.scratch.is_some()
                && self.request_page_table.is_some()
                && self.metadata.is_some(),
            "qwen3 Main GQA state resources are not loaded"
        );
        self.request_page_table
            .take()
            .expect("qwen3 Main GQA request page-table state must be loaded");
        self.metadata.take();
        self.scratch.take();
        self.backend.take();
    }

    pub fn allocate_resources(&mut self, device: &Device) {
        assert!(
            self.backend.is_none()
                && self.scratch.is_none()
                && self.request_page_table.is_none()
                && self.metadata.is_none(),
            "qwen3 Main GQA state resources are already loaded"
        );
        let backend = Rc::new(UngatedGQA::new(
            device,
            self.core.clone(),
            self.metal,
            self.max_tokens,
            self.page_table_layout,
        ));
        let scratch = Rc::new(backend.new_scratch());
        let metadata = backend.new_metadata();
        self.replay_bucket_policy = backend.replay_bucket_policy();
        self.backend = Some(backend);
        self.scratch = Some(scratch);
        self.request_page_table = Some(Rc::new(GQARequestPageTable::new(device, self.page_table_layout)));
        self.metadata = Some(metadata);
    }
}
