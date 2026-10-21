//! Persistent and replay state for BiDiBlockGQA.

use std::rc::Rc;

use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::attn::BiDiBlockCapacity;
use inference_executor_core::attn::BiDiBlockGQACore;
use inference_executor_core::attn::BiDiBlockGQAMetadata;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::bidi_block_gqa::backend::BiDiBlockGQA;
use crate::attn::bidi_block_gqa::backend::BiDiBlockGQAMetalConfig;
use crate::attn::bidi_block_gqa::backend::add_bidi_block_gqa_replay_arguments;
use crate::attn::bidi_block_gqa::capacity::BiDiBlockGQACapacity;
use crate::attn::bidi_block_gqa::kv_cache_write::BiDiBlockGQAKVCacheWriteScratch;
use crate::attn::bidi_block_gqa::metadata::BiDiBlockGQAMetadataBuffers;
use crate::attn::bidi_block_gqa::scratch::BiDiBlockGQAScratch;
use crate::attn::bidi_block_gqa::sdpa::Selector as SDPASelector;
use crate::attn::gqa::batch_metadata::GQAReplayShape;
use crate::attn::gqa::request_page_table::GQAPageTableLayout;
use crate::attn::gqa::request_page_table::GQARequestPageTable;

mod file_io;

pub struct BiDiBlockGQAState {
    sdpa_execution: backend_sdpa::ExecutionVariant,
    bidi_block_scratch: Option<Rc<BiDiBlockGQAScratch>>,
    kv_cache_write_scratch: Option<Rc<BiDiBlockGQAKVCacheWriteScratch>>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
    metadata: Option<BiDiBlockGQAMetadataBuffers>,
    core: BiDiBlockGQACore,
    sdpa_config: backend_sdpa::Config,
    capacity: BiDiBlockGQACapacity,
    max_context_tokens: usize,
    page_table_layout: GQAPageTableLayout,
    num_cache_pages: usize,
}

impl BiDiBlockGQAState {
    pub fn new(
        device: &Device,
        core: BiDiBlockGQACore,
        sdpa_config: backend_sdpa::Config,
        page_table_layout: GQAPageTableLayout,
        capacity: BiDiBlockCapacity,
        max_context_tokens: usize,
        num_cache_pages: usize,
    ) -> Self {
        assert!(
            max_context_tokens > 0,
            "BiDiBlockGQA KV-cache-write scratch requires tokens"
        );
        assert!(num_cache_pages > 0, "BiDiBlockGQA state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "BiDiBlockGQA cache page IDs must fit u32"
        );
        core.validate();
        sdpa_config.validate();
        page_table_layout.validate();
        assert_eq!(
            core.block_size, capacity.block_size,
            "BiDiBlockGQA state core and capacity block sizes must match"
        );
        let attention = &core.attention;
        assert_eq!(sdpa_config.num_q_heads as usize, attention.num_q_heads);
        assert_eq!(sdpa_config.num_kv_heads as usize, attention.num_kv_heads);
        assert_eq!(sdpa_config.head_dim as usize, attention.head_dim);
        let sdpa_selection = SDPASelector::new(backend_sdpa::Registry::new(sdpa_config), capacity).select();
        let sdpa_execution = sdpa_selection.variant();
        let gqa_capacity = sdpa_selection.capacity();
        Self {
            sdpa_execution,
            bidi_block_scratch: Some(Rc::new(BiDiBlockGQAScratch::new(
                device,
                &core,
                sdpa_config.io_dtype,
                gqa_capacity,
            ))),
            kv_cache_write_scratch: Some(Rc::new(BiDiBlockGQAKVCacheWriteScratch::new(
                device,
                &core,
                sdpa_config.io_dtype,
                max_context_tokens,
            ))),
            request_page_table: Some(Rc::new(GQARequestPageTable::new(device, page_table_layout))),
            metadata: Some(BiDiBlockGQAMetadataBuffers::new(device, gqa_capacity, sdpa_execution)),
            core,
            sdpa_config,
            capacity: gqa_capacity,
            max_context_tokens,
            page_table_layout,
            num_cache_pages,
        }
    }

    pub fn num_tokens_per_page(&self) -> usize {
        self.sdpa_config.tokens_per_page as usize
    }

    pub fn max_sdpa_map_task_templates(&self) -> u32 {
        self.capacity.max_sdpa_map_task_templates as u32
    }

    pub fn prepare_bidi_block(&self, block: &BiDiBlockGQAMetadata) -> GQAReplayShape {
        self.metadata().update(block)
    }

    pub fn prepare_bidi_block_with_active_requests(
        &self,
        block: &BiDiBlockGQAMetadata,
        num_active_requests: usize,
    ) -> GQAReplayShape {
        self.metadata().update_with_active_requests(block, num_active_requests)
    }

    pub fn add_replay_arguments(&self, arguments: &mut ReplayArguments) {
        add_bidi_block_gqa_replay_arguments(self.metadata().replay_shape(), arguments);
    }

    pub fn new_gqa(&self, device: &Device, core: BiDiBlockGQACore, metal: BiDiBlockGQAMetalConfig) -> BiDiBlockGQA {
        let shared = &self.core.attention;
        let attention = &core.attention;
        assert_eq!(core.block_size, self.core.block_size);
        assert_eq!(attention.hidden_dim, shared.hidden_dim);
        assert_eq!(attention.head_dim, shared.head_dim);
        assert_eq!(attention.num_q_heads, shared.num_q_heads);
        assert_eq!(attention.num_kv_heads, shared.num_kv_heads);
        assert_eq!(attention.scale, shared.scale);
        assert_eq!(metal.io_dtype, self.sdpa_config.io_dtype);
        BiDiBlockGQA::new(device, core, metal, self.sdpa_execution)
    }

    pub fn write_page_ids(&self, req_slot: u32, block_index: usize, page_ids: &[u32]) {
        let request_page_table = self.request_page_table_ref();
        let num_page_ids_per_layer = request_page_table.num_page_ids_per_block();
        let expected_page_ids = request_page_table.num_layers() * num_page_ids_per_layer;
        assert_eq!(
            page_ids.len(),
            expected_page_ids,
            "BiDiBlockGQA cache block must contain all layer page IDs"
        );
        assert!(
            page_ids
                .iter()
                .all(|&page_id| (page_id as usize) < self.num_cache_pages),
            "runtime supplied a BiDiBlockGQA page ID outside the cache-page buffer"
        );
        for (layer_index, layer_page_ids) in page_ids.chunks_exact(num_page_ids_per_layer).enumerate() {
            request_page_table.write_page_ids(req_slot, layer_index, block_index, layer_page_ids);
        }
    }

    pub fn read_page_ids(&self, req_slot: u32, block_index: usize) -> Vec<u32> {
        let request_page_table = self.request_page_table_ref();
        let mut page_ids =
            Vec::with_capacity(request_page_table.num_layers() * request_page_table.num_page_ids_per_block());
        for layer_index in 0..request_page_table.num_layers() {
            page_ids.extend(request_page_table.read_page_ids(req_slot, layer_index, block_index));
        }
        page_ids
    }

    pub fn reset_req_slots(&self, req_slots: &[RawRequestSlot]) {
        self.request_page_table_ref().reset_req_slots(req_slots);
    }

    pub fn bidi_block_scratch(&self) -> Rc<BiDiBlockGQAScratch> {
        Rc::clone(
            self.bidi_block_scratch
                .as_ref()
                .expect("BiDiBlockGQA block scratch state must be loaded"),
        )
    }

    pub fn kv_cache_write_scratch(&self) -> Rc<BiDiBlockGQAKVCacheWriteScratch> {
        Rc::clone(
            self.kv_cache_write_scratch
                .as_ref()
                .expect("BiDiBlockGQA KV-cache-write scratch state must be loaded"),
        )
    }

    pub fn request_page_table(&self) -> Rc<GQARequestPageTable> {
        Rc::clone(self.request_page_table_ref())
    }

    pub fn metadata(&self) -> &BiDiBlockGQAMetadataBuffers {
        self.metadata
            .as_ref()
            .expect("BiDiBlockGQA metadata state must be loaded")
    }

    pub fn release_resources(&mut self) {
        assert!(
            self.bidi_block_scratch.is_some()
                && self.kv_cache_write_scratch.is_some()
                && self.request_page_table.is_some()
                && self.metadata.is_some(),
            "BiDiBlockGQA state resources are not loaded"
        );
        self.request_page_table
            .take()
            .expect("BiDiBlockGQA request page-table state must be loaded");
        self.metadata.take();
        self.kv_cache_write_scratch.take();
        self.bidi_block_scratch.take();
    }

    pub fn allocate_resources(&mut self, device: &Device) {
        assert!(
            self.bidi_block_scratch.is_none()
                && self.kv_cache_write_scratch.is_none()
                && self.request_page_table.is_none()
                && self.metadata.is_none(),
            "BiDiBlockGQA state resources are already loaded"
        );
        self.bidi_block_scratch = Some(Rc::new(BiDiBlockGQAScratch::new(
            device,
            &self.core,
            self.sdpa_config.io_dtype,
            self.capacity,
        )));
        self.kv_cache_write_scratch = Some(Rc::new(BiDiBlockGQAKVCacheWriteScratch::new(
            device,
            &self.core,
            self.sdpa_config.io_dtype,
            self.max_context_tokens,
        )));
        self.request_page_table = Some(Rc::new(GQARequestPageTable::new(device, self.page_table_layout)));
        self.metadata = Some(BiDiBlockGQAMetadataBuffers::new(
            device,
            self.capacity,
            self.sdpa_execution,
        ));
    }

    fn request_page_table_ref(&self) -> &Rc<GQARequestPageTable> {
        self.request_page_table
            .as_ref()
            .expect("BiDiBlockGQA request page-table state must be loaded")
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;
    use inference_executor_core::attn::BiDiBlockCapacity;
    use inference_executor_core::attn::BiDiBlockGQACore;
    use inference_executor_core::attn::UngatedGQACore;

    use super::BiDiBlockGQAState;
    use crate::attn::gqa::request_page_table::GQAPageTableLayout;

    #[test]
    fn test_write_read_page_ids_uses_complete_bidi_block_gqa_block() {
        let device = Device::system_default();
        let state = new_state(&device);

        state.write_page_ids(1, 1, &[30, 31, 40, 41]);

        assert_eq!(state.read_page_ids(1, 1), vec![30, 31, 40, 41]);
    }

    #[test]
    #[should_panic(expected = "runtime supplied a BiDiBlockGQA page ID outside the cache-page buffer")]
    fn test_write_page_ids_rejects_page_id_outside_cache() {
        let device = Device::system_default();
        let state = new_state(&device);

        state.write_page_ids(1, 1, &[30, 31, 40, 64]);
    }

    fn new_state(device: &Device) -> BiDiBlockGQAState {
        BiDiBlockGQAState::new(
            device,
            BiDiBlockGQACore::new(UngatedGQACore::new(0, 128, 128, 1, 1, 1.0), 1),
            backend_sdpa::Config {
                io_dtype: Dtype::Bfloat16,
                num_q_heads: 1,
                num_kv_heads: 1,
                head_dim: 128,
                tokens_per_page: 8,
            },
            GQAPageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 2,
                num_blocks: 2,
                num_page_ids_per_block: 2,
            },
            BiDiBlockCapacity::new(2, 1),
            2,
            64,
        )
    }
}
