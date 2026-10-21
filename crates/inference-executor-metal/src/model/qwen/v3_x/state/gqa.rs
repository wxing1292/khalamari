use std::rc::Rc;

use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::attn::GQACore;
use inference_runtime_core::runtime::RawRequestSlot;

use crate::attn::gqa::backend::GQA;
use crate::attn::gqa::backend::GQAMetalConfig;
use crate::attn::gqa::backend::GQAReplayTopology;
use crate::attn::gqa::backend::add_gqa_private_replay_arguments;
use crate::attn::gqa::backend::add_gqa_replay_arguments;
use crate::attn::gqa::batch_metadata::GQAMetadataBuffers;
use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;
use crate::attn::gqa::batch_metadata::GQAReplayShape;
use crate::attn::gqa::request_page_table::GQAPageTableLayout;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::attn::gqa::scratch::GQAScratch;

mod file_io;

pub struct Qwen3xGQAState {
    backend: Option<Rc<GQA>>,
    scratch: Option<Rc<GQAScratch>>,
    request_page_table: Option<Rc<GQARequestPageTable>>,
    metadata: Option<GQAMetadataBuffers>,
    core: GQACore,
    metal: GQAMetalConfig,
    page_table_layout: GQAPageTableLayout,
    max_tokens: usize,
    replay_bucket_policy: GQAReplayBucketPolicy,
    num_cache_pages: usize,
}

impl Qwen3xGQAState {
    pub fn new(
        device: &Device,
        core: GQACore,
        metal: GQAMetalConfig,
        page_table_layout: GQAPageTableLayout,
        max_tokens: usize,
        num_cache_pages: usize,
    ) -> Self {
        assert!(num_cache_pages > 0, "qwen3.x GQA state requires cache pages");
        assert!(
            u32::try_from(num_cache_pages - 1).is_ok(),
            "qwen3.x cache page IDs must fit u32"
        );
        page_table_layout.validate();
        let backend = Rc::new(GQA::new(device, core.clone(), metal, max_tokens, page_table_layout));
        let scratch = Rc::new(backend.new_scratch());
        let metadata = backend.new_metadata();
        let replay_bucket_policy = backend.replay_bucket_policy();
        Self {
            backend: Some(backend),
            scratch: Some(scratch),
            request_page_table: Some(Rc::new(GQARequestPageTable::new(device, page_table_layout))),
            metadata: Some(metadata),
            core,
            metal,
            page_table_layout,
            max_tokens,
            replay_bucket_policy,
            num_cache_pages,
        }
    }

    pub fn backend(&self) -> &Rc<GQA> {
        self.backend.as_ref().expect("Qwen3.x GQA backend state must be loaded")
    }

    pub fn scratch(&self) -> &Rc<GQAScratch> {
        self.scratch.as_ref().expect("Qwen3.x GQA scratch state must be loaded")
    }

    pub fn request_page_table(&self) -> &Rc<GQARequestPageTable> {
        self.request_page_table
            .as_ref()
            .expect("Qwen3.x GQA request page-table state must be loaded")
    }

    pub fn metadata(&self) -> &GQAMetadataBuffers {
        self.metadata
            .as_ref()
            .expect("Qwen3.x GQA metadata state must be loaded")
    }

    pub fn write_page_ids(&self, req_slot: u32, block_index: usize, page_ids: &[u32]) {
        let request_page_table = self.request_page_table();
        let num_page_ids_per_layer = request_page_table.num_page_ids_per_block();
        let expected_page_ids = request_page_table.num_layers() * num_page_ids_per_layer;
        assert_eq!(
            page_ids.len(),
            expected_page_ids,
            "qwen3.x GQA cache block must contain all layer page IDs"
        );
        assert!(
            page_ids
                .iter()
                .all(|&page_id| (page_id as usize) < self.num_cache_pages),
            "runtime supplied a qwen3.x GQA page ID outside the cache-page buffer"
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

    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        self.backend().replay_token_topology_boundaries()
    }

    pub fn replay_topology(&self) -> GQAReplayTopology {
        self.backend().replay_topology(self.metadata())
    }

    pub fn add_replay_arguments(&self, arguments: &mut ReplayArguments) {
        add_gqa_replay_arguments(self.metadata().replay_shape(), self.replay_topology(), arguments);
    }

    pub fn add_private_replay_arguments(&self, arguments: &mut ReplayArguments) {
        add_gqa_private_replay_arguments(self.metadata().replay_shape(), self.replay_topology(), arguments);
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
            "Qwen3.x GQA state resources are not loaded"
        );
        self.request_page_table
            .take()
            .expect("Qwen3.x GQA request page-table state must be loaded");
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
            "Qwen3.x GQA state resources are already loaded"
        );
        let backend = Rc::new(GQA::new(
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use inference_backend_metal::components::rms_norm_rope::RopeScaling;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;
    use inference_executor_core::attn::GQACore;
    use inference_runtime_core::compute::ExecutorHibernationPlan;

    use super::Qwen3xGQAState;
    use crate::attn::gqa::backend::GQAMetalConfig;
    use crate::attn::gqa::request_page_table::GQAPageTableLayout;
    use crate::model::state_snapshot::FullStateIO;
    use crate::model::state_snapshot::GQAStateSnapshotFiles;
    use crate::model::state_snapshot::StateSnapshotFile;
    use crate::model::state_snapshot::StateSnapshotReader;
    use crate::model::state_snapshot::StateSnapshotWriter;

    const NUM_REQ_SLOTS: usize = 2;
    const NUM_GQA_LAYERS: usize = 2;
    const NUM_BLOCKS: usize = 2;
    const NUM_PAGE_IDS_PER_BLOCK: usize = 2;
    const NUM_CACHE_PAGES: u32 = 64;
    const SNAPSHOT_FILES: GQAStateSnapshotFiles =
        GQAStateSnapshotFiles::new(StateSnapshotFile::MainGQARequestPageTable);

    #[test]
    fn test_full_state_unload_load_fixed() {
        assert_unload_load("fixed", vec![1, 7, 11, 29, 31, 63, 4, 19, 2, 8, 12, 30, 32, 62, 5, 20]);
    }

    #[test]
    fn test_full_state_unload_load_random() {
        let mut random = TestRandom::new(0x4751_415f_5354_4154);
        let page_ids = (0..num_page_ids())
            .map(|_| random.next_u32() % NUM_CACHE_PAGES)
            .collect();
        assert_unload_load("random", page_ids);
    }

    #[test]
    fn test_write_read_page_ids_uses_complete_gqa_role_block() {
        let device = Device::system_default();
        let state = new_state(&device);

        state.write_page_ids(1, 1, &[10, 11, 20, 21]);

        assert_eq!(state.read_page_ids(1, 1), vec![10, 11, 20, 21]);
        assert_eq!(state.request_page_table().read_page_ids(1, 0, 1), vec![10, 11]);
        assert_eq!(state.request_page_table().read_page_ids(1, 1, 1), vec![20, 21]);
    }

    #[test]
    #[should_panic(expected = "qwen3.x GQA cache block must contain all layer page IDs")]
    fn test_write_page_ids_rejects_wrong_role_block_length() {
        let device = Device::system_default();
        let state = new_state(&device);

        state.write_page_ids(1, 1, &[10, 11, 20]);
    }

    #[test]
    #[should_panic(expected = "runtime supplied a qwen3.x GQA page ID outside the cache-page buffer")]
    fn test_write_page_ids_rejects_page_id_outside_cache() {
        let device = Device::system_default();
        let state = new_state(&device);

        state.write_page_ids(1, 1, &[10, 11, 20, NUM_CACHE_PAGES]);
    }

    fn assert_unload_load(name: &str, expected_page_ids: Vec<u32>) {
        assert_eq!(expected_page_ids.len(), num_page_ids());
        assert!(expected_page_ids.iter().all(|&page_id| page_id < NUM_CACHE_PAGES));

        let device = Device::system_default();
        let mut state = new_state(&device);
        write_page_ids(&state, &expected_page_ids);

        let snapshot_path = snapshot_path(name);
        let buffer_io = inference_backend_metal::metal::BufferIO::new(&device);
        let snapshot_files = [SNAPSHOT_FILES.request_page_table()];
        let mut writer = StateSnapshotWriter::new(
            &snapshot_path,
            &snapshot_files,
            &ExecutorHibernationPlan::All,
            &buffer_io,
        )
        .unwrap();
        state.write_full_state(&mut writer, SNAPSHOT_FILES).unwrap();
        writer.commit().unwrap();

        state.release_resources();
        state.allocate_resources(&device);

        let mut reader = StateSnapshotReader::open(
            &snapshot_path,
            &snapshot_files,
            &ExecutorHibernationPlan::All,
            &buffer_io,
        )
        .unwrap();
        state.read_full_state(&mut reader, SNAPSHOT_FILES).unwrap();
        reader.finish().unwrap();

        assert_eq!(read_page_ids(&state), expected_page_ids);
        std::fs::remove_dir_all(snapshot_path).unwrap();
    }

    fn new_state(device: &Device) -> Qwen3xGQAState {
        Qwen3xGQAState::new(
            device,
            GQACore::new(0, 128, 128, 1, 1, 1.0),
            GQAMetalConfig {
                group_size: 32,
                bits: 4,
                page_bytes: 4096,
                rope_dim: 128,
                norm_eps: 1.0e-6,
                rope_theta: 1_000_000.0,
                rope_scaling: RopeScaling::Default,
                io_dtype: Dtype::Bfloat16,
            },
            GQAPageTableLayout {
                num_req_slots: NUM_REQ_SLOTS as u32,
                num_gqa_layers: NUM_GQA_LAYERS as u32,
                num_blocks: NUM_BLOCKS as u32,
                num_page_ids_per_block: NUM_PAGE_IDS_PER_BLOCK as u32,
            },
            2,
            NUM_CACHE_PAGES as usize,
        )
    }

    fn write_page_ids(state: &Qwen3xGQAState, page_ids: &[u32]) {
        let (entries, remainder) = page_ids.as_chunks::<NUM_PAGE_IDS_PER_BLOCK>();
        assert!(remainder.is_empty());
        let mut entries = entries.iter();
        for req_slot in 0..NUM_REQ_SLOTS as u32 {
            for layer_index in 0..NUM_GQA_LAYERS {
                for block_index in 0..NUM_BLOCKS {
                    state.request_page_table().write_page_ids(
                        req_slot,
                        layer_index,
                        block_index,
                        entries.next().unwrap(),
                    );
                }
            }
        }
        assert!(entries.next().is_none());
    }

    fn read_page_ids(state: &Qwen3xGQAState) -> Vec<u32> {
        let mut page_ids = Vec::with_capacity(num_page_ids());
        for req_slot in 0..NUM_REQ_SLOTS as u32 {
            for layer_index in 0..NUM_GQA_LAYERS {
                for block_index in 0..NUM_BLOCKS {
                    page_ids.extend(
                        state
                            .request_page_table()
                            .read_page_ids(req_slot, layer_index, block_index),
                    );
                }
            }
        }
        page_ids
    }

    fn num_page_ids() -> usize {
        NUM_REQ_SLOTS * NUM_GQA_LAYERS * NUM_BLOCKS * NUM_PAGE_IDS_PER_BLOCK
    }

    fn snapshot_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "psi-dec-gqa-state-unload-load-{}-{name}.state",
            std::process::id()
        ))
    }

    struct TestRandom(u64);

    impl TestRandom {
        fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 as u32
        }
    }
}
