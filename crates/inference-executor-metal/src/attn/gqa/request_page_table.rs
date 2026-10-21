use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;

mod file_io;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQAPageTableLayout {
    /// Init-time page-table storage layout. Treat the memory as a tensor with
    /// shape `[num_req_slots, num_gqa_layers, num_blocks, num_page_ids_per_block]`.
    ///
    /// One physical page stores both K and V:
    /// `[K/V, kv_head, page_token_index, head_dim]`.
    pub num_req_slots: u32,
    /// Number of GQA page-table layer slots.
    pub num_gqa_layers: u32,
    /// Logical cache-block capacity per request slot and GQA layer.
    pub num_blocks: u32,
    /// Physical page IDs assigned to one request/GQA-layer/cache-block tuple.
    pub num_page_ids_per_block: u32,
}

impl GQAPageTableLayout {
    pub fn validate(self) {
        assert!(self.num_req_slots > 0);
        assert!(self.num_blocks > 0);
        assert!(self.num_gqa_layers > 0);
        assert!(self.num_page_ids_per_block > 0);
    }

    pub fn num_page_ids(self) -> usize {
        (self.num_req_slots as usize)
            .checked_mul(self.num_gqa_layers as usize)
            .and_then(|count| count.checked_mul(self.num_blocks as usize))
            .and_then(|count| count.checked_mul(self.num_page_ids_per_block as usize))
            .expect("GQA page-table element count must fit usize")
    }

    /// Number of physical GQA pages addressable by one request/GQA-layer table.
    pub fn num_physical_pages_per_request(self) -> usize {
        (self.num_blocks as usize)
            .checked_mul(self.num_page_ids_per_block as usize)
            .expect("GQA per-request physical page count must fit usize")
    }
}

#[derive(Debug)]
pub struct GQARequestPageTable {
    layout: GQAPageTableLayout,
    page_ids: Buffer,
}

impl GQARequestPageTable {
    pub fn new(device: &Device, layout: GQAPageTableLayout) -> Self {
        layout.validate();
        Self {
            layout,
            page_ids: Buffer::new_zeroed_elements(
                device,
                layout.num_page_ids(),
                inference_backend_metal::metal::Dtype::Uint32,
            ),
        }
    }

    pub fn num_req_slots(&self) -> usize {
        self.layout.num_req_slots as usize
    }

    pub fn num_layers(&self) -> usize {
        self.layout.num_gqa_layers as usize
    }

    pub fn num_blocks(&self) -> usize {
        self.layout.num_blocks as usize
    }

    pub fn num_page_ids_per_block(&self) -> usize {
        self.layout.num_page_ids_per_block as usize
    }

    pub fn layout(&self) -> GQAPageTableLayout {
        self.layout
    }

    pub fn read_page_ids(&self, req_slot: u32, layer_index: usize, block_index: usize) -> Vec<u32> {
        self.assert_req_slot(req_slot);
        self.assert_layer_index(layer_index);
        self.assert_block_index(block_index);
        let start = self.page_ids_start_index(req_slot, layer_index, block_index);
        self.page_ids_buffer().read_typed(start, self.num_page_ids_per_block())
    }

    pub fn page_ids_buffer(&self) -> &Buffer {
        &self.page_ids
    }

    pub fn write_page_ids(&self, req_slot: u32, layer_index: usize, block_index: usize, page_ids: &[u32]) {
        self.assert_req_slot(req_slot);
        self.assert_layer_index(layer_index);
        self.assert_block_index(block_index);
        assert_eq!(
            page_ids.len(),
            self.num_page_ids_per_block(),
            "GQA page-id count must match one request/GQA-layer/block entry"
        );
        let start = self.page_ids_start_index(req_slot, layer_index, block_index);
        self.page_ids_buffer().write_typed(start, page_ids);
    }

    pub fn reset_req_slot(&self, req_slot: u32) {
        self.assert_req_slot(req_slot);
        let start = self.page_ids_start_index(req_slot, 0, 0) * size_of::<u32>();
        let len = self.page_ids_per_request() * size_of::<u32>();
        debug_assert!(start + len <= self.page_ids_buffer().len_bytes());
        self.page_ids_buffer().zero_bytes(start, len);
    }

    pub fn reset_req_slots(&self, req_slots: &[u32]) {
        for &req_slot in req_slots {
            self.reset_req_slot(req_slot);
        }
    }

    fn assert_req_slot(&self, req_slot: u32) {
        if req_slot as usize >= self.num_req_slots() {
            panic!(
                "GQA request page table req_slot out of range: req_slot={req_slot} num_req_slots={}",
                self.num_req_slots()
            );
        }
    }

    fn assert_layer_index(&self, layer_index: usize) {
        if layer_index >= self.num_layers() {
            panic!(
                "GQA request page table layer out of range: layer_index={layer_index} num_layers={}",
                self.num_layers()
            );
        }
    }

    fn assert_block_index(&self, block_index: usize) {
        if block_index >= self.num_blocks() {
            panic!(
                "GQA request page table block out of range: block_index={block_index} num_blocks={}",
                self.num_blocks()
            );
        }
    }

    fn page_ids_start_index(&self, req_slot: u32, layer_index: usize, block_index: usize) -> usize {
        let index = (((req_slot as usize * self.num_layers()) + layer_index) * self.num_blocks() + block_index)
            * self.num_page_ids_per_block();
        debug_assert!(index + self.num_page_ids_per_block() <= self.layout.num_page_ids());
        index
    }

    fn page_ids_per_request(&self) -> usize {
        self.num_layers() * self.num_blocks() * self.num_page_ids_per_block()
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Device;
    use inference_runtime_core::compute::ExecutorHibernationPlan;

    use super::GQARequestPageTable;
    use crate::attn::gqa::request_page_table::GQAPageTableLayout;
    use crate::model::state_snapshot::GQAStateSnapshotFiles;
    use crate::model::state_snapshot::SelectedStateIO;
    use crate::model::state_snapshot::StateSnapshotFile;
    use crate::model::state_snapshot::StateSnapshotReader;
    use crate::model::state_snapshot::StateSnapshotWriter;

    const SNAPSHOT_FILES: GQAStateSnapshotFiles =
        GQAStateSnapshotFiles::new(StateSnapshotFile::MainGQARequestPageTable);

    #[test]
    fn test_page_table_lifecycle_resets_selected_requests_and_preserves_neighbors() {
        let device = Device::system_default();
        let page_table = GQARequestPageTable::new(
            &device,
            GQAPageTableLayout {
                num_req_slots: 4,
                num_gqa_layers: 3,
                num_blocks: 6,
                num_page_ids_per_block: 2,
            },
        );

        page_table.write_page_ids(1, 0, 0, &[10, 11]);
        page_table.write_page_ids(1, 0, 1, &[12, 13]);
        page_table.write_page_ids(1, 2, 0, &[20, 21]);
        page_table.write_page_ids(2, 1, 0, &[30, 31]);
        page_table.write_page_ids(3, 0, 0, &[40, 41]);

        assert_eq!(page_table.num_req_slots(), 4);
        assert_eq!(page_table.num_layers(), 3);
        assert_eq!(page_table.num_blocks(), 6);
        assert_eq!(page_table.num_page_ids_per_block(), 2);
        assert_eq!(page_table.read_page_ids(1, 0, 0), vec![10, 11]);
        assert_eq!(page_table.read_page_ids(1, 0, 1), vec![12, 13]);
        assert_eq!(page_table.read_page_ids(1, 2, 0), vec![20, 21]);

        page_table.reset_req_slot(1);

        assert_eq!(page_table.read_page_ids(1, 0, 0), vec![0, 0]);
        assert_eq!(page_table.read_page_ids(1, 0, 1), vec![0, 0]);
        assert_eq!(page_table.read_page_ids(1, 2, 0), vec![0, 0]);
        assert_eq!(page_table.read_page_ids(2, 1, 0), vec![30, 31]);
        assert_eq!(page_table.read_page_ids(3, 0, 0), vec![40, 41]);
    }

    #[test]
    fn test_selected_state_unload_load() {
        let device = Device::system_default();
        let layout = GQAPageTableLayout {
            num_req_slots: 4,
            num_gqa_layers: 2,
            num_blocks: 2,
            num_page_ids_per_block: 2,
        };
        let source = GQARequestPageTable::new(&device, layout);
        let source_values = (100_u32..132).collect::<Vec<_>>();
        source.page_ids_buffer().write_typed(0, &source_values);
        let selected_request_slot_ranges = [1..2, 3..4];
        let plan = ExecutorHibernationPlan::selected(selected_request_slot_ranges.to_vec(), Vec::new());
        let snapshot_path =
            std::env::temp_dir().join(format!("psi-dec-gqa-selected-state-{}.state", std::process::id()));
        let buffer_io = inference_backend_metal::metal::BufferIO::new(&device);
        let snapshot_files = [SNAPSHOT_FILES.request_page_table()];

        let mut writer = StateSnapshotWriter::new(&snapshot_path, &snapshot_files, &plan, &buffer_io).unwrap();
        source
            .write_selected_state(&mut writer, SNAPSHOT_FILES, &selected_request_slot_ranges)
            .unwrap();
        writer.commit().unwrap();

        let mut restored = GQARequestPageTable::new(&device, layout);
        let mut reader = StateSnapshotReader::open(&snapshot_path, &snapshot_files, &plan, &buffer_io).unwrap();
        restored
            .read_selected_state(&mut reader, SNAPSHOT_FILES, &selected_request_slot_ranges)
            .unwrap();
        reader.finish().unwrap();

        assert_eq!(
            restored.page_ids_buffer().read_typed::<u32>(0, 32),
            [
                vec![0; 8],
                source_values[8..16].to_vec(),
                vec![0; 8],
                source_values[24..32].to_vec()
            ]
            .concat()
        );
        std::fs::remove_dir_all(snapshot_path).unwrap();
    }
}
