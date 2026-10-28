use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;

mod file_io;

pub struct PageArena {
    pages: Option<Buffer>,
    num_pages: usize,
    page_bytes: usize,
}

impl PageArena {
    pub fn new(device: &Device, num_pages: usize, page_bytes: usize) -> Self {
        assert!(num_pages > 0, "page arena requires pages");
        assert!(page_bytes > 0, "page arena requires nonzero page size");
        let len_bytes = num_pages
            .checked_mul(page_bytes)
            .expect("page arena byte length must fit usize");
        Self {
            pages: Some(Buffer::new_zeroed(device, len_bytes)),
            num_pages,
            page_bytes,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        self.pages
            .as_ref()
            .expect("page arena state must be loaded before execution")
    }

    pub fn release_resources(&mut self) {
        assert!(self.pages.is_some(), "page arena state is not loaded");
        self.pages.take();
    }

    pub fn allocate_resources(&mut self, device: &Device) {
        assert!(self.pages.is_none(), "page arena state is already loaded");
        self.pages = Some(Buffer::new_zeroed(
            device,
            self.num_pages
                .checked_mul(self.page_bytes)
                .expect("page arena byte length must fit usize"),
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use inference_backend_metal::metal::Device;
    use inference_runtime_core::compute::ExecutorHibernationPlan;

    use super::PageArena;
    use crate::model::state_snapshot::FullStateIO;
    use crate::model::state_snapshot::PageArenaStateSnapshotFiles;
    use crate::model::state_snapshot::SelectedStateIO;
    use crate::model::state_snapshot::StateSnapshotFile;
    use crate::model::state_snapshot::StateSnapshotReader;
    use crate::model::state_snapshot::StateSnapshotWriter;

    const NUM_PAGES: usize = 3;
    const PAGE_BYTES: usize = 16;
    const SNAPSHOT_FILES: PageArenaStateSnapshotFiles = PageArenaStateSnapshotFiles::new(StateSnapshotFile::PageArena);

    #[test]
    fn test_full_state_unload_load_fixed() {
        let expected = (0..NUM_PAGES * PAGE_BYTES)
            .map(|index| u8::try_from(index * 5 + 3).unwrap())
            .collect();
        assert_unload_load("fixed", expected);
    }

    #[test]
    fn test_full_state_unload_load_random() {
        let mut random = TestRandom::new(0x5041_4745_4152_454e);
        let expected = (0..NUM_PAGES * PAGE_BYTES).map(|_| random.next_u32() as u8).collect();
        assert_unload_load("random", expected);
    }

    #[test]
    fn test_selected_state_unload_load() {
        let device = Device::system_default();
        let mut arena = PageArena::new(&device, NUM_PAGES, PAGE_BYTES);
        let source = (0..NUM_PAGES * PAGE_BYTES)
            .map(|index| u8::try_from(index + 1).unwrap())
            .collect::<Vec<_>>();
        arena.buffer().write_bytes(0, &source);
        let selected_page_id_ranges = [0..1, 2..3];
        let plan = ExecutorHibernationPlan::selected(Vec::new(), selected_page_id_ranges.to_vec());
        let snapshot_path = snapshot_path("selected");
        let buffer_io = inference_backend_metal::metal::BufferIO::new(&device);
        let snapshot_files = [SNAPSHOT_FILES.pages()];

        let mut writer = StateSnapshotWriter::new(&snapshot_path, &snapshot_files, &plan, &buffer_io).unwrap();
        arena
            .write_selected_state(&mut writer, SNAPSHOT_FILES, &selected_page_id_ranges)
            .unwrap();
        writer.commit().unwrap();

        arena.release_resources();
        arena.allocate_resources(&device);
        let mut reader = StateSnapshotReader::open(&snapshot_path, &snapshot_files, &plan, &buffer_io).unwrap();
        arena
            .read_selected_state(&mut reader, SNAPSHOT_FILES, &selected_page_id_ranges)
            .unwrap();
        reader.finish().unwrap();

        let mut restored = vec![0; source.len()];
        arena.buffer().read_bytes(0, &mut restored);
        assert_eq!(&restored[..PAGE_BYTES], &source[..PAGE_BYTES]);
        assert_eq!(&restored[PAGE_BYTES..2 * PAGE_BYTES], &[0; PAGE_BYTES]);
        assert_eq!(&restored[2 * PAGE_BYTES..], &source[2 * PAGE_BYTES..]);
        std::fs::remove_dir_all(snapshot_path).unwrap();
    }

    fn assert_unload_load(name: &str, expected: Vec<u8>) {
        let device = Device::system_default();
        let mut arena = PageArena::new(&device, NUM_PAGES, PAGE_BYTES);
        arena.buffer().write_bytes(0, &expected);

        let snapshot_path = snapshot_path(name);
        let buffer_io = inference_backend_metal::metal::BufferIO::new(&device);
        let snapshot_files = [SNAPSHOT_FILES.pages()];
        let mut writer = StateSnapshotWriter::new(
            &snapshot_path,
            &snapshot_files,
            &ExecutorHibernationPlan::All,
            &buffer_io,
        )
        .unwrap();
        arena.write_full_state(&mut writer, SNAPSHOT_FILES).unwrap();
        writer.commit().unwrap();

        arena.release_resources();
        arena.allocate_resources(&device);

        let mut reader = StateSnapshotReader::open(
            &snapshot_path,
            &snapshot_files,
            &ExecutorHibernationPlan::All,
            &buffer_io,
        )
        .unwrap();
        arena.read_full_state(&mut reader, SNAPSHOT_FILES).unwrap();
        reader.finish().unwrap();

        let mut actual = vec![0; expected.len()];
        arena.buffer().read_bytes(0, &mut actual);
        assert_eq!(actual, expected);
        std::fs::remove_dir_all(snapshot_path).unwrap();
    }

    fn snapshot_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "khalamari-page-arena-unload-load-{}-{name}.state",
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
