const PAGE_FILES: PageArenaStateSnapshotFiles = PageArenaStateSnapshotFiles::new(StateSnapshotFile::PageArena);
const KV_FILES: GQAStateSnapshotFiles = GQAStateSnapshotFiles::new(StateSnapshotFile::MainGQARequestPageTable);
const GDN_FILES: GDNStateSnapshotFiles = GDNStateSnapshotFiles::new(
    StateSnapshotFile::MainGDNRequestStateTable,
    StateSnapshotFile::MainGDNRecurrentState,
    StateSnapshotFile::MainGDNConvState,
);
const STATE_FILES: &[StateSnapshotFile] = &[
    StateSnapshotFile::PageArena,
    StateSnapshotFile::MainGQARequestPageTable,
    StateSnapshotFile::MainGDNRequestStateTable,
    StateSnapshotFile::MainGDNRecurrentState,
    StateSnapshotFile::MainGDNConvState,
    StateSnapshotFile::Qwen4xAuxState,
    StateSnapshotFile::Qwen4xRequestState,
];

impl Qwen4xExecutor {
    fn write_state(&mut self, path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        self.state.finish_commit();
        self.gdn.finish_commit();
        assert!(
            self.transactions.is_empty(),
            "Qwen4x snapshot requires committed transactions"
        );
        let device = self.runtime.device();
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        self.state.pack_snapshot(device, &runtime);
        let mut writer = StateSnapshotWriter::new(path, STATE_FILES, plan, self.runtime.buffer_io())?;
        let slots = match plan {
            ExecutorHibernationPlan::All => {
                self.state.page_table().write_full_state(&mut writer, KV_FILES)?;
                self.gdn.write_full_state(&mut writer, GDN_FILES)?;
                self.pages.write_full_state(&mut writer, PAGE_FILES)?;
                std::iter::once(0..self.config.max_requests as u32).collect()
            },
            ExecutorHibernationPlan::Selected {
                request_slot_ranges,
                page_id_ranges,
            } => {
                self.state
                    .page_table()
                    .write_selected_state(&mut writer, KV_FILES, request_slot_ranges)?;
                self.gdn
                    .write_selected_state(&mut writer, GDN_FILES, request_slot_ranges)?;
                self.pages
                    .write_selected_state(&mut writer, PAGE_FILES, page_id_ranges)?;
                request_slot_ranges.clone()
            },
        };
        writer.write_selected_buffer(
            StateSnapshotFile::Qwen4xAuxState,
            self.state.packed(),
            &slots,
            self.state.state_bytes(),
        )?;
        writer.write_metadata(
            StateSnapshotFile::Qwen4xRequestState,
            &self.state.snapshot_metadata(&slots),
        )?;
        writer.commit()
    }
    fn read_state(&mut self, path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError> {
        let mut reader = StateSnapshotReader::open(path, STATE_FILES, plan, self.runtime.buffer_io())?;
        let device = self.runtime.device();
        self.state.allocate(device, &self.model_config)?;
        self.gdn.allocate_resources(device);
        self.pages.allocate_resources(device);
        let result = (|| {
            let slots = match plan {
                ExecutorHibernationPlan::All => {
                    self.state.page_table_mut().read_full_state(&mut reader, KV_FILES)?;
                    self.gdn.read_full_state(&mut reader, GDN_FILES)?;
                    self.pages.read_full_state(&mut reader, PAGE_FILES)?;
                    std::iter::once(0..self.config.max_requests as u32).collect()
                },
                ExecutorHibernationPlan::Selected {
                    request_slot_ranges,
                    page_id_ranges,
                } => {
                    self.state
                        .page_table_mut()
                        .read_selected_state(&mut reader, KV_FILES, request_slot_ranges)?;
                    self.gdn
                        .read_selected_state(&mut reader, GDN_FILES, request_slot_ranges)?;
                    self.pages
                        .read_selected_state(&mut reader, PAGE_FILES, page_id_ranges)?;
                    request_slot_ranges.clone()
                },
            };
            reader.read_selected_buffer(
                StateSnapshotFile::Qwen4xAuxState,
                self.state.packed(),
                &slots,
                self.state.state_bytes(),
            )?;
            let metadata: Qwen4xRequestState = reader.read_metadata(StateSnapshotFile::Qwen4xRequestState)?;
            self.state.load_metadata(metadata, &self.model_config)?;
            reader.finish()
        })();
        if let Err(error) = result {
            self.pages.release_resources();
            self.gdn.release_resources();
            self.state.release();
            return Err(error);
        }
        self.state
            .restore_snapshot(device, &MetalReplayRuntime::new(self.runtime.stream()));
        self.main.component_mut().attach_gdn(&self.gdn);
        Ok(())
    }
}
