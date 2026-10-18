use std::ops::Range;

use inference_executor_core::def::ModelExecutorError;

use crate::model::state_snapshot::StateSnapshotFile;
use crate::model::state_snapshot::StateSnapshotReader;
use crate::model::state_snapshot::StateSnapshotWriter;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageArenaStateSnapshotFiles {
    pages: StateSnapshotFile,
}

impl PageArenaStateSnapshotFiles {
    pub const fn new(pages: StateSnapshotFile) -> Self {
        assert!(matches!(pages, StateSnapshotFile::PageArena));
        Self { pages }
    }

    pub const fn pages(self) -> StateSnapshotFile {
        self.pages
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQAStateSnapshotFiles {
    request_page_table: StateSnapshotFile,
}

impl GQAStateSnapshotFiles {
    pub const fn new(request_page_table: StateSnapshotFile) -> Self {
        assert!(matches!(
            request_page_table,
            StateSnapshotFile::MainGQARequestPageTable
                | StateSnapshotFile::MTPGQARequestPageTable
                | StateSnapshotFile::DFlash2GQARequestPageTable
        ));
        Self { request_page_table }
    }

    pub const fn request_page_table(self) -> StateSnapshotFile {
        self.request_page_table
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GDNStateSnapshotFiles {
    request_state_table: StateSnapshotFile,
    recurrent_state: StateSnapshotFile,
    conv_state: StateSnapshotFile,
}

impl GDNStateSnapshotFiles {
    pub const fn new(
        request_state_table: StateSnapshotFile,
        recurrent_state: StateSnapshotFile,
        conv_state: StateSnapshotFile,
    ) -> Self {
        assert!(matches!(
            request_state_table,
            StateSnapshotFile::MainGDNRequestStateTable
        ));
        assert!(matches!(recurrent_state, StateSnapshotFile::MainGDNRecurrentState));
        assert!(matches!(conv_state, StateSnapshotFile::MainGDNConvState));
        Self {
            request_state_table,
            recurrent_state,
            conv_state,
        }
    }

    pub const fn request_state_table(self) -> StateSnapshotFile {
        self.request_state_table
    }

    pub const fn recurrent_state(self) -> StateSnapshotFile {
        self.recurrent_state
    }

    pub const fn conv_state(self) -> StateSnapshotFile {
        self.conv_state
    }
}

pub trait FullStateIO {
    type Files: Copy;

    fn write_full_state(&self, writer: &mut StateSnapshotWriter, files: Self::Files) -> Result<(), ModelExecutorError>;

    fn read_full_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
    ) -> Result<(), ModelExecutorError>;
}

pub trait SelectedStateIO: FullStateIO {
    type ID;

    fn write_selected_state(
        &self,
        writer: &mut StateSnapshotWriter,
        files: Self::Files,
        id_ranges: &[Range<Self::ID>],
    ) -> Result<(), ModelExecutorError>;

    fn read_selected_state(
        &mut self,
        reader: &mut StateSnapshotReader,
        files: Self::Files,
        id_ranges: &[Range<Self::ID>],
    ) -> Result<(), ModelExecutorError>;
}
