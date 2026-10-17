use std::collections::BTreeSet;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Seek;
use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::BufferIO;
use inference_backend_metal::metal::BufferIOFile;
use inference_backend_metal::metal::BufferIOFileCacheMode;
use inference_executor_core::def::ModelExecutorError;
use inference_runtime_core::compute::ExecutorHibernationPlan;
use inference_runtime_core::runtime::RawPageID;
use inference_runtime_core::runtime::RawRequestSlot;
use wincode::SchemaRead;
use wincode::SchemaReadOwned;
use wincode::SchemaWrite;
use wincode::config::Configuration;
use wincode::config::PREALLOCATION_SIZE_LIMIT_DISABLED;
#[cfg(target_endian = "big")]
use wincode::int_encoding::BigEndian as NativeEndian;
use wincode::int_encoding::FixInt;
#[cfg(target_endian = "little")]
use wincode::int_encoding::LittleEndian as NativeEndian;
use wincode::io::std_read::ReadAdapter;
use wincode::io::std_write::WriteAdapter;
use wincode::len::FixIntLen;

mod state_io;
pub use state_io::FullStateIO;
pub use state_io::GDNStateSnapshotFiles;
pub use state_io::GQAStateSnapshotFiles;
pub use state_io::PageArenaStateSnapshotFiles;
pub use state_io::SelectedStateIO;

const SNAPSHOT_MAGIC: [u8; 8] = *b"PSISTATE";
const SNAPSHOT_VERSION: u32 = 3;
const MANIFEST_FILE_NAME: &str = "manifest";

static NEXT_TEMP_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

type StateSnapshotWincodeConfig =
    Configuration<true, PREALLOCATION_SIZE_LIMIT_DISABLED, FixIntLen<u32>, NativeEndian, FixInt, u8>;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, SchemaRead, SchemaWrite)]
#[wincode(tag_encoding = "u8")]
pub enum StateSnapshotFile {
    PageArena,
    MainGQARequestPageTable,
    MainGDNRequestStateTable,
    MainGDNRecurrentState,
    MainGDNConvState,
    MTPGQARequestPageTable,
    DSparkGQARequestPageTable,
    DFlash2GQARequestPageTable,
    Qwen4xAuxState,
    Qwen4xRequestState,
}

impl StateSnapshotFile {
    pub fn file_name(self) -> &'static str {
        match self {
            Self::PageArena => "page-arena",
            Self::MainGQARequestPageTable => "main-gqa-request-page-table",
            Self::MainGDNRequestStateTable => "main-gdn-request-state-table",
            Self::MainGDNRecurrentState => "main-gdn-recurrent-state",
            Self::MainGDNConvState => "main-gdn-conv-state",
            Self::MTPGQARequestPageTable => "mtp-gqa-request-page-table",
            Self::DSparkGQARequestPageTable => "dspark-gqa-request-page-table",
            Self::DFlash2GQARequestPageTable => "dflash2-gqa-request-page-table",
            Self::Qwen4xAuxState => "qwen4x-aux-state",
            Self::Qwen4xRequestState => "qwen4x-request-state",
        }
    }

    fn kind(self) -> StateSnapshotFileKind {
        match self {
            Self::MainGDNRequestStateTable | Self::Qwen4xRequestState => StateSnapshotFileKind::Metadata,
            Self::PageArena
            | Self::MainGQARequestPageTable
            | Self::MainGDNRecurrentState
            | Self::MainGDNConvState
            | Self::MTPGQARequestPageTable
            | Self::DSparkGQARequestPageTable
            | Self::DFlash2GQARequestPageTable
            | Self::Qwen4xAuxState => StateSnapshotFileKind::Buffer,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, SchemaRead, SchemaWrite)]
#[wincode(tag_encoding = "u8")]
enum StateSnapshotFileKind {
    Buffer,
    Metadata,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, SchemaRead, SchemaWrite)]
struct StateSnapshotManifestEntry {
    file: StateSnapshotFile,
    kind: StateSnapshotFileKind,
    payload_bytes: u64,
}

#[derive(Debug, Eq, PartialEq, SchemaRead, SchemaWrite)]
struct StateSnapshotManifest {
    magic: [u8; 8],
    version: u32,
    plan: StateSnapshotPlan,
    files: Vec<StateSnapshotManifestEntry>,
}

#[derive(Debug, Eq, PartialEq, SchemaRead, SchemaWrite)]
#[wincode(tag_encoding = "u8")]
enum StateSnapshotPlan {
    All,
    Selected {
        request_slot_ranges: Vec<Range<RawRequestSlot>>,
        page_id_ranges: Vec<Range<RawPageID>>,
    },
}

impl StateSnapshotPlan {
    fn from_model_plan(plan: &ExecutorHibernationPlan) -> Self {
        match plan {
            ExecutorHibernationPlan::All => Self::All,
            ExecutorHibernationPlan::Selected {
                request_slot_ranges,
                page_id_ranges,
            } => {
                Self::Selected {
                    request_slot_ranges: request_slot_ranges.clone(),
                    page_id_ranges: page_id_ranges.clone(),
                }
            },
        }
    }

    fn matches(&self, plan: &ExecutorHibernationPlan) -> bool {
        match (self, plan) {
            (Self::All, ExecutorHibernationPlan::All) => true,
            (
                Self::Selected {
                    request_slot_ranges: snapshot_request_slot_ranges,
                    page_id_ranges: snapshot_page_id_ranges,
                },
                ExecutorHibernationPlan::Selected {
                    request_slot_ranges,
                    page_id_ranges,
                },
            ) => snapshot_request_slot_ranges == request_slot_ranges && snapshot_page_id_ranges == page_id_ranges,
            _ => false,
        }
    }
}

enum StateSnapshotWriteFile {
    Buffer(BufferIOFile),
    Metadata(File),
}

impl StateSnapshotWriteFile {
    fn sync_all(&self) -> std::io::Result<()> {
        match self {
            Self::Buffer(file) => file.sync_all(),
            Self::Metadata(file) => file.sync_all(),
        }
    }
}

pub struct StateSnapshotWriter<'a> {
    destination: PathBuf,
    temp_path: PathBuf,
    buffer_io: &'a BufferIO,
    plan: StateSnapshotPlan,
    expected_files: Box<[StateSnapshotFile]>,
    files: Vec<(StateSnapshotManifestEntry, StateSnapshotWriteFile)>,
    published: bool,
}

pub struct StateSnapshotReader<'a> {
    path: PathBuf,
    buffer_io: &'a BufferIO,
    files: Vec<StateSnapshotManifestEntry>,
    consumed: Vec<bool>,
}

impl<'a> StateSnapshotWriter<'a> {
    pub fn new(
        destination: &Path,
        expected_files: &[StateSnapshotFile],
        plan: &ExecutorHibernationPlan,
        buffer_io: &'a BufferIO,
    ) -> Result<Self, ModelExecutorError> {
        plan.assert_valid();
        validate_expected_file_set(expected_files)?;
        let parent = destination.parent().ok_or_else(|| {
            ModelExecutorError::custom(format!("model state snapshot path has no parent: {destination:?}"))
        })?;
        destination.file_name().ok_or_else(|| {
            ModelExecutorError::custom(format!("model state snapshot path has no file name: {destination:?}"))
        })?;
        std::fs::create_dir_all(parent).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to create model state snapshot parent directory {parent:?}: {error}"
            ))
        })?;
        match std::fs::symlink_metadata(destination) {
            Ok(_) => {
                return Err(ModelExecutorError::custom(format!(
                    "model state snapshot destination already exists: {destination:?}"
                )));
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
            Err(error) => {
                return Err(ModelExecutorError::custom(format!(
                    "unable to inspect model state snapshot destination {destination:?}: {error}"
                )));
            },
        }

        let temp_path = temp_path(destination);
        std::fs::create_dir(&temp_path).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to create temporary model state snapshot directory {temp_path:?}: {error}"
            ))
        })?;
        Ok(Self {
            destination: destination.to_path_buf(),
            temp_path,
            buffer_io,
            plan: StateSnapshotPlan::from_model_plan(plan),
            expected_files: expected_files.into(),
            files: Vec::new(),
            published: false,
        })
    }

    pub fn write_full_buffer(
        &mut self,
        snapshot_file: StateSnapshotFile,
        buffer: &Buffer,
    ) -> Result<(), ModelExecutorError> {
        if buffer.len_bytes() == 0 {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot buffer must not be empty: file={snapshot_file:?}"
            )));
        }
        let range = 0..buffer.len_bytes_u64();
        self.write_buffer_ranges(snapshot_file, buffer, std::slice::from_ref(&range))
    }

    pub fn write_selected_buffer(
        &mut self,
        snapshot_file: StateSnapshotFile,
        buffer: &Buffer,
        entry_ranges: &[Range<u32>],
        entry_bytes: usize,
    ) -> Result<(), ModelExecutorError> {
        let ranges = selected_buffer_ranges(buffer, entry_ranges, entry_bytes);
        self.write_buffer_ranges(snapshot_file, buffer, &ranges)
    }

    fn write_buffer_ranges(
        &mut self,
        snapshot_file: StateSnapshotFile,
        buffer: &Buffer,
        ranges: &[Range<u64>],
    ) -> Result<(), ModelExecutorError> {
        if snapshot_file.kind() != StateSnapshotFileKind::Buffer {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file does not contain a buffer: file={snapshot_file:?}"
            )));
        }
        self.assert_file_is_new(snapshot_file)?;
        let path = self.temp_path.join(snapshot_file.file_name());
        let output_file = self
            .buffer_io
            .create(&path, BufferIOFileCacheMode::Uncached)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to create model state buffer file {snapshot_file:?} at {path:?}: {error}"
                ))
            })?;
        let mut payload_bytes = 0_u64;
        for range in ranges {
            let len_bytes = range.end - range.start;
            self.buffer_io
                .buffer_to_file(buffer, range.start, &output_file, payload_bytes, len_bytes)
                .map_err(|error| {
                    ModelExecutorError::custom(format!(
                        "unable to write model state buffer file {snapshot_file:?} to {path:?}: {error}"
                    ))
                })?;
            payload_bytes = payload_bytes
                .checked_add(len_bytes)
                .expect("model state snapshot payload length must fit u64");
        }
        self.files.push((
            StateSnapshotManifestEntry {
                file: snapshot_file,
                kind: StateSnapshotFileKind::Buffer,
                payload_bytes,
            },
            StateSnapshotWriteFile::Buffer(output_file),
        ));
        Ok(())
    }

    pub fn write_metadata<T>(
        &mut self,
        snapshot_file: StateSnapshotFile,
        metadata: &T,
    ) -> Result<(), ModelExecutorError>
    where
        T: SchemaWrite<StateSnapshotWincodeConfig, Src = T> + ?Sized,
    {
        if snapshot_file.kind() != StateSnapshotFileKind::Metadata {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file does not contain metadata: file={snapshot_file:?}"
            )));
        }
        self.assert_file_is_new(snapshot_file)?;
        let path = self.temp_path.join(snapshot_file.file_name());
        let mut output_file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to create model state metadata file {snapshot_file:?} at {path:?}: {error}"
                ))
            })?;
        wincode::config::serialize_into(
            WriteAdapter::new(&mut output_file),
            metadata,
            state_snapshot_wincode_config(),
        )
        .map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to encode model state metadata file {snapshot_file:?} to {path:?}: {error}"
            ))
        })?;
        let payload_bytes = output_file.stream_position().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to inspect model state metadata file {snapshot_file:?} at {path:?}: {error}"
            ))
        })?;
        if payload_bytes == 0 {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot metadata must not be empty: file={snapshot_file:?}"
            )));
        }
        self.files.push((
            StateSnapshotManifestEntry {
                file: snapshot_file,
                kind: StateSnapshotFileKind::Metadata,
                payload_bytes,
            },
            StateSnapshotWriteFile::Metadata(output_file),
        ));
        Ok(())
    }

    pub fn commit(mut self) -> Result<(), ModelExecutorError> {
        self.files.sort_unstable_by_key(|(entry, _)| entry.file);
        let manifest_entries = self.files.iter().map(|(entry, _)| *entry).collect::<Vec<_>>();
        validate_expected_files(&manifest_entries, &self.expected_files)?;
        for (entry, file) in &self.files {
            file.sync_all().map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to sync model state file {:?} in {:?}: {error}",
                    entry.file, self.temp_path
                ))
            })?;
        }

        let manifest = StateSnapshotManifest {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            plan: std::mem::replace(&mut self.plan, StateSnapshotPlan::All),
            files: manifest_entries,
        };
        let manifest_path = self.temp_path.join(MANIFEST_FILE_NAME);
        let mut manifest_file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&manifest_path)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to create model state manifest {manifest_path:?}: {error}"
                ))
            })?;
        wincode::config::serialize_into(
            WriteAdapter::new(&mut manifest_file),
            &manifest,
            state_snapshot_wincode_config(),
        )
        .map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to encode model state manifest {manifest_path:?}: {error}"
            ))
        })?;
        manifest_file.sync_all().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to sync model state manifest {manifest_path:?}: {error}"
            ))
        })?;
        drop(manifest_file);
        self.files.clear();

        File::open(&self.temp_path)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to sync temporary model state snapshot directory {:?}: {error}",
                    self.temp_path
                ))
            })?;
        std::fs::rename(&self.temp_path, &self.destination).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to publish model state snapshot directory {:?} as {:?}: {error}",
                self.temp_path, self.destination
            ))
        })?;
        self.published = true;

        let parent = self
            .destination
            .parent()
            .expect("model state snapshot destination parent must remain available");
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to sync model state snapshot parent directory {parent:?}: {error}"
                ))
            })?;
        Ok(())
    }

    fn assert_file_is_new(&self, snapshot_file: StateSnapshotFile) -> Result<(), ModelExecutorError> {
        if self.files.iter().any(|(entry, _)| entry.file == snapshot_file) {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file was written twice: file={snapshot_file:?}"
            )));
        }
        Ok(())
    }
}

impl Drop for StateSnapshotWriter<'_> {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_dir_all(&self.temp_path);
        }
    }
}

impl<'a> StateSnapshotReader<'a> {
    pub fn open(
        path: &Path,
        expected_files: &[StateSnapshotFile],
        plan: &ExecutorHibernationPlan,
        buffer_io: &'a BufferIO,
    ) -> Result<Self, ModelExecutorError> {
        plan.assert_valid();
        validate_expected_file_set(expected_files)?;
        let directory_metadata = std::fs::symlink_metadata(path).map_err(|error| {
            ModelExecutorError::custom(format!("unable to inspect model state snapshot {path:?}: {error}"))
        })?;
        if !directory_metadata.file_type().is_dir() {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot must be a directory: {path:?}"
            )));
        }

        let manifest_path = path.join(MANIFEST_FILE_NAME);
        let mut manifest_file = File::open(&manifest_path).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to open model state manifest {manifest_path:?}: {error}"
            ))
        })?;
        let manifest: StateSnapshotManifest =
            wincode::config::deserialize_from(ReadAdapter::new(&mut manifest_file), state_snapshot_wincode_config())
                .map_err(|error| {
                    ModelExecutorError::custom(format!(
                        "unable to decode model state manifest {manifest_path:?}: {error}"
                    ))
                })?;
        let manifest_bytes = manifest_file.stream_position().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to inspect model state manifest {manifest_path:?}: {error}"
            ))
        })?;
        let actual_manifest_bytes = manifest_file
            .metadata()
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to inspect model state manifest {manifest_path:?}: {error}"
                ))
            })?
            .len();
        if manifest_bytes != actual_manifest_bytes {
            return Err(ModelExecutorError::custom(format!(
                "model state manifest contains trailing bytes: path={manifest_path:?} \
                 file_bytes={actual_manifest_bytes} consumed={manifest_bytes}"
            )));
        }
        validate_manifest(&manifest)?;
        if !manifest.plan.matches(plan) {
            return Err(ModelExecutorError::custom(
                "model state snapshot hibernation plan does not match the requested plan",
            ));
        }
        validate_expected_files(&manifest.files, expected_files)?;
        validate_snapshot_directory(path, &manifest.files)?;

        let consumed = vec![false; manifest.files.len()];
        Ok(Self {
            path: path.to_path_buf(),
            buffer_io,
            files: manifest.files,
            consumed,
        })
    }

    pub fn read_full_buffer(
        &mut self,
        snapshot_file: StateSnapshotFile,
        buffer: &Buffer,
    ) -> Result<(), ModelExecutorError> {
        let range = 0..buffer.len_bytes_u64();
        self.read_buffer_ranges(snapshot_file, buffer, std::slice::from_ref(&range))
    }

    pub fn read_selected_buffer(
        &mut self,
        snapshot_file: StateSnapshotFile,
        buffer: &Buffer,
        entry_ranges: &[Range<u32>],
        entry_bytes: usize,
    ) -> Result<(), ModelExecutorError> {
        let ranges = selected_buffer_ranges(buffer, entry_ranges, entry_bytes);
        self.read_buffer_ranges(snapshot_file, buffer, &ranges)
    }

    fn read_buffer_ranges(
        &mut self,
        snapshot_file: StateSnapshotFile,
        buffer: &Buffer,
        ranges: &[Range<u64>],
    ) -> Result<(), ModelExecutorError> {
        let file_index = self.file_index(snapshot_file, StateSnapshotFileKind::Buffer)?;
        let entry = self.files[file_index];
        let expected_payload_bytes = ranges
            .iter()
            .try_fold(0_u64, |total, range| total.checked_add(range.end - range.start));
        let expected_payload_bytes = expected_payload_bytes.expect("model state snapshot payload length must fit u64");
        if entry.payload_bytes != expected_payload_bytes {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file length mismatch: file={snapshot_file:?} expected={} actual={}",
                expected_payload_bytes, entry.payload_bytes
            )));
        }
        if ranges.is_empty() {
            self.consumed[file_index] = true;
            return Ok(());
        }
        let path = self.path.join(snapshot_file.file_name());
        let input_file = self
            .buffer_io
            .open(&path, BufferIOFileCacheMode::Uncached)
            .map_err(|error| {
                ModelExecutorError::custom(format!(
                    "unable to open model state buffer file {snapshot_file:?} at {path:?}: {error}"
                ))
            })?;
        let mut file_offset_bytes = 0_u64;
        for range in ranges {
            let len_bytes = range.end - range.start;
            self.buffer_io
                .file_to_buffer(&input_file, file_offset_bytes, buffer, range.start, len_bytes)
                .map_err(|error| {
                    ModelExecutorError::custom(format!(
                        "unable to read model state buffer file {snapshot_file:?} from {path:?}: {error}"
                    ))
                })?;
            file_offset_bytes = file_offset_bytes
                .checked_add(len_bytes)
                .expect("model state snapshot file offset must fit u64");
        }
        self.consumed[file_index] = true;
        Ok(())
    }

    pub fn read_metadata<T>(&mut self, snapshot_file: StateSnapshotFile) -> Result<T, ModelExecutorError>
    where
        T: SchemaReadOwned<StateSnapshotWincodeConfig, Dst = T>,
    {
        let file_index = self.file_index(snapshot_file, StateSnapshotFileKind::Metadata)?;
        let entry = self.files[file_index];
        let path = self.path.join(snapshot_file.file_name());
        let mut input_file = File::open(&path).map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to open model state metadata file {snapshot_file:?} at {path:?}: {error}"
            ))
        })?;
        let metadata =
            wincode::config::deserialize_from(ReadAdapter::new(&mut input_file), state_snapshot_wincode_config())
                .map_err(|error| {
                    ModelExecutorError::custom(format!(
                        "unable to decode model state metadata file {snapshot_file:?} from {path:?}: {error}"
                    ))
                })?;
        let consumed_bytes = input_file.stream_position().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to inspect model state metadata file {snapshot_file:?} at {path:?}: {error}"
            ))
        })?;
        if consumed_bytes != entry.payload_bytes {
            return Err(ModelExecutorError::custom(format!(
                "model state metadata file contains trailing bytes: file={snapshot_file:?} expected={} \
                 consumed={consumed_bytes}",
                entry.payload_bytes
            )));
        }
        self.consumed[file_index] = true;
        Ok(metadata)
    }

    pub fn finish(self) -> Result<(), ModelExecutorError> {
        if let Some((entry, _)) = self.files.iter().zip(self.consumed).find(|(_, consumed)| !consumed) {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file was not read: file={:?}",
                entry.file
            )));
        }
        Ok(())
    }

    fn file_index(
        &self,
        snapshot_file: StateSnapshotFile,
        expected_kind: StateSnapshotFileKind,
    ) -> Result<usize, ModelExecutorError> {
        let file_index = self
            .files
            .binary_search_by_key(&snapshot_file, |entry| entry.file)
            .map_err(|_| {
                ModelExecutorError::custom(format!("model state snapshot file is missing: file={snapshot_file:?}"))
            })?;
        if self.consumed[file_index] {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file was read twice: file={snapshot_file:?}"
            )));
        }
        let actual_kind = self.files[file_index].kind;
        if actual_kind != expected_kind {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file kind mismatch: file={snapshot_file:?} expected={expected_kind:?} \
                 actual={actual_kind:?}"
            )));
        }
        Ok(file_index)
    }
}

const fn state_snapshot_wincode_config() -> StateSnapshotWincodeConfig {
    Configuration::default()
        .disable_preallocation_size_limit()
        .with_length_encoding::<FixIntLen<u32>>()
        .with_platform_endian()
        .with_fixint_encoding()
        .with_tag_encoding::<u8>()
}

#[cfg(test)]
fn encode_manifest(manifest: &StateSnapshotManifest) -> Result<Vec<u8>, ModelExecutorError> {
    wincode::config::serialize(manifest, state_snapshot_wincode_config())
        .map_err(|error| ModelExecutorError::custom(format!("unable to encode model state manifest: {error}")))
}

fn validate_manifest(manifest: &StateSnapshotManifest) -> Result<(), ModelExecutorError> {
    if manifest.magic != SNAPSHOT_MAGIC {
        return Err(ModelExecutorError::custom("model state snapshot magic mismatch"));
    }
    if manifest.version != SNAPSHOT_VERSION {
        return Err(ModelExecutorError::custom(format!(
            "model state snapshot version mismatch: expected={SNAPSHOT_VERSION} actual={}",
            manifest.version
        )));
    }
    validate_snapshot_plan(&manifest.plan)?;
    if manifest.files.is_empty() {
        return Err(ModelExecutorError::custom(
            "model state snapshot must contain at least one file",
        ));
    }
    if manifest
        .files
        .windows(2)
        .any(|entries| entries[0].file >= entries[1].file)
    {
        return Err(ModelExecutorError::custom(
            "model state snapshot files must be unique and sorted",
        ));
    }
    if manifest
        .files
        .iter()
        .any(|entry| entry.kind == StateSnapshotFileKind::Metadata && entry.payload_bytes == 0)
    {
        return Err(ModelExecutorError::custom(
            "model state snapshot metadata files must not be empty",
        ));
    }
    if manifest.files.iter().any(|entry| entry.kind != entry.file.kind()) {
        return Err(ModelExecutorError::custom(
            "model state snapshot file kind does not match its semantic file",
        ));
    }
    Ok(())
}

fn validate_snapshot_plan(plan: &StateSnapshotPlan) -> Result<(), ModelExecutorError> {
    let StateSnapshotPlan::Selected {
        request_slot_ranges,
        page_id_ranges,
    } = plan
    else {
        return Ok(());
    };
    if !are_canonical_ranges(request_slot_ranges) {
        return Err(ModelExecutorError::custom(
            "model state snapshot request slot ranges must be nonempty, sorted, disjoint, and nonadjacent",
        ));
    }
    if !are_canonical_ranges(page_id_ranges) {
        return Err(ModelExecutorError::custom(
            "model state snapshot page ID ranges must be nonempty, sorted, disjoint, and nonadjacent",
        ));
    }
    Ok(())
}

fn selected_buffer_ranges(buffer: &Buffer, entry_ranges: &[Range<u32>], entry_bytes: usize) -> Vec<Range<u64>> {
    assert!(
        entry_bytes > 0,
        "model state snapshot buffer entry size must be positive"
    );
    let entry_bytes = entry_bytes as u64;
    assert!(
        are_canonical_ranges(entry_ranges),
        "model state snapshot buffer entry ranges must be nonempty, sorted, disjoint, and nonadjacent"
    );
    entry_ranges
        .iter()
        .map(|entry_range| {
            let start = (entry_range.start as u64)
                .checked_mul(entry_bytes)
                .expect("model state snapshot buffer entry offset must fit u64");
            let end = (entry_range.end as u64)
                .checked_mul(entry_bytes)
                .expect("model state snapshot buffer entry end must fit u64");
            assert!(
                end <= buffer.len_bytes_u64(),
                "model state snapshot buffer entry range is out of bounds: entry_range={entry_range:?} entry_bytes={} \
                 buffer_bytes={}",
                entry_bytes,
                buffer.len_bytes_u64()
            );
            start..end
        })
        .collect()
}

fn are_canonical_ranges(ranges: &[Range<u32>]) -> bool {
    ranges.iter().all(|range| range.start < range.end) && ranges.windows(2).all(|pair| pair[0].end < pair[1].start)
}

fn validate_snapshot_directory(path: &Path, files: &[StateSnapshotManifestEntry]) -> Result<(), ModelExecutorError> {
    let mut expected_names = files
        .iter()
        .map(|entry| entry.file.file_name())
        .collect::<BTreeSet<_>>();
    expected_names.insert(MANIFEST_FILE_NAME);

    for entry in std::fs::read_dir(path).map_err(|error| {
        ModelExecutorError::custom(format!("unable to enumerate model state snapshot {path:?}: {error}"))
    })? {
        let entry = entry.map_err(|error| {
            ModelExecutorError::custom(format!("unable to enumerate model state snapshot {path:?}: {error}"))
        })?;
        let file_name = entry.file_name();
        let file_name = file_name.to_str().ok_or_else(|| {
            ModelExecutorError::custom(format!(
                "model state snapshot contains a non-UTF-8 file name: {file_name:?}"
            ))
        })?;
        if !expected_names.remove(file_name) {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot contains an unexpected file: {file_name:?}"
            )));
        }
        let file_type = entry.file_type().map_err(|error| {
            ModelExecutorError::custom(format!(
                "unable to inspect model state snapshot entry {:?}: {error}",
                entry.path()
            ))
        })?;
        if !file_type.is_file() {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot entry must be a regular file: {:?}",
                entry.path()
            )));
        }
    }
    if let Some(file_name) = expected_names.first() {
        return Err(ModelExecutorError::custom(format!(
            "model state snapshot file is missing: {file_name:?}"
        )));
    }

    for entry in files {
        let file_path = path.join(entry.file.file_name());
        let actual_bytes = std::fs::metadata(&file_path)
            .map_err(|error| {
                ModelExecutorError::custom(format!("unable to inspect model state file {:?}: {error}", entry.file))
            })?
            .len();
        if actual_bytes != entry.payload_bytes {
            return Err(ModelExecutorError::custom(format!(
                "model state snapshot file length mismatch: file={:?} expected={} actual={actual_bytes}",
                entry.file, entry.payload_bytes
            )));
        }
    }
    Ok(())
}

fn validate_expected_files(
    actual_files: &[StateSnapshotManifestEntry],
    expected_files: &[StateSnapshotFile],
) -> Result<(), ModelExecutorError> {
    let expected = validate_expected_file_set(expected_files)?;
    let actual = actual_files.iter().map(|entry| entry.file).collect::<BTreeSet<_>>();
    if let Some(file) = expected.difference(&actual).next() {
        return Err(ModelExecutorError::custom(format!(
            "model state snapshot file is missing: file={file:?}"
        )));
    }
    if let Some(file) = actual.difference(&expected).next() {
        return Err(ModelExecutorError::custom(format!(
            "model state snapshot contains an unexpected file: file={file:?}"
        )));
    }
    Ok(())
}

fn validate_expected_file_set(
    expected_files: &[StateSnapshotFile],
) -> Result<BTreeSet<StateSnapshotFile>, ModelExecutorError> {
    let expected = expected_files.iter().copied().collect::<BTreeSet<_>>();
    if expected.len() != expected_files.len() {
        return Err(ModelExecutorError::custom(
            "model state snapshot expected file set contains duplicates",
        ));
    }
    if expected.is_empty() {
        return Err(ModelExecutorError::custom(
            "model state snapshot expected file set must not be empty",
        ));
    }
    Ok(expected)
}

fn temp_path(destination: &Path) -> PathBuf {
    let file_name = destination
        .file_name()
        .expect("model state snapshot destination must have a file name")
        .to_string_lossy();
    let id = NEXT_TEMP_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
    destination.with_file_name(format!(".{file_name}.tmp-{}-{id}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Write as _;
    use std::mem::size_of;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::BufferIO;
    use inference_backend_metal::metal::Device;
    use inference_runtime_core::compute::ExecutorHibernationPlan;
    use wincode::SchemaRead;
    use wincode::SchemaWrite;

    use super::StateSnapshotFile;
    use super::StateSnapshotReader;
    use super::StateSnapshotWriter;

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

    #[derive(Debug, Eq, PartialEq, SchemaRead, SchemaWrite)]
    struct TestMetadata {
        slots: Vec<u32>,
    }

    const PAGE_ARENA_FILES: &[StateSnapshotFile] = &[StateSnapshotFile::PageArena];
    const PAGE_ARENA_AND_GDN_METADATA_FILES: &[StateSnapshotFile] = &[
        StateSnapshotFile::PageArena,
        StateSnapshotFile::MainGDNRequestStateTable,
    ];
    const GDN_METADATA_FILES: &[StateSnapshotFile] = &[StateSnapshotFile::MainGDNRequestStateTable];

    #[test]
    fn test_directory_snapshot_unload_load() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source_values = (0..4096).map(|index| (index % 251) as u8).collect::<Vec<_>>();
        let source = Buffer::from_slice(&device, &source_values);
        let restored = Buffer::new_zeroed(&device, source_values.len());
        let metadata = TestMetadata {
            slots: vec![3, 5, 8, 13],
        };
        let path = test_path("unload-load");

        let mut writer = StateSnapshotWriter::new(
            &path,
            PAGE_ARENA_AND_GDN_METADATA_FILES,
            &ExecutorHibernationPlan::All,
            &buffer_io,
        )
        .unwrap();
        writer.write_full_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer
            .write_metadata(StateSnapshotFile::MainGDNRequestStateTable, &metadata)
            .unwrap();
        writer.commit().unwrap();

        assert!(path.join("manifest").is_file());
        assert!(path.join("page-arena").is_file());
        assert!(path.join("main-gdn-request-state-table").is_file());

        let mut reader = StateSnapshotReader::open(
            &path,
            PAGE_ARENA_AND_GDN_METADATA_FILES,
            &ExecutorHibernationPlan::All,
            &buffer_io,
        )
        .unwrap();
        reader
            .read_full_buffer(StateSnapshotFile::PageArena, &restored)
            .unwrap();
        let restored_metadata: TestMetadata = reader
            .read_metadata(StateSnapshotFile::MainGDNRequestStateTable)
            .unwrap();
        reader.finish().unwrap();

        assert_eq!(restored.read_typed::<u8>(0, source_values.len()), source_values);
        assert_eq!(restored_metadata, metadata);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_selected_buffer_entries_unload_load() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source_values = (10_u32..18).collect::<Vec<_>>();
        let source = Buffer::from_slice(&device, &source_values);
        let restored = Buffer::from_slice(&device, &[u32::MAX; 8]);
        let selected_entry_ranges = [1..3, 5..6];
        let request_slot_ranges = std::iter::once(3..4).collect();
        let plan = ExecutorHibernationPlan::selected(request_slot_ranges, selected_entry_ranges.to_vec());
        let path = test_path("selected-buffer-entries");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &plan, &buffer_io).unwrap();
        writer
            .write_selected_buffer(
                StateSnapshotFile::PageArena,
                &source,
                &selected_entry_ranges,
                size_of::<u32>(),
            )
            .unwrap();
        writer.commit().unwrap();

        assert_eq!(std::fs::metadata(path.join("page-arena")).unwrap().len(), 12);
        let mut reader = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &plan, &buffer_io).unwrap();
        reader
            .read_selected_buffer(
                StateSnapshotFile::PageArena,
                &restored,
                &selected_entry_ranges,
                size_of::<u32>(),
            )
            .unwrap();
        reader.finish().unwrap();

        assert_eq!(
            restored.read_typed::<u32>(0, 8),
            vec![u32::MAX, 11, 12, u32::MAX, u32::MAX, 15, u32::MAX, u32::MAX]
        );
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    #[should_panic(
        expected = "model state snapshot buffer entry ranges must be nonempty, sorted, disjoint, and nonadjacent"
    )]
    fn test_selected_buffer_entries_reject_adjacent_ranges() {
        let device = Device::system_default();
        let source = Buffer::new_zeroed_elements(&device, 8, inference_backend_metal::metal::Dtype::Uint32);
        let _ = super::selected_buffer_ranges(&source, &[1..2, 2..3], size_of::<u32>());
    }

    #[test]
    #[should_panic(expected = "model state snapshot buffer entry range is out of bounds")]
    fn test_selected_buffer_entries_reject_out_of_bounds_range() {
        let device = Device::system_default();
        let source = Buffer::new_zeroed_elements(&device, 8, inference_backend_metal::metal::Dtype::Uint32);
        let entry_range = 8..9;
        let _ = super::selected_buffer_ranges(&source, std::slice::from_ref(&entry_range), size_of::<u32>());
    }

    #[test]
    fn test_empty_selected_buffer_unload_load() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let plan = ExecutorHibernationPlan::selected(Vec::new(), Vec::new());
        let path = test_path("empty-selected-buffer");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &plan, &buffer_io).unwrap();
        writer
            .write_selected_buffer(StateSnapshotFile::PageArena, &source, &[], 4)
            .unwrap();
        writer.commit().unwrap();

        assert_eq!(std::fs::metadata(path.join("page-arena")).unwrap().len(), 0);
        let mut reader = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &plan, &buffer_io).unwrap();
        reader
            .read_selected_buffer(StateSnapshotFile::PageArena, &source, &[], 4)
            .unwrap();
        reader.finish().unwrap();

        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_reader_rejects_mismatched_plan() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let written_plan =
            ExecutorHibernationPlan::selected(std::iter::once(1..2).collect(), std::iter::once(2..3).collect());
        let requested_plan =
            ExecutorHibernationPlan::selected(std::iter::once(1..2).collect(), std::iter::once(3..4).collect());
        let path = test_path("hibernation-plan-mismatch");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &written_plan, &buffer_io).unwrap();
        let entry_range = 2..3;
        writer
            .write_selected_buffer(
                StateSnapshotFile::PageArena,
                &source,
                std::slice::from_ref(&entry_range),
                4,
            )
            .unwrap();
        writer.commit().unwrap();

        let error = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &requested_plan, &buffer_io)
            .err()
            .expect("mismatched plan must fail snapshot validation");
        assert!(error.to_string().contains("plan does not match"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_reader_rejects_noncanonical_plan_ranges() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let plan = ExecutorHibernationPlan::selected(Vec::new(), std::iter::once(2..3).collect());
        let path = test_path("noncanonical-plan-ranges");

        let mut writer = StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &plan, &buffer_io).unwrap();
        let entry_range = 2..3;
        writer
            .write_selected_buffer(
                StateSnapshotFile::PageArena,
                &source,
                std::slice::from_ref(&entry_range),
                4,
            )
            .unwrap();
        writer.commit().unwrap();

        let mut manifest = read_manifest(&path);
        manifest.plan = super::StateSnapshotPlan::Selected {
            request_slot_ranges: Vec::new(),
            page_id_ranges: vec![2..4, 4..5],
        };
        std::fs::write(path.join("manifest"), super::encode_manifest(&manifest).unwrap()).unwrap();

        let error = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &plan, &buffer_io)
            .err()
            .expect("noncanonical manifest plan ranges must fail snapshot validation");
        assert!(
            error
                .to_string()
                .contains("page ID ranges must be nonempty, sorted, disjoint, and nonadjacent")
        );
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_reader_rejects_unexpected_file() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("unexpected-file");

        let mut writer =
            StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();
        writer.write_full_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer.commit().unwrap();
        std::fs::write(path.join("unexpected"), b"unexpected").unwrap();

        let error = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io)
            .err()
            .expect("unexpected file must fail snapshot validation");
        assert!(error.to_string().contains("unexpected file"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_reader_rejects_truncated_file_before_restore() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("truncated-file");

        let mut writer =
            StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();
        writer.write_full_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer.commit().unwrap();
        OpenOptions::new()
            .write(true)
            .open(path.join("page-arena"))
            .unwrap()
            .set_len(8)
            .unwrap();

        let error = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io)
            .err()
            .expect("truncated file must fail snapshot validation");
        assert!(error.to_string().contains("length mismatch"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_writer_rejects_semantic_file_kind_mismatch() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("writer-kind-mismatch");
        let mut writer =
            StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();

        let buffer_error = writer
            .write_full_buffer(StateSnapshotFile::MainGDNRequestStateTable, &source)
            .unwrap_err();
        assert!(buffer_error.to_string().contains("does not contain a buffer"));
        let metadata_error = writer
            .write_metadata(StateSnapshotFile::PageArena, &TestMetadata { slots: vec![1] })
            .unwrap_err();
        assert!(metadata_error.to_string().contains("does not contain metadata"));

        drop(writer);
        assert!(!path.exists());
    }

    #[test]
    fn test_reader_rejects_manifest_kind_mismatch() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("manifest-kind-mismatch");

        let mut writer =
            StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();
        writer.write_full_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer.commit().unwrap();

        let mut manifest = read_manifest(&path);
        manifest.files[0].kind = super::StateSnapshotFileKind::Metadata;
        std::fs::write(path.join("manifest"), super::encode_manifest(&manifest).unwrap()).unwrap();

        let error = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io)
            .err()
            .expect("semantic file kind mismatch must fail snapshot validation");
        assert!(error.to_string().contains("kind does not match"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_reader_rejects_manifest_trailing_bytes() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("manifest-trailing-bytes");

        let mut writer =
            StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();
        writer.write_full_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer.commit().unwrap();
        OpenOptions::new()
            .append(true)
            .open(path.join("manifest"))
            .unwrap()
            .write_all(&[0xaa])
            .unwrap();

        let error = StateSnapshotReader::open(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io)
            .err()
            .expect("manifest trailing bytes must fail snapshot validation");
        assert!(error.to_string().contains("manifest contains trailing bytes"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_reader_rejects_duplicate_expected_file() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("duplicate-expected-file");

        let mut writer =
            StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();
        writer.write_full_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer.commit().unwrap();

        let error = StateSnapshotReader::open(
            &path,
            &[StateSnapshotFile::PageArena, StateSnapshotFile::PageArena],
            &ExecutorHibernationPlan::All,
            &buffer_io,
        )
        .err()
        .expect("duplicate expected file must fail snapshot validation");
        assert!(error.to_string().contains("contains duplicates"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_writer_rejects_incomplete_expected_file_set() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("incomplete-expected-file-set");

        let mut writer = StateSnapshotWriter::new(
            &path,
            PAGE_ARENA_AND_GDN_METADATA_FILES,
            &ExecutorHibernationPlan::All,
            &buffer_io,
        )
        .unwrap();
        writer.write_full_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        let error = writer.commit().unwrap_err();

        assert!(error.to_string().contains("file is missing"));
        assert!(!path.exists());
    }

    #[test]
    fn test_writer_rejects_unexpected_file() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("writer-unexpected-file");

        let mut writer =
            StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();
        writer.write_full_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        writer
            .write_full_buffer(StateSnapshotFile::MainGQARequestPageTable, &source)
            .unwrap();
        let error = writer.commit().unwrap_err();

        assert!(error.to_string().contains("unexpected file"));
        assert!(!path.exists());
    }

    #[test]
    fn test_writer_rejects_duplicate_file() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source = Buffer::new_zeroed(&device, 16);
        let path = test_path("writer-duplicate-file");

        let mut writer =
            StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();
        writer.write_full_buffer(StateSnapshotFile::PageArena, &source).unwrap();
        let error = writer
            .write_full_buffer(StateSnapshotFile::PageArena, &source)
            .unwrap_err();

        assert!(error.to_string().contains("written twice"));
        drop(writer);
        assert!(!path.exists());
    }

    #[test]
    fn test_metadata_reader_rejects_trailing_bytes() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let path = test_path("metadata-trailing-bytes");

        let mut writer =
            StateSnapshotWriter::new(&path, GDN_METADATA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();
        writer
            .write_metadata(
                StateSnapshotFile::MainGDNRequestStateTable,
                &TestMetadata { slots: vec![1, 2, 3] },
            )
            .unwrap();
        writer.commit().unwrap();

        let metadata_path = path.join("main-gdn-request-state-table");
        OpenOptions::new()
            .append(true)
            .open(&metadata_path)
            .unwrap()
            .write_all(&[0xaa])
            .unwrap();
        let mut manifest = read_manifest(&path);
        manifest.files[0].payload_bytes += 1;
        std::fs::write(path.join("manifest"), super::encode_manifest(&manifest).unwrap()).unwrap();

        let mut reader =
            StateSnapshotReader::open(&path, GDN_METADATA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();
        let error = reader
            .read_metadata::<TestMetadata>(StateSnapshotFile::MainGDNRequestStateTable)
            .unwrap_err();
        assert!(error.to_string().contains("trailing bytes"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_uncommitted_snapshot_drop_removes_temporary_directory() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let path = test_path("uncommitted-drop");
        let writer =
            StateSnapshotWriter::new(&path, PAGE_ARENA_FILES, &ExecutorHibernationPlan::All, &buffer_io).unwrap();
        let temp_path = writer.temp_path.clone();

        assert!(temp_path.is_dir());
        assert!(!path.exists());
        drop(writer);
        assert!(!temp_path.exists());
        assert!(!path.exists());
    }

    fn read_manifest(path: &std::path::Path) -> super::StateSnapshotManifest {
        let bytes = std::fs::read(path.join("manifest")).unwrap();
        wincode::config::deserialize_exact(&bytes, super::state_snapshot_wincode_config()).unwrap()
    }

    fn test_path(name: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "psi-dec-state-snapshot-test-{}-{name}-{id}",
            std::process::id()
        ))
    }
}
