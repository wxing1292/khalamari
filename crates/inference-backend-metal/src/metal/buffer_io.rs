use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::slice;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSURL;
use objc2_metal::MTLDevice;
use objc2_metal::MTLIOCommandBuffer;
use objc2_metal::MTLIOCommandQueue;
use objc2_metal::MTLIOCommandQueueDescriptor;
use objc2_metal::MTLIOCommandQueueType;
use objc2_metal::MTLIOFileHandle;
use objc2_metal::MTLIOStatus;

use crate::metal::Buffer;
use crate::metal::Device;

// Metal I/O rejects one `loadBuffer` command when its size reaches 2^31 bytes. Use 1 GiB commands so large public
// transfers stay below that internal boundary.
const FILE_TO_BUFFER_COMMAND_BYTES: u64 = 1 << 30;

/// Transfers byte ranges between files and shared Metal buffers.
///
/// One `BufferIO` owns one serial Metal I/O queue. `file_to_buffer` uses that
/// queue. `buffer_to_file` writes directly from the shared buffer pointer.
/// Callers must synchronize prior GPU access before they call either method.
#[derive(Debug)]
pub struct BufferIO {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLIOCommandQueue>>,
}

/// Owns the POSIX and Metal handles for one file used by `BufferIO`.
#[derive(Debug)]
pub struct BufferIOFile {
    file: File,
    metal_handle: Retained<ProtocolObject<dyn MTLIOFileHandle>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferIOFileCacheMode {
    /// Uses the default macOS data-cache behavior.
    Cached,
    /// Applies `F_NOCACHE` and `F_GLOBAL_NOCACHE` before Metal opens the file.
    Uncached,
}

impl BufferIO {
    pub fn new(device: &Device) -> Self {
        let descriptor = MTLIOCommandQueueDescriptor::new();
        descriptor.setType(MTLIOCommandQueueType::Serial);
        // Each command buffer contains one synchronous load. Bound each queue's I/O worker allocation.
        descriptor.setMaxCommandsInFlight(1);
        let queue = device
            .as_raw()
            .newIOCommandQueueWithDescriptor_error(&descriptor)
            .unwrap_or_else(|error| panic!("Metal I/O command queue allocation failed: {error}"));
        Self {
            device: device.as_raw_retained(),
            queue,
        }
    }

    /// Creates one new file for buffer-to-file transfers.
    pub fn create(&self, file_path: &Path, cache_mode: BufferIOFileCacheMode) -> io::Result<BufferIOFile> {
        let mut options = OpenOptions::new();
        options.create_new(true).read(true).write(true);
        self.open_with_options(file_path, &options, cache_mode)
    }

    /// Opens one existing file for file-to-buffer transfers.
    pub fn open(&self, file_path: &Path, cache_mode: BufferIOFileCacheMode) -> io::Result<BufferIOFile> {
        let mut options = OpenOptions::new();
        options.read(true);
        self.open_with_options(file_path, &options, cache_mode)
    }

    fn open_with_options(
        &self,
        file_path: &Path,
        options: &OpenOptions,
        cache_mode: BufferIOFileCacheMode,
    ) -> io::Result<BufferIOFile> {
        let file_url = NSURL::from_file_path(file_path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Metal I/O file path is invalid: file_path={file_path:?}"),
            )
        })?;
        let file = options.open(file_path)?;
        set_file_cache_mode(&file, cache_mode)?;
        let metal_handle = self
            .device
            .newIOFileHandleWithURL_error(&file_url)
            .map_err(|error| io::Error::other(format!("unable to open Metal I/O file {file_path:?}: {error}")))?;
        Ok(BufferIOFile { file, metal_handle })
    }

    /// Loads one file range directly into one shared Metal buffer range.
    ///
    /// This method returns after the Metal I/O command completes. It does not
    /// synchronize later GPU consumers.
    pub fn file_to_buffer(
        &self,
        file: &BufferIOFile,
        file_offset_bytes: u64,
        buffer: &Buffer,
        buffer_offset_bytes: u64,
        len_bytes: u64,
    ) -> io::Result<()> {
        validate_buffer_range(buffer_offset_bytes, len_bytes, buffer.len_bytes_u64())?;
        validate_file_range(file_offset_bytes, len_bytes, file.len_bytes()?)?;
        if len_bytes == 0 {
            return Ok(());
        }

        let mut transferred_bytes = 0_u64;
        while transferred_bytes < len_bytes {
            let command_bytes = (len_bytes - transferred_bytes).min(FILE_TO_BUFFER_COMMAND_BYTES);
            self.file_to_buffer_command(
                file,
                file_offset_bytes + transferred_bytes,
                buffer,
                buffer_offset_bytes + transferred_bytes,
                command_bytes,
            )?;
            transferred_bytes += command_bytes;
        }
        Ok(())
    }

    fn file_to_buffer_command(
        &self,
        file: &BufferIOFile,
        file_offset_bytes: u64,
        buffer: &Buffer,
        buffer_offset_bytes: u64,
        len_bytes: u64,
    ) -> io::Result<()> {
        let command_buffer = self.queue.commandBuffer();
        let buffer_offset = to_usize(buffer_offset_bytes, "buffer_offset_bytes")?;
        let len = to_usize(len_bytes, "len_bytes")?;
        let file_offset = to_usize(file_offset_bytes, "file_offset_bytes")?;
        unsafe {
            command_buffer.loadBuffer_offset_size_sourceHandle_sourceHandleOffset(
                buffer.as_raw(),
                buffer_offset,
                len,
                &file.metal_handle,
                file_offset,
            );
        }
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        if command_buffer.status() == MTLIOStatus::Complete {
            return Ok(());
        }

        let error = command_buffer.error().map_or_else(
            || format!("status={:?}", command_buffer.status()),
            |error| error.to_string(),
        );
        Err(io::Error::other(format!(
            "Metal file-to-buffer command failed: file_offset_bytes={file_offset_bytes} \
             buffer_offset_bytes={buffer_offset_bytes} len_bytes={len_bytes}: {error}"
        )))
    }

    /// Writes one shared Metal buffer range directly to one file range.
    ///
    /// This method returns after all bytes reach the file. It does not sync the
    /// file. The snapshot owner controls file sync and publication.
    pub fn buffer_to_file(
        &self,
        buffer: &Buffer,
        buffer_offset_bytes: u64,
        file: &BufferIOFile,
        file_offset_bytes: u64,
        len_bytes: u64,
    ) -> io::Result<()> {
        validate_buffer_range(buffer_offset_bytes, len_bytes, buffer.len_bytes_u64())?;
        if len_bytes == 0 {
            return Ok(());
        }

        let buffer_offset = to_usize(buffer_offset_bytes, "buffer_offset_bytes")?;
        let len = to_usize(len_bytes, "len_bytes")?;
        let source =
            unsafe { slice::from_raw_parts(buffer.contents().cast::<u8>().add(buffer_offset).cast_const(), len) };
        write_all_at(&file.file, source, file_offset_bytes)
    }
}

fn set_file_cache_mode(file: &File, cache_mode: BufferIOFileCacheMode) -> io::Result<()> {
    match cache_mode {
        BufferIOFileCacheMode::Cached => Ok(()),
        BufferIOFileCacheMode::Uncached => {
            enable_file_flag(file, libc::F_NOCACHE, "F_NOCACHE")?;
            enable_file_flag(file, libc::F_GLOBAL_NOCACHE, "F_GLOBAL_NOCACHE")
        },
    }
}

fn enable_file_flag(file: &File, command: libc::c_int, name: &'static str) -> io::Result<()> {
    // SAFETY: `file` owns a live descriptor. Both commands used here accept one integer argument.
    let result = unsafe { libc::fcntl(file.as_raw_fd(), command, 1) };
    if result != -1 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    Err(io::Error::new(
        error.kind(),
        format!("unable to enable {name} for BufferIO file: {error}"),
    ))
}

impl BufferIOFile {
    pub fn sync_all(&self) -> io::Result<()> {
        self.file.sync_all()
    }

    fn len_bytes(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }
}

fn validate_buffer_range(offset_bytes: u64, len_bytes: u64, capacity_bytes: u64) -> io::Result<()> {
    let end_bytes = checked_end_bytes("Metal buffer", offset_bytes, len_bytes)?;
    if end_bytes <= capacity_bytes {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "Metal buffer I/O range is out of bounds: offset_bytes={offset_bytes} len_bytes={len_bytes} \
             capacity_bytes={capacity_bytes}"
        ),
    ))
}

fn validate_file_range(offset_bytes: u64, len_bytes: u64, file_bytes: u64) -> io::Result<()> {
    let end_bytes = checked_end_bytes("file", offset_bytes, len_bytes)?;
    if end_bytes <= file_bytes {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        format!(
            "file I/O range is truncated: offset_bytes={offset_bytes} len_bytes={len_bytes} file_bytes={file_bytes}"
        ),
    ))
}

fn checked_end_bytes(owner: &'static str, offset_bytes: u64, len_bytes: u64) -> io::Result<u64> {
    offset_bytes.checked_add(len_bytes).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{owner} I/O range overflows u64: offset_bytes={offset_bytes} len_bytes={len_bytes}"),
        )
    })
}

fn to_usize(value: u64, name: &'static str) -> io::Result<usize> {
    value.try_into().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name} does not fit the Metal NSUInteger boundary: value={value}"),
        )
    })
}

fn write_all_at(file: &File, mut bytes: &[u8], mut offset_bytes: u64) -> io::Result<()> {
    while !bytes.is_empty() {
        match file.write_at(bytes, offset_bytes) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero)),
            Ok(written_bytes) => {
                bytes = &bytes[written_bytes..];
                offset_bytes = offset_bytes
                    .checked_add(written_bytes as u64)
                    .expect("completed file write offset must fit u64");
            },
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {},
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    use super::*;

    static NEXT_TEST_FILE_ID: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn test_file_to_buffer_with_multiple_queues() {
        let device = Device::system_default();
        // Keep enough independent owners alive to expose excessive Metal I/O worker allocation.
        let queues = (0..32).map(|_| BufferIO::new(&device)).collect::<Vec<_>>();
        let values = (1_u32..=32).collect::<Vec<_>>();
        let source = Buffer::from_slice(&device, &values);
        let restored = Buffer::new_zeroed(&device, 4);
        let file_path = test_file_path();
        let file = queues[0].create(&file_path, BufferIOFileCacheMode::Cached).unwrap();
        queues[0]
            .buffer_to_file(&source, 0, &file, 0, source.len_bytes_u64())
            .unwrap();
        file.sync_all().unwrap();

        for (index, queue) in queues.iter().enumerate() {
            queue.file_to_buffer(&file, index as u64 * 4, &restored, 0, 4).unwrap();
            assert_eq!(restored.read_typed::<u32>(0, 1), [values[index]]);
        }

        drop(file);
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_buffer_file_round_trip_preserves_selected_range() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source_values = (0..4096).map(|index| (index % 251) as u8).collect::<Vec<_>>();
        let source = Buffer::from_slice(&device, &source_values);
        let restored = Buffer::new_zeroed(&device, 4096);
        let file_path = test_file_path();
        let file = buffer_io.create(&file_path, BufferIOFileCacheMode::Cached).unwrap();

        buffer_io.buffer_to_file(&source, 512, &file, 8192, 3072).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let file = buffer_io.open(&file_path, BufferIOFileCacheMode::Cached).unwrap();
        buffer_io.file_to_buffer(&file, 8192, &restored, 256, 3072).unwrap();

        let mut restored_values = vec![0; 4096];
        restored.read_bytes(0, &mut restored_values);
        assert_eq!(&source_values[512..3584], &restored_values[256..3328]);
        assert!(restored_values[..256].iter().all(|value| *value == 0));
        assert!(restored_values[3328..].iter().all(|value| *value == 0));

        drop(file);
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_uncached_buffer_file_round_trip() {
        const LEN_BYTES: usize = 4 * 1024 * 1024;

        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let source_values = (0..LEN_BYTES).map(|index| (index % 251) as u8).collect::<Vec<_>>();
        let source = Buffer::from_slice(&device, &source_values);
        let restored = Buffer::new_zeroed(&device, LEN_BYTES);
        let file_path = test_file_path();
        let file = buffer_io.create(&file_path, BufferIOFileCacheMode::Uncached).unwrap();
        let len_bytes = LEN_BYTES as u64;

        buffer_io.buffer_to_file(&source, 0, &file, 0, len_bytes).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let file = buffer_io.open(&file_path, BufferIOFileCacheMode::Uncached).unwrap();
        buffer_io.file_to_buffer(&file, 0, &restored, 0, len_bytes).unwrap();

        let mut restored_values = vec![0; LEN_BYTES];
        restored.read_bytes(0, &mut restored_values);
        assert_eq!(restored_values, source_values);

        drop(file);
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_buffer_to_file_rejects_out_of_bounds_buffer_range() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let buffer = Buffer::new_zeroed(&device, 16);
        let file_path = test_file_path();
        let file = buffer_io.create(&file_path, BufferIOFileCacheMode::Cached).unwrap();

        let result = buffer_io.buffer_to_file(&buffer, 8, &file, 0, 9);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput);

        drop(file);
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_file_to_buffer_rejects_out_of_bounds_buffer_range() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let buffer = Buffer::new_zeroed(&device, 16);
        let file_path = test_file_path();
        let file = buffer_io.create(&file_path, BufferIOFileCacheMode::Cached).unwrap();

        let result = buffer_io.file_to_buffer(&file, 0, &buffer, 8, 9);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput);

        drop(file);
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_file_to_buffer_rejects_truncated_file_range() {
        let device = Device::system_default();
        let buffer_io = BufferIO::new(&device);
        let buffer = Buffer::new_zeroed(&device, 16);
        let file_path = test_file_path();
        let file = buffer_io.create(&file_path, BufferIOFileCacheMode::Cached).unwrap();

        let result = buffer_io.file_to_buffer(&file, 0, &buffer, 0, 1);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);

        drop(file);
        std::fs::remove_file(file_path).unwrap();
    }

    fn test_file_path() -> std::path::PathBuf {
        let id = NEXT_TEST_FILE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("psi-dec-buffer-io-test-{}-{id}", std::process::id()))
    }
}
