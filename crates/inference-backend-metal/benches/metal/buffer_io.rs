use std::env;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::BufferIO;
use inference_backend_metal::metal::BufferIOFile;
use inference_backend_metal::metal::BufferIOFileCacheMode;
use inference_backend_metal::metal::Device;

const BENCH_DIR_ENV: &str = "KHALAMARI_BUFFER_IO_BENCH_DIR";
const BUFFER_MIB_VALUES: [u64; 3] = [4, 64, 128];
const MIB_BYTES: u64 = 1024 * 1024;
const PATTERN_LEN_BYTES: usize = 1024 * 1024;

fn bench_buffer_io(criterion: &mut Criterion) {
    let device = Device::system_default();
    let bench_dir = bench_dir();
    std::fs::create_dir_all(&bench_dir).unwrap();
    let mut group = criterion.benchmark_group("metal/buffer-io");

    for buffer_mib in BUFFER_MIB_VALUES {
        let len_bytes = buffer_mib.checked_mul(MIB_BYTES).unwrap();
        let fixture = BufferIOFixture::new(&device, &bench_dir, len_bytes);
        group.throughput(Throughput::Bytes(len_bytes));

        group.bench_function(format!("file_to_buffer_uncached/{buffer_mib}mib"), |bencher| {
            bencher.iter(|| {
                fixture.file_to_buffer();
                black_box(&fixture.destination);
            });
        });
        group.bench_function(format!("buffer_to_file_uncached_sync/{buffer_mib}mib"), |bencher| {
            bencher.iter(|| {
                fixture.buffer_to_file_and_sync();
                black_box(&fixture.write_file);
            });
        });

        let file_path = fixture.file_path.clone();
        drop(fixture);
        std::fs::remove_file(file_path).unwrap();
    }

    group.finish();
}

struct BufferIOFixture {
    buffer_io: BufferIO,
    read_file: BufferIOFile,
    write_file: BufferIOFile,
    source: Buffer,
    destination: Buffer,
    file_path: PathBuf,
    len_bytes: u64,
}

impl BufferIOFixture {
    fn new(device: &Device, bench_dir: &std::path::Path, len_bytes: u64) -> Self {
        let buffer_io = BufferIO::new(device);
        let source = pattern_buffer(device, len_bytes);
        let destination = Buffer::new_zeroed(device, len_bytes);
        let file_path = bench_file_path(bench_dir, len_bytes);
        let write_file = buffer_io.create(&file_path, BufferIOFileCacheMode::Uncached).unwrap();
        buffer_io.buffer_to_file(&source, 0, &write_file, 0, len_bytes).unwrap();
        write_file.sync_all().unwrap();
        let read_file = buffer_io.open(&file_path, BufferIOFileCacheMode::Uncached).unwrap();
        let fixture = Self {
            buffer_io,
            read_file,
            write_file,
            source,
            destination,
            file_path,
            len_bytes,
        };
        fixture.file_to_buffer();
        fixture.validate_destination();
        fixture
    }

    fn file_to_buffer(&self) {
        self.buffer_io
            .file_to_buffer(&self.read_file, 0, &self.destination, 0, self.len_bytes)
            .unwrap();
    }

    fn buffer_to_file(&self) {
        self.buffer_io
            .buffer_to_file(&self.source, 0, &self.write_file, 0, self.len_bytes)
            .unwrap();
    }

    fn buffer_to_file_and_sync(&self) {
        self.buffer_to_file();
        self.write_file.sync_all().unwrap();
    }

    fn validate_destination(&self) {
        let pattern = byte_pattern();
        let mut actual = vec![0; PATTERN_LEN_BYTES];
        let mut offset_bytes = 0_u64;
        while offset_bytes < self.len_bytes {
            let chunk_bytes = usize::try_from((self.len_bytes - offset_bytes).min(MIB_BYTES)).unwrap();
            self.destination
                .read_bytes(usize::try_from(offset_bytes).unwrap(), &mut actual[..chunk_bytes]);
            assert_eq!(&actual[..chunk_bytes], &pattern[..chunk_bytes]);
            offset_bytes = offset_bytes.checked_add(u64::try_from(chunk_bytes).unwrap()).unwrap();
        }
    }
}

fn pattern_buffer(device: &Device, len_bytes: u64) -> Buffer {
    let buffer = Buffer::new_uninit(device, len_bytes);
    let pattern = byte_pattern();
    let mut offset_bytes = 0_u64;
    while offset_bytes < len_bytes {
        let chunk_bytes = usize::try_from((len_bytes - offset_bytes).min(MIB_BYTES)).unwrap();
        buffer.write_bytes(usize::try_from(offset_bytes).unwrap(), &pattern[..chunk_bytes]);
        offset_bytes = offset_bytes.checked_add(u64::try_from(chunk_bytes).unwrap()).unwrap();
    }
    buffer
}

fn byte_pattern() -> Vec<u8> {
    (0..PATTERN_LEN_BYTES)
        .map(|index| u8::try_from((index * 131 + 17) % 251).unwrap())
        .collect()
}

fn bench_dir() -> PathBuf {
    env::var_os(BENCH_DIR_ENV).map_or_else(env::temp_dir, PathBuf::from)
}

fn bench_file_path(bench_dir: &std::path::Path, len_bytes: u64) -> PathBuf {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    bench_dir.join(format!(
        "khalamari-buffer-io-bench-{}-{timestamp}-{len_bytes}",
        std::process::id()
    ))
}

criterion_group!(benches, bench_buffer_io);
criterion_main!(benches);
