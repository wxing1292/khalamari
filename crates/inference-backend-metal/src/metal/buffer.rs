use std::collections::HashMap;
use std::ffi::c_void;
use std::panic::Location;
use std::slice;
use std::sync::Mutex;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLDevice;
use objc2_metal::MTLResourceOptions;

use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::MetalBufferElement;

#[derive(Debug)]
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len_bytes: u64,
    allocation_site: GpuAllocationSite,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GpuAllocationSite {
    kind: &'static str,
    file: &'static str,
    line: u32,
    column: u32,
}

#[derive(Clone, Debug)]
pub struct BufferAllocationSummary {
    pub kind: &'static str,
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
    pub live_count: usize,
    pub live_bytes: u64,
}

#[derive(Clone, Debug, Default)]
struct BufferAllocationStats {
    live_count: usize,
    live_bytes: u64,
}

static BUFFER_ALLOCATION_STATS: OnceLock<Mutex<HashMap<GpuAllocationSite, BufferAllocationStats>>> = OnceLock::new();

impl Buffer {
    #[track_caller]
    pub fn new_zeroed_elements<N>(device: &Device, num_elements: N, dtype: Dtype) -> Self
    where
        N: TryInto<u64>,
    {
        let num_elements = to_u64(num_elements, "buffer element count must fit u64");
        let len_bytes = num_elements
            .checked_mul(dtype.item_size() as u64)
            .expect("buffer element byte length must fit u64");
        Self::new_zeroed(device, len_bytes)
    }

    #[track_caller]
    pub fn new_zeroed<N>(device: &Device, len_bytes: N) -> Self
    where
        N: TryInto<u64>,
    {
        let len_bytes = to_u64(len_bytes, "zeroed buffer byte length must fit u64");
        let buffer = Self::new_uninit(device, len_bytes);
        buffer.zero_bytes(
            0,
            len_bytes
                .try_into()
                .expect("zeroed buffer byte length must fit host usize"),
        );
        buffer
    }

    #[track_caller]
    pub fn new_uninit<N>(device: &Device, len_bytes: N) -> Self
    where
        N: TryInto<u64>,
    {
        let len_bytes = to_u64(len_bytes, "buffer byte length must fit u64");
        assert!(len_bytes > 0);
        assert!(
            len_bytes <= device.max_buffer_length(),
            "MTLBuffer allocation exceeds device maxBufferLength: len_bytes={len_bytes} max_buffer_length={}",
            device.max_buffer_length()
        );
        let len: usize = len_bytes
            .try_into()
            .expect("MTLBuffer allocation byte length must fit host usize");
        let options = MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared;
        let caller = Location::caller();
        let allocation_site = GpuAllocationSite {
            kind: "buffer",
            file: caller.file(),
            line: caller.line(),
            column: caller.column(),
        };
        let raw = device
            .as_raw()
            .newBufferWithLength_options(len, options)
            .unwrap_or_else(|| {
                panic!(
                    "MTLBuffer allocation failed: len_bytes={len_bytes} caller={}:{}:{}",
                    caller.file(),
                    caller.line(),
                    caller.column()
                );
            });
        assert_eq!(raw.length(), len);
        record_gpu_alloc(allocation_site, len_bytes);
        Self {
            raw,
            len_bytes,
            allocation_site,
        }
    }

    #[track_caller]
    pub fn from_slice<T: MetalBufferElement>(device: &Device, values: &[T]) -> Self {
        assert!(!values.is_empty());
        let len_bytes = (values.len() as u64)
            .checked_mul(T::DTYPE.item_size() as u64)
            .expect("buffer byte length must fit u64");
        let buffer = Self::new_uninit(device, len_bytes);
        buffer.write_typed(0, values);
        buffer
    }

    pub fn as_raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    pub fn as_raw_retained(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.raw.clone()
    }

    pub fn as_raw_ptr(&self) -> *mut c_void {
        Retained::as_ptr(&self.raw).cast::<c_void>().cast_mut()
    }

    pub fn len_bytes(&self) -> usize {
        self.len_bytes as usize
    }

    pub fn len_bytes_u64(&self) -> u64 {
        self.len_bytes
    }

    pub fn contents(&self) -> *mut c_void {
        self.raw.contents().as_ptr()
    }

    pub fn zero_bytes(&self, start_bytes: usize, len_bytes: usize) {
        let buffer_len = self.len_bytes();
        assert!(start_bytes <= buffer_len);
        let end_bytes = start_bytes
            .checked_add(len_bytes)
            .expect("zeroed write byte range must fit usize");
        assert!(end_bytes <= buffer_len);

        let dst = unsafe { self.contents().cast::<u8>().add(start_bytes) };
        unsafe {
            dst.write_bytes(0, len_bytes);
        }
    }

    pub fn write_bytes(&self, start_bytes: usize, values: &[u8]) {
        let buffer_len = self.len_bytes();
        assert!(start_bytes <= buffer_len);
        let end_bytes = start_bytes
            .checked_add(values.len())
            .expect("byte buffer write end offset must fit usize");
        assert!(end_bytes <= buffer_len);

        let dst = unsafe { self.contents().cast::<u8>().add(start_bytes) };
        unsafe {
            dst.copy_from_nonoverlapping(values.as_ptr(), values.len());
        }
    }

    pub fn read_bytes(&self, start_bytes: usize, output: &mut [u8]) {
        let buffer_len = self.len_bytes();
        assert!(start_bytes <= buffer_len);
        let end_bytes = start_bytes
            .checked_add(output.len())
            .expect("byte buffer read end offset must fit usize");
        assert!(end_bytes <= buffer_len);

        let src = unsafe { self.contents().cast::<u8>().add(start_bytes) };
        unsafe {
            output.as_mut_ptr().copy_from_nonoverlapping(src, output.len());
        }
    }

    pub fn write_typed<T: MetalBufferElement>(&self, start: usize, values: &[T]) {
        let start_bytes = start
            .checked_mul(T::DTYPE.item_size())
            .expect("typed buffer write start byte offset must fit usize");
        let total_bytes = values
            .len()
            .checked_mul(T::DTYPE.item_size())
            .expect("typed buffer write byte length must fit usize");
        let buffer_len = self.len_bytes();
        assert!(start_bytes <= buffer_len);
        let end_bytes = start_bytes
            .checked_add(total_bytes)
            .expect("typed buffer write end byte offset must fit usize");
        assert!(end_bytes <= buffer_len);

        let dst = unsafe { self.contents().cast::<u8>().add(start_bytes).cast::<T>() };
        unsafe {
            dst.copy_from_nonoverlapping(values.as_ptr(), values.len());
        }
    }

    pub fn read_typed<T: MetalBufferElement>(&self, start: usize, len: usize) -> Vec<T> {
        let start_bytes = start
            .checked_mul(T::DTYPE.item_size())
            .expect("typed buffer read start byte offset must fit usize");
        let total_bytes = len
            .checked_mul(T::DTYPE.item_size())
            .expect("typed buffer read byte length must fit usize");
        let buffer_len = self.len_bytes();
        assert!(start_bytes <= buffer_len);
        let end_bytes = start_bytes
            .checked_add(total_bytes)
            .expect("typed buffer read end byte offset must fit usize");
        assert!(end_bytes <= buffer_len);

        let src = unsafe { self.contents().cast::<u8>().add(start_bytes).cast::<T>() };
        unsafe { slice::from_raw_parts(src, len).to_vec() }
    }

    pub fn view(&self, dtype: Dtype, shape: Vec<i32>, start_bytes: usize) -> BufferView<'_> {
        BufferView::new(self, dtype, shape, start_bytes)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        record_gpu_free(self.allocation_site, self.len_bytes);
    }
}

pub fn buffer_allocation_summary() -> Vec<BufferAllocationSummary> {
    let stats = buffer_allocation_stats()
        .lock()
        .expect("buffer allocation stats mutex poisoned");
    let mut summary = stats
        .iter()
        .filter(|(_, stats)| stats.live_bytes > 0 || stats.live_count > 0)
        .map(|(site, stats)| {
            BufferAllocationSummary {
                kind: site.kind,
                file: site.file,
                line: site.line,
                column: site.column,
                live_count: stats.live_count,
                live_bytes: stats.live_bytes,
            }
        })
        .collect::<Vec<_>>();
    summary.sort_by_key(|entry| std::cmp::Reverse(entry.live_bytes));
    summary
}

#[track_caller]
pub fn record_gpu_buffer_alloc<N>(kind: &'static str, len_bytes: N) -> GpuAllocationSite
where
    N: TryInto<u64>,
{
    let len_bytes = to_u64(len_bytes, "GPU allocation byte length must fit u64");
    let caller = Location::caller();
    let site = GpuAllocationSite {
        kind,
        file: caller.file(),
        line: caller.line(),
        column: caller.column(),
    };
    record_gpu_alloc(site, len_bytes);
    site
}

pub fn record_gpu_buffer_free<N>(site: GpuAllocationSite, len_bytes: N)
where
    N: TryInto<u64>,
{
    record_gpu_free(site, to_u64(len_bytes, "GPU allocation byte length must fit u64"));
}

fn buffer_allocation_stats() -> &'static Mutex<HashMap<GpuAllocationSite, BufferAllocationStats>> {
    BUFFER_ALLOCATION_STATS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn record_gpu_alloc(site: GpuAllocationSite, len_bytes: u64) {
    let mut stats = buffer_allocation_stats()
        .lock()
        .expect("buffer allocation stats mutex poisoned");
    let entry = stats.entry(site).or_default();
    entry.live_count += 1;
    entry.live_bytes = entry
        .live_bytes
        .checked_add(len_bytes)
        .expect("buffer allocation stats byte count overflow");
}

fn record_gpu_free(site: GpuAllocationSite, len_bytes: u64) {
    let mut stats = buffer_allocation_stats()
        .lock()
        .expect("buffer allocation stats mutex poisoned");
    let entry = stats
        .get_mut(&site)
        .expect("buffer allocation stats missing allocation site");
    assert!(entry.live_count > 0, "buffer allocation stats live count underflow");
    assert!(
        entry.live_bytes >= len_bytes,
        "buffer allocation stats live bytes underflow"
    );
    entry.live_count -= 1;
    entry.live_bytes -= len_bytes;
}

fn to_u64<N>(value: N, message: &str) -> u64
where
    N: TryInto<u64>,
{
    value.try_into().unwrap_or_else(|_| panic!("{message}"))
}

#[derive(Clone, Debug)]
pub struct BufferView<'a> {
    buffer: &'a Buffer,
    dtype: Dtype,
    shape: Vec<i32>,
    offset_bytes: usize,
}

impl<'a> BufferView<'a> {
    pub fn new(buffer: &'a Buffer, dtype: Dtype, shape: Vec<i32>, start_bytes: usize) -> Self {
        assert!(
            start_bytes.is_multiple_of(dtype.item_size()),
            "buffer view byte offset must align to its dtype"
        );
        assert!(shape.iter().all(|dim| *dim >= 0));
        let num_elements = shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim as usize))
            .expect("element count must fit usize");
        let len_bytes = num_elements
            .checked_mul(dtype.item_size())
            .expect("view byte length must fit usize");
        assert!(start_bytes <= buffer.len_bytes());
        let end_bytes = start_bytes
            .checked_add(len_bytes)
            .expect("view byte range must fit usize");
        assert!(end_bytes <= buffer.len_bytes());

        Self {
            buffer,
            dtype,
            shape,
            offset_bytes: start_bytes,
        }
    }

    pub fn buffer(&self) -> &'a Buffer {
        self.buffer
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    pub fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&dim| dim as usize).product()
    }

    pub fn len_bytes(&self) -> usize {
        self.num_elements() * self.dtype.item_size()
    }

    pub fn write_typed<T: MetalBufferElement>(&self, values: &[T]) {
        assert_eq!(self.dtype, T::DTYPE);
        assert_eq!(values.len(), self.num_elements());
        self.buffer
            .write_typed(self.offset_bytes / T::DTYPE.item_size(), values);
    }

    pub fn read_typed<T: MetalBufferElement>(&self) -> Vec<T> {
        assert_eq!(self.dtype, T::DTYPE);
        self.buffer
            .read_typed(self.offset_bytes / T::DTYPE.item_size(), self.num_elements())
    }
}
