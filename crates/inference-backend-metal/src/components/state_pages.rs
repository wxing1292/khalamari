//! Batched copies between executor state buffers and runtime-owned pages.

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Operator;

/// Byte ranges. Owners prove allocation bounds before they upload jobs.
/// Offsets and lengths must be multiples of four.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CopyJob {
    pub src_offset_bytes: u64,
    pub dst_offset_bytes: u64,
    pub len_bytes: u64,
}

pub struct Compute {
    copy: CompiledKernel,
}
impl Compute {
    pub fn new(device: &Device) -> Self {
        Self {
            copy: CompiledKernel::new(device, include_str!("metal/state_pages.metal"), "copy_state_ranges"),
        }
    }
    pub fn copy<'a>(&'a self, src: &'a Buffer, dst: &'a Buffer, jobs: Buffer, num_jobs: u32) -> CopyInvocation<'a> {
        assert!(num_jobs > 0 && jobs.len_bytes() / size_of::<CopyJob>() >= num_jobs as usize);
        CopyInvocation {
            compute: self,
            src,
            dst,
            jobs,
            num_jobs,
        }
    }
}
pub struct CopyInvocation<'a> {
    compute: &'a Compute,
    src: &'a Buffer,
    dst: &'a Buffer,
    jobs: Buffer,
    num_jobs: u32,
}
impl Operator for CopyInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.copy);
        recorder.set_buffer_read(0, self.src, 0);
        recorder.set_buffer_write(1, self.dst, 0);
        recorder.set_buffer_read(2, &self.jobs, 0);
        recorder.dispatch_threadblocks((self.num_jobs as usize, 1, 1), (256, 1, 1));
    }
}
