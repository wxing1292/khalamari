use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTL4CommandQueue;
use objc2_metal::MTLAllocation;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLDevice;
use objc2_metal::MTLIndirectCommandBuffer;
use objc2_metal::MTLResidencySet;
use objc2_metal::MTLResidencySetDescriptor;

#[derive(Debug)]
pub struct ResidencySet {
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    raw: Retained<ProtocolObject<dyn MTLResidencySet>>,
    allocation_ref_counts: RefCell<HashMap<usize, usize>>,
}

#[derive(Debug)]
pub struct Residency {
    set: Rc<ResidencySet>,
    allocations: Vec<ResidencyAllocation>,
}

#[derive(Debug)]
enum ResidencyAllocation {
    Buffer(Retained<ProtocolObject<dyn MTLBuffer>>),
    Pipeline(Retained<ProtocolObject<dyn MTLComputePipelineState>>),
    IndirectCommandBuffer(Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>),
}

impl ResidencySet {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    ) -> Rc<Self> {
        let descriptor = MTLResidencySetDescriptor::new();
        let residency_set = device
            .newResidencySetWithDescriptor_error(&descriptor)
            .expect("Metal residency-set allocation failed");
        queue.addResidencySet(&residency_set);
        Rc::new(Self {
            queue,
            raw: residency_set,
            allocation_ref_counts: RefCell::new(HashMap::new()),
        })
    }

    pub fn register(
        self: &Rc<Self>,
        buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        pipelines: &[Retained<ProtocolObject<dyn MTLComputePipelineState>>],
        indirect_command_buffer: &Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    ) -> Rc<Residency> {
        let icb_allocation = ResidencyAllocation::IndirectCommandBuffer(indirect_command_buffer.clone());
        let icb_identity = icb_allocation.identity();
        let mut allocations = Vec::with_capacity(buffers.len() + pipelines.len() + 1);
        allocations.extend(buffers.iter().cloned().map(ResidencyAllocation::Buffer));
        allocations.extend(pipelines.iter().cloned().map(ResidencyAllocation::Pipeline));
        allocations.push(icb_allocation);

        let mut ref_counts = self.allocation_ref_counts.borrow_mut();
        assert!(
            !ref_counts.contains_key(&icb_identity),
            "Metal replay ICB must be registered exactly once"
        );
        for allocation in &allocations {
            let count = ref_counts.entry(allocation.identity()).or_default();
            if *count == 0 {
                self.raw.addAllocation(allocation.as_raw());
            }
            *count = count.checked_add(1).expect("Metal residency ref count overflow");
        }
        drop(ref_counts);
        self.raw.commit();
        self.raw.requestResidency();

        Rc::new(Residency {
            set: self.clone(),
            allocations,
        })
    }

    #[cfg(test)]
    pub fn allocation_count(&self) -> usize {
        self.raw.allocationCount()
    }
}

impl Drop for ResidencySet {
    fn drop(&mut self) {
        assert!(
            self.allocation_ref_counts.get_mut().is_empty(),
            "Metal residency set dropped with live leases"
        );
        self.queue.removeResidencySet(&self.raw);
        self.raw.endResidency();
    }
}

impl Residency {
    pub fn belongs_to(&self, set: &Rc<ResidencySet>) -> bool {
        Rc::ptr_eq(&self.set, set)
    }
}

impl Drop for Residency {
    fn drop(&mut self) {
        let mut ref_counts = self.set.allocation_ref_counts.borrow_mut();
        for allocation in &self.allocations {
            let identity = allocation.identity();
            let count = ref_counts
                .get_mut(&identity)
                .expect("Metal residency allocation is not registered");
            assert!(*count > 0, "Metal residency ref count underflow");
            *count -= 1;
            if *count == 0 {
                self.set.raw.removeAllocation(allocation.as_raw());
                ref_counts.remove(&identity);
            }
        }
        drop(ref_counts);
        self.set.raw.commit();
    }
}

impl ResidencyAllocation {
    fn identity(&self) -> usize {
        match self {
            Self::Buffer(allocation) => Retained::as_ptr(allocation).cast::<()>() as usize,
            Self::Pipeline(allocation) => Retained::as_ptr(allocation).cast::<()>() as usize,
            Self::IndirectCommandBuffer(allocation) => Retained::as_ptr(allocation).cast::<()>() as usize,
        }
    }

    fn as_raw(&self) -> &ProtocolObject<dyn MTLAllocation> {
        match self {
            Self::Buffer(allocation) => ProtocolObject::from_ref(&**allocation),
            Self::Pipeline(allocation) => ProtocolObject::from_ref(&**allocation),
            Self::IndirectCommandBuffer(allocation) => ProtocolObject::from_ref(&**allocation),
        }
    }
}

pub fn retain_buffer_once(
    buffers: &mut Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
) {
    let identity = Retained::as_ptr(buffer);
    if !buffers.iter().any(|existing| Retained::as_ptr(existing) == identity) {
        buffers.push(buffer.clone());
    }
}

pub fn retain_pipeline_once(
    pipelines: &mut Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
) {
    let identity = Retained::as_ptr(pipeline);
    if !pipelines.iter().any(|existing| Retained::as_ptr(existing) == identity) {
        pipelines.push(pipeline.clone());
    }
}
