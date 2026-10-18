use crate::runtime::RawComputeSlotID;
use crate::runtime::RawComputeSlotSeq;
use crate::runtime::RawRequestID;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeSlotSeq {
    Ready { seq_id: RawComputeSlotSeq },
    Pending { seq_id: RawComputeSlotSeq },
}

pub struct ComputeSlot {
    id: RawComputeSlotID,
    seq: Option<RawComputeSlotSeq>,
    sticky_req_ids: Vec<RawRequestID>,
}

impl ComputeSlot {
    pub fn new(id: RawComputeSlotID) -> Self {
        Self {
            id,
            seq: None,
            sticky_req_ids: Vec::new(),
        }
    }

    pub fn id(&self) -> RawComputeSlotID {
        self.id
    }

    pub fn seq(&self) -> Option<RawComputeSlotSeq> {
        self.seq
    }

    pub fn sticky_req_ids_ref(&self) -> &[RawRequestID] {
        &self.sticky_req_ids
    }

    pub fn sticky_req_ids_mut(&mut self) -> &mut Vec<RawRequestID> {
        &mut self.sticky_req_ids
    }

    pub fn prepare(&mut self, seq: RawComputeSlotSeq) {
        debug_assert!(self.seq.is_none(), "compute slot must be free before prepare");
        self.seq = Some(seq);
    }

    pub fn reset(&mut self) {
        debug_assert!(self.seq.is_some(), "compute slot must be in use before reset");
        self.seq = None;
    }
}
