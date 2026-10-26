use crate::config::SamplingConfig;
use crate::runtime::RawComputeSlotSeq;
use crate::runtime::RawRequestID;
use crate::runtime::RawRequestSlot;

mod decoder;
pub use decoder::DecoderSyncBlocks;
pub use decoder::QueryTokens;
pub use decoder::SampledTokens;

mod resource;
pub use resource::DeviceResourcePlacement;

#[mockall::automock]
pub trait DevReq: Send + 'static {
    fn id(&self) -> RawRequestID;

    fn request_cost(&self) -> usize;
    fn token_cost(&self) -> usize;
}

#[mockall::automock]
pub trait BatchDevReq<DeviceReq>: Send + 'static
where
    DeviceReq: DevReq,
{
    fn seq(&self) -> RawComputeSlotSeq;

    fn request_cost(&self) -> usize;
    fn token_cost(&self) -> usize;

    /// Packs the selected requests. Implementations may reorder requests while preserving each request's identity.
    fn from_parts(seq: RawComputeSlotSeq, dev_reqs: Vec<DeviceReq>) -> Self;
    fn into_inner(self) -> (RawComputeSlotSeq, Vec<DeviceReq>);
}

pub struct DeviceRequest {
    pub req_id: RawRequestID,
    pub req_slot: RawRequestSlot,
    pub decoder_query_tokens: QueryTokens,
    pub decoder_sync_blocks: DecoderSyncBlocks,
    pub resource_placements: Vec<DeviceResourcePlacement>,
    pub sampling_config: SamplingConfig,
}

impl DeviceRequest {
    pub fn new(
        req_id: RawRequestID,
        req_slot: RawRequestSlot,
        decoder_query_tokens: QueryTokens,
        decoder_sync_blocks: DecoderSyncBlocks,
        resource_placements: Vec<DeviceResourcePlacement>,
        sampling_config: SamplingConfig,
    ) -> Self {
        Self {
            req_id,
            req_slot,
            decoder_query_tokens,
            decoder_sync_blocks,
            resource_placements,
            sampling_config,
        }
    }
}

impl DevReq for DeviceRequest {
    fn id(&self) -> RawRequestID {
        self.req_id
    }

    fn request_cost(&self) -> usize {
        1
    }

    fn token_cost(&self) -> usize {
        self.decoder_query_tokens.token_consumption()
    }
}

pub struct BatchDeviceRequest {
    pub seq: RawComputeSlotSeq,
    pub dev_reqs: Vec<DeviceRequest>,
}

impl BatchDeviceRequest {
    pub fn new<I>(seq: RawComputeSlotSeq, dev_reqs: I) -> Self
    where
        I: IntoIterator<Item = DeviceRequest> + 'static,
    {
        Self {
            seq,
            dev_reqs: dev_reqs.into_iter().collect(),
        }
    }
}

impl BatchDevReq<DeviceRequest> for BatchDeviceRequest {
    fn seq(&self) -> RawComputeSlotSeq {
        self.seq
    }

    fn request_cost(&self) -> usize {
        self.dev_reqs.len()
    }

    fn token_cost(&self) -> usize {
        self.dev_reqs.iter().fold(0, |sum, req| sum + req.token_cost())
    }

    fn from_parts(seq: RawComputeSlotSeq, dev_reqs: Vec<DeviceRequest>) -> Self {
        Self::new(seq, dev_reqs)
    }

    fn into_inner(self) -> (RawComputeSlotSeq, Vec<DeviceRequest>) {
        (self.seq, self.dev_reqs)
    }
}
