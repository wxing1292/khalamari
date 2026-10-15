mod single_lane;
pub use single_lane::AllocateSingleLaneMutableBlockResult;
pub use single_lane::CommitSingleLaneMutableBlockResult;
pub use single_lane::CommitSingleLaneSemiImmutableBlockResult;
pub use single_lane::ReserveSingleLaneSemiImmutableBlockResult;
pub use single_lane::SingleLaneBlockCache;

mod multi_lane;
pub use multi_lane::AllocateMultiLaneMutableBlockResult;
pub use multi_lane::CommitMultiLaneMutableBlockResult;
pub use multi_lane::CommitMultiLaneSemiImmutableBlockResult;
pub use multi_lane::MultiLaneBlockCache;
pub use multi_lane::ReserveMultiLaneSemiImmutableBlockResult;

mod single_lane_impl;
pub use single_lane_impl::SingleLaneTrieBlockCache;

mod multi_lane_impl;
pub use multi_lane_impl::MultiLaneTrieBlockCache;

mod reservation;
pub use reservation::Reservation;
pub use reservation::ReservationKey;

#[cfg(test)]
mod eviction_policy_test;
