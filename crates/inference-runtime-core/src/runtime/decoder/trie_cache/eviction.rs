use std::fmt::Debug;
use std::sync::Arc;

/// Selects eviction candidates. The trie owns node and page removal.
///
/// ```text
/// policy -> reserve candidate K
/// trie   -> check ownership and try removal
///         -> accept K after removal, or reject K
/// ```
///
/// A candidate must remain reserved until the caller accepts or rejects it.
/// The caller must resolve each returned candidate exactly once.
/// During shutdown, candidate selection returns `None` if the policy stops.
/// Candidate resolution must also tolerate policy shutdown.
pub trait EvictionPolicy<K>: Debug + Send + Sync + 'static {
    /// The concrete handle for this policy's node notifications.
    type Entry: EvictionEntry;

    /// Creates an unregistered entry. The entry starts pinned.
    fn new_entry(&self, key: K) -> Arc<Self::Entry>;

    /// Reserves a registered key, or returns `None` when none is available.
    fn evict_candidate(&self) -> Option<K>;

    /// Releases a rejected candidate and re-evaluates its eligibility.
    fn reject_candidate(&self, key: K);

    /// Forgets a candidate after the trie removes its node and pages.
    fn accept_candidate(&self, key: K);
}

/// Receives lifecycle and reuse notifications for one cache node.
///
/// Pin and unpin report eligibility transitions, not reference counts.
/// The caller must register each entry once, before its first unpin.
/// Registration and notifications must tolerate runtime shutdown.
/// The trie can hold a node lock while it calls these methods.
/// Implementations must not call back into the trie.
pub trait EvictionEntry: Debug + Send + Sync + 'static {
    /// Registers this pinned entry with its policy.
    fn register(self: Arc<Self>);

    /// Marks the entry as ineligible for eviction.
    fn pin(&self);

    /// Reports that node ownership permits eviction.
    fn unpin(&self);

    /// Records a cache reuse for the policy.
    fn touch(&self);
}
