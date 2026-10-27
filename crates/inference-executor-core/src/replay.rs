const BASE_BUCKETS: [u32; 14] = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayBucketPolicy {
    max_capacity: u32,
    buckets: Box<[u32]>,
}

impl ReplayBucketPolicy {
    pub fn new(max_capacity: u32) -> Self {
        Self::with_topology_boundaries(max_capacity, &[])
    }

    /// `topology_boundaries` contains the exclusive upper boundary of each
    /// preceding recorded topology. Boundary `b` separates the half-open
    /// topology domains `[.., b)` and `[b, ..)`.
    ///
    /// Unlike the repository-default half-open interval representation, each
    /// replay bucket is a positive inclusive upper capacity. `capacity` selects
    /// the first bucket that is greater than or equal to the active count. The
    /// policy therefore inserts `boundary - 1` as the final bucket for the
    /// preceding topology. Boundary `1` has no preceding positive capacity.
    /// Boundaries above `max_capacity` cannot affect this policy and are
    /// ignored.
    pub fn with_topology_boundaries(max_capacity: u32, topology_boundaries: &[u32]) -> Self {
        assert!(max_capacity > 0, "replay bucket capacity must be positive");
        let mut buckets = default_buckets(max_capacity);
        for &boundary in topology_boundaries {
            assert!(boundary > 0, "replay topology boundary must be positive");
            if boundary > 1 && boundary <= max_capacity {
                let preceding_topology_inclusive_max_capacity = boundary - 1;
                buckets.push(preceding_topology_inclusive_max_capacity);
            }
        }
        buckets.push(max_capacity);
        buckets.sort_unstable();
        buckets.dedup();
        Self {
            max_capacity,
            buckets: buckets.into_boxed_slice(),
        }
    }

    /// Returns the configured allocation limit.
    pub fn max_capacity(&self) -> u32 {
        self.max_capacity
    }

    /// Returns the strictly increasing positive inclusive upper capacities.
    /// The final capacity is `max_capacity`.
    pub fn buckets(&self) -> &[u32] {
        &self.buckets
    }

    pub fn capacity(&self, active: u32) -> u32 {
        assert!(active > 0, "replay bucket requires active work");
        assert!(active <= self.max_capacity, "replay active work exceeds capacity");
        let index = self.buckets.partition_point(|&bucket| bucket < active);
        self.buckets[index]
    }

    /// Returns zero when the work domain is absent. Zero is not a replay
    /// capacity and must not record or dispatch work.
    pub fn capacity_allow_zero(&self, active: u32) -> u32 {
        if active == 0 { 0 } else { self.capacity(active) }
    }
}

fn default_buckets(max_capacity: u32) -> Vec<u32> {
    let mut buckets = BASE_BUCKETS
        .into_iter()
        .take_while(|&bucket| bucket <= max_capacity)
        .collect::<Vec<_>>();
    let mut lower = 64u32;
    while lower < max_capacity {
        let step = lower / 4;
        for part in 1..=4 {
            let candidate = lower.saturating_add(step * part);
            if candidate >= max_capacity {
                return buckets;
            }
            buckets.push(candidate);
        }
        lower = lower.saturating_mul(2);
    }
    buckets
}

#[cfg(test)]
mod tests {
    use super::ReplayBucketPolicy;

    #[test]
    fn test_default_policy_extends_past_64_and_includes_terminal_capacity() {
        assert_eq!(
            ReplayBucketPolicy::new(128).buckets(),
            [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128]
        );
        assert_eq!(ReplayBucketPolicy::new(70).buckets().last(), Some(&70));
        assert_eq!(ReplayBucketPolicy::new(6).buckets(), [1, 2, 4, 6]);
    }

    #[test]
    fn test_buckets_remain_ordered_at_extension_boundaries_and_u32_limit() {
        for max_capacity in [65, 79, 80, 81, 127, 128, 129, u32::MAX] {
            let policy = ReplayBucketPolicy::new(max_capacity);

            assert_eq!(policy.buckets().last(), Some(&max_capacity));
            assert_eq!(policy.capacity(max_capacity), max_capacity);
            assert!(policy.buckets().windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn test_capacity_uses_the_first_bucket_that_contains_active_work() {
        let policy = ReplayBucketPolicy::new(64);

        assert_eq!(policy.capacity(1), 1);
        assert_eq!(policy.capacity(3), 4);
        assert_eq!(policy.capacity(5), 6);
        assert_eq!(policy.capacity(9), 12);
        assert_eq!(policy.capacity(25), 32);
        assert_eq!(policy.capacity(64), 64);
        assert_eq!(policy.capacity_allow_zero(0), 0);
    }

    #[test]
    fn test_topology_boundaries_prevent_cross_topology_padding() {
        let policy = ReplayBucketPolicy::with_topology_boundaries(64, &[5, 6, 10, 12, 18]);

        assert!(!policy.buckets().contains(&0));
        assert_eq!(policy.capacity(5), 5);
        assert_eq!(policy.capacity(9), 9);
        assert_eq!(policy.capacity(10), 11);
        assert_eq!(policy.capacity(11), 11);
        assert_eq!(policy.capacity(17), 17);
        assert_eq!(policy.capacity(18), 20);
    }

    #[test]
    fn test_first_topology_boundary_has_no_preceding_bucket() {
        let policy = ReplayBucketPolicy::with_topology_boundaries(6, &[1]);

        assert_eq!(policy.buckets(), [1, 2, 4, 6]);
    }

    #[test]
    fn test_exclusive_topology_boundary_adds_the_preceding_inclusive_capacity() {
        let policy = ReplayBucketPolicy::with_topology_boundaries(8, &[6]);

        assert_eq!(policy.buckets(), [1, 2, 4, 5, 6, 8]);
        assert_eq!(policy.capacity(5), 5);
        assert_eq!(policy.capacity(6), 6);
        assert_eq!(policy.capacity(7), 8);
    }

    #[test]
    fn test_default_policy_keeps_inactive_lanes_at_or_below_one_quarter() {
        let policy = ReplayBucketPolicy::new(256);

        for active in 1..=policy.max_capacity() {
            let total = policy.capacity(active);
            assert!((total - active) * 4 <= total, "active={active} total={total}");
        }
    }

    #[test]
    #[should_panic(expected = "replay active work exceeds capacity")]
    fn test_capacity_rejects_active_work_above_limit() {
        ReplayBucketPolicy::new(6).capacity(7);
    }
}
