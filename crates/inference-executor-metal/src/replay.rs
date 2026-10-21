use std::fmt::Debug;
use std::hash::Hash;

use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use quick_cache::OptionsBuilder;
use quick_cache::unsync::Cache;

const REPLAY_CACHE_CAPACITY: usize = 128;

pub trait ReplayComponent {
    type Key: Clone + Debug + Eq + Hash;
    type Input<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key;

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>);
}

pub struct Replay<T: ReplayComponent> {
    name: &'static str,
    component: T,
    cache: Cache<T::Key, ReplayProgram>,
}

impl<T: ReplayComponent> Replay<T> {
    pub fn new(name: &'static str, component: T) -> Self {
        // Ghost reinsertion can evict the new program before submission.
        // Without ghosts, insertion evicts existing programs first.
        let options = OptionsBuilder::new()
            .estimated_items_capacity(REPLAY_CACHE_CAPACITY)
            .weight_capacity(REPLAY_CACHE_CAPACITY as u64)
            .ghost_allocation(0.0)
            .build()
            .expect("replay cache options must be valid");
        Self {
            name,
            component,
            cache: Cache::with_options(options, Default::default(), Default::default(), Default::default()),
        }
    }

    pub fn component(&self) -> &T {
        &self.component
    }

    pub fn component_mut(&mut self) -> &mut T {
        &mut self.component
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Records or reuses a program. The returned key remains available until
    /// the next `record` or `clear` call on this owner.
    pub fn record<'a>(&'a mut self, runtime: &Stream, input: &T::Input<'a>) -> (T::Key, bool) {
        let key = self.component.replay_key(input);
        let cache_hit = self.cache.get(&key).is_some();
        if !cache_hit {
            let mut recorder = ReplayRecorder::new(runtime.create_replay_program());
            self.component.record(&mut recorder, input);
            self.cache.insert(key.clone(), recorder.build());
        }
        (key, cache_hit)
    }

    pub fn replay(&self, key: &T::Key) -> &ReplayProgram {
        self.cache
            .peek(key)
            .unwrap_or_else(|| panic!("{} replay cache missing recorded batch for key {:?}", self.name, key))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use inference_backend_metal::components::ReplayOp;
    use inference_backend_metal::components::ReplayRecorder;
    use inference_backend_metal::metal::Buffer;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Stream;
    use inference_backend_metal::operators::bf16_concat_rows;

    use super::Replay;
    use super::ReplayComponent;

    const NUM_ACTIVE_ROWS: inference_backend_metal::metal::ReplayParameterKey =
        inference_backend_metal::metal::ReplayParameterKey::new("test.replay.num_active_rows");

    #[test]
    fn test_record_is_idempotent_and_replay_is_strict() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = &stream;
        let mut replay = Replay::new("test", component(&device));

        let (key, cache_hit) = replay.record(runtime, &7);
        assert!(!cache_hit);
        assert_eq!(replay.component().records.get(), 1);
        assert_eq!(replay.record(runtime, &7), (key, true));
        assert_eq!(replay.component().records.get(), 1);
        let _ = replay.replay(&key);

        replay.clear();
        replay.cache.set_capacity(3);
        for key in 0..4 {
            assert_eq!(replay.record(runtime, &key), (key, false));
            let _ = replay.replay(&key);
        }
        assert_eq!(replay.cache.len(), 3);
        let evicted_key = (0..4).find(|key| replay.cache.peek(key).is_none()).unwrap();
        let resident_keys = replay.cache.iter().map(|(key, _)| *key).collect::<Vec<_>>();
        // Warm resident entries before revisiting an evicted key. The rebuilt program
        // must remain available even when the older programs have higher counts.
        for key in resident_keys {
            assert_eq!(replay.record(runtime, &key), (key, true));
            assert_eq!(replay.record(runtime, &key), (key, true));
        }
        assert_eq!(replay.record(runtime, &evicted_key), (evicted_key, false));
        let _ = replay.replay(&evicted_key);
        assert_eq!(replay.cache.len(), 3);
        assert_eq!(replay.component().records.get(), 6);

        replay.clear();
        assert!(replay.cache.is_empty());
        assert_eq!(replay.record(runtime, &evicted_key), (evicted_key, false));
        assert_eq!(replay.component().records.get(), 7);
    }

    #[test]
    #[should_panic(expected = "replay cache missing recorded batch")]
    fn test_replay_panics_before_record() {
        let device = Device::system_default();
        let replay = Replay::new("test", component(&device));
        let _ = replay.replay(&1);
    }

    struct CountedComponent {
        records: Cell<usize>,
        kernel: bf16_concat_rows::Kernel,
        lhs: Buffer,
        rhs: Buffer,
        output: Buffer,
    }

    impl ReplayComponent for CountedComponent {
        type Key = u32;
        type Input<'a> = u32;

        fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
            *input
        }

        fn record<'a>(&'a self, recorder: &mut ReplayRecorder, _input: &Self::Input<'a>) {
            self.records.set(self.records.get() + 1);
            recorder.record(ReplayOp::opaque(self.kernel.invoke(
                1,
                inference_backend_metal::metal::ReplayU32::Parameter(NUM_ACTIVE_ROWS),
                bf16_concat_rows::Buffers {
                    lhs: &self.lhs,
                    rhs: &self.rhs,
                    output: &self.output,
                },
            )));
        }
    }

    fn component(device: &Device) -> CountedComponent {
        CountedComponent {
            records: Cell::new(0),
            kernel: bf16_concat_rows::Kernel::new(device, bf16_concat_rows::Config { num_columns: 4 }),
            lhs: Buffer::new_zeroed_elements(device, 4, inference_backend_metal::metal::Dtype::Bfloat16),
            rhs: Buffer::new_zeroed_elements(device, 4, inference_backend_metal::metal::Dtype::Bfloat16),
            output: Buffer::new_zeroed_elements(device, 8, inference_backend_metal::metal::Dtype::Bfloat16),
        }
    }
}
