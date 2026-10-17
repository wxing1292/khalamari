use std::cell::RefCell;
use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLDevice;
use objc2_metal::MTLResourceOptions;

use crate::metal::record_gpu_buffer_alloc;
use crate::metal::stream::PARAMETER_BUFFER_ALIGNMENT;
use crate::metal::stream::TrackedGpuAllocation;

/// Stable lookup key for one submission-time replay value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayParameterKey(&'static str);

impl ReplayParameterKey {
    pub const fn new(name: &'static str) -> Self {
        assert!(!name.is_empty(), "Metal replay parameter name must not be empty");
        Self(name)
    }
}

/// Source for one scalar kernel argument in a replay program.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayValue<T> {
    Fixed(T),
    Parameter(ReplayParameterKey),
}

/// Source for one `u32` kernel argument in a replay program.
pub type ReplayU32 = ReplayValue<u32>;

impl ReplayU32 {
    /// Validate a fixed value or the declared domain before recording.
    /// Submission validates values supplied for parameters.
    #[track_caller]
    pub fn validate_range(self, min_value: u32, max_value: u32) {
        assert!(min_value <= max_value, "Metal replay u32 domain is empty");
        if let Self::Fixed(value) = self {
            assert!(
                (min_value..=max_value).contains(&value),
                "Metal replay u32 value {value} is outside {min_value}..={max_value}"
            );
        }
    }
}

/// Source for one `u64` kernel argument in a replay program.
pub type ReplayU64 = ReplayValue<u64>;

/// Source for one `i32` kernel argument in a replay program.
pub type ReplayI32 = ReplayValue<i32>;

/// Source for one `i64` kernel argument in a replay program.
pub type ReplayI64 = ReplayValue<i64>;

/// Source for one `f32` kernel argument in a replay program.
pub type ReplayF32 = ReplayValue<f32>;

/// Submission values keyed by the replay parameter table declared while recording.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct ReplayArguments {
    values: HashMap<ReplayParameterKey, ReplayArgumentValue>,
}

impl ReplayArguments {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_u32(&mut self, key: ReplayParameterKey, value: u32) {
        self.insert(key, ReplayArgumentValue::U32(value));
    }

    pub fn with_u32(mut self, key: ReplayParameterKey, value: u32) -> Self {
        self.set_u32(key, value);
        self
    }

    pub fn set_u64(&mut self, key: ReplayParameterKey, value: u64) {
        self.insert(key, ReplayArgumentValue::U64(value));
    }

    pub fn with_u64(mut self, key: ReplayParameterKey, value: u64) -> Self {
        self.set_u64(key, value);
        self
    }

    pub fn set_i32(&mut self, key: ReplayParameterKey, value: i32) {
        self.insert(key, ReplayArgumentValue::I32(value));
    }

    pub fn with_i32(mut self, key: ReplayParameterKey, value: i32) -> Self {
        self.set_i32(key, value);
        self
    }

    pub fn set_i64(&mut self, key: ReplayParameterKey, value: i64) {
        self.insert(key, ReplayArgumentValue::I64(value));
    }

    pub fn with_i64(mut self, key: ReplayParameterKey, value: i64) -> Self {
        self.set_i64(key, value);
        self
    }

    pub fn set_f32(&mut self, key: ReplayParameterKey, value: f32) {
        self.insert(key, ReplayArgumentValue::F32(value.to_bits()));
    }

    pub fn with_f32(mut self, key: ReplayParameterKey, value: f32) -> Self {
        self.set_f32(key, value);
        self
    }

    fn insert(&mut self, key: ReplayParameterKey, value: ReplayArgumentValue) {
        let collision = self.values.insert(key, value);
        assert!(collision.is_none(), "Metal replay argument {:?} was set twice", key);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayArgumentValue {
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(u32),
}

#[derive(Debug)]
pub struct ReplayParameterTable {
    entries: HashMap<ReplayParameterKey, ReplayParameterEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReplayParameterEntry {
    offset_bytes: usize,
    domain: ReplayParameterDomain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayParameterDomain {
    U32 { min_value: u32, max_value: u32 },
    U64 { min_value: u64, max_value: u64 },
    I32 { min_value: i32, max_value: i32 },
    I64 { min_value: i64, max_value: i64 },
    F32 { min_bits: u32, max_bits: u32 },
}

impl ReplayParameterDomain {
    fn contains(self, value: ReplayArgumentValue) -> bool {
        match (self, value) {
            (Self::U32 { min_value, max_value }, ReplayArgumentValue::U32(value)) => {
                value >= min_value && value <= max_value
            },
            (Self::U64 { min_value, max_value }, ReplayArgumentValue::U64(value)) => {
                value >= min_value && value <= max_value
            },
            (Self::I32 { min_value, max_value }, ReplayArgumentValue::I32(value)) => {
                value >= min_value && value <= max_value
            },
            (Self::I64 { min_value, max_value }, ReplayArgumentValue::I64(value)) => {
                value >= min_value && value <= max_value
            },
            (Self::F32 { min_bits, max_bits }, ReplayArgumentValue::F32(value_bits)) => {
                let value = f32::from_bits(value_bits);
                value >= f32::from_bits(min_bits) && value <= f32::from_bits(max_bits)
            },
            _ => false,
        }
    }
}

impl ReplayParameterTable {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn validate(&self, arguments: &ReplayArguments) {
        assert_eq!(
            arguments.values.len(),
            self.entries.len(),
            "Metal replay submission must provide every declared parameter exactly once"
        );
        for (&key, entry) in &self.entries {
            let value = arguments
                .values
                .get(&key)
                .copied()
                .unwrap_or_else(|| panic!("Metal replay submission is missing parameter {:?}", key));
            assert!(
                entry.domain.contains(value),
                "Metal replay parameter {key:?} value {value:?} has the wrong type or is outside domain {:?}",
                entry.domain
            );
        }
    }

    pub fn write(&self, buffer: &ProtocolObject<dyn MTLBuffer>, arguments: &ReplayArguments) {
        for (&key, entry) in &self.entries {
            write_argument_value(buffer, entry.offset_bytes, arguments.values[&key]);
        }
    }
}

/// Build-time host packer for fixed kernel arguments and dynamic replay parameters.
#[derive(Debug, Default)]
pub struct CommandParameterLayoutBuilder {
    bytes: RefCell<Vec<u8>>,
    replay_entries: RefCell<HashMap<ReplayParameterKey, ReplayParameterEntry>>,
}

#[derive(Debug)]
pub struct CommandParameterLayout {
    pub bytes: Vec<u8>,
    pub replay_parameter_table: ReplayParameterTable,
}

impl CommandParameterLayoutBuilder {
    pub fn build(self) -> CommandParameterLayout {
        CommandParameterLayout {
            bytes: self.bytes.into_inner(),
            replay_parameter_table: ReplayParameterTable {
                entries: self.replay_entries.into_inner(),
            },
        }
    }

    pub fn push_bytes<T>(&self, values: &[T]) -> usize {
        let len_bytes = std::mem::size_of_val(values);
        assert!(len_bytes > 0);
        let mut bytes = self.bytes.borrow_mut();
        let offset = align_up(bytes.len(), PARAMETER_BUFFER_ALIGNMENT);
        let end = offset
            .checked_add(len_bytes)
            .expect("Metal command parameter byte length overflow");
        bytes.resize(end, 0);
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), bytes.as_mut_ptr().add(offset), len_bytes);
        }
        offset
    }

    pub fn bind_u32(&self, key: ReplayParameterKey, min_value: u32, max_value: u32) -> usize {
        assert!(min_value <= max_value, "Metal replay parameter domain is empty");
        self.bind(key, ReplayParameterDomain::U32 { min_value, max_value }, 0_u32)
    }

    pub fn bind_u64(&self, key: ReplayParameterKey, min_value: u64, max_value: u64) -> usize {
        assert!(min_value <= max_value, "Metal replay parameter domain is empty");
        self.bind(key, ReplayParameterDomain::U64 { min_value, max_value }, 0_u64)
    }

    pub fn bind_i32(&self, key: ReplayParameterKey, min_value: i32, max_value: i32) -> usize {
        assert!(min_value <= max_value, "Metal replay parameter domain is empty");
        self.bind(key, ReplayParameterDomain::I32 { min_value, max_value }, 0_i32)
    }

    pub fn bind_i64(&self, key: ReplayParameterKey, min_value: i64, max_value: i64) -> usize {
        assert!(min_value <= max_value, "Metal replay parameter domain is empty");
        self.bind(key, ReplayParameterDomain::I64 { min_value, max_value }, 0_i64)
    }

    pub fn bind_f32(&self, key: ReplayParameterKey, min_value: f32, max_value: f32) -> usize {
        assert!(min_value <= max_value, "Metal replay parameter domain is empty");
        self.bind(
            key,
            ReplayParameterDomain::F32 {
                min_bits: min_value.to_bits(),
                max_bits: max_value.to_bits(),
            },
            0_f32,
        )
    }

    fn bind<T>(&self, key: ReplayParameterKey, domain: ReplayParameterDomain, zero: T) -> usize {
        if let Some(entry) = self.replay_entries.borrow().get(&key).copied() {
            assert_eq!(
                entry.domain, domain,
                "Metal replay parameter {:?} has inconsistent domains",
                key
            );
            return entry.offset_bytes;
        }

        let offset_bytes = self.push_bytes(&[zero]);
        let previous = self
            .replay_entries
            .borrow_mut()
            .insert(key, ReplayParameterEntry { offset_bytes, domain });
        assert!(previous.is_none());
        offset_bytes
    }
}

fn write_argument_value(buffer: &ProtocolObject<dyn MTLBuffer>, offset_bytes: usize, value: ReplayArgumentValue) {
    match value {
        ReplayArgumentValue::U32(value) | ReplayArgumentValue::F32(value) => {
            write_argument(buffer, offset_bytes, value)
        },
        ReplayArgumentValue::U64(value) => write_argument(buffer, offset_bytes, value),
        ReplayArgumentValue::I32(value) => write_argument(buffer, offset_bytes, value),
        ReplayArgumentValue::I64(value) => write_argument(buffer, offset_bytes, value),
    }
}

fn write_argument<T>(buffer: &ProtocolObject<dyn MTLBuffer>, offset_bytes: usize, value: T) {
    assert!(
        offset_bytes + std::mem::size_of::<T>() <= buffer.length(),
        "Metal replay parameter offset exceeds parameter buffer"
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            std::ptr::from_ref(&value).cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>().add(offset_bytes),
            std::mem::size_of::<T>(),
        );
    }
}

pub fn allocate_parameter_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    allocation_kind: &'static str,
) -> Option<(Retained<ProtocolObject<dyn MTLBuffer>>, TrackedGpuAllocation)> {
    if bytes.is_empty() {
        return None;
    }

    let len_bytes = align_up(bytes.len(), PARAMETER_BUFFER_ALIGNMENT);
    let buffer = device
        .newBufferWithLength_options(
            len_bytes,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        )
        .expect("Metal parameter buffer allocation failed");
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().as_ptr().cast::<u8>(), bytes.len());
    }
    let allocation_site = record_gpu_buffer_alloc(allocation_kind, len_bytes);
    Some((buffer, TrackedGpuAllocation::new(allocation_site, len_bytes)))
}

fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::CommandParameterLayoutBuilder;
    use super::ReplayArguments;
    use super::ReplayParameterKey;

    const NUM_ACTIVE_THREADS: ReplayParameterKey = ReplayParameterKey::new("test.num_active_threads");
    const OFFSET_BYTES: ReplayParameterKey = ReplayParameterKey::new("test.offset_bytes");
    const SIGNED_INDEX: ReplayParameterKey = ReplayParameterKey::new("test.signed_index");
    const SIGNED_OFFSET: ReplayParameterKey = ReplayParameterKey::new("test.signed_offset");
    const SCALE: ReplayParameterKey = ReplayParameterKey::new("test.scale");

    #[test]
    fn test_key_reuse() {
        let builder = CommandParameterLayoutBuilder::default();
        let first = builder.bind_u32(NUM_ACTIVE_THREADS, 64, 128);
        let second = builder.bind_u32(NUM_ACTIVE_THREADS, 64, 128);
        builder.bind_u64(OFFSET_BYTES, 0, 1024);
        builder.bind_i32(SIGNED_INDEX, -8, 8);
        builder.bind_i64(SIGNED_OFFSET, -1024, 1024);
        builder.bind_f32(SCALE, 0.0, 1.0);

        assert_eq!(first, second);
        assert_eq!(builder.build().replay_parameter_table.len(), 5);
    }

    #[test]
    #[should_panic(expected = "wrong type or is outside domain")]
    fn test_argument_type_must_match_parameter_type() {
        let builder = CommandParameterLayoutBuilder::default();
        builder.bind_u64(OFFSET_BYTES, 0, 1024);
        builder
            .build()
            .replay_parameter_table
            .validate(&ReplayArguments::new().with_u32(OFFSET_BYTES, 4));
    }
}
