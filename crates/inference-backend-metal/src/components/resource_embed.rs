use std::mem::size_of;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("metal/resource_embed.metal");
const NUM_U32S_PER_MAPPING: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KernelConstants {
    thread_block: ThreadBlockConstants,
}

impl KernelConstants {
    fn current() -> Self {
        Self {
            thread_block: ThreadBlockConstants { required_threads: 256 },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub hidden_dim: u32,
    pub io_dtype: Dtype,
}

impl Config {
    pub fn validate(self) {
        assert!(self.hidden_dim > 0);
        match self.io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 resource embedding is not implemented"),
            dtype => panic!("unsupported resource embedding IO dtype {dtype:?}"),
        }
    }

    pub fn hidden_dim_bytes(self) -> u64 {
        self.validate();
        u64::from(self.hidden_dim) * self.io_dtype.item_size() as u64
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Shape {
    pub num_total_mappings: u32,
}

impl Shape {
    pub fn validate(self, config: Config) {
        config.validate();
        assert!(self.num_total_mappings > 0);
        let _ = self
            .num_total_mappings
            .checked_mul(config.hidden_dim)
            .expect("resource embed grid value count must fit the shader u32 domain");
        let _ = self
            .num_total_mappings
            .checked_mul(NUM_U32S_PER_MAPPING as u32)
            .expect("resource embed mapping index must fit the shader u32 domain");
    }

    fn mapping_bytes(self) -> u64 {
        u64::from(self.num_total_mappings) * NUM_U32S_PER_MAPPING as u64 * size_of::<u32>() as u64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mapping {
    /// Row in the destination hidden buffer.
    pub destination_row: u32,
    /// Byte offset from the start of the resource arena.
    pub source_offset_bytes: u64,
}

pub struct MappingTable {
    config: Config,
    shape: Shape,
    required_resource_arena_bytes: u64,
    required_hidden_bytes: u64,
    // Each mapping is [destination_row, source_offset_low, source_offset_high].
    encoded_u32s: Vec<u32>,
}

impl MappingTable {
    pub fn new(config: Config, mut mappings: Vec<Mapping>) -> Self {
        let num_total_mappings =
            u32::try_from(mappings.len()).expect("resource embed mapping count must fit the shader u32 domain");
        let shape = Shape { num_total_mappings };
        shape.validate(config);

        // The active mapping count selects a prefix. Canonical row order makes that prefix deterministic.
        mappings.sort_unstable_by_key(|mapping| mapping.destination_row);
        let hidden_dim_bytes = u64::from(config.hidden_dim) * config.io_dtype.item_size() as u64;
        let item_size = config.io_dtype.item_size() as u64;
        let mut previous_destination_row = None;
        let mut required_resource_arena_bytes = 0;
        let mut required_hidden_bytes = 0;
        let mut encoded_u32s = Vec::with_capacity(mappings.len() * NUM_U32S_PER_MAPPING);
        for mapping in mappings {
            assert_ne!(
                previous_destination_row,
                Some(mapping.destination_row),
                "resource embed destination rows must be unique"
            );
            assert_eq!(
                mapping.source_offset_bytes % item_size,
                0,
                "resource embed source byte offset must align to its dtype"
            );
            let source_end_bytes = mapping
                .source_offset_bytes
                .checked_add(hidden_dim_bytes)
                .expect("resource embed source byte range must fit u64");
            let hidden_end_bytes = (u64::from(mapping.destination_row) + 1)
                .checked_mul(hidden_dim_bytes)
                .expect("resource embed destination byte range must fit u64");

            previous_destination_row = Some(mapping.destination_row);
            required_resource_arena_bytes = required_resource_arena_bytes.max(source_end_bytes);
            required_hidden_bytes = required_hidden_bytes.max(hidden_end_bytes);
            encoded_u32s.push(mapping.destination_row);
            encoded_u32s.push(mapping.source_offset_bytes as u32);
            encoded_u32s.push((mapping.source_offset_bytes >> 32) as u32);
        }
        Self {
            config,
            shape,
            required_resource_arena_bytes,
            required_hidden_bytes,
            encoded_u32s,
        }
    }

    pub const fn shape(&self) -> Shape {
        self.shape
    }

    /// Returns the Metal mapping ABI as consecutive `u32` values.
    pub fn encoded_u32s(&self) -> &[u32] {
        &self.encoded_u32s
    }
}

#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    pub resource_arena: &'a Buffer,
    pub mappings: &'a Buffer,
    pub hidden: &'a Buffer,
}

pub struct Compute {
    config: Config,
    constants: KernelConstants,
    kernel: CompiledKernel,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            constants: KernelConstants::current(),
            kernel: CompiledKernel::new(device, SOURCE, "resource_embed_bf16"),
        }
    }

    pub fn invoke<'a>(
        &'a self,
        mappings: &MappingTable,
        num_active_mappings: ReplayU32,
        buffers: Buffers<'a>,
    ) -> Invocation<'a> {
        assert_eq!(
            mappings.config, self.config,
            "resource embed mapping table configuration must match compute"
        );
        let invocation = Invocation {
            compute: self,
            shape: mappings.shape,
            required_resource_arena_bytes: mappings.required_resource_arena_bytes,
            required_hidden_bytes: mappings.required_hidden_bytes,
            num_active_mappings,
            buffers,
        };
        invocation.validate();
        invocation
    }
}

pub struct Invocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    required_resource_arena_bytes: u64,
    required_hidden_bytes: u64,
    num_active_mappings: ReplayU32,
    buffers: Buffers<'a>,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.kernel);
        recorder.set_buffer_read(0, self.buffers.resource_arena, 0);
        recorder.set_buffer_read(1, self.buffers.mappings, 0);
        recorder.set_buffer_read_write(2, self.buffers.hidden, 0);
        match self.num_active_mappings {
            ReplayU32::Fixed(value) => recorder.set_u32(3, value),
            ReplayU32::Parameter(key) => recorder.bind_u32(3, key, 1, self.shape.num_total_mappings),
        }
        recorder.set_u32(4, self.compute.config.hidden_dim);
        let num_total_values = self.shape.num_total_mappings * self.compute.config.hidden_dim;
        recorder.dispatch_1d(
            num_total_values as usize,
            self.compute.constants.thread_block.required_threads as usize,
        );
    }
}

impl Invocation<'_> {
    fn validate(&self) {
        self.num_active_mappings
            .validate_range(1, self.shape.num_total_mappings);
        assert!(self.buffers.resource_arena.len_bytes_u64() >= self.required_resource_arena_bytes);
        assert!(self.buffers.mappings.len_bytes_u64() >= self.shape.mapping_bytes());
        assert!(self.buffers.hidden.len_bytes_u64() >= self.required_hidden_bytes);
    }
}

#[cfg(test)]
#[path = "resource_embed_test.rs"]
mod tests;
