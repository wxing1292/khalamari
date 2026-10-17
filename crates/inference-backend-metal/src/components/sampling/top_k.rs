use super::MAX_TOP_K;
use super::SAMPLING_SOURCE;
use super::checked_bytes;
use super::checked_num_threads;
use super::checked_product;
use super::failure::SamplingFailure;
use super::failure::SamplingOperation;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;

const TOP_K_REDUCTION_LIMIT: u32 = 32;
const TOP_K_VOCAB_TILE_SIZE: u32 = 256;
pub const MAP_NUM_ACTIVE_THREADS_KEY: ReplayParameterKey =
    ReplayParameterKey::new("top_k_sampling.tile_num_active_threads");
const REDUCE_NUM_ACTIVE_THREADS_KEY: ReplayParameterKey =
    ReplayParameterKey::new("top_k_sampling.merge_num_active_threads");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PartialCandidateLayout {
    vocab_partition_size: u32,
}

impl PartialCandidateLayout {
    pub fn new(vocab_partition_size: u32) -> Self {
        assert!(vocab_partition_size > 0);
        Self { vocab_partition_size }
    }

    pub fn vocab_partition_size(self) -> u32 {
        self.vocab_partition_size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MapThreadBlockConstants {
    max_vocab_tokens: u32,
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MapAlgorithm {
    Reduction,
    Bitonic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MapKernelConstants {
    logits_dtype: Dtype,
    algorithm: MapAlgorithm,
    thread_block: MapThreadBlockConstants,
}

impl MapKernelConstants {
    fn new(logits_dtype: Dtype, algorithm: MapAlgorithm) -> Self {
        assert!(matches!(logits_dtype, Dtype::Float32 | Dtype::Bfloat16));
        Self {
            logits_dtype,
            algorithm,
            thread_block: MapThreadBlockConstants::current(),
        }
    }

    fn partial_candidate_layout(self) -> PartialCandidateLayout {
        PartialCandidateLayout::new(self.thread_block.max_vocab_tokens)
    }
}

impl MapThreadBlockConstants {
    fn current() -> Self {
        Self {
            max_vocab_tokens: TOP_K_VOCAB_TILE_SIZE,
            required_threads: 256,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReduceThreadBlockConstants {
    required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReduceKernelConstants {
    thread_block: ReduceThreadBlockConstants,
}

impl ReduceKernelConstants {
    fn current() -> Self {
        Self {
            thread_block: ReduceThreadBlockConstants { required_threads: 256 },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub num_total_sampling_inputs: u32,
    pub vocab_size: u32,
    pub top_k: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
    Merge,
    Sample,
    WriteDistribution,
    SampleAndWriteDistribution,
}

impl Shape {
    pub fn validate(self) {
        assert!(self.num_total_sampling_inputs > 0);
        assert!(self.vocab_size > 0);
        assert!(self.top_k > 0);
        assert!(self.top_k <= self.vocab_size);
        assert!(self.top_k <= MAX_TOP_K);
        assert!(
            i32::try_from(self.vocab_size).is_ok(),
            "Metal top-k vocab index must fit i32"
        );
        checked_product(
            "Metal top-k logits element count",
            &[self.num_total_sampling_inputs as usize, self.vocab_size as usize],
        );
        checked_product(
            "Metal top-k partial candidate count",
            &[
                self.num_total_sampling_inputs as usize,
                self.vocab_size
                    .div_ceil(standard_partial_candidate_layout().vocab_partition_size()) as usize,
                self.top_k as usize,
            ],
        );
    }
}

fn standard_partial_candidate_layout() -> PartialCandidateLayout {
    PartialCandidateLayout::new(MapThreadBlockConstants::current().max_vocab_tokens)
}

fn num_vocab_partitions(shape: Shape, layout: PartialCandidateLayout) -> u32 {
    shape.vocab_size.div_ceil(layout.vocab_partition_size())
}

fn partial_candidate_count(shape: Shape, layout: PartialCandidateLayout) -> usize {
    checked_product(
        "Metal top-k partial candidate count",
        &[
            shape.num_total_sampling_inputs as usize,
            num_vocab_partitions(shape, layout) as usize,
            shape.top_k as usize,
        ],
    )
}

#[derive(Clone, Copy)]
pub struct MapBuffers<'a> {
    pub logits: &'a Buffer,
    pub logits_offset_bytes: usize,
    pub tile_token_ids: &'a Buffer,
    pub tile_logits: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct SampleBuffers<'a> {
    pub tile_token_ids: &'a Buffer,
    pub tile_logits: &'a Buffer,
    pub token_ids: &'a Buffer,
    pub token_probs: &'a Buffer,
    pub params: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub sample_positions: &'a Buffer,
    pub sample_position_increment: u32,
    pub sampling_domain: u32,
}

#[derive(Clone, Copy)]
pub struct MergeBuffers<'a> {
    pub tile_token_ids: &'a Buffer,
    pub tile_logits: &'a Buffer,
    pub token_ids: &'a Buffer,
    pub logits: &'a Buffer,
}

#[derive(Clone, Copy)]
pub struct WriteDistributionBuffers<'a> {
    pub tile_token_ids: &'a Buffer,
    pub tile_logits: &'a Buffer,
    pub distribution_token_ids: &'a Buffer,
    pub distribution_probs: &'a Buffer,
    pub params: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub output_distribution_indices: &'a Buffer,
    pub max_k: u32,
    pub num_output_distributions: u32,
}

#[derive(Clone, Copy)]
pub struct SampleAndWriteDistributionBuffers<'a> {
    pub tile_token_ids: &'a Buffer,
    pub tile_logits: &'a Buffer,
    pub sampled_token_ids: &'a Buffer,
    pub sampled_token_probs: &'a Buffer,
    pub distribution_token_ids: &'a Buffer,
    pub distribution_probs: &'a Buffer,
    pub params: &'a Buffer,
    pub req_slots: &'a Buffer,
    pub sample_positions: &'a Buffer,
    pub sample_position_increment: u32,
    pub sampling_domain: u32,
    pub output_distribution_indices: &'a Buffer,
    pub max_k: u32,
    pub num_output_distributions: u32,
}

fn assert_map_buffers_fit(
    shape: Shape,
    layout: PartialCandidateLayout,
    buffers: MapBuffers<'_>,
    logits_item_size: usize,
) {
    assert!(
        buffers.logits_offset_bytes.is_multiple_of(logits_item_size),
        "top-k logits byte offset must align to its dtype"
    );
    let logits_bytes = checked_product(
        "Metal top-k logits byte length",
        &[
            shape.num_total_sampling_inputs as usize,
            shape.vocab_size as usize,
            logits_item_size,
        ],
    );
    assert!(
        buffers.logits.len_bytes()
            >= buffers
                .logits_offset_bytes
                .checked_add(logits_bytes)
                .expect("Metal top-k logits region must fit usize"),
        "top-k logits buffer is too short for total sampling inputs"
    );
    let candidates = partial_candidate_count(shape, layout);
    assert!(
        buffers.tile_token_ids.len_bytes() >= checked_bytes("Metal top-k partial token", candidates, size_of::<i32>()),
        "top-k partial-token buffer is too short"
    );
    assert!(
        buffers.tile_logits.len_bytes() >= checked_bytes("Metal top-k partial logit", candidates, size_of::<f32>()),
        "top-k partial-logit buffer is too short"
    );
}

fn assert_reduce_inputs_fit(
    shape: Shape,
    tile_token_ids: &Buffer,
    tile_logits: &Buffer,
    params: &Buffer,
    req_slots: &Buffer,
    layout: PartialCandidateLayout,
) {
    let candidates = partial_candidate_count(shape, layout);
    assert!(
        tile_token_ids.len_bytes() >= checked_bytes("Metal top-k reduce token", candidates, size_of::<i32>()),
        "top-k partial-token buffer is too short"
    );
    assert!(
        tile_logits.len_bytes() >= checked_bytes("Metal top-k reduce logit", candidates, size_of::<f32>()),
        "top-k partial-logit buffer is too short"
    );
    assert!(
        params.len_bytes() >= 4 * size_of::<u32>(),
        "top-k sampling parameter store is empty"
    );
    assert!(
        req_slots.len_bytes()
            >= checked_bytes(
                "Metal top-k request-slot row",
                shape.num_total_sampling_inputs as usize,
                size_of::<u32>(),
            ),
        "top-k request-slot buffer is too short"
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VariantKey {
    F32Reduction,
    F32Bitonic,
    Bf16Reduction,
    Bf16Bitonic,
}

struct Variant {
    constants: MapKernelConstants,
    logits_item_size: usize,
    kernel: CompiledKernel,
}

struct Registry {
    entries: Vec<(VariantKey, Variant)>,
}

struct Selector;

impl Registry {
    fn new(device: &crate::metal::Device) -> Self {
        Self {
            entries: vec![
                (
                    VariantKey::F32Reduction,
                    Variant {
                        constants: MapKernelConstants::new(Dtype::Float32, MapAlgorithm::Reduction),
                        logits_item_size: size_of::<f32>(),
                        kernel: CompiledKernel::new(device, SAMPLING_SOURCE, "top_k_logits_tiles"),
                    },
                ),
                (
                    VariantKey::F32Bitonic,
                    Variant {
                        constants: MapKernelConstants::new(Dtype::Float32, MapAlgorithm::Bitonic),
                        logits_item_size: size_of::<f32>(),
                        kernel: CompiledKernel::new(device, SAMPLING_SOURCE, "top_k_logits_tiles_bitonic"),
                    },
                ),
                (
                    VariantKey::Bf16Reduction,
                    Variant {
                        constants: MapKernelConstants::new(Dtype::Bfloat16, MapAlgorithm::Reduction),
                        logits_item_size: size_of::<u16>(),
                        kernel: CompiledKernel::new(device, SAMPLING_SOURCE, "top_k_logits_tiles_bf16"),
                    },
                ),
                (
                    VariantKey::Bf16Bitonic,
                    Variant {
                        constants: MapKernelConstants::new(Dtype::Bfloat16, MapAlgorithm::Bitonic),
                        logits_item_size: size_of::<u16>(),
                        kernel: CompiledKernel::new(device, SAMPLING_SOURCE, "top_k_logits_tiles_bf16_bitonic"),
                    },
                ),
            ],
        }
    }

    fn get(&self, key: VariantKey) -> &Variant {
        self.entries
            .iter()
            .find_map(|(candidate_key, variant)| (*candidate_key == key).then_some(variant))
            .unwrap_or_else(|| panic!("missing top-k Map execution variant {key:?}"))
    }
}

impl Selector {
    fn select(registry: &Registry, shape: Shape, logits_dtype: Dtype, operation: Operation) -> (VariantKey, &Variant) {
        let key = Self::key(shape, logits_dtype, operation);
        (key, registry.get(key))
    }

    fn key(shape: Shape, logits_dtype: Dtype, operation: Operation) -> VariantKey {
        shape.validate();
        let algorithm = match operation {
            Operation::Merge | Operation::Sample if shape.top_k <= TOP_K_REDUCTION_LIMIT => MapAlgorithm::Reduction,
            Operation::Merge
            | Operation::Sample
            | Operation::WriteDistribution
            | Operation::SampleAndWriteDistribution => MapAlgorithm::Bitonic,
        };
        match (logits_dtype, algorithm) {
            (Dtype::Float32, MapAlgorithm::Reduction) => VariantKey::F32Reduction,
            (Dtype::Float32, MapAlgorithm::Bitonic) => VariantKey::F32Bitonic,
            (Dtype::Bfloat16, MapAlgorithm::Reduction) => VariantKey::Bf16Reduction,
            (Dtype::Bfloat16, MapAlgorithm::Bitonic) => VariantKey::Bf16Bitonic,
            (dtype, _) => panic!("unsupported top-k logits dtype {dtype:?}"),
        }
    }
}

pub struct MapCompute {
    registry: Registry,
}

impl MapCompute {
    pub fn new(device: &crate::metal::Device) -> Self {
        Self {
            registry: Registry::new(device),
        }
    }

    pub fn invoke_replay<'a>(
        &'a self,
        shape: Shape,
        logits_dtype: Dtype,
        operation: Operation,
        buffers: MapBuffers<'a>,
    ) -> MapInvocation<'a> {
        let (_, variant) = Selector::select(&self.registry, shape, logits_dtype, operation);
        let constants = variant.constants;
        let layout = constants.partial_candidate_layout();
        assert_map_buffers_fit(shape, layout, buffers, variant.logits_item_size);
        let num_partitions = num_vocab_partitions(shape, layout);
        let num_threads_per_row = checked_num_threads(num_partitions, constants.thread_block.required_threads);
        checked_num_threads(shape.num_total_sampling_inputs, num_threads_per_row);
        MapInvocation {
            variant,
            shape,
            buffers,
        }
    }

    pub fn candidate_count(&self, shape: Shape) -> usize {
        shape.validate();
        partial_candidate_count(shape, self.partial_candidate_layout())
    }

    pub fn partial_candidate_layout(&self) -> PartialCandidateLayout {
        standard_partial_candidate_layout()
    }

    /// Adds active values for the same shape used to record the replay.
    pub fn add_replay_arguments(&self, shape: Shape, num_active_sampling_inputs: u32, arguments: &mut ReplayArguments) {
        assert!(
            num_active_sampling_inputs > 0 && num_active_sampling_inputs <= shape.num_total_sampling_inputs,
            "top-k active sampling inputs must fit the recorded capacity"
        );
        let num_partitions = num_vocab_partitions(shape, self.partial_candidate_layout());
        let required_threads = MapThreadBlockConstants::current().required_threads;
        let num_threads_per_row = num_partitions * required_threads;
        let num_active_threads = checked_num_threads(num_active_sampling_inputs, num_threads_per_row);
        arguments.set_u32(MAP_NUM_ACTIVE_THREADS_KEY, num_active_threads);
    }
}

pub struct MapInvocation<'a> {
    variant: &'a Variant,
    shape: Shape,
    buffers: MapBuffers<'a>,
}

impl Operator for MapInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let constants = self.variant.constants;
        let layout = constants.partial_candidate_layout();
        let num_partitions = self.shape.vocab_size.div_ceil(layout.vocab_partition_size());
        recorder.set_kernel(&self.variant.kernel);
        recorder.set_buffer_read(0, self.buffers.logits, self.buffers.logits_offset_bytes);
        recorder.set_buffer_write(1, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_write(2, self.buffers.tile_logits, 0);
        recorder.set_u32(4, self.shape.vocab_size);
        recorder.set_u32(5, self.shape.top_k);
        recorder.set_u32(6, layout.vocab_partition_size());
        recorder.set_u32(7, num_partitions);
        let required_threads = constants.thread_block.required_threads;
        let num_threads_per_row = num_partitions * required_threads;
        let num_total_threads = self.shape.num_total_sampling_inputs * num_threads_per_row;
        recorder.bind_u32(3, MAP_NUM_ACTIVE_THREADS_KEY, num_threads_per_row, num_total_threads);
        recorder.dispatch_1d(num_total_threads as usize, required_threads as usize);
    }
}

pub struct ReduceCompute {
    constants: ReduceKernelConstants,
    merge: CompiledKernel,
    sample: CompiledKernel,
    write_distribution: CompiledKernel,
    sample_and_write_distribution: CompiledKernel,
}

impl ReduceCompute {
    pub fn new(device: &crate::metal::Device) -> Self {
        Self {
            constants: ReduceKernelConstants::current(),
            merge: CompiledKernel::new(device, SAMPLING_SOURCE, "top_k_merge_tiles"),
            sample: CompiledKernel::new(device, SAMPLING_SOURCE, "top_k_sample_tiles"),
            write_distribution: CompiledKernel::new(device, SAMPLING_SOURCE, "top_k_write_distribution_tiles"),
            sample_and_write_distribution: CompiledKernel::new(
                device,
                SAMPLING_SOURCE,
                "top_k_sample_and_write_distribution_tiles",
            ),
        }
    }

    pub fn invoke_merge<'a>(&'a self, shape: Shape, buffers: MergeBuffers<'a>) -> MergeInvocation<'a> {
        let partial_candidate_layout = standard_partial_candidate_layout();
        shape.validate();
        let candidates = partial_candidate_count(shape, partial_candidate_layout);
        assert!(
            buffers.tile_token_ids.len_bytes()
                >= checked_bytes("Metal top-k merge partial token", candidates, size_of::<i32>())
        );
        assert!(
            buffers.tile_logits.len_bytes()
                >= checked_bytes("Metal top-k merge partial logit", candidates, size_of::<f32>())
        );
        let outputs = checked_product(
            "Metal top-k merge output count",
            &[shape.num_total_sampling_inputs as usize, shape.top_k as usize],
        );
        assert!(buffers.token_ids.len_bytes() >= checked_bytes("Metal top-k merge token", outputs, size_of::<i32>()));
        assert!(buffers.logits.len_bytes() >= checked_bytes("Metal top-k merge logit", outputs, size_of::<f32>()));
        checked_num_threads(
            shape.num_total_sampling_inputs,
            self.constants.thread_block.required_threads,
        );
        MergeInvocation {
            kernel: &self.merge,
            constants: self.constants,
            partial_candidate_layout,
            shape,
            buffers,
        }
    }

    pub fn invoke_sample<'a>(&'a self, shape: Shape, buffers: SampleBuffers<'a>) -> SampleInvocation<'a> {
        self.invoke_sample_with_layout(shape, buffers, standard_partial_candidate_layout())
    }

    pub fn invoke_sample_with_layout<'a>(
        &'a self,
        shape: Shape,
        buffers: SampleBuffers<'a>,
        partial_candidate_layout: PartialCandidateLayout,
    ) -> SampleInvocation<'a> {
        shape.validate();
        assert_reduce_inputs_fit(
            shape,
            buffers.tile_token_ids,
            buffers.tile_logits,
            buffers.params,
            buffers.req_slots,
            partial_candidate_layout,
        );
        assert!(
            buffers.sample_positions.len_bytes()
                >= checked_bytes(
                    "Metal top-k sample position",
                    shape.num_total_sampling_inputs as usize,
                    size_of::<u32>(),
                ),
            "top-k sample-position buffer is too short"
        );
        assert!(
            buffers.token_ids.len_bytes()
                >= checked_bytes(
                    "Metal sampled token",
                    shape.num_total_sampling_inputs as usize,
                    size_of::<i32>(),
                ),
            "top-k sampled-token buffer is too short"
        );
        assert!(
            buffers.token_probs.len_bytes()
                >= checked_bytes(
                    "Metal sampled probability",
                    shape.num_total_sampling_inputs as usize,
                    size_of::<f32>(),
                ),
            "top-k sampled-probability buffer is too short"
        );
        checked_num_threads(
            shape.num_total_sampling_inputs,
            self.constants.thread_block.required_threads,
        );
        SampleInvocation {
            kernel: &self.sample,
            constants: self.constants,
            partial_candidate_layout,
            shape,
            buffers,
        }
    }

    pub fn invoke_write_distribution<'a>(
        &'a self,
        shape: Shape,
        buffers: WriteDistributionBuffers<'a>,
    ) -> WriteDistributionInvocation<'a> {
        self.invoke_write_distribution_with_layout(shape, buffers, standard_partial_candidate_layout())
    }

    pub fn invoke_write_distribution_with_layout<'a>(
        &'a self,
        shape: Shape,
        buffers: WriteDistributionBuffers<'a>,
        partial_candidate_layout: PartialCandidateLayout,
    ) -> WriteDistributionInvocation<'a> {
        shape.validate();
        assert_reduce_inputs_fit(
            shape,
            buffers.tile_token_ids,
            buffers.tile_logits,
            buffers.params,
            buffers.req_slots,
            partial_candidate_layout,
        );
        assert!(
            buffers.max_k >= shape.top_k,
            "top-k write-distribution slots must cover active top_k"
        );
        assert!(
            buffers.num_output_distributions > 0,
            "top-k write-distribution output requires distributions"
        );
        let output_elements = checked_product(
            "Metal top-k write-distribution output element count",
            &[buffers.num_output_distributions as usize, buffers.max_k as usize],
        );
        assert!(
            buffers.output_distribution_indices.len_bytes()
                >= checked_bytes(
                    "Metal write-distribution output index",
                    shape.num_total_sampling_inputs as usize,
                    size_of::<u32>(),
                ),
            "top-k write-distribution output-index buffer too short"
        );
        assert!(
            buffers.distribution_token_ids.len_bytes()
                >= output_elements
                    .checked_mul(size_of::<i32>())
                    .expect("Metal top-k write-distribution token bytes must fit usize"),
            "top-k write-distribution token buffer too short for declared outputs"
        );
        assert!(
            buffers.distribution_probs.len_bytes()
                >= output_elements
                    .checked_mul(size_of::<f32>())
                    .expect("Metal top-k write-distribution probability bytes must fit usize"),
            "top-k write-distribution prob buffer too short for declared outputs"
        );
        checked_num_threads(
            shape.num_total_sampling_inputs,
            self.constants.thread_block.required_threads,
        );
        WriteDistributionInvocation {
            kernel: &self.write_distribution,
            constants: self.constants,
            partial_candidate_layout,
            shape,
            buffers,
        }
    }

    pub fn invoke_sample_and_write_distribution<'a>(
        &'a self,
        shape: Shape,
        buffers: SampleAndWriteDistributionBuffers<'a>,
    ) -> SampleAndWriteDistributionInvocation<'a> {
        self.invoke_sample_and_write_distribution_with_layout(shape, buffers, standard_partial_candidate_layout())
    }

    pub fn invoke_sample_and_write_distribution_with_layout<'a>(
        &'a self,
        shape: Shape,
        buffers: SampleAndWriteDistributionBuffers<'a>,
        partial_candidate_layout: PartialCandidateLayout,
    ) -> SampleAndWriteDistributionInvocation<'a> {
        shape.validate();
        assert_reduce_inputs_fit(
            shape,
            buffers.tile_token_ids,
            buffers.tile_logits,
            buffers.params,
            buffers.req_slots,
            partial_candidate_layout,
        );
        assert!(
            buffers.sample_positions.len_bytes()
                >= checked_bytes(
                    "Metal top-k sample position",
                    shape.num_total_sampling_inputs as usize,
                    size_of::<u32>(),
                ),
            "top-k sample-position buffer is too short"
        );
        assert!(
            buffers.sampled_token_ids.len_bytes()
                >= checked_bytes(
                    "Metal sampled token",
                    shape.num_total_sampling_inputs as usize,
                    size_of::<i32>(),
                ),
            "top-k sample-and-write-distribution sampled-token buffer is too short"
        );
        assert!(
            buffers.sampled_token_probs.len_bytes()
                >= checked_bytes(
                    "Metal sampled probability",
                    shape.num_total_sampling_inputs as usize,
                    size_of::<f32>(),
                ),
            "top-k sample-and-write-distribution sampled-probability buffer is too short"
        );
        assert!(
            buffers.max_k >= shape.top_k,
            "top-k sample-and-write-distribution slots must cover batch top_k"
        );
        assert!(
            buffers.num_output_distributions > 0,
            "top-k sample-and-write-distribution output requires distributions"
        );
        let output_elements = checked_product(
            "Metal top-k sample-and-write-distribution output element count",
            &[buffers.num_output_distributions as usize, buffers.max_k as usize],
        );
        assert!(
            buffers.output_distribution_indices.len_bytes()
                >= checked_bytes(
                    "Metal write-distribution output index",
                    shape.num_total_sampling_inputs as usize,
                    size_of::<u32>(),
                ),
            "top-k sample-and-write-distribution output-index buffer too short"
        );
        assert!(
            buffers.distribution_token_ids.len_bytes()
                >= output_elements
                    .checked_mul(size_of::<i32>())
                    .expect("Metal top-k sample-and-write-distribution token bytes must fit usize"),
            "top-k sample-and-write-distribution token buffer too short"
        );
        assert!(
            buffers.distribution_probs.len_bytes()
                >= output_elements
                    .checked_mul(size_of::<f32>())
                    .expect("Metal top-k sample-and-write-distribution probability bytes must fit usize"),
            "top-k sample-and-write-distribution probability buffer too short"
        );
        checked_num_threads(
            shape.num_total_sampling_inputs,
            self.constants.thread_block.required_threads,
        );
        SampleAndWriteDistributionInvocation {
            kernel: &self.sample_and_write_distribution,
            constants: self.constants,
            partial_candidate_layout,
            shape,
            buffers,
        }
    }

    /// Adds active values for the same shape used to record the replay.
    pub fn add_replay_arguments(&self, shape: Shape, num_active_sampling_inputs: u32, arguments: &mut ReplayArguments) {
        assert!(
            num_active_sampling_inputs > 0 && num_active_sampling_inputs <= shape.num_total_sampling_inputs,
            "top-k active sampling inputs must fit the recorded capacity"
        );
        let num_active_threads =
            checked_num_threads(num_active_sampling_inputs, self.constants.thread_block.required_threads);
        arguments.set_u32(REDUCE_NUM_ACTIVE_THREADS_KEY, num_active_threads);
    }
}

pub struct MergeInvocation<'a> {
    kernel: &'a CompiledKernel,
    constants: ReduceKernelConstants,
    partial_candidate_layout: PartialCandidateLayout,
    shape: Shape,
    buffers: MergeBuffers<'a>,
}

impl Operator for MergeInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let num_partitions = num_vocab_partitions(self.shape, self.partial_candidate_layout);
        recorder.set_kernel(self.kernel);
        SamplingFailure::record(
            recorder,
            self.kernel,
            SamplingOperation::Merge,
            format!("{:?}", self.shape),
        );
        recorder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.tile_logits, 0);
        recorder.set_buffer_write(2, self.buffers.token_ids, 0);
        recorder.set_buffer_write(3, self.buffers.logits, 0);
        let required_threads = self.constants.thread_block.required_threads;
        let num_total_threads = self.shape.num_total_sampling_inputs * required_threads;
        recorder.bind_u32(4, REDUCE_NUM_ACTIVE_THREADS_KEY, required_threads, num_total_threads);
        recorder.set_u32(5, self.shape.top_k);
        recorder.set_u32(6, num_partitions);
        recorder.set_u32(7, self.shape.top_k);
        recorder.set_u32(8, self.partial_candidate_layout.vocab_partition_size());
        recorder.dispatch_1d(num_total_threads as usize, required_threads as usize);
    }
}

pub struct SampleInvocation<'a> {
    kernel: &'a CompiledKernel,
    constants: ReduceKernelConstants,
    partial_candidate_layout: PartialCandidateLayout,
    shape: Shape,
    buffers: SampleBuffers<'a>,
}

impl Operator for SampleInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let num_partitions = num_vocab_partitions(self.shape, self.partial_candidate_layout);
        let num_candidates_per_partition = self.shape.top_k;
        recorder.set_kernel(self.kernel);
        SamplingFailure::record(
            recorder,
            self.kernel,
            SamplingOperation::Sample,
            format!("{:?}", self.shape),
        );
        recorder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.tile_logits, 0);
        recorder.set_buffer_write(2, self.buffers.token_ids, 0);
        recorder.set_buffer_write(3, self.buffers.token_probs, 0);
        recorder.set_buffer_read(4, self.buffers.params, 0);
        recorder.set_buffer_read(5, self.buffers.req_slots, 0);
        recorder.set_buffer_read(6, self.buffers.sample_positions, 0);
        recorder.set_u32(8, self.buffers.sample_position_increment);
        recorder.set_u32(9, self.buffers.sampling_domain);
        recorder.set_u32(10, self.shape.top_k);
        recorder.set_u32(11, num_partitions);
        recorder.set_u32(12, num_candidates_per_partition);
        recorder.set_u32(13, self.partial_candidate_layout.vocab_partition_size());
        let num_threads_per_row = self.constants.thread_block.required_threads;
        let num_total_threads = self.shape.num_total_sampling_inputs * num_threads_per_row;
        recorder.bind_u32(7, REDUCE_NUM_ACTIVE_THREADS_KEY, num_threads_per_row, num_total_threads);
        recorder.dispatch_1d(num_total_threads as usize, num_threads_per_row as usize);
    }
}

pub struct WriteDistributionInvocation<'a> {
    kernel: &'a CompiledKernel,
    constants: ReduceKernelConstants,
    partial_candidate_layout: PartialCandidateLayout,
    shape: Shape,
    buffers: WriteDistributionBuffers<'a>,
}

impl Operator for WriteDistributionInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let num_partitions = num_vocab_partitions(self.shape, self.partial_candidate_layout);
        let num_candidates_per_partition = self.shape.top_k;
        recorder.set_kernel(self.kernel);
        SamplingFailure::record(
            recorder,
            self.kernel,
            SamplingOperation::WriteDistribution,
            format!("{:?}", self.shape),
        );
        recorder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.tile_logits, 0);
        recorder.set_buffer_write(2, self.buffers.distribution_token_ids, 0);
        recorder.set_buffer_write(3, self.buffers.distribution_probs, 0);
        recorder.set_buffer_read(4, self.buffers.params, 0);
        recorder.set_buffer_read(5, self.buffers.req_slots, 0);
        recorder.set_buffer_read(6, self.buffers.output_distribution_indices, 0);
        recorder.set_u32(8, self.shape.top_k);
        recorder.set_u32(9, num_partitions);
        recorder.set_u32(10, num_candidates_per_partition);
        recorder.set_u32(11, self.partial_candidate_layout.vocab_partition_size());
        recorder.set_u32(12, self.buffers.max_k);
        recorder.set_u32(13, self.buffers.num_output_distributions);
        let num_threads_per_row = self.constants.thread_block.required_threads;
        let num_total_threads = self.shape.num_total_sampling_inputs * num_threads_per_row;
        recorder.bind_u32(7, REDUCE_NUM_ACTIVE_THREADS_KEY, num_threads_per_row, num_total_threads);
        recorder.dispatch_1d(num_total_threads as usize, num_threads_per_row as usize);
    }
}

pub struct SampleAndWriteDistributionInvocation<'a> {
    kernel: &'a CompiledKernel,
    constants: ReduceKernelConstants,
    partial_candidate_layout: PartialCandidateLayout,
    shape: Shape,
    buffers: SampleAndWriteDistributionBuffers<'a>,
}

impl Operator for SampleAndWriteDistributionInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let num_partitions = num_vocab_partitions(self.shape, self.partial_candidate_layout);
        let num_candidates_per_partition = self.shape.top_k;
        recorder.set_kernel(self.kernel);
        SamplingFailure::record(
            recorder,
            self.kernel,
            SamplingOperation::SampleAndWriteDistribution,
            format!("{:?}", self.shape),
        );
        recorder.set_buffer_read(0, self.buffers.tile_token_ids, 0);
        recorder.set_buffer_read(1, self.buffers.tile_logits, 0);
        recorder.set_buffer_write(2, self.buffers.sampled_token_ids, 0);
        recorder.set_buffer_write(3, self.buffers.sampled_token_probs, 0);
        recorder.set_buffer_write(4, self.buffers.distribution_token_ids, 0);
        recorder.set_buffer_write(5, self.buffers.distribution_probs, 0);
        recorder.set_buffer_read(6, self.buffers.params, 0);
        recorder.set_buffer_read(7, self.buffers.req_slots, 0);
        recorder.set_buffer_read(8, self.buffers.sample_positions, 0);
        recorder.set_buffer_read(9, self.buffers.output_distribution_indices, 0);
        recorder.set_u32(11, self.buffers.sample_position_increment);
        recorder.set_u32(12, self.buffers.sampling_domain);
        recorder.set_u32(13, self.shape.top_k);
        recorder.set_u32(14, num_partitions);
        recorder.set_u32(15, num_candidates_per_partition);
        recorder.set_u32(16, self.partial_candidate_layout.vocab_partition_size());
        recorder.set_u32(17, self.buffers.max_k);
        recorder.set_u32(18, self.buffers.num_output_distributions);
        let num_threads_per_row = self.constants.thread_block.required_threads;
        let num_total_threads = self.shape.num_total_sampling_inputs * num_threads_per_row;
        recorder.bind_u32(
            10,
            REDUCE_NUM_ACTIVE_THREADS_KEY,
            num_threads_per_row,
            num_total_threads,
        );
        recorder.dispatch_1d(num_total_threads as usize, num_threads_per_row as usize);
    }
}

#[cfg(test)]
#[path = "top_k_test.rs"]
mod tests;
