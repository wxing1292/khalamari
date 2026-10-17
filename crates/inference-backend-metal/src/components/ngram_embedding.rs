use crate::components::assert_u32_index_domain;
use crate::components::checked_product;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("metal/ngram_embedding.metal");
const THREADS: usize = 256;
const RESIDENT_GROUPS: usize = 8;
const RESIDENT_THREADS: usize = 160;

/// Stable identity of the lookup dispatch recorded for one table layout and capacity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelKind {
    Sharded,
    ResidentH1,
    ResidentH4,
}

impl KernelKind {
    fn heads_per_threadblock(self) -> u32 {
        match self {
            Self::ResidentH1 => 1,
            Self::ResidentH4 => 4,
            Self::Sharded => unreachable!("sharded lookup uses a flat value grid"),
        }
    }
}

struct Selector;

impl Selector {
    const H1_CAPACITY: u32 = 256;

    fn select(config: Config, num_total_tokens: u32, shards: &[Shard<'_>]) -> KernelKind {
        if !config.supports_resident_lookup()
            || shards.len() > RESIDENT_GROUPS
            || !shards.iter().all(|shard| {
                matches!(
                    shard.weights,
                    TableWeights::Affine {
                        bits: 4,
                        group_size: 32,
                        ..
                    }
                )
            })
        {
            KernelKind::Sharded
        } else if num_total_tokens <= Self::H1_CAPACITY {
            KernelKind::ResidentH1
        } else {
            KernelKind::ResidentH4
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub ngram_size: u32,
    pub heads_per_ngram: u32,
    pub embedding_dim: u32,
    pub eos_token_id: u32,
}

impl Config {
    pub fn validate(self) {
        assert!(self.ngram_size >= 2 && self.heads_per_ngram > 0);
        let heads = (self.ngram_size - 1)
            .checked_mul(self.heads_per_ngram)
            .expect("n-gram heads must fit u32");
        assert!(self.embedding_dim > 0 && self.embedding_dim.is_multiple_of(heads));
    }
    fn num_heads(self) -> u32 {
        (self.ngram_size - 1) * self.heads_per_ngram
    }
    fn head_dim(self) -> u32 {
        self.embedding_dim / self.num_heads()
    }

    fn supports_resident_lookup(self) -> bool {
        self.num_heads() == 16 && self.head_dim() == 160
    }
}

/// Each query reads its own request's source history and contiguous input segment.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Token {
    pub batch_start: u32,
    pub history_slot: u32,
}

#[derive(Clone, Copy)]
pub struct HashBuffers<'a> {
    pub token_ids: &'a Buffer,
    pub previous_tokens: &'a Buffer,
    pub tokens: &'a Buffer,
    pub multipliers: &'a Buffer,
    pub head_vocab_sizes: &'a Buffer,
    pub head_offsets: &'a Buffer,
    pub row_ids: &'a Buffer,
}

#[derive(Clone, Copy)]
pub enum TableWeights<'a> {
    Bf16(&'a Buffer),
    Affine {
        bits: u32,
        group_size: u32,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
    },
}

#[derive(Clone, Copy)]
pub struct Shard<'a> {
    pub row_start: u32,
    pub num_rows: u32,
    pub weights: TableWeights<'a>,
}

pub struct Compute {
    config: Config,
    hash: CompiledKernel,
    lookup_bf16: CompiledKernel,
    lookup_affine: CompiledKernel,
    resident: Vec<(KernelKind, CompiledKernel)>,
}

impl Compute {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        let source = format!(
            "{}\nconstant uint ngram_size={}u, heads_per_ngram={}u, num_heads={}u, head_dim={}u, \
             eos_token_id={}u;\n{SOURCE}",
            include_str!("metal/quantized_embedding.metal"),
            config.ngram_size,
            config.heads_per_ngram,
            config.num_heads(),
            config.head_dim(),
            config.eos_token_id
        );
        let resident = if config.supports_resident_lookup() {
            [KernelKind::ResidentH1, KernelKind::ResidentH4]
                .into_iter()
                .map(|kind| {
                    let source = format!(
                        "#include <metal_stdlib>\nusing namespace metal;\nconstant uint heads_per_threadblock={}u;\n{}",
                        kind.heads_per_threadblock(),
                        include_str!("metal/ngram_embedding_resident.metal"),
                    );
                    (kind, CompiledKernel::new(device, &source, "ngram_lookup_resident"))
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            config,
            hash: CompiledKernel::new(device, &source, "ngram_hash"),
            lookup_bf16: CompiledKernel::new(device, &source, "ngram_lookup_bf16"),
            lookup_affine: CompiledKernel::new(device, &source, "ngram_lookup_affine"),
            resident,
        }
    }

    /// Table layout and total capacity are record-time facts. Active count does not select a kernel.
    pub fn lookup_topology(&self, num_total_tokens: u32, shards: &[Shard<'_>]) -> KernelKind {
        assert!(num_total_tokens > 0 && !shards.is_empty());
        Selector::select(self.config, num_total_tokens, shards)
    }

    /// Geometry-level boundaries. Table format can remove a resident path, but cannot add another boundary.
    pub fn lookup_token_topology_boundaries(&self) -> Box<[u32]> {
        if self.config.supports_resident_lookup() {
            Box::new([Selector::H1_CAPACITY + 1])
        } else {
            Box::new([])
        }
    }

    /// First capacity where the recorded lookup topology changes for this table layout.
    pub fn lookup_topology_boundaries(&self, shards: &[Shard<'_>]) -> Box<[u32]> {
        assert!(!shards.is_empty());
        if Selector::select(self.config, 1, shards) == KernelKind::Sharded {
            Box::new([])
        } else {
            Box::new([Selector::H1_CAPACITY + 1])
        }
    }

    pub fn hash<'a>(
        &'a self,
        num_total_tokens: u32,
        num_history_slots: u32,
        num_active_tokens: ReplayU32,
        buffers: HashBuffers<'a>,
    ) -> HashInvocation<'a> {
        let config = self.config;
        validate_active(num_total_tokens, num_active_tokens);
        assert_u32_index_domain(
            checked_product(
                "n-gram hash values",
                &[num_total_tokens as usize, config.num_heads() as usize],
            ),
            "n-gram hash values",
        );
        assert!(num_history_slots > 0);
        assert!(buffers.tokens.len_bytes() >= bytes(num_total_tokens, size_of::<Token>()));
        assert!(buffers.token_ids.len_bytes() >= bytes(num_total_tokens, 4));
        assert!(buffers.previous_tokens.len_bytes() >= bytes(num_history_slots, (config.ngram_size - 1) as usize * 4));
        assert!(buffers.multipliers.len_bytes() >= config.ngram_size as usize * 8);
        assert!(buffers.head_vocab_sizes.len_bytes() >= config.num_heads() as usize * 4);
        assert!(buffers.head_offsets.len_bytes() >= config.num_heads() as usize * 4);
        assert!(buffers.row_ids.len_bytes() >= bytes(num_total_tokens, config.num_heads() as usize * 4));
        HashInvocation {
            compute: self,
            num_total_tokens,
            num_active_tokens,
            buffers,
        }
    }

    pub fn lookup<'a>(
        &'a self,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        ids: &'a Buffer,
        output: &'a Buffer,
        scale: f32,
        shards: &[Shard<'a>],
    ) -> LookupInvocation<'a> {
        validate_active(num_total_tokens, num_active_tokens);
        assert_u32_index_domain(
            checked_product(
                "n-gram output values",
                &[num_total_tokens as usize, self.config.embedding_dim as usize],
            ),
            "n-gram output values",
        );
        assert!(scale.is_finite());
        assert!(!shards.is_empty());
        let dim = self.config.head_dim() as usize;
        assert!(ids.len_bytes() >= bytes(num_total_tokens, self.config.num_heads() as usize * 4));
        assert!(output.len_bytes() >= bytes(num_total_tokens, self.config.embedding_dim as usize * 2));
        let mut end = 0_u32;
        for shard in shards {
            assert_eq!(
                shard.row_start, end,
                "n-gram shards must partition the global row interval"
            );
            assert!(shard.num_rows > 0);
            end = end
                .checked_add(shard.num_rows)
                .expect("n-gram shard interval must fit u32");
            match shard.weights {
                TableWeights::Bf16(weight) => assert!(weight.len_bytes() >= bytes(shard.num_rows, dim * 2)),
                TableWeights::Affine {
                    bits,
                    group_size,
                    weight,
                    scales,
                    biases,
                } => {
                    assert!(matches!(bits, 2 | 3 | 4 | 6 | 8));
                    assert!(matches!(group_size, 32 | 64 | 128) && dim.is_multiple_of(group_size as usize));
                    assert!(weight.len_bytes() >= bytes(shard.num_rows, dim * bits as usize / 8));
                    let affine_bytes = bytes(shard.num_rows, dim / group_size as usize * 2);
                    assert!(scales.len_bytes() >= affine_bytes && biases.len_bytes() >= affine_bytes);
                },
            }
        }
        LookupInvocation {
            compute: self,
            kind: Selector::select(self.config, num_total_tokens, shards),
            num_total_tokens,
            num_active_tokens,
            ids,
            output,
            scale,
            shards: shards.to_vec(),
        }
    }
}

pub struct HashInvocation<'a> {
    compute: &'a Compute,
    num_total_tokens: u32,
    num_active_tokens: ReplayU32,
    buffers: HashBuffers<'a>,
}
impl Operator for HashInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.hash);
        for (index, buffer) in [
            self.buffers.token_ids,
            self.buffers.previous_tokens,
            self.buffers.tokens,
            self.buffers.multipliers,
            self.buffers.head_vocab_sizes,
            self.buffers.head_offsets,
        ]
        .into_iter()
        .enumerate()
        {
            recorder.set_buffer_read(index, buffer, 0);
        }
        recorder.set_buffer_write(6, self.buffers.row_ids, 0);
        bind_active(recorder, 7, self.num_active_tokens, self.num_total_tokens);
        let work = u64::from(self.num_total_tokens) * u64::from(self.compute.config.num_heads());
        recorder.dispatch_threadblocks((work.div_ceil(THREADS as u64) as usize, 1, 1), (THREADS, 1, 1));
    }
}

pub struct LookupInvocation<'a> {
    compute: &'a Compute,
    kind: KernelKind,
    num_total_tokens: u32,
    num_active_tokens: ReplayU32,
    ids: &'a Buffer,
    output: &'a Buffer,
    scale: f32,
    shards: Vec<Shard<'a>>,
}
impl Operator for LookupInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        if self.kind != KernelKind::Sharded {
            self.record_resident(recorder);
            return;
        }
        // Shard intervals are disjoint. Every selected ID has exactly one writer.
        recorder.record_disjoint_buffers(&[self.output], || {
            for shard in &self.shards {
                let weight = match shard.weights {
                    TableWeights::Bf16(weight) => {
                        recorder.set_kernel(&self.compute.lookup_bf16);
                        weight
                    },
                    TableWeights::Affine {
                        bits,
                        group_size,
                        weight,
                        scales,
                        biases,
                    } => {
                        recorder.set_kernel(&self.compute.lookup_affine);
                        recorder.set_buffer_read(7, scales, 0);
                        recorder.set_buffer_read(8, biases, 0);
                        recorder.set_u32(9, bits);
                        recorder.set_u32(10, group_size);
                        weight
                    },
                };
                recorder.set_buffer_read(0, self.ids, 0);
                recorder.set_buffer_read(1, weight, 0);
                recorder.set_buffer_write(2, self.output, 0);
                bind_active(recorder, 3, self.num_active_tokens, self.num_total_tokens);
                recorder.set_u32(4, shard.row_start);
                recorder.set_u32(5, shard.num_rows);
                recorder.set_f32(6, self.scale);
                let work = u64::from(self.num_total_tokens) * u64::from(self.compute.config.embedding_dim);
                recorder.dispatch_threadblocks((work.div_ceil(THREADS as u64) as usize, 1, 1), (THREADS, 1, 1));
            }
        });
    }
}

impl LookupInvocation<'_> {
    fn record_resident(self, recorder: &CommandRecorder<'_>) {
        let kernel = &self
            .compute
            .resident
            .iter()
            .find(|(kind, _)| *kind == self.kind)
            .expect("missing resident n-gram lookup variant")
            .1;
        recorder.set_kernel(kernel);
        // Unused bindings alias the final group. Repeated terminal starts make them unreachable.
        let last = self.shards.last().unwrap();
        let mut starts = [last.row_start + last.num_rows; RESIDENT_GROUPS + 1];
        for (group, start) in starts[..RESIDENT_GROUPS].iter_mut().enumerate() {
            let shard = &self.shards[group.min(self.shards.len() - 1)];
            if group < self.shards.len() {
                *start = shard.row_start;
            }
            let TableWeights::Affine {
                weight, scales, biases, ..
            } = shard.weights
            else {
                unreachable!("resident lookup requires affine Q4 groups")
            };
            recorder.set_buffer_read(group, weight, 0);
            recorder.set_buffer_read(8 + group, scales, 0);
            recorder.set_buffer_read(16 + group, biases, 0);
        }
        recorder.set_buffer_read(24, self.ids, 0);
        recorder.set_u32_slice(25, &starts);
        recorder.set_buffer_write(26, self.output, 0);
        recorder.set_f32(27, self.scale);
        bind_active(recorder, 28, self.num_active_tokens, self.num_total_tokens);
        recorder.dispatch_threadblocks(
            (
                self.num_total_tokens as usize,
                (16 / self.kind.heads_per_threadblock()) as usize,
                1,
            ),
            (RESIDENT_THREADS, 1, 1),
        );
    }
}

fn validate_active(capacity: u32, num_active_tokens: ReplayU32) {
    assert!(capacity > 0);
    if let ReplayU32::Fixed(value) = num_active_tokens {
        assert!(value > 0 && value <= capacity);
    }
}
fn bind_active(recorder: &CommandRecorder<'_>, index: usize, num_active_tokens: ReplayU32, capacity: u32) {
    match num_active_tokens {
        ReplayU32::Fixed(value) => recorder.set_u32(index, value),
        ReplayU32::Parameter(key) => recorder.bind_u32(index, key, 1, capacity),
    }
}
fn bytes(rows: u32, width: usize) -> usize {
    checked_product("n-gram buffer bytes", &[rows as usize, width])
}

#[cfg(test)]
#[path = "ngram_embedding_test.rs"]
mod tests;
