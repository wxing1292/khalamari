use half::bf16;
use inference_executor_core::ngram_embedding::core::HashParameters;
use inference_executor_core::ngram_embedding::core::NGramEmbeddingCore;
use inference_executor_core::ngram_embedding::reference;

use super::*;
use crate::metal::ReplayArguments;
use crate::metal::ReplayParameterKey;
use crate::metal::Stream;
use crate::test_support::ReplayTestCache;

const ACTIVE: ReplayParameterKey = ReplayParameterKey::new("test.ngram.tokens");

#[test]
fn test_hash_matches_fixed_and_random_reference() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    for (ngram_size, heads_per_ngram, eos_token_id) in [(3, 2, 0), (3, 8, 7), (5, 3, 0)] {
        let core = NGramEmbeddingCore {
            vocab_size: 32,
            eos_token_id,
            ngram_size,
            heads_per_ngram,
            embedding_dim: (ngram_size - 1) * heads_per_ngram * 32,
        };
        let compute = Compute::new(&device, config(core));
        let parameters = HashParameters::new(core, 11, 0, 1234);
        for random in [false, true] {
            let mut seed = 0x18A1_7113;
            let input: Vec<u32> = (0..259)
                .map(|i| {
                    if random {
                        next(&mut seed) % 32
                    } else if i % 7 == 2 {
                        eos_token_id
                    } else {
                        i % 32
                    }
                })
                .collect();
            let previous = vec![eos_token_id; 2 * (ngram_size as usize - 1)];
            let split = 129;
            let expected = [
                reference::row_ids(core, &parameters, &previous[..ngram_size as usize - 1], &input[..split]),
                reference::row_ids(core, &parameters, &previous[ngram_size as usize - 1..], &input[split..]),
            ]
            .concat();
            let tokens: Vec<_> = (0..input.len())
                .map(|i| {
                    if i < split {
                        Token {
                            batch_start: 0,
                            history_slot: 0,
                        }
                    } else {
                        Token {
                            batch_start: split as u32,
                            history_slot: 1,
                        }
                    }
                })
                .collect();
            // Token has an explicit two-u32 host/shader ABI.
            let tokens = tokens
                .iter()
                .flat_map(|t| [t.batch_start, t.history_slot])
                .collect::<Vec<_>>();
            let input = Buffer::from_slice(&device, &input);
            let history = Buffer::from_slice(&device, &previous);
            let tokens = Buffer::from_slice(&device, &tokens);
            let multipliers = Buffer::from_slice(&device, &parameters.multipliers);
            let sizes = Buffer::from_slice(&device, &parameters.head_vocab_sizes);
            let offsets = Buffer::from_slice(&device, &parameters.head_offsets);
            let ids = Buffer::new_zeroed(&device, expected.len() * 4);
            let mut builder = stream.create_replay_program();
            builder.record(compute.hash(
                259,
                2,
                ReplayU32::Fixed(259),
                HashBuffers {
                    token_ids: &input,
                    previous_tokens: &history,
                    tokens: &tokens,
                    multipliers: &multipliers,
                    head_vocab_sizes: &sizes,
                    head_offsets: &offsets,
                    row_ids: &ids,
                },
            ));
            stream.submit_replay(&builder.build()).wait();
            assert_eq!(ids.read_typed::<u32>(0, expected.len()), expected);
        }
    }
}

#[test]
fn test_lookup_matches_fixed_and_random_reference() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    for (heads, dim) in [(4, 32), (16, 160), (16, 256)] {
        let core = NGramEmbeddingCore {
            vocab_size: 32,
            eos_token_id: 0,
            ngram_size: 3,
            heads_per_ngram: heads / 2,
            embedding_dim: heads * dim,
        };
        let compute = Compute::new(&device, config(core));
        for (bits, group_size) in [(0, 32), (2, 32), (3, 32), (4, 32), (6, 32), (8, 32), (4, 64), (4, 128)] {
            if dim % group_size != 0 {
                continue;
            }
            for groups in [1, 2, 3, 7, 8, 9] {
                for random in [false, true] {
                    let table = fixture_table(&device, dim as usize, bits, group_size, groups, random);
                    let shards = table.shards();
                    for total in [1, 256, 257] {
                        let ids = fixture_ids(&table, total as usize * heads as usize, random);
                        let expected = reference_lookup(&table, &ids);
                        let ids = Buffer::from_slice(&device, &ids);
                        let output = Buffer::new_zeroed(&device, expected.len() * 2);
                        let mut builder = stream.create_replay_program();
                        builder.record(compute.lookup(total, ReplayU32::Fixed(total), &ids, &output, 0.5, &shards));
                        stream.submit_replay(&builder.build()).wait();
                        assert_eq!(
                            output.read_typed::<u16>(0, expected.len()),
                            expected,
                            "heads={heads}, dim={dim}, bits={bits}, groups={groups}, total={total}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_lookup_replay_reuses_capacity_and_topology() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let core = NGramEmbeddingCore {
        vocab_size: 32,
        eos_token_id: 0,
        ngram_size: 3,
        heads_per_ngram: 8,
        embedding_dim: 2560,
    };
    let compute = Compute::new(&device, config(core));
    for (bits, groups, h1, h4) in [
        (4, 8, KernelKind::ResidentH1, KernelKind::ResidentH4),
        (4, 9, KernelKind::Sharded, KernelKind::Sharded),
        (0, 2, KernelKind::Sharded, KernelKind::Sharded),
    ] {
        let table = fixture_table(&device, 160, bits, 32, groups, true);
        let shards = table.shards();
        let ids = fixture_ids(&table, 512 * 16, true);
        let expected = reference_lookup(&table, &ids);
        let ids = Buffer::from_slice(&device, &ids);
        let output = Buffer::new_zeroed(&device, expected.len() * 2);
        let boundaries: &[u32] = if groups == 8 { &[257] } else { &[] };
        assert_eq!(&*compute.lookup_topology_boundaries(&shards), boundaries);
        let mut cache = ReplayTestCache::new();
        for (total, hit, kind) in [(256, false, h1), (257, false, h4), (512, false, h4), (256, true, h1)] {
            let topology = compute.lookup_topology(total, &shards);
            assert_eq!(topology, kind);
            let (replay, cache_hit) = cache.record((total, topology), || {
                let mut builder = stream.create_replay_program();
                builder.record(compute.lookup(total, ReplayU32::Parameter(ACTIVE), &ids, &output, 0.5, &shards));
                builder.build()
            });
            assert_eq!(cache_hit, hit);
            for active in [1, total, 3, 65, 2] {
                stream
                    .submit_replay_with_arguments(replay, &ReplayArguments::new().with_u32(ACTIVE, active))
                    .wait();
                let count = active as usize * 2560;
                assert_eq!(output.read_typed::<u16>(0, count), expected[..count]);
            }
        }
    }
}

struct TableFixture {
    dim: usize,
    bits: u32,
    group_size: u32,
    starts: Vec<u32>,
    weight: Vec<Buffer>,
    scales: Vec<Buffer>,
    biases: Vec<Buffer>,
    decoded: Vec<u16>,
}

impl TableFixture {
    fn shards(&self) -> Vec<Shard<'_>> {
        self.starts
            .windows(2)
            .enumerate()
            .map(|(i, range)| {
                Shard {
                    row_start: range[0],
                    num_rows: range[1] - range[0],
                    weights: if self.bits == 0 {
                        TableWeights::Bf16(&self.weight[i])
                    } else {
                        TableWeights::Affine {
                            bits: self.bits,
                            group_size: self.group_size,
                            weight: &self.weight[i],
                            scales: &self.scales[i],
                            biases: &self.biases[i],
                        }
                    },
                }
            })
            .collect()
    }
}

fn config(core: NGramEmbeddingCore) -> Config {
    Config {
        ngram_size: core.ngram_size,
        heads_per_ngram: core.heads_per_ngram,
        embedding_dim: core.embedding_dim,
        eos_token_id: core.eos_token_id,
    }
}

fn next(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

fn fixture_ids(table: &TableFixture, count: usize, random: bool) -> Vec<u32> {
    let mut seed = 0x4182_0173;
    let rows = *table.starts.last().unwrap();
    (0..count)
        .map(|i| {
            if random {
                next(&mut seed) % rows
            } else {
                let group = i / 2 % (table.starts.len() - 1);
                if i % 2 == 0 {
                    table.starts[group]
                } else {
                    table.starts[group + 1] - 1
                }
            }
        })
        .collect()
}

fn reference_lookup(table: &TableFixture, ids: &[u32]) -> Vec<u16> {
    ids.iter()
        .flat_map(|&id| {
            (0..table.dim).map(move |d| {
                bf16::from_f32(bf16::from_bits(table.decoded[id as usize * table.dim + d]).to_f32() * 0.5).to_bits()
            })
        })
        .collect()
}

fn fixture_table(device: &Device, dim: usize, bits: u32, group_size: u32, groups: usize, random: bool) -> TableFixture {
    let rows = 89;
    let params = dim / group_size as usize;
    let mut seed = 0x1357_1731;
    let scales: Vec<_> = (0..rows * params)
        .map(|i| {
            bf16::from_f32(
                (if random {
                    next(&mut seed) as usize % 7 + 1
                } else {
                    i % 3 + 1
                }) as f32
                    / 64.0,
            )
            .to_bits()
        })
        .collect();
    let biases: Vec<_> = (0..rows * params)
        .map(|i| {
            bf16::from_f32(-((if random { next(&mut seed) as usize % 13 } else { i % 5 }) as f32) / 16.0).to_bits()
        })
        .collect();
    let mut words = vec![0_u8; rows * dim * bits as usize / 8];
    let mut decoded = Vec::new();
    for row in 0..rows {
        for d in 0..dim {
            let value = if random {
                next(&mut seed) as usize
            } else {
                row * 17 + d * 13
            };
            if bits == 0 {
                decoded.push(bf16::from_f32((value % 31) as f32 / 32.0 - 0.5).to_bits());
            } else {
                let code = (value % (1 << bits)) as u32;
                for b in 0..bits {
                    let at = (row * dim + d) * bits as usize + b as usize;
                    words[at / 8] |= (((code >> b) & 1) as u8) << (at % 8);
                }
                let affine = row * params + d / group_size as usize;
                let product = bf16::from_f32(bf16::from_bits(scales[affine]).to_f32() * code as f32).to_f32();
                decoded.push(bf16::from_f32(product + bf16::from_bits(biases[affine]).to_f32()).to_bits());
            }
        }
    }
    let starts: Vec<_> = (0..=groups).map(|g| (g * rows / groups) as u32).collect();
    let weight = starts
        .windows(2)
        .map(|range| {
            let (a, b) = (range[0] as usize, range[1] as usize);
            if bits == 0 {
                Buffer::from_slice(device, &decoded[a * dim..b * dim])
            } else {
                Buffer::from_slice(device, &words[a * dim * bits as usize / 8..b * dim * bits as usize / 8])
            }
        })
        .collect();
    let affine_buffers = |values: &[u16]| {
        starts
            .windows(2)
            .map(|r| Buffer::from_slice(device, &values[r[0] as usize * params..r[1] as usize * params]))
            .collect()
    };
    let scales = affine_buffers(&scales);
    let biases = affine_buffers(&biases);
    TableFixture {
        dim,
        bits,
        group_size,
        starts,
        weight,
        scales,
        biases,
        decoded,
    }
}
