use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use half::bf16;
use inference_backend_metal::components::ngram_embedding;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_executor_core::ngram_embedding::core::HashParameters;
use inference_executor_core::ngram_embedding::core::NGramEmbeddingCore;

const NUM_INVOCATIONS: u32 = 32;

fn bench_ngram_embedding(c: &mut Criterion) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let core = NGramEmbeddingCore {
        vocab_size: 151936,
        eos_token_id: 151645,
        ngram_size: 3,
        heads_per_ngram: 8,
        embedding_dim: 2560,
    };
    let parameters = HashParameters::new(core, 65536, 0, 1234);
    let compute = ngram_embedding::Compute::new(
        &device,
        ngram_embedding::Config {
            ngram_size: core.ngram_size,
            heads_per_ngram: core.heads_per_ngram,
            embedding_dim: core.embedding_dim,
            eos_token_id: core.eos_token_id,
        },
    );
    let starts: Vec<_> = (0..=8)
        .map(|g| (u64::from(parameters.num_rows()) * g / 8) as u32)
        .collect();
    let weights: Vec<_> = starts
        .windows(2)
        .map(|range| {
            let rows = (range[1] - range[0]) as usize;
            [
                Buffer::from_slice(&device, &vec![0x7654_3210_u32; rows * 20]),
                Buffer::from_slice(&device, &vec![bf16::from_f32(0.0625).to_bits(); rows * 5]),
                Buffer::from_slice(&device, &vec![bf16::from_f32(-0.5).to_bits(); rows * 5]),
            ]
        })
        .collect();
    let shards: Vec<_> = starts
        .windows(2)
        .enumerate()
        .map(|(i, range)| {
            ngram_embedding::Shard {
                row_start: range[0],
                num_rows: range[1] - range[0],
                weights: ngram_embedding::TableWeights::Affine {
                    bits: 4,
                    group_size: 32,
                    weight: &weights[i][0],
                    scales: &weights[i][1],
                    biases: &weights[i][2],
                },
            }
        })
        .collect();
    let multipliers = Buffer::from_slice(&device, &parameters.multipliers);
    let sizes = Buffer::from_slice(&device, &parameters.head_vocab_sizes);
    let offsets = Buffer::from_slice(&device, &parameters.head_offsets);
    let history = Buffer::from_slice(&device, &[0_u32, 0]);
    let mut group = c.benchmark_group("metal/ngram_embedding");
    for total in [1, 8, 32, 128, 256, 257, 512, 1024, 2048, 16384, 65536] {
        let input: Vec<u32> = (0..total).map(|i| (i * 7919 + 17) % core.vocab_size).collect();
        let ids = inference_executor_core::ngram_embedding::host::row_ids(core, &parameters, &[0, 0], &input);
        let input = Buffer::from_slice(&device, &input);
        let tokens = Buffer::from_slice(&device, &vec![0_u32; total as usize * 2]);
        let ids = Buffer::from_slice(&device, &ids);
        let output = Buffer::new_uninit(&device, total as usize * 2560 * 2);
        let active = ReplayU32::Fixed(total);
        let hash = ngram_embedding::HashBuffers {
            token_ids: &input,
            previous_tokens: &history,
            tokens: &tokens,
            multipliers: &multipliers,
            head_vocab_sizes: &sizes,
            head_offsets: &offsets,
            row_ids: &ids,
        };
        for phase in ["lookup", "hash_lookup"] {
            let mut builder = stream.create_replay_program();
            for _ in 0..NUM_INVOCATIONS {
                if phase == "hash_lookup" {
                    builder.record_with_barrier_before(compute.hash(total, 1, active, hash));
                }
                builder.record_with_barrier_before(compute.lookup(total, active, &ids, &output, 0.5, &shards));
            }
            let replay = builder.build();
            let kind = compute.lookup_topology(total, &shards);
            group.throughput(Throughput::Elements(u64::from(total) * u64::from(NUM_INVOCATIONS)));
            group.bench_function(format!("{phase}/{kind:?}/replay{NUM_INVOCATIONS}/tokens{total}"), |b| {
                b.iter(|| stream.submit_replay(black_box(&replay)).wait());
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_ngram_embedding);
criterion_main!(benches);
