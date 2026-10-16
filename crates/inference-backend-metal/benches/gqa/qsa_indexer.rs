use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use half::bf16;
use inference_backend_metal::components::qsa_indexer;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;

const NUM_INVOCATIONS: usize = 32;
const HEADS: u32 = 4;
const DIM: u32 = 128;
const TOP_K: u32 = 512;
const RATIO: u32 = 4;
const PAGE_BYTES: u32 = 32768;
const GROUPS_PER_PAGE: u32 = PAGE_BYTES / (DIM * 2);

fn bench_qsa_indexer(c: &mut Criterion) {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let compute = qsa_indexer::Compute::new(
        &device,
        qsa_indexer::Config {
            num_q_heads: HEADS,
            head_dim: DIM,
            compress_ratio: RATIO,
            top_k: TOP_K,
            rope_dim: DIM,
            rope_theta: 10000.0,
            norm_eps: 1.0e-6,
            dtype: Dtype::Bfloat16,
        },
    );
    let mut group = c.benchmark_group("metal/qsa_indexer/select");
    for requests in [1, 2, 4, 8] {
        let cases = [4096, 16384, 65536]
            .into_iter()
            .flat_map(|history| [1, 8, 32, 128, 256].map(|queries| (history, queries)))
            .chain([(0, 4096), (0, 16384)]);
        for (history, queries) in cases {
            let total = requests * queries;
            let groups = (history + queries) / RATIO;
            let sparse_start = ((TOP_K + 1) * RATIO - 1).saturating_sub(history).min(queries);
            let sparse = queries - sparse_start;
            let layout = qsa_indexer::CacheLayout {
                num_req_slots: requests,
                num_pages_per_request: groups.div_ceil(GROUPS_PER_PAGE),
                num_state_slots: total + requests,
                page_bytes: PAGE_BYTES,
            };
            // Task capacity is independent of the number of active query tokens.
            let mut shape = qsa_indexer::Shape {
                num_total_tokens: total,
                num_total_sparse_queries: requests * sparse,
                num_total_request_sparse_queries: sparse,
                num_total_groups: groups,
                num_total_score_task_templates: total,
            };
            let geometry = qsa_indexer::ScoreGeometry::for_shape(HEADS, DIM, Dtype::Bfloat16, shape, layout);
            let width = geometry.num_queries_per_task(sparse);
            let mut tokens = Vec::new();
            let mut tasks = Vec::new();
            for request in 0..requests {
                for query in 0..queries {
                    tokens.extend_from_slice(
                        &qsa_indexer::Token {
                            batch_start: request * queries,
                            token_index: history + query,
                            req_slot: request,
                            src_state_slot: request,
                            dst_state_slot: requests + request * queries + query,
                            group_start: 0,
                        }
                        .to_array(),
                    );
                }
                for start in (sparse_start..queries).step_by(width as usize) {
                    tasks.extend_from_slice(
                        &qsa_indexer::ScoreTaskTemplate {
                            query_start: request * queries + start,
                            num_queries: width.min(queries - start),
                        }
                        .to_array(),
                    );
                }
            }
            shape.num_total_score_task_templates = (tasks.len() / 2) as u32;
            let q = fixture_buffer(&device, total as usize * HEADS as usize * DIM as usize, 13);
            let page_count = requests * layout.num_pages_per_request;
            let pages = fixture_buffer(&device, page_count as usize * PAGE_BYTES as usize / 2, 17);
            let page_ids = Buffer::from_slice(&device, &(0..page_count).rev().collect::<Vec<_>>());
            let tokens = Buffer::from_slice(&device, &tokens);
            let tasks = Buffer::from_slice(&device, &tasks);
            let ids = Buffer::new_zeroed(&device, total as usize * TOP_K as usize * 4);
            let scores = Buffer::new_zeroed(&device, total as usize * groups as usize * 4);
            let mut builder = stream.create_replay_program();
            for _ in 0..NUM_INVOCATIONS {
                builder.record_with_barrier_before(compute.select(
                    shape,
                    layout,
                    ReplayU32::Fixed(total),
                    ReplayU32::Fixed(shape.num_total_score_task_templates),
                    qsa_indexer::SelectBuffers {
                        q: &q,
                        tokens: &tokens,
                        score_task_templates: &tasks,
                        pages: &pages,
                        page_ids: &page_ids,
                        group_ids: &ids,
                    },
                    &scores,
                ));
            }
            let replay = builder.build();
            group.throughput(Throughput::Elements(
                requests as u64 * sparse as u64 * NUM_INVOCATIONS as u64,
            ));
            group.bench_function(
                format!("replay{NUM_INVOCATIONS}/r{requests}/q{queries}/h{history}"),
                |b| {
                    b.iter(|| {
                        stream.submit_replay(&replay).wait();
                        black_box(&ids);
                    });
                },
            );
        }
    }
    group.finish();
}

fn fixture_buffer(device: &Device, len: usize, seed: usize) -> Buffer {
    Buffer::from_slice(
        device,
        &(0..len)
            .map(|index| bf16::from_f32((((index * seed + index / 11) % 127) as f32 - 63.0) / 64.0).to_bits())
            .collect::<Vec<_>>(),
    )
}

criterion_group!(benches, bench_qsa_indexer);
criterion_main!(benches);
