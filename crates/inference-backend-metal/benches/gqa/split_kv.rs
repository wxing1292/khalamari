use std::hint::black_box;
use std::mem::size_of;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use half::bf16;
use inference_backend_metal::components::gqa::fp8::bf16_to_fp8_e4m3;
use inference_backend_metal::components::gqa::fp8::f32_to_bf16;
use inference_backend_metal::components::gqa::kv_page_write as backend_kv_page_write;
use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::components::gqa::sdpa::sliced_kv as backend_sliced_kv;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;

const GQA_CONTEXT_LENS: [u32; 4] = [128, 1024, 2048, 4096];
const GQA_NUM_Q_HEADS: u32 = 32;
const GQA_NUM_KV_HEADS: u32 = 8;
const GQA_HEAD_DIM: u32 = 128;
const GQA_TOKENS_PER_PAGE: u32 = 16;
const GQA_KV_TOKEN_TILE_SIZE: u32 = 512;
const GQA_MAX_KV_SPLITS: u32 = 64;
const GQA_NUM_THREADS_PER_THREADBLOCK: u32 = 128;
const GQA_Q_HEAD_TILE_SIZE_CAP: u32 = 8;

fn bench_gqa_split_kv(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/gqa-split-kv");

    for context_len in GQA_CONTEXT_LENS {
        let fixture = GQAFixture::new(&device, &[context_len]);
        group.throughput(Throughput::Elements(context_len as u64));
        group.bench_function(format!("sliced-kv/replay/ctx{context_len}"), |b| {
            b.iter(|| {
                fixture.run_replay();
                black_box(&fixture.output);
            });
        });
    }

    let ragged_context_lens = [128, 128, 128, 128, 128, 128, 128, 4096];
    let fixture = GQAFixture::new(&device, &ragged_context_lens);
    group.throughput(Throughput::Elements(
        ragged_context_lens.iter().map(|&context_len| context_len as u64).sum(),
    ));
    group.bench_function("sliced-kv/replay/ragged-7x128-1x4096", |b| {
        b.iter(|| {
            fixture.run_replay();
            black_box(&fixture.output);
        });
    });

    group.finish();
}

struct GQAFixture {
    stream: Stream,
    output: Buffer,
    replay: ReplayProgram,
}

impl GQAFixture {
    fn new(device: &Device, context_lens: &[u32]) -> Self {
        assert!(!context_lens.is_empty());
        let max_context_len = context_lens.iter().copied().max().unwrap();
        let num_tokens = context_lens.len() as u32;
        let num_blocks = max_context_len.div_ceil(GQA_TOKENS_PER_PAGE);
        let (mut kv_split_values, cu_kv_split_values) = split_kv_plan(context_lens);
        let num_kv_splits = kv_split_values.len() as u32 / 3;
        let num_total_kv_splits = num_kv_splits.next_power_of_two();
        kv_split_values.resize(num_total_kv_splits as usize * 3, u32::MAX);
        let page_table_layout = backend_kv_page_write::PageTableLayout {
            num_req_slots: num_tokens,
            num_blocks,
            num_gqa_layers: 1,
            num_page_ids_per_block: 1,
        };
        let config = backend_sliced_kv::Config {
            num_q_heads: GQA_NUM_Q_HEADS,
            num_kv_heads: GQA_NUM_KV_HEADS,
            head_dim: GQA_HEAD_DIM,
            scale: 1.0 / (GQA_HEAD_DIM as f32).sqrt(),
            page_bytes: 2 * GQA_NUM_KV_HEADS * GQA_TOKENS_PER_PAGE * GQA_HEAD_DIM * size_of::<u8>() as u32,
            page_table_layout,
            dtype: Dtype::Bfloat16,
        };
        let execution = backend_sdpa::ExecutionVariant::sliced_kv(
            backend_sdpa::Config {
                io_dtype: config.dtype,
                num_q_heads: config.num_q_heads,
                num_kv_heads: config.num_kv_heads,
                head_dim: config.head_dim,
                tokens_per_page: config.num_tokens_per_page(),
            },
            GQA_KV_TOKEN_TILE_SIZE,
            GQA_NUM_THREADS_PER_THREADBLOCK,
            (GQA_NUM_Q_HEADS / GQA_NUM_KV_HEADS).min(GQA_Q_HEAD_TILE_SIZE_CAP),
        );
        let shape = backend_sliced_kv::Shape {
            num_total_tokens: num_tokens,
            num_total_sdpa_map_task_templates: num_total_kv_splits,
        };
        shape.validate(config);

        let q = bf16_pattern_buffer(device, config.num_output_values(shape), 0.003);
        let kv_pages = fp8_pattern_buffer(
            device,
            page_table_layout.num_req_slots as usize
                * page_table_layout.num_blocks as usize
                * config.page_bytes as usize,
            0.002,
        );
        let req_slots = Buffer::from_slice(device, &(0..num_tokens).collect::<Vec<_>>());
        let num_page_ids = page_table_layout.num_req_slots * page_table_layout.num_blocks;
        let page_ids = Buffer::from_slice(device, &(0..num_page_ids).collect::<Vec<_>>());
        let kv_splits = Buffer::from_slice(device, &kv_split_values);
        let cu_kv_splits = Buffer::from_slice(device, &cu_kv_split_values);
        let scratch = backend_sliced_kv::Scratch::new(device, config, shape);
        let output = Buffer::new_zeroed(device, config.q_bytes(shape));
        let stream = Stream::new(device);
        let kernels = backend_sliced_kv::Compute::new(device, config, execution, shape);
        let replay = build_gqa_replay(
            &stream,
            &kernels,
            &q,
            &kv_pages,
            &req_slots,
            &page_ids,
            &kv_splits,
            &cu_kv_splits,
            num_tokens,
            num_kv_splits,
            &scratch,
            &output,
        );

        let fixture = Self { stream, output, replay };
        fixture.run_replay();
        fixture
    }

    fn run_replay(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }
}

fn split_kv_plan(context_lens: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let num_kv_token_tiles = context_lens
        .iter()
        .map(|&context_len| context_len.div_ceil(GQA_KV_TOKEN_TILE_SIZE) as usize)
        .collect::<Vec<_>>();
    let mut num_kv_splits_by_q_token_tile = vec![1_usize; context_lens.len()];
    let mut num_kv_splits = num_kv_splits_by_q_token_tile.len();
    while num_kv_splits < GQA_MAX_KV_SPLITS as usize {
        let Some(q_token_tile_index) = (0..num_kv_token_tiles.len())
            .filter(|&q_token_tile_index| {
                num_kv_splits_by_q_token_tile[q_token_tile_index] < num_kv_token_tiles[q_token_tile_index]
            })
            .max_by_key(|&q_token_tile_index| {
                num_kv_token_tiles[q_token_tile_index].div_ceil(num_kv_splits_by_q_token_tile[q_token_tile_index])
            })
        else {
            break;
        };
        num_kv_splits_by_q_token_tile[q_token_tile_index] += 1;
        num_kv_splits += 1;
    }

    let mut kv_splits = Vec::with_capacity(num_kv_splits * 3);
    let mut cu_kv_splits = vec![0];
    for (q_token_tile_index, &context_len) in context_lens.iter().enumerate() {
        for kv_split_index in 0..num_kv_splits_by_q_token_tile[q_token_tile_index] {
            let kv_token_tile_begin = num_kv_token_tiles[q_token_tile_index] * kv_split_index
                / num_kv_splits_by_q_token_tile[q_token_tile_index];
            let kv_token_tile_end = num_kv_token_tiles[q_token_tile_index] * (kv_split_index + 1)
                / num_kv_splits_by_q_token_tile[q_token_tile_index];
            kv_splits.extend_from_slice(&[
                q_token_tile_index as u32,
                kv_token_tile_begin as u32 * GQA_KV_TOKEN_TILE_SIZE,
                context_len.min(kv_token_tile_end as u32 * GQA_KV_TOKEN_TILE_SIZE),
            ]);
        }
        cu_kv_splits.push((kv_splits.len() / 3) as u32);
    }
    (kv_splits, cu_kv_splits)
}

#[allow(clippy::too_many_arguments)]
fn build_gqa_replay(
    stream: &Stream,
    kernels: &backend_sliced_kv::Compute,
    q: &Buffer,
    kv_pages: &Buffer,
    req_slots: &Buffer,
    page_ids: &Buffer,
    kv_splits: &Buffer,
    cu_kv_splits: &Buffer,
    num_tokens: u32,
    num_kv_splits: u32,
    scratch: &backend_sliced_kv::Scratch,
    output: &Buffer,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    builder.record(kernels.invoke_map(
        scratch.map_buffers(q, kv_pages, req_slots, page_ids, kv_splits),
        inference_backend_metal::metal::ReplayU32::Fixed(0),
        inference_backend_metal::metal::ReplayU32::Fixed(num_tokens),
        inference_backend_metal::metal::ReplayU32::Fixed(num_kv_splits),
    ));
    builder.record(kernels.invoke_reduce(
        scratch.reduce_buffers(cu_kv_splits, output),
        inference_backend_metal::metal::ReplayU32::Fixed(num_tokens),
    ));
    builder.build()
}

fn bf16_pattern_buffer(device: &Device, len: usize, scale: f32) -> Buffer {
    let v = (0..len)
        .map(|index| {
            let value = (index % 251) as f32 - 125.0;
            bf16::from_f32(value * scale).to_bits()
        })
        .collect::<Vec<_>>();
    Buffer::from_slice(device, &v)
}

fn fp8_pattern_buffer(device: &Device, len: usize, scale: f32) -> Buffer {
    let values = (0..len)
        .map(|index| bf16_to_fp8_e4m3(f32_to_bf16(((index % 251) as f32 - 125.0) * scale)))
        .collect::<Vec<_>>();
    Buffer::from_slice(device, &values)
}

criterion_group!(benches, bench_gqa_split_kv);
criterion_main!(benches);
