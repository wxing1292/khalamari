use std::hint::black_box;

use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use half::bf16;
use inference_backend_metal::components::gdn::compute as backend_compute;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;

const GDN_CASES: [(u32, u32); 4] = [(1, 1), (4, 1), (1, 16), (4, 4)];
const GDN_CONV_DIM: u32 = 4096;
const GDN_V_HEADS: u32 = 16;
const GDN_V_HEAD_DIM: u32 = 128;
const GDN_QK_HEAD_DIM: u32 = 128;
const GDN_QK_HEADS: u32 = 8;
const GDN_CONV_STATE_LEN: u32 = 3;
const GDN_CONV_KERNEL_SIZE: u32 = 4;
// An odd prime period keeps the fixture centered at zero and avoids row-aligned repetition.
const FIXTURE_PATTERN_PERIOD: usize = 251;
const FIXTURE_PATTERN_CENTER: f32 = 125.0;

fn bench_gdn_attn(c: &mut Criterion) {
    let device = Device::system_default();
    let mut group = c.benchmark_group("metal/gdn-attn");

    for (batch, tokens) in GDN_CASES {
        for num_invocations in [1, 128] {
            let fixture = GDNFixture::new(&device, batch, tokens, num_invocations);
            let path = if num_invocations == 1 {
                "replay".to_owned()
            } else {
                format!("replay{num_invocations}")
            };
            group.throughput(Throughput::Elements(
                batch as u64 * tokens as u64 * num_invocations as u64,
            ));
            group.bench_function(
                format!("core-ragged_recurrent/{path}/batch{batch}/tokens{tokens}"),
                |b| {
                    b.iter(|| {
                        fixture.run_with_state_replay();
                        black_box(&fixture.norm_gated_output);
                    });
                },
            );
            group.bench_function(
                format!("forward_candidate_state_update/{path}/batch{batch}/tokens{tokens}"),
                |b| {
                    b.iter(|| {
                        fixture.run_forward_candidate_state_update_replay();
                        black_box(&fixture.recurrent_state_arena);
                    });
                },
            );
        }
    }

    group.finish();
}

struct GDNFixture {
    stream: Stream,
    recurrent_state_arena: Buffer,
    norm_gated_output: Buffer,
    with_state_replay: ReplayProgram,
    forward_candidate_state_update_replay: ReplayProgram,
}

impl GDNFixture {
    fn new(device: &Device, batch: u32, tokens: u32, num_invocations: usize) -> Self {
        let shape = backend_compute::Shape {
            num_total_reqs: batch,
            num_total_tokens: batch * tokens,
        };
        let config = backend_compute::Config {
            num_qk_heads: GDN_QK_HEADS,
            qk_head_dim: GDN_QK_HEAD_DIM,
            num_v_heads: GDN_V_HEADS,
            v_head_dim: GDN_V_HEAD_DIM,
            conv_kernel_size: GDN_CONV_KERNEL_SIZE,
            q_scale: 1.0 / (GDN_QK_HEAD_DIM as f32).sqrt(),
            norm_eps: 1.0e-6,
            output_gate: backend_compute::OutputGate::Silu,
        };
        let cu_token_values = (0..=shape.num_total_reqs)
            .map(|req_index| (req_index * tokens) as i32)
            .collect::<Vec<_>>();
        let src_state_slot_values = (0..shape.num_total_reqs).collect::<Vec<_>>();
        let dst_slot_id_values = (shape.num_total_reqs..shape.num_total_reqs * 2).collect::<Vec<_>>();
        let mut flat_final_state_slot_values = vec![u32::MAX; shape.num_total_tokens as usize];
        for (req_index, &dst_state_slot) in dst_slot_id_values.iter().enumerate() {
            flat_final_state_slot_values[(req_index as u32 * tokens + tokens - 1) as usize] = dst_state_slot;
        }
        let candidate_dst_slot_id_values = (0..shape.num_total_tokens)
            .map(|flat_token_index| shape.num_total_reqs * 2 + flat_token_index)
            .collect::<Vec<_>>();
        let state_slot_count = shape.num_total_reqs * 2 + shape.num_total_tokens;

        let stream = Stream::new(device);
        let kernels = backend_compute::Compute::new(device, config);
        let qkv = bf16_pattern_buffer(device, config.num_qkv_values(shape), 0.001);
        let a = bf16_pattern_buffer(
            device,
            shape.num_total_tokens as usize * config.num_v_heads as usize,
            0.002,
        );
        let b = bf16_pattern_buffer(
            device,
            shape.num_total_tokens as usize * config.num_v_heads as usize,
            -0.001,
        );
        let z = bf16_pattern_buffer(device, config.num_recurrent_output_values(shape), 0.0015);
        let conv_weight = bf16_pattern_buffer(
            device,
            config.qkv_dim() as usize * config.conv_kernel_size as usize,
            0.0005,
        );
        let norm_weight = bf16_constant_buffer(device, config.v_head_dim as usize, 1.0);
        let a_log = bf16_constant_buffer(device, config.num_v_heads as usize, -0.01);
        let dt_bias = bf16_constant_buffer(device, config.num_v_heads as usize, 0.02);
        let cu_tokens = Buffer::from_slice(device, &cu_token_values);
        let src_state_slots = Buffer::from_slice(device, &src_state_slot_values);
        let flat_final_state_slots = Buffer::from_slice(device, &flat_final_state_slot_values);
        let candidate_dst_slot_ids = Buffer::from_slice(device, &candidate_dst_slot_id_values);
        let conv_state = bf16_pattern_buffer(device, config.num_conv_state_values(shape), 0.001);
        let next_conv_state = Buffer::new_zeroed(
            device,
            state_slot_count as usize * config.qkv_dim() as usize * config.conv_state_len() as usize * size_of::<u16>(),
        );
        let recurrent_state_arena = bf16_pattern_buffer(
            device,
            config.recurrent_state_stride() * state_slot_count as usize,
            0.0001,
        );
        let conv_qkv = Buffer::new_zeroed(device, config.num_qkv_values(shape) * size_of::<u16>());
        let recurrent_output = Buffer::new_zeroed(device, config.num_recurrent_output_values(shape) * size_of::<u16>());
        let norm_gated_output =
            Buffer::new_zeroed(device, config.num_recurrent_output_values(shape) * size_of::<u16>());
        let buffers = backend_compute::Buffers {
            qkv: &qkv,
            a: &a,
            b: &b,
            z: &z,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            cu_tokens: &cu_tokens,
            src_recurrent_state_slots: &src_state_slots,
            src_conv_state_slots: &src_state_slots,
            flat_recurrent_state_write_slots: &flat_final_state_slots,
            flat_conv_state_write_slots: &flat_final_state_slots,
            conv_state: &conv_state,
            conv_state_offset_bytes: 0,
            next_conv_state: &next_conv_state,
            next_conv_state_offset_bytes: 0,
            recurrent_state_arena: &recurrent_state_arena,
            recurrent_state_arena_offset_bytes: 0,
            conv_qkv: &conv_qkv,
            recurrent_output: &recurrent_output,
            norm_gated_output: &norm_gated_output,
        };
        let with_state_replay = build_gdn_with_state_replay(&stream, &kernels, shape, buffers, num_invocations);
        let forward_candidate_state_update_replay = build_gdn_forward_candidate_state_update_replay(
            &stream,
            &kernels,
            shape,
            backend_compute::Buffers {
                flat_recurrent_state_write_slots: &candidate_dst_slot_ids,
                flat_conv_state_write_slots: &candidate_dst_slot_ids,
                ..buffers
            },
            num_invocations,
        );

        let fixture = Self {
            stream,
            recurrent_state_arena,
            norm_gated_output,
            with_state_replay,
            forward_candidate_state_update_replay,
        };
        fixture.run_with_state_replay();
        fixture.run_forward_candidate_state_update_replay();
        fixture
    }

    fn run_with_state_replay(&self) {
        self.stream.submit_replay(&self.with_state_replay).wait();
    }

    fn run_forward_candidate_state_update_replay(&self) {
        self.stream
            .submit_replay(&self.forward_candidate_state_update_replay)
            .wait();
    }
}

fn build_gdn_with_state_replay(
    stream: &Stream,
    kernels: &backend_compute::Compute,
    shape: backend_compute::Shape,
    buffers: backend_compute::Buffers<'_>,
    num_invocations: usize,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    for _ in 0..num_invocations {
        builder.record(kernels.invoke(
            shape,
            buffers,
            ReplayU32::Fixed(shape.num_total_reqs),
            ReplayU32::Fixed(shape.num_total_tokens),
        ));
    }
    builder.build()
}

fn build_gdn_forward_candidate_state_update_replay(
    stream: &Stream,
    kernels: &backend_compute::Compute,
    shape: backend_compute::Shape,
    buffers: backend_compute::Buffers<'_>,
    num_invocations: usize,
) -> ReplayProgram {
    let mut builder = stream.create_replay_program();
    for _ in 0..num_invocations {
        builder.record(kernels.invoke_with_candidate_state_update(
            shape,
            buffers,
            ReplayU32::Fixed(shape.num_total_reqs),
            ReplayU32::Fixed(shape.num_total_tokens),
        ));
    }
    builder.build()
}

fn bf16_pattern_buffer(device: &Device, len: usize, scale: f32) -> Buffer {
    let values = (0..len)
        .map(|index| {
            let value = (index % FIXTURE_PATTERN_PERIOD) as f32 - FIXTURE_PATTERN_CENTER;
            bf16::from_f32(value * scale).to_bits()
        })
        .collect::<Vec<_>>();
    Buffer::from_slice(device, &values)
}

fn bf16_constant_buffer(device: &Device, len: usize, value: f32) -> Buffer {
    Buffer::from_slice(device, &vec![bf16::from_f32(value).to_bits(); len])
}

criterion_group!(benches, bench_gdn_attn);
criterion_main!(benches);
