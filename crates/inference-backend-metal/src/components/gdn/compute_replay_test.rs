use super::*;
use crate::components::gdn::compute::ReplayBuffers;
use crate::components::gdn::state_replay;

#[test]
fn test_replay_commits_selected_prefixes_across_layers() {
    for projection_layout in [
        super::super::ProjectionLayout::Separate,
        super::super::ProjectionLayout::PackedQkvabz,
    ] {
        for (qk_head_dim, v_head_dim) in [(64, 8), (128, 128), (256, 8)] {
            assert_replay_commits(
                Config {
                    projection_layout,
                    num_qk_heads: 2,
                    num_v_heads: 4,
                    qk_head_dim,
                    v_head_dim,
                    ..fixture_config()
                },
                4,
            );
        }
    }
}

#[test]
fn test_replay_commits_across_token_tiles() {
    for tokens in [17, 33] {
        assert_replay_commits(
            Config {
                projection_layout: super::super::ProjectionLayout::PackedQkvabz,
                num_qk_heads: 2,
                num_v_heads: 4,
                qk_head_dim: 128,
                v_head_dim: 128,
                ..fixture_config()
            },
            tokens,
        );
    }
}

fn assert_replay_commits(config: Config, replay_tokens: u32) {
    const NUM_LAYERS: usize = 2;
    let num_slots = replay_tokens as usize + 10;
    const CANARY: f32 = -777.0;
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let shape = fixture_shape(3, replay_tokens + 8);
    let commit_config = state_replay::Config {
        num_gdn_layers: NUM_LAYERS as u32,
        num_state_slots: num_slots as u32,
        max_replay_tokens: replay_tokens,
        num_qk_heads: config.num_qk_heads,
        qk_head_dim: config.qk_head_dim,
        num_v_heads: config.num_v_heads,
        v_head_dim: config.v_head_dim,
        conv_state_len: config.conv_state_len(),
    };
    let alpha = Buffer::new_zeroed(&device, commit_config.alpha_bytes());
    let k = Buffer::new_zeroed(&device, commit_config.k_bytes());
    let u = Buffer::new_zeroed(&device, commit_config.u_bytes());
    let log_qkv = Buffer::new_zeroed(&device, commit_config.qkv_bytes());
    let recurrent_stride = config.recurrent_state_stride();
    let conv_stride = config.qkv_dim() as usize * config.conv_state_len() as usize;
    let recurrent_values = bf16_round_trip(&fixture_values(
        NUM_LAYERS * num_slots * recurrent_stride,
        0.0078125,
        19,
    ));
    let conv_values = bf16_round_trip(&fixture_values(NUM_LAYERS * num_slots * conv_stride, 0.015625, 17));
    let recurrent = bf16_buffer(&device, &recurrent_values);
    let conv = bf16_buffer(&device, &conv_values);
    let reference_recurrent = bf16_buffer(&device, &recurrent_values);
    let reference_conv = bf16_buffer(&device, &conv_values);
    let cu_tokens = Buffer::from_slice(&device, &[0_u32, 5, 5 + replay_tokens, u32::MAX]);
    let src_recurrent = Buffer::from_slice(&device, &[1_u32, 0, u32::MAX]);
    let src_conv = Buffer::from_slice(&device, &[0_u32, 1, u32::MAX]);
    let mut reference_slots = (3..8 + replay_tokens).collect::<Vec<_>>();
    reference_slots.extend([u32::MAX; 3]);
    let reference_write = Buffer::from_slice(&device, &reference_slots);
    // Two recurrent boundaries. Replay must leave all full-state slots unchanged.
    let mut replay_slots = vec![u32::MAX; shape.num_total_tokens as usize];
    replay_slots[1] = 4;
    replay_slots[4] = 7;
    let replay_write = Buffer::from_slice(&device, &replay_slots);
    let kernels = Compute::new(&device, config);
    let reference_kernels = Compute::new(
        &device,
        Config {
            projection_layout: super::super::ProjectionLayout::Separate,
            ..config
        },
    );
    let mut expected_outputs = Vec::new();
    for layer in 0..NUM_LAYERS {
        let qkv = bf16_buffer(
            &device,
            &fixture_values(config.num_qkv_values(shape), 0.03125, 3 + layer),
        );
        let a = bf16_buffer(
            &device,
            &fixture_values(
                shape.num_total_tokens as usize * config.num_v_heads as usize,
                0.0625,
                5 + layer,
            ),
        );
        let b = bf16_buffer(
            &device,
            &fixture_values(
                shape.num_total_tokens as usize * config.num_v_heads as usize,
                0.0625,
                7 + layer,
            ),
        );
        let z = bf16_buffer(
            &device,
            &fixture_values(config.num_recurrent_output_values(shape), 0.03125, 11),
        );
        let widths = [
            config.qkv_dim() as usize,
            config.num_v_heads as usize,
            config.num_v_heads as usize,
            (config.num_v_heads * config.v_head_dim) as usize,
        ];
        let parts = [&qkv, &a, &b, &z]
            .into_iter()
            .zip(widths)
            .map(|(buffer, width)| buffer.read_typed::<u16>(0, shape.num_total_tokens as usize * width))
            .collect::<Vec<_>>();
        let mut packed = Vec::new();
        for token in 0..shape.num_total_tokens as usize {
            for (part, width) in parts.iter().zip(widths) {
                packed.extend_from_slice(&part[token * width..(token + 1) * width]);
            }
        }
        let packed = Buffer::from_slice(&device, &packed);
        let conv_weight = bf16_buffer(
            &device,
            &fixture_values(
                config.qkv_dim() as usize * config.conv_kernel_size as usize,
                0.00390625,
                13,
            ),
        );
        let norm_weight = bf16_buffer(&device, &vec![1.0; config.v_head_dim as usize]);
        let a_log = bf16_buffer(&device, &vec![-0.25; config.num_v_heads as usize]);
        let dt_bias = bf16_buffer(&device, &vec![0.125; config.num_v_heads as usize]);
        let conv_qkv = Buffer::new_zeroed_elements(&device, config.num_qkv_values(shape), Dtype::Bfloat16);
        let recurrent_output =
            Buffer::new_zeroed_elements(&device, config.num_recurrent_output_values(shape), Dtype::Bfloat16);
        let output = bf16_buffer(&device, &vec![CANARY; config.num_recurrent_output_values(shape)]);
        let buffers = Buffers {
            qkv: &qkv,
            a: &a,
            b: &b,
            z: &z,
            conv_weight: &conv_weight,
            norm_weight: &norm_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            cu_tokens: &cu_tokens,
            src_recurrent_state_slots: &src_recurrent,
            src_conv_state_slots: &src_conv,
            flat_recurrent_state_write_slots: &reference_write,
            flat_conv_state_write_slots: &reference_write,
            conv_state: &reference_conv,
            conv_state_offset_bytes: (layer * num_slots * conv_stride * 2) as u64,
            next_conv_state: &reference_conv,
            next_conv_state_offset_bytes: (layer * num_slots * conv_stride * 2) as u64,
            recurrent_state_arena: &reference_recurrent,
            recurrent_state_arena_offset_bytes: (layer * num_slots * recurrent_stride * 2) as u64,
            conv_qkv: &conv_qkv,
            recurrent_output: &recurrent_output,
            norm_gated_output: &output,
        };
        let mut reference = stream.create_replay_program();
        reference.record(reference_kernels.invoke_with_candidate_state_update(
            shape,
            buffers,
            ReplayU32::Fixed(2),
            ReplayU32::Fixed(5 + replay_tokens),
        ));
        stream.submit_replay(&reference.build()).wait();
        expected_outputs.push(read_bf16(&output, 0, config.num_recurrent_output_values(shape)));
        write_bf16(&output, 0, &vec![CANARY; config.num_recurrent_output_values(shape)]);
        let input_buffers = if config.projection_layout == super::super::ProjectionLayout::PackedQkvabz {
            Buffers {
                qkv: &packed,
                a: &packed,
                b: &packed,
                z: &packed,
                ..buffers
            }
        } else {
            buffers
        };
        let mut forward = stream.create_replay_program();
        forward.record(kernels.invoke_with_replay(
            shape,
            Buffers {
                conv_state: &conv,
                next_conv_state: &conv,
                recurrent_state_arena: &recurrent,
                flat_recurrent_state_write_slots: &replay_write,
                flat_conv_state_write_slots: &replay_write,
                ..input_buffers
            },
            ReplayBuffers {
                alpha: &alpha,
                k: &k,
                u: &u,
                qkv: &log_qkv,
                token_offset: layer as u64 * commit_config.max_replay_tokens as u64,
                num_total_tokens: commit_config.max_replay_tokens,
            },
            ReplayU32::Parameter(NUM_ACTIVE_REQUESTS),
            ReplayU32::Parameter(NUM_ACTIVE_TOKENS),
            ReplayU32::Parameter(NUM_ACTIVE_RECURRENT_REQUESTS),
        ));
        let arguments = ReplayArguments::new()
            .with_u32(NUM_ACTIVE_REQUESTS, 2)
            .with_u32(NUM_ACTIVE_TOKENS, 5 + replay_tokens)
            .with_u32(NUM_ACTIVE_RECURRENT_REQUESTS, 1);
        stream.submit_replay_with_arguments(&forward.build(), &arguments).wait();
        assert_close(
            &read_bf16(&output, 0, config.num_recurrent_output_values(shape)),
            &expected_outputs[layer],
            0.004,
        );
        for slot in 0..num_slots {
            if [4, 7].contains(&slot) {
                let offset = (layer * num_slots + slot) * recurrent_stride;
                assert_close(
                    &read_bf16(&recurrent, offset, recurrent_stride),
                    &read_bf16(&reference_recurrent, offset, recurrent_stride),
                    0.001,
                );
                let offset = (layer * num_slots + slot) * conv_stride;
                assert_eq!(
                    read_bf16(&conv, offset, conv_stride),
                    read_bf16(&reference_conv, offset, conv_stride)
                );
                continue;
            }
            let offset = (layer * num_slots + slot) * recurrent_stride;
            assert_eq!(
                read_bf16(&recurrent, offset, recurrent_stride),
                recurrent_values[offset..offset + recurrent_stride]
            );
            let offset = (layer * num_slots + slot) * conv_stride;
            assert_eq!(
                read_bf16(&conv, offset, conv_stride),
                conv_values[offset..offset + conv_stride]
            );
        }
    }
    // One zero-prefix target and every positive accepted prefix, including all accepted.
    let jobs = (0..=replay_tokens)
        .map(|num_tokens| {
            state_replay::Job {
                src_recurrent_state_slot: 0,
                src_conv_state_slot: 1,
                dst_recurrent_state_slot: 8 + num_tokens,
                dst_conv_state_slot: 8 + num_tokens,
                replay_token_begin: 0,
                num_tokens,
            }
        })
        .collect::<Vec<_>>();
    let job_buffer = Buffer::new_zeroed(&device, jobs.len() * size_of::<state_replay::Job>());
    state_replay::write_jobs(&job_buffer, &jobs);
    let commit = state_replay::Commit::new(&device, commit_config);
    let mut builder = stream.create_replay_program();
    builder.record(commit.invoke(
        jobs.len() as u32,
        ReplayU32::Fixed(jobs.len() as u32),
        state_replay::Buffers {
            recurrent_states: &recurrent,
            conv_states: &conv,
            alpha: &alpha,
            k: &k,
            u: &u,
            qkv: &log_qkv,
            jobs: &job_buffer,
        },
    ));
    stream.submit_replay(&builder.build()).wait();
    for layer in 0..NUM_LAYERS {
        for prefix in 0..=replay_tokens as usize {
            let dst = layer * num_slots + 8 + prefix;
            let rec_src = layer * num_slots + if prefix == 0 { 0 } else { 7 + prefix };
            let conv_src = layer * num_slots + if prefix == 0 { 1 } else { 7 + prefix };
            assert_close(
                &read_bf16(&recurrent, dst * recurrent_stride, recurrent_stride),
                &read_bf16(&reference_recurrent, rec_src * recurrent_stride, recurrent_stride),
                0.0001,
            );
            assert_eq!(
                read_bf16(&conv, dst * conv_stride, conv_stride),
                read_bf16(&reference_conv, conv_src * conv_stride, conv_stride)
            );
        }
    }
}
