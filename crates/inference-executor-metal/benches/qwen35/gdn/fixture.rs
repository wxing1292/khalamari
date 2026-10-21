use std::fs::File;
use std::mem::size_of;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gdn::compute as backend_compute;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::attn::GDNCore;
use inference_executor_core::attn::gdn::GDNOutputGate;
use inference_executor_metal::attn::gdn::backend::GDN;
use inference_executor_metal::attn::gdn::backend::GDNInput;
use inference_executor_metal::attn::gdn::backend::GDNLayerStateBindings;
use inference_executor_metal::attn::gdn::backend::GDNMetalConfig;
use inference_executor_metal::attn::gdn::backend::GDNWeights;
use inference_executor_metal::attn::gdn::batch_metadata::GDNMetadataBuffers;
use inference_executor_metal::attn::gdn::scratch::GDNScratchBindings;
use inference_executor_metal::def::layer::ReplayLayer;
use safetensors::SafeTensors;

use crate::Args;
use crate::BITS;
use crate::GDN_CONV_KERNEL_SIZE;
use crate::GDN_EPS;
use crate::GDN_LAYER;
use crate::GDN_QK_HEAD_DIM;
use crate::GDN_QK_HEADS;
use crate::GDN_V_HEAD_DIM;
use crate::GDNModelProfile;
use crate::GROUP_SIZE;
use crate::build_single_invocation_replay;
use crate::concat_parts;
use crate::cu_tokens;
use crate::gdn_conv_state_fixture;
use crate::gdn_output_affine_config;
use crate::gdn_qkvabz_affine_config;
use crate::gdn_recurrent_state_fixture;
use crate::hidden_fixture;
use crate::measure_runs;
use crate::median;
use crate::print_named_perf;
use crate::print_perf;
use crate::print_skip;
use crate::request_token_counts;
use crate::tensor_bytes;
use crate::valid_num_reqs;
use crate::validate_qkvabz_sizes;

pub fn run(args: Args) {
    let device = Device::system_default();
    let mapped = MappedFile::open(&args.model_dir.join(args.model.shard));
    let tensors = SafeTensors::deserialize(mapped.as_bytes()).unwrap_or_else(|err| {
        panic!(
            "unable to deserialize safetensors shard {}: {err:?}",
            args.model_dir.join(args.model.shard).display()
        )
    });
    let weights = RealGDNWeights::load(&device, &tensors, args.model);
    let contexts = if args.contexts.is_empty() {
        vec![0]
    } else {
        args.contexts
    };

    let request_shapes = if let Some(counts) = args.tokens_per_req {
        vec![counts]
    } else {
        let mut shapes = Vec::new();
        for num_tokens in args.tokens {
            for &num_reqs in &args.num_reqs {
                if !valid_num_reqs(num_tokens, num_reqs) {
                    print_skip(num_tokens, num_reqs, None, None, "num_reqs_exceeds_tokens");
                    continue;
                }
                shapes.push(request_token_counts(num_tokens, num_reqs));
            }
        }
        shapes
    };
    for num_tokens_per_req in request_shapes {
        for &existing_context_len in &contexts {
            let fixture = RealGDNFixture::new(
                &device,
                &num_tokens_per_req,
                args.prefill_requests,
                existing_context_len,
                &weights,
                args.model,
            );
            println!(
                "bench_shape component=gdn tokens_per_req={num_tokens_per_req:?} prefill_requests={}",
                args.prefill_requests
            );
            if args.compare_recurrent {
                fixture.measure_recurrent_comparison(args.warmup_iters, args.iters, args.runs);
            } else {
                fixture.measure(args.warmup_iters, args.iters, args.runs);
            }
            if args.subcomponents {
                fixture.measure_subcomponents(args.warmup_iters, args.iters, args.runs);
            }
        }
    }
}

struct RealGDNFixture<'a> {
    device: Device,
    stream: Stream,
    num_tokens: u32,
    num_reqs: u32,
    num_prefill_requests: u32,
    existing_context_len: u32,
    model: GDNModelProfile,
    next_hidden_state: Buffer,
    replay: ReplayProgram,
    hidden_state: Buffer,
    batch_metadata: GDNMetadataBuffers,
    conv_state: Buffer,
    next_conv_state: Buffer,
    recurrent_state_arena: Buffer,
    qkvabz: Buffer,
    conv_qkv: Buffer,
    recurrent_output: Buffer,
    norm_gated_output: Buffer,
    replay_alpha: Buffer,
    replay_k: Buffer,
    replay_u: Buffer,
    replay_qkv: Buffer,
    weights: &'a RealGDNWeights,
}

impl<'a> RealGDNFixture<'a> {
    fn new(
        device: &Device,
        num_tokens_per_req: &[u32],
        num_prefill_requests: u32,
        existing_context_len: u32,
        weights: &'a RealGDNWeights,
        model: GDNModelProfile,
    ) -> Self {
        let num_tokens = num_tokens_per_req.iter().sum();
        let num_reqs = u32::try_from(num_tokens_per_req.len()).expect("request count must fit u32");
        assert!(
            valid_num_reqs(num_tokens, num_reqs),
            "GDN bench requires 1 <= num_reqs <= num_tokens"
        );
        let stream = Stream::new(device);
        let core = GDNCore {
            model_layer_index: GDN_LAYER,
            hidden_dim: model.hidden_dim,
            num_qk_heads: GDN_QK_HEADS,
            qk_head_dim: GDN_QK_HEAD_DIM,
            num_v_heads: model.num_v_heads,
            v_head_dim: GDN_V_HEAD_DIM,
            conv_kernel_size: GDN_CONV_KERNEL_SIZE,
            q_scale: (GDN_QK_HEAD_DIM as f32).sqrt().recip(),
            output_gate: GDNOutputGate::Silu,
        };
        let config = GDNMetalConfig {
            group_size: GROUP_SIZE,
            bits: BITS,
            norm_eps: GDN_EPS,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            qkvabz_scale_bias_dtype: Dtype::Bfloat16,
            output_scale_bias_dtype: Dtype::Bfloat16,
        };
        let backend = GDN::new(device, core, config);
        let hidden_state = Buffer::from_slice(device, &hidden_fixture(num_tokens as usize, model.hidden_dim));
        let next_hidden_state = Buffer::new_zeroed(
            device,
            num_tokens as usize * model.hidden_dim * Dtype::Bfloat16.item_size(),
        );
        let batch_metadata = GDNMetadataBuffers::new(device, num_reqs as usize, num_tokens as usize);
        let cu_tokens = cu_tokens(num_tokens_per_req)
            .into_iter()
            .map(|value| value as u32)
            .collect::<Vec<_>>();
        let mut flat_materialized_state_slots = vec![u32::MAX; num_tokens as usize];
        for (req_index, &flat_end) in cu_tokens.iter().skip(1).take(num_prefill_requests as usize).enumerate() {
            flat_materialized_state_slots[flat_end as usize - 1] = num_reqs + req_index as u32;
        }
        batch_metadata.update(
            &cu_tokens,
            num_prefill_requests,
            &(0..num_reqs).collect::<Vec<_>>(),
            &(0..num_reqs).collect::<Vec<_>>(),
            &flat_materialized_state_slots,
            &flat_materialized_state_slots,
            num_reqs,
            num_tokens,
        );
        let num_state_slots = num_reqs
            .checked_mul(2)
            .expect("GDN bench state-slot count must fit u32");
        let conv_state = Buffer::from_slice(
            device,
            &gdn_conv_state_fixture(
                model,
                existing_context_len,
                num_reqs as usize,
                num_state_slots as usize * model.conv_dim() * (GDN_CONV_KERNEL_SIZE - 1),
            ),
        );
        let next_conv_state = Buffer::new_zeroed(
            device,
            num_state_slots as usize * model.conv_dim() * (GDN_CONV_KERNEL_SIZE - 1) * size_of::<u16>(),
        );
        let recurrent_state_arena = Buffer::from_slice(
            device,
            &gdn_recurrent_state_fixture(
                model,
                existing_context_len,
                num_reqs as usize,
                num_state_slots as usize * model.num_v_heads * GDN_V_HEAD_DIM * GDN_QK_HEAD_DIM,
            ),
        );
        let qkvabz = Buffer::new_zeroed(device, num_tokens as usize * model.qkvabz_dim() * size_of::<u16>());
        let conv_qkv = Buffer::new_zeroed(device, num_tokens as usize * model.conv_dim() * size_of::<u16>());
        let recurrent_output = Buffer::new_zeroed(device, num_tokens as usize * model.v_dim() * size_of::<u16>());
        let norm_gated_output = Buffer::new_zeroed(device, num_tokens as usize * model.v_dim() * size_of::<u16>());
        let replay_alpha = Buffer::new_zeroed_elements(device, num_tokens as usize * model.num_v_heads, Dtype::Float32);
        let replay_k = Buffer::new_zeroed_elements(
            device,
            num_tokens as usize * GDN_QK_HEADS * GDN_QK_HEAD_DIM,
            Dtype::Float32,
        );
        let replay_u = Buffer::new_zeroed_elements(device, num_tokens as usize * model.v_dim(), Dtype::Float32);
        let replay_qkv = Buffer::new_zeroed_elements(device, num_tokens as usize * model.conv_dim(), Dtype::Bfloat16);
        let mut recorder = ReplayRecorder::new(stream.create_replay_program());
        let _ = <GDN as ReplayLayer>::record(
            &backend,
            &mut recorder,
            GDNInput {
                hidden_state: &hidden_state,
                next_hidden_state: &next_hidden_state,
                scratch: GDNScratchBindings {
                    qkvabz: &qkvabz,
                    conv_qkv: &conv_qkv,
                    recurrent_output: &recurrent_output,
                    norm_gated_output: &norm_gated_output,
                },
                batch_metadata: &batch_metadata,
                state: GDNLayerStateBindings {
                    conv_state: &conv_state,
                    conv_state_offset_bytes: 0,
                    next_conv_state: &next_conv_state,
                    next_conv_state_offset_bytes: 0,
                    recurrent_state_arena: &recurrent_state_arena,
                    recurrent_state_arena_offset_bytes: 0,
                    replay: inference_backend_metal::components::gdn::compute::ReplayBuffers {
                        alpha: &replay_alpha,
                        k: &replay_k,
                        u: &replay_u,
                        qkv: &replay_qkv,
                        token_offset: 0,
                        num_total_tokens: num_tokens,
                    },
                },
                weights: weights.as_borrowed(),
                num_active_tokens: ReplayU32::Fixed(num_tokens),
            },
        );
        let replay = recorder.build();
        let fixture = Self {
            model,
            device: device.clone(),
            stream,
            num_tokens,
            num_reqs,
            num_prefill_requests,
            existing_context_len,
            next_hidden_state,
            replay,
            hidden_state,
            batch_metadata,
            conv_state,
            next_conv_state,
            recurrent_state_arena,
            qkvabz,
            conv_qkv,
            recurrent_output,
            norm_gated_output,
            replay_alpha,
            replay_k,
            replay_u,
            replay_qkv,
            weights,
        };
        fixture.run();
        fixture
    }

    fn run(&self) {
        self.stream.submit_replay(&self.replay).wait();
    }

    fn measure(&self, warmup_iters: usize, iters: usize, runs: usize) {
        let samples = measure_runs(runs, warmup_iters, iters, || self.run());
        let _ = self.next_hidden_state.len_bytes();
        print_perf(
            self.num_tokens,
            self.num_reqs,
            Some(self.existing_context_len),
            Some("mixed_replay"),
            iters,
            &samples,
        );
    }

    fn measure_recurrent_comparison(&self, warmup_iters: usize, iters: usize, runs: usize) {
        let recurrent = self.recurrent_replay();
        let replays = [&recurrent, &self.replay];
        let run = |index: usize| self.stream.submit_replay(replays[index]).wait();
        let read = |buffer: &Buffer| buffer.read_typed::<u16>(0, buffer.len_bytes() / size_of::<u16>());
        let initial_recurrent = read(&self.recurrent_state_arena);
        let initial_conv = read(&self.next_conv_state);
        run(0);
        let expected_hidden = read(&self.next_hidden_state);
        let expected_recurrent = read(&self.recurrent_state_arena);
        let expected_conv = read(&self.next_conv_state);
        self.recurrent_state_arena.write_typed(0, &initial_recurrent);
        self.next_conv_state.write_typed(0, &initial_conv);
        run(1);
        let actual_hidden = read(&self.next_hidden_state);
        let actual_recurrent = read(&self.recurrent_state_arena);
        assert_eq!(
            read(&self.next_conv_state),
            expected_conv,
            "short convolution must remain unchanged"
        );
        let source_state_len = self.num_reqs as usize * self.model.num_v_heads * GDN_V_HEAD_DIM * GDN_QK_HEAD_DIM;
        assert_eq!(
            &actual_recurrent[..source_state_len],
            &initial_recurrent[..source_state_len]
        );
        assert_eq!(
            &expected_recurrent[..source_state_len],
            &initial_recurrent[..source_state_len]
        );
        let prefill_tokens = self
            .batch_metadata
            .cu_tokens()
            .read_typed::<u32>(self.num_prefill_requests as usize, 1)[0] as usize;
        assert_eq!(
            &actual_hidden[prefill_tokens * self.model.hidden_dim..],
            &expected_hidden[prefill_tokens * self.model.hidden_dim..],
            "replay forward must preserve the recurrent output bits",
        );
        let state_stride = self.model.num_v_heads * GDN_V_HEAD_DIM * GDN_QK_HEAD_DIM;
        let write_slots = self
            .batch_metadata
            .flat_recurrent_state_write_slots()
            .read_typed::<u32>(0, self.num_tokens as usize);
        for &slot in &write_slots[prefill_tokens..] {
            if slot != u32::MAX {
                let start = slot as usize * state_stride;
                assert_eq!(
                    &actual_recurrent[start..start + state_stride],
                    &expected_recurrent[start..start + state_stride]
                );
            }
        }
        for (name, actual, expected, absolute_tolerance) in [
            ("hidden", &actual_hidden, &expected_hidden, 0.0625_f32),
            ("recurrent_state", &actual_recurrent, &expected_recurrent, 0.005_f32),
        ] {
            let mut max_absolute_error = 0.0_f32;
            let mut squared_error = 0.0_f64;
            let mut squared_reference = 0.0_f64;
            for (&actual, &expected) in actual.iter().zip(expected) {
                let actual = bf16::from_bits(actual).to_f32();
                let expected = bf16::from_bits(expected).to_f32();
                let error = (actual - expected).abs();
                assert!(actual.is_finite() && expected.is_finite());
                assert!(
                    error <= absolute_tolerance + 0.02 * expected.abs(),
                    "{name}: {actual} != {expected}"
                );
                max_absolute_error = max_absolute_error.max(error);
                squared_error += f64::from(error).powi(2);
                squared_reference += f64::from(expected).powi(2);
            }
            println!(
                "bench_check component=gdn check=recurrent-comparison tensor={name} max_abs={max_absolute_error:.8} \
                 relative_l2={:.8} status=pass",
                (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt()
            );
        }
        println!(
            "bench_compare component=gdn baseline_commands={} mixed_commands={} prefill_requests={} order=alternating",
            recurrent.command_count(),
            self.replay.command_count(),
            self.num_prefill_requests,
        );
        let mut samples = [Vec::with_capacity(runs), Vec::with_capacity(runs)];
        for run_index in 0..runs {
            for iteration in 0..warmup_iters {
                let first = (run_index + iteration) % 2;
                run(first);
                run(1 - first);
            }
            let mut elapsed = [0.0; 2];
            for iteration in 0..iters {
                let first = (run_index + iteration) % 2;
                for index in [first, 1 - first] {
                    let start = Instant::now();
                    run(index);
                    elapsed[index] += start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;
                }
            }
            for index in 0..2 {
                samples[index].push(elapsed[index]);
            }
        }
        for (name, sample) in ["recurrent", "mixed"].into_iter().zip(&samples) {
            print_perf(
                self.num_tokens,
                self.num_reqs,
                Some(self.existing_context_len),
                Some(name),
                iters,
                sample,
            );
        }
        let ratios = samples[1]
            .iter()
            .zip(&samples[0])
            .map(|(mixed, recurrent)| mixed / recurrent)
            .collect::<Vec<_>>();
        println!(
            "bench_compare component=gdn mixed_over_recurrent={:.6} ratios={ratios:?}",
            median(&ratios)
        );
    }

    fn recurrent_replay(&self) -> ReplayProgram {
        let device = &self.device;
        let mut builder = self.stream.create_replay_program();
        let qkvabz_config = gdn_qkvabz_affine_config(self.model);
        let qkvabz = affine_quantized::Matmul::new(device, qkvabz_config);
        let compute = backend_compute::Compute::new(device, gdn_compute_config(self.model));
        let output_config = gdn_output_affine_config(self.model);
        let output = affine_quantized::Matmul::new(device, output_config);

        builder.record_with_barrier_before(qkvabz.invoke(
            self.num_tokens,
            ReplayU32::Fixed(self.num_tokens),
            &self.qkvabz,
            0,
            &self.hidden_state,
            0,
            &self.weights.qkvabz_weight,
            0,
            &self.weights.qkvabz_scales,
            0,
            &self.weights.qkvabz_biases,
            0,
        ));
        let compute_shape = backend_compute::Shape {
            num_total_reqs: self.num_reqs,
            num_total_tokens: self.num_tokens,
        };
        let compute_buffers = backend_compute::Buffers {
            qkv: &self.qkvabz,
            a: &self.qkvabz,
            b: &self.qkvabz,
            z: &self.qkvabz,
            conv_weight: &self.weights.conv_weight,
            norm_weight: &self.weights.norm_weight,
            a_log: &self.weights.a_log,
            dt_bias: &self.weights.dt_bias,
            cu_tokens: self.batch_metadata.cu_tokens(),
            src_recurrent_state_slots: self.batch_metadata.src_recurrent_state_slots(),
            src_conv_state_slots: self.batch_metadata.src_conv_state_slots(),
            flat_recurrent_state_write_slots: self.batch_metadata.flat_recurrent_state_write_slots(),
            flat_conv_state_write_slots: self.batch_metadata.flat_conv_state_write_slots(),
            conv_state: &self.conv_state,
            conv_state_offset_bytes: 0,
            next_conv_state: &self.next_conv_state,
            next_conv_state_offset_bytes: 0,
            recurrent_state_arena: &self.recurrent_state_arena,
            recurrent_state_arena_offset_bytes: 0,
            conv_qkv: &self.conv_qkv,
            recurrent_output: &self.recurrent_output,
            norm_gated_output: &self.norm_gated_output,
        };
        // Reference recurrence uses the same full-state write map; decode writes no full state.
        builder.record_with_barrier_before(compute.invoke_with_candidate_state_update(
            compute_shape,
            compute_buffers,
            ReplayU32::Fixed(self.num_reqs),
            ReplayU32::Fixed(self.num_tokens),
        ));
        builder.record_with_barrier_before(output.invoke(
            self.num_tokens,
            ReplayU32::Fixed(self.num_tokens),
            &self.next_hidden_state,
            0,
            &self.norm_gated_output,
            0,
            &self.weights.output_weight,
            0,
            &self.weights.output_scales,
            0,
            &self.weights.output_biases,
            0,
        ));

        builder.build()
    }

    fn measure_subcomponents(&self, warmup_iters: usize, iters: usize, runs: usize) {
        let device = &self.device;
        let qkvabz_config = gdn_qkvabz_affine_config(self.model);
        let qkvabz = affine_quantized::Matmul::new(device, qkvabz_config);
        let compute = backend_compute::Compute::new(device, gdn_compute_config(self.model));
        let output_config = gdn_output_affine_config(self.model);
        let output = affine_quantized::Matmul::new(device, output_config);

        let qkvabz_replay = build_single_invocation_replay(
            &self.stream,
            qkvabz.invoke(
                self.num_tokens,
                ReplayU32::Fixed(self.num_tokens),
                &self.qkvabz,
                0,
                &self.hidden_state,
                0,
                &self.weights.qkvabz_weight,
                0,
                &self.weights.qkvabz_scales,
                0,
                &self.weights.qkvabz_biases,
                0,
            ),
        );
        let compute_shape = backend_compute::Shape {
            num_total_reqs: self.num_reqs,
            num_total_tokens: self.num_tokens,
        };
        let compute_buffers = backend_compute::Buffers {
            qkv: &self.qkvabz,
            a: &self.qkvabz,
            b: &self.qkvabz,
            z: &self.qkvabz,
            conv_weight: &self.weights.conv_weight,
            norm_weight: &self.weights.norm_weight,
            a_log: &self.weights.a_log,
            dt_bias: &self.weights.dt_bias,
            cu_tokens: self.batch_metadata.cu_tokens(),
            src_recurrent_state_slots: self.batch_metadata.src_recurrent_state_slots(),
            src_conv_state_slots: self.batch_metadata.src_conv_state_slots(),
            flat_recurrent_state_write_slots: self.batch_metadata.flat_recurrent_state_write_slots(),
            flat_conv_state_write_slots: self.batch_metadata.flat_conv_state_write_slots(),
            conv_state: &self.conv_state,
            conv_state_offset_bytes: 0,
            next_conv_state: &self.next_conv_state,
            next_conv_state_offset_bytes: 0,
            recurrent_state_arena: &self.recurrent_state_arena,
            recurrent_state_arena_offset_bytes: 0,
            conv_qkv: &self.conv_qkv,
            recurrent_output: &self.recurrent_output,
            norm_gated_output: &self.norm_gated_output,
        };
        let compute_replay = build_single_invocation_replay(
            &self.stream,
            compute.invoke_with_replay(
                compute_shape,
                compute_buffers,
                backend_compute::ReplayBuffers {
                    alpha: &self.replay_alpha,
                    k: &self.replay_k,
                    u: &self.replay_u,
                    qkv: &self.replay_qkv,
                    token_offset: 0,
                    num_total_tokens: self.num_tokens,
                },
                ReplayU32::Fixed(self.num_reqs),
                ReplayU32::Fixed(self.num_tokens),
                ReplayU32::Fixed(self.num_prefill_requests),
            ),
        );
        let output_replay = build_single_invocation_replay(
            &self.stream,
            output.invoke(
                self.num_tokens,
                ReplayU32::Fixed(self.num_tokens),
                &self.next_hidden_state,
                0,
                &self.norm_gated_output,
                0,
                &self.weights.output_weight,
                0,
                &self.weights.output_scales,
                0,
                &self.weights.output_biases,
                0,
            ),
        );

        self.measure_subcomponent("qkvabz", &qkvabz_replay, warmup_iters, iters, runs);
        self.measure_subcomponent("compute_mixed_replay", &compute_replay, warmup_iters, iters, runs);
        self.measure_subcomponent("output", &output_replay, warmup_iters, iters, runs);
    }

    fn measure_subcomponent(&self, name: &str, replay: &ReplayProgram, warmup_iters: usize, iters: usize, runs: usize) {
        let samples = measure_runs(runs, warmup_iters, iters, || {
            self.stream.submit_replay(replay).wait();
        });
        print_named_perf(
            &format!("gdn.{name}"),
            self.num_tokens,
            self.num_reqs,
            Some(self.existing_context_len),
            iters,
            &samples,
        );
    }
}

fn gdn_compute_config(model: GDNModelProfile) -> backend_compute::Config {
    backend_compute::Config {
        projection_layout: backend_compute::ProjectionLayout::PackedQkvabz,
        num_qk_heads: GDN_QK_HEADS.try_into().expect("GDN qk heads must fit u32"),
        qk_head_dim: GDN_QK_HEAD_DIM.try_into().expect("GDN qk head dim must fit u32"),
        num_v_heads: model.num_v_heads.try_into().expect("GDN V heads must fit u32"),
        v_head_dim: GDN_V_HEAD_DIM.try_into().expect("GDN V head dim must fit u32"),
        conv_kernel_size: GDN_CONV_KERNEL_SIZE
            .try_into()
            .expect("GDN conv kernel size must fit u32"),
        q_scale: (GDN_QK_HEAD_DIM as f32).sqrt().recip(),
        output_gate: backend_compute::OutputGate::Silu,
        norm_eps: GDN_EPS,
    }
}

struct RealGDNWeights {
    qkvabz_weight: Buffer,
    qkvabz_scales: Buffer,
    qkvabz_biases: Buffer,
    conv_weight: Buffer,
    norm_weight: Buffer,
    a_log: Buffer,
    dt_bias: Buffer,
    output_weight: Buffer,
    output_scales: Buffer,
    output_biases: Buffer,
}

impl RealGDNWeights {
    fn load(device: &Device, tensors: &SafeTensors<'_>, model: GDNModelProfile) -> Self {
        let prefix = format!("language_model.model.layers.{GDN_LAYER}.linear_attn");
        let qkv_weight = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_qkv.weight"),
            safetensors::Dtype::U32,
        );
        let a_weight = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_a.weight"),
            safetensors::Dtype::U32,
        );
        let b_weight = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_b.weight"),
            safetensors::Dtype::U32,
        );
        let z_weight = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_z.weight"),
            safetensors::Dtype::U32,
        );
        let qkv_scales = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_qkv.scales"),
            safetensors::Dtype::BF16,
        );
        let a_scales = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_a.scales"),
            safetensors::Dtype::BF16,
        );
        let b_scales = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_b.scales"),
            safetensors::Dtype::BF16,
        );
        let z_scales = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_z.scales"),
            safetensors::Dtype::BF16,
        );
        let qkv_biases = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_qkv.biases"),
            safetensors::Dtype::BF16,
        );
        let a_biases = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_a.biases"),
            safetensors::Dtype::BF16,
        );
        let b_biases = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_b.biases"),
            safetensors::Dtype::BF16,
        );
        let z_biases = tensor_bytes(
            model,
            tensors,
            &format!("{prefix}.in_proj_z.biases"),
            safetensors::Dtype::BF16,
        );
        let qkvabz_weight = concat_parts(&[&qkv_weight, &a_weight, &b_weight, &z_weight]);
        let qkvabz_scales = concat_parts(&[&qkv_scales, &a_scales, &b_scales, &z_scales]);
        let qkvabz_biases = concat_parts(&[&qkv_biases, &a_biases, &b_biases, &z_biases]);
        validate_qkvabz_sizes(model, &qkvabz_weight, &qkvabz_scales, &qkvabz_biases);
        Self {
            qkvabz_weight: Buffer::from_slice(device, &qkvabz_weight),
            qkvabz_scales: Buffer::from_slice(device, &qkvabz_scales),
            qkvabz_biases: Buffer::from_slice(device, &qkvabz_biases),
            conv_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    model,
                    tensors,
                    &format!("{prefix}.conv1d.weight"),
                    safetensors::Dtype::BF16,
                ),
            ),
            norm_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    model,
                    tensors,
                    &format!("{prefix}.norm.weight"),
                    safetensors::Dtype::BF16,
                ),
            ),
            a_log: Buffer::from_slice(
                device,
                &tensor_bytes(model, tensors, &format!("{prefix}.A_log"), safetensors::Dtype::BF16),
            ),
            dt_bias: Buffer::from_slice(
                device,
                &tensor_bytes(model, tensors, &format!("{prefix}.dt_bias"), safetensors::Dtype::BF16),
            ),
            output_weight: Buffer::from_slice(
                device,
                &tensor_bytes(
                    model,
                    tensors,
                    &format!("{prefix}.out_proj.weight"),
                    safetensors::Dtype::U32,
                ),
            ),
            output_scales: Buffer::from_slice(
                device,
                &tensor_bytes(
                    model,
                    tensors,
                    &format!("{prefix}.out_proj.scales"),
                    safetensors::Dtype::BF16,
                ),
            ),
            output_biases: Buffer::from_slice(
                device,
                &tensor_bytes(
                    model,
                    tensors,
                    &format!("{prefix}.out_proj.biases"),
                    safetensors::Dtype::BF16,
                ),
            ),
        }
    }

    fn as_borrowed(&self) -> GDNWeights<'_> {
        GDNWeights {
            qkvabz_weight: &self.qkvabz_weight,
            qkvabz_scales: &self.qkvabz_scales,
            qkvabz_biases: &self.qkvabz_biases,
            conv_weight: &self.conv_weight,
            norm_weight: &self.norm_weight,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            output_weight: &self.output_weight,
            output_scales: &self.output_scales,
            output_biases: &self.output_biases,
        }
    }
}

struct MappedFile {
    ptr: *mut libc::c_void,
    len: usize,
}

impl MappedFile {
    fn open(path: &Path) -> Self {
        let file = File::open(path).unwrap_or_else(|err| panic!("unable to open {}: {err}", path.display()));
        let len = file
            .metadata()
            .unwrap_or_else(|err| panic!("unable to stat {}: {err}", path.display()))
            .len() as usize;
        assert!(len > 0, "safetensors shard must not be empty");
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            panic!("unable to mmap {}: {}", path.display(), std::io::Error::last_os_error());
        }
        unsafe {
            let _ = libc::madvise(ptr, len, libc::MADV_RANDOM);
        }
        Self { ptr, len }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}
