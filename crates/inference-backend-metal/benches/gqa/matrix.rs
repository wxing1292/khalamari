//! Serial, model-independent GQA candidate runner. One JSON command per stdin line.
use std::io::BufRead;
use std::io::Write;
use std::io::{self};
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::gqa::fp8::bf16_to_fp8_e4m3;
use inference_backend_metal::components::gqa::fp8::fp8_e4m3_to_bf16;
use inference_backend_metal::components::gqa::kv_page_write::PageTableLayout;
use inference_backend_metal::components::gqa::sdpa::ExecutionVariant;
use inference_backend_metal::components::gqa::sdpa::MapKernelKind;
use inference_backend_metal::components::gqa::sdpa::sliced_kv;
use inference_backend_metal::components::gqa::sdpa::sliced_q;
use inference_backend_metal::components::gqa::sdpa::{self};
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::metal::Stream;
use serde_json::Value;
use serde_json::json;

fn u(v: &Value, name: &str) -> u32 {
    u32::try_from(v[name].as_u64().unwrap_or_else(|| panic!("missing u32 field {name}"))).unwrap()
}

struct Inputs {
    workload: Value,
    config: sliced_q::Config,
    q: Buffer,
    kv: Buffer,
    slots: Buffer,
    pages: Buffer,
    visible: Buffer,
    q_values: Vec<u16>,
    kv_values: Vec<u8>,
    num_tokens: u32,
    oracle: Vec<(usize, f32)>,
}

impl Inputs {
    fn new(device: &Device, workload: Value) -> Self {
        let (history, queries, requests, ratio, hkv, dim, page) = (
            u(&workload, "history"),
            u(&workload, "queries"),
            u(&workload, "requests"),
            u(&workload, "ratio"),
            u(&workload, "kv_heads"),
            u(&workload, "dim"),
            u(&workload, "page"),
        );
        assert!(queries > 0 && requests > 0 && hkv > 0);
        let total_kv = history.checked_add(queries).unwrap();
        let blocks = total_kv.div_ceil(page);
        let num_tokens = queries.checked_mul(requests).unwrap();
        let config = sliced_q::Config {
            num_q_heads: ratio.checked_mul(hkv).unwrap(),
            num_kv_heads: hkv,
            head_dim: dim,
            scale: (dim as f32).sqrt().recip(),
            page_bytes: hkv
                .checked_mul(page)
                .and_then(|n| n.checked_mul(dim))
                .and_then(|n| n.checked_mul(2))
                .expect("KV page byte length must fit u32"),
            dtype: Dtype::Bfloat16,
            page_table_layout: PageTableLayout {
                num_req_slots: requests,
                num_gqa_layers: 1,
                num_blocks: blocks,
                num_page_ids_per_block: 1,
            },
        };
        config.validate();
        let mut state = 17_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bf16::from_f32((state >> 8) as f32 / 16_777_216.0 - 0.5).to_bits()
        };
        let q_values = (0..num_tokens as usize * config.num_q_heads as usize * dim as usize)
            .map(|_| next())
            .collect::<Vec<_>>();
        let num_pages = requests.checked_mul(blocks).expect("page count must fit u32");
        let kv_values = (0..num_pages as usize * config.page_bytes as usize)
            .map(|index| {
                let value = bf16::from_bits(next());
                // Positive V makes a zero-output kernel fail the numerical gate even at long contexts.
                let value = if index % config.page_bytes as usize >= config.page_bytes as usize / 2 {
                    bf16::from_f32(0.25 + 0.25 * value.to_f32())
                } else {
                    value
                };
                bf16_to_fp8_e4m3(value)
            })
            .collect::<Vec<_>>();
        let slots = (0..num_tokens).map(|t| t / queries).collect::<Vec<_>>();
        let visible = (0..num_tokens)
            .flat_map(|t| [0, history + t % queries + 1])
            .collect::<Vec<_>>();
        let pages = (0..num_pages).rev().collect::<Vec<_>>();
        let mut inputs = Self {
            workload,
            config,
            q: Buffer::from_slice(device, &q_values),
            kv: Buffer::from_slice(device, &kv_values),
            slots: Buffer::from_slice(device, &slots),
            pages: Buffer::from_slice(device, &pages),
            visible: Buffer::from_slice(device, &visible),
            q_values,
            kv_values,
            num_tokens,
            oracle: Vec::new(),
        };
        // Check all dimensions of one head at first/middle/last flat Q tokens.
        // The paged cache is reversed. Each Q token has its own causal end.
        for token in [0, num_tokens / 2, num_tokens - 1] {
            let head = (token % hkv) * ratio + ratio - 1;
            let end = history + token % queries + 1;
            let req = token / queries;
            let q_begin = (token as usize * config.num_q_heads as usize + head as usize) * dim as usize;
            let kv_head = head / ratio;
            let read = |t: u32, d: u32, value: bool| {
                let physical = num_pages - 1 - (req * blocks + t / page);
                let offset = physical as usize * config.page_bytes as usize
                    + ((kv_head + if value { hkv } else { 0 }) * page + t % page) as usize * dim as usize
                    + d as usize;
                fp8_e4m3_to_bf16(inputs.kv_values[offset]).to_f32()
            };
            let scores = (0..end)
                .map(|t| {
                    (0..dim)
                        .map(|d| bf16::from_bits(inputs.q_values[q_begin + d as usize]).to_f32() * read(t, d, false))
                        .sum::<f32>()
                        * config.scale
                })
                .collect::<Vec<_>>();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
            let sum = weights.iter().sum::<f32>();
            for d in 0..dim {
                let expected = weights
                    .iter()
                    .enumerate()
                    .map(|(t, w)| w * read(t as u32, d, true))
                    .sum::<f32>()
                    / sum;
                inputs.oracle.push((q_begin + d as usize, expected));
            }
        }
        inputs
    }

    fn execution(&self, c: &Value) -> ExecutionVariant {
        let config = sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: self.config.num_q_heads,
            num_kv_heads: self.config.num_kv_heads,
            head_dim: self.config.head_dim,
            tokens_per_page: self.config.num_tokens_per_page(),
        };
        match c["layout"].as_str().unwrap() {
            "single" => ExecutionVariant::sliced_kv(config, self.config.head_dim, 128, u(c, "heads")),
            layout => {
                ExecutionVariant::sliced_q(
                    config,
                    match layout {
                        "token" => MapKernelKind::QTokensPerHead,
                        "head" => MapKernelKind::QHeadsPerToken,
                        "joint" => MapKernelKind::QTokensAndHeads,
                        _ => panic!("unknown Q-token/head assignment"),
                    },
                    u(c, "tile"),
                    u(c, "stage"),
                    u(c, "heads"),
                )
            },
        }
    }

    fn program(&self, device: &Device, stream: &Stream, c: &Value, inner: u32) -> (ReplayProgram, Buffer, usize, u32) {
        let execution = self.execution(c);
        let tile = execution.map.thread_block.max_q_tokens;
        let queries = u(&self.workload, "queries");
        let history = u(&self.workload, "history");
        let slice = u(c, "slice");
        assert!(slice > 0);
        let mut ranges = Vec::new();
        let mut templates = Vec::new();
        let mut cu = vec![0_u32];
        for req in 0..u(&self.workload, "requests") {
            for local in (0..queries).step_by(tile as usize) {
                let local_end = (local + tile).min(queries);
                let range = (ranges.len() / 2) as u32;
                ranges.extend([req * queries + local, req * queries + local_end]);
                let end = history + local_end;
                for begin in (0..end).step_by(slice as usize) {
                    templates.extend([range, begin, (begin + slice).min(end)]);
                }
                cu.push((templates.len() / 3) as u32);
            }
        }
        let nq = (ranges.len() / 2) as u32;
        let nm = (templates.len() / 3) as u32;
        let ranges = Buffer::from_slice(device, &ranges);
        let templates = Buffer::from_slice(device, &templates);
        let cu = Buffer::from_slice(device, &cu);
        let output = Buffer::new_zeroed(
            device,
            self.num_tokens as usize * self.config.num_q_heads as usize * self.config.head_dim as usize * 2,
        );
        let mut builder = stream.create_replay_program();
        let fixed = ReplayU32::Fixed;
        let direct = c["direct"].as_bool().unwrap();
        let scratch;
        if execution.map.kind == MapKernelKind::SlicedKv {
            assert!(!direct);
            let config = sliced_kv::Config {
                num_q_heads: self.config.num_q_heads,
                num_kv_heads: self.config.num_kv_heads,
                head_dim: self.config.head_dim,
                scale: self.config.scale,
                page_bytes: self.config.page_bytes,
                dtype: self.config.dtype,
                page_table_layout: self.config.page_table_layout,
            };
            let shape = sliced_kv::Shape {
                num_total_tokens: self.num_tokens,
                num_total_sdpa_map_task_templates: nm,
            };
            let kernels = sliced_kv::Compute::new(device, config, execution, shape);
            let partial = Buffer::new_zeroed(
                device,
                nm as usize * config.num_q_heads as usize * config.head_dim as usize * 2,
            );
            let sums = Buffer::new_zeroed(device, nm as usize * config.num_q_heads as usize * 4);
            let maxs = Buffer::new_zeroed(device, sums.len_bytes());
            scratch = partial.len_bytes() + sums.len_bytes() + maxs.len_bytes();
            for _ in 0..inner {
                builder.record_with_barrier_before(kernels.invoke_map(
                    sliced_kv::MapBuffers {
                        q: &self.q,
                        kv_pages: &self.kv,
                        req_slots: &self.slots,
                        page_ids: &self.pages,
                        sdpa_map_task_templates: &templates,
                        partial_output: &partial,
                        partial_exp_sums: &sums,
                        partial_max_logits: &maxs,
                    },
                    fixed(0),
                    fixed(self.num_tokens),
                    fixed(nm),
                ));
                builder.record_with_barrier_before(kernels.invoke_reduce(
                    sliced_kv::ReduceBuffers {
                        partial_output: &partial,
                        partial_exp_sums: &sums,
                        partial_max_logits: &maxs,
                        cu_sdpa_partial_outputs: &cu,
                        output: &output,
                    },
                    fixed(self.num_tokens),
                ));
            }
        } else {
            let shape = sliced_q::Shape {
                num_total_tokens: self.num_tokens,
                num_total_q_token_tiles: nq,
                num_total_sdpa_map_task_templates: nm,
            };
            let config = self.config;
            let kernels = if direct {
                sliced_q::Compute::new_direct(device, config, execution, shape)
            } else {
                sliced_q::Compute::new(device, config, execution, shape)
            };
            if direct {
                scratch = 0;
                for _ in 0..inner {
                    builder.record_with_barrier_before(kernels.invoke_direct(
                        sliced_q::DirectBuffers {
                            q: &self.q,
                            kv_pages: &self.kv,
                            req_slots: &self.slots,
                            page_ids: &self.pages,
                            visible_kv_token_ranges: &self.visible,
                            q_token_ranges: &ranges,
                            sdpa_map_task_templates: &templates,
                            output: &output,
                        },
                        fixed(0),
                        fixed(self.num_tokens),
                        fixed(nq),
                    ));
                }
            } else {
                let partial = Buffer::new_zeroed(device, config.partial_output_bytes(execution, shape));
                let sums = Buffer::new_zeroed(device, config.partial_output_stats_bytes(execution, shape));
                let maxs = Buffer::new_zeroed(device, sums.len_bytes());
                scratch = partial.len_bytes() + sums.len_bytes() + maxs.len_bytes();
                for _ in 0..inner {
                    builder.record_with_barrier_before(kernels.invoke_map(
                        sliced_q::MapBuffers {
                            q: &self.q,
                            kv_pages: &self.kv,
                            req_slots: &self.slots,
                            page_ids: &self.pages,
                            visible_kv_token_ranges: &self.visible,
                            q_token_ranges: &ranges,
                            sdpa_map_task_templates: &templates,
                            partial_output: &partial,
                            partial_exp_sums: &sums,
                            partial_max_logits: &maxs,
                        },
                        fixed(0),
                        fixed(self.num_tokens),
                        fixed(nq),
                        fixed(nm),
                    ));
                    builder.record_with_barrier_before(kernels.invoke_reduce(
                        sliced_q::ReduceBuffers {
                            partial_output: &partial,
                            partial_exp_sums: &sums,
                            partial_max_logits: &maxs,
                            q_token_ranges: &ranges,
                            cu_sdpa_partial_outputs: &cu,
                            output: &output,
                        },
                        fixed(nq),
                    ));
                }
            }
        }
        (builder.build(), output, scratch, nm)
    }
}

fn check(a: &[u16], b: &[u16]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0_f32;
    let mut sum = 0.0_f32;
    for (&a, &b) in a.iter().zip(b) {
        let error = (bf16::from_bits(a).to_f32() - bf16::from_bits(b).to_f32()).abs();
        assert!(error.is_finite());
        max = max.max(error);
        sum += error;
    }
    let mean = sum / a.len() as f32;
    assert!(max <= 0.02 && mean <= 0.005, "numeric mismatch max={max} mean={mean}");
    (max, mean)
}

fn measure(stream: &Stream, replay: &ReplayProgram, inner: u32) -> f64 {
    let start = Instant::now();
    stream.submit_replay(replay).wait();
    start.elapsed().as_secs_f64() * 1000.0 / inner as f64
}

pub fn main() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let mut inputs: Option<Inputs> = None;
    for line in io::stdin().lock().lines() {
        let command: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if inputs.as_ref().is_none_or(|i| i.workload != command["workload"]) {
                inputs = Some(Inputs::new(&device, command["workload"].clone()));
            }
            let inputs = inputs.as_ref().unwrap();
            let baseline = json!({"layout":"single","tile":1,"heads":u(&inputs.workload,"ratio").min(8),
                "stage":inputs.config.head_dim,"slice":u(&inputs.workload,"history")+u(&inputs.workload,"queries"),"direct":false});
            let inner = u(&command, "inner");
            assert!(inner > 0);
            let (base, base_out, ..) = inputs.program(&device, &stream, &baseline, inner);
            let (candidate, out, scratch, tasks) = inputs.program(&device, &stream, &command["candidate"], inner);
            for _ in 0..2 {
                stream.submit_replay(&base).wait();
                stream.submit_replay(&candidate).wait();
            }
            let actual = out.read_typed::<u16>(0, inputs.q_values.len());
            let reference = base_out.read_typed::<u16>(0, actual.len());
            let (max_error, mean_error) = check(&actual, &reference);
            for &(index, expected) in &inputs.oracle {
                assert!(
                    (bf16::from_bits(actual[index]).to_f32() - expected).abs() <= 0.02,
                    "CPU reference mismatch at {index}"
                );
                assert!(
                    (bf16::from_bits(reference[index]).to_f32() - expected).abs() <= 0.02,
                    "baseline CPU reference mismatch at {index}"
                );
            }
            let mut samples = Vec::new();
            for repeat in 0..u(&command, "repeats") {
                // Alternate baseline/candidate/baseline with candidate/baseline/candidate.
                if repeat % 2 == 0 {
                    let before = measure(&stream, &base, inner);
                    let current = measure(&stream, &candidate, inner);
                    let after = measure(&stream, &base, inner);
                    samples.push(json!({"before_ms":before,"candidate_ms":current,"after_ms":after}));
                } else {
                    let first = measure(&stream, &candidate, inner);
                    let base_ms = measure(&stream, &base, inner);
                    let last = measure(&stream, &candidate, inner);
                    samples.push(json!({"before_ms":base_ms,"candidate_ms":(first+last)/2.0,
                        "after_ms":base_ms,"candidate_first_ms":first,"candidate_last_ms":last}));
                }
            }
            json!({"status":"measured","device":device.name(),"metric":"amortized_replay_wall_ms","inner":inner,
                "baseline":baseline,"samples":samples,"scratch_bytes":scratch,"map_tasks":tasks,
                "max_abs_error":max_error,"mean_abs_error":mean_error,"cpu_reference_values":inputs.oracle.len()})
        }));
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                json!({"status":"failed","reason":
            error.downcast_ref::<String>().cloned().or_else(||error.downcast_ref::<&str>().map(|s|s.to_string())).unwrap_or_default()})
            },
        };
        println!("{result}");
        io::stdout().flush().unwrap();
    }
}
