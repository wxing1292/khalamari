use std::mem::size_of;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::Operator;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_backend_metal::operators::affine_quantized;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

const GROUP_SIZE: u32 = 64;
const BITS: u32 = 4;

const TOKENS_PER_PAGE: u32 = 16;
const KV_TOKEN_TILE_SIZE: u32 = 256;
const Q_HEAD_TILE_SIZE_CAP: u32 = 8;
const Q_TOKEN_TILE_SIZE: u32 = 8;
const TILED_KV_TOKEN_TILE_SIZE: u32 = 16;

const GDN_LAYER: usize = 0;
const GDN_QK_HEADS: usize = 16;
const GDN_QK_HEAD_DIM: usize = 128;
const GDN_V_HEAD_DIM: usize = 128;
const GDN_QK_DIM: usize = GDN_QK_HEADS * GDN_QK_HEAD_DIM;
const GDN_CONV_KERNEL_SIZE: usize = 4;
const GDN_EPS: f32 = 1.0e-6;

#[derive(Clone, Copy)]
struct GDNModelProfile {
    shard: &'static str,
    hidden_dim: usize,
    num_v_heads: usize,
}

impl GDNModelProfile {
    fn parse(value: &str) -> Self {
        match value {
            "27b" => {
                Self {
                    shard: "model-00001-of-00003.safetensors",
                    hidden_dim: 5120,
                    num_v_heads: 48,
                }
            },
            "35b" => {
                Self {
                    shard: "model-00001-of-00004.safetensors",
                    hidden_dim: 2048,
                    num_v_heads: 32,
                }
            },
            _ => panic!("expected --gdn-model 27b or 35b"),
        }
    }
    fn v_dim(self) -> usize {
        self.num_v_heads * GDN_V_HEAD_DIM
    }
    fn conv_dim(self) -> usize {
        2 * GDN_QK_DIM + self.v_dim()
    }
    fn qkvabz_dim(self) -> usize {
        self.conv_dim() + 2 * self.num_v_heads + self.v_dim()
    }
}

#[path = "gdn/fixture.rs"]
mod fixture;

fn main() {
    fixture::run(Args::parse());
}
struct Args {
    model: GDNModelProfile,
    model_dir: PathBuf,
    tokens: Vec<u32>,
    contexts: Vec<u32>,
    num_reqs: Vec<u32>,
    tokens_per_req: Option<Vec<u32>>,
    prefill_requests: u32,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
    subcomponents: bool,
    compare_recurrent: bool,
}

impl Args {
    fn parse() -> Self {
        let mut args = Self {
            model: GDNModelProfile::parse("35b"),
            model_dir: PathBuf::new(),
            tokens: vec![1, 2, 4, 8, 16, 32, 64],
            contexts: Vec::new(),
            num_reqs: vec![1],
            tokens_per_req: None,
            prefill_requests: 0,
            iters: 50,
            warmup_iters: 20,
            runs: 1,
            subcomponents: false,
            compare_recurrent: false,
        };
        let mut values = std::env::args().skip(1);
        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(),
                "--gdn-model" => args.model = GDNModelProfile::parse(&next_arg(&mut values, &arg)),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut values, &arg)),
                "--tokens" => args.tokens = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--contexts" => args.contexts = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--num-reqs" => args.num_reqs = parse_u32_list(&next_arg(&mut values, &arg), &arg),
                "--tokens-per-req" => args.tokens_per_req = Some(parse_u32_list(&next_arg(&mut values, &arg), &arg)),
                "--prefill-requests" => args.prefill_requests = parse_u32(&next_arg(&mut values, &arg), &arg),
                "--iters" => args.iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--runs" => args.runs = parse_usize(&next_arg(&mut values, &arg), &arg),
                "--subcomponents" => args.subcomponents = true,
                "--compare-recurrent" => args.compare_recurrent = true,
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(
            !args.tokens.is_empty(),
            "--tokens must include at least one token count"
        );
        assert!(
            !args.num_reqs.is_empty(),
            "--num-reqs must include at least one request count"
        );
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        for &num_reqs in &args.num_reqs {
            assert!(num_reqs > 0, "--num-reqs entries must be positive");
        }
        if let Some(counts) = &args.tokens_per_req {
            assert!(
                counts.iter().all(|&count| count > 0),
                "request segments must be non-empty"
            );
            assert!(
                args.prefill_requests as usize <= counts.len(),
                "prefill requests must be a request prefix"
            );
            let _ = counts
                .iter()
                .try_fold(0_u32, |total, &count| total.checked_add(count))
                .expect("request token total must fit u32");
        } else {
            assert!(
                args.num_reqs.iter().all(|&count| args.prefill_requests <= count),
                "prefill requests must be a request prefix"
            );
        }
        args
    }
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> String {
    iter.next()
        .unwrap_or_else(|| panic!("{name} requires a value; pass --help for usage"))
}

fn parse_string_list(value: &str, name: &str) -> Vec<String> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert!(
        !values.is_empty(),
        "{name} must contain at least one comma-separated value"
    );
    values
}

fn parse_u32_list(value: &str, name: &str) -> Vec<u32> {
    parse_string_list(value, name)
        .into_iter()
        .map(|part| parse_u32(&part, name))
        .collect()
}

fn parse_u32(value: &str, name: &str) -> u32 {
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid {name} value {value:?}: {err}"))
}

fn parse_usize(value: &str, name: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid {name} value {value:?}: {err}"))
}

fn print_help_and_exit() -> ! {
    println!(
        r#"qwen35_gdn bench
--model-dir PATH
--gdn-model 27b|35b
--tokens 1,2,4,8,16,32,64
--contexts 0,128
--num-reqs 1,2,4
--tokens-per-req 9,7,1,4  (overrides --tokens and --num-reqs)
--prefill-requests N      (first N requests; default 0)
--compare-recurrent      (all-recurrent reference, alternating with chunkwise/replay)
--subcomponents
--iters N
--warmup-iters N
--runs N"#
    );
    std::process::exit(0);
}

fn hidden_fixture(tokens: usize, hidden_dim: usize) -> Vec<u16> {
    (0..tokens * hidden_dim)
        .map(|index| bf16::from_f32(((index % 23) as f32 - 11.0) * 0.03125).to_bits())
        .collect()
}

fn valid_num_reqs(num_tokens: u32, num_reqs: u32) -> bool {
    num_reqs > 0 && num_reqs <= num_tokens
}

fn request_token_counts(num_tokens: u32, num_reqs: u32) -> Vec<u32> {
    assert!(
        valid_num_reqs(num_tokens, num_reqs),
        "request token counts require 1 <= num_reqs <= tokens"
    );
    let base = num_tokens / num_reqs;
    let remainder = num_tokens % num_reqs;
    (0..num_reqs)
        .map(|req_index| base + u32::from(req_index < remainder))
        .collect()
}

fn cu_tokens(num_tokens_per_req: &[u32]) -> Vec<i32> {
    let mut cu = Vec::with_capacity(num_tokens_per_req.len() + 1);
    cu.push(0);
    let mut total = 0u32;
    for &num_req_tokens in num_tokens_per_req {
        assert!(num_req_tokens > 0, "bench request segments must be non-empty");
        total = total
            .checked_add(num_req_tokens)
            .expect("bench cu_tokens total overflow");
        cu.push(total.try_into().expect("cu_tokens value must fit i32"));
    }
    cu
}

fn gdn_conv_state_fixture(model: GDNModelProfile, existing_context_len: u32, num_reqs: usize, len: usize) -> Vec<u16> {
    gdn_state_fixture(
        existing_context_len,
        len,
        num_reqs,
        model.conv_dim() * (GDN_CONV_KERNEL_SIZE - 1),
        0.00390625,
    )
}

fn gdn_recurrent_state_fixture(
    model: GDNModelProfile,
    existing_context_len: u32,
    num_reqs: usize,
    len: usize,
) -> Vec<u16> {
    gdn_state_fixture(
        existing_context_len,
        len,
        num_reqs,
        model.num_v_heads * GDN_V_HEAD_DIM * GDN_QK_HEAD_DIM,
        0.000_244_140_63,
    )
}

fn gdn_state_fixture(existing_context_len: u32, len: usize, num_reqs: usize, slot_len: usize, scale: f32) -> Vec<u16> {
    let mut values = vec![0; len];
    if existing_context_len == 0 {
        return values;
    }
    let src_state_len = num_reqs * slot_len;
    assert!(
        src_state_len <= len,
        "GDN state fixture source state slot cannot exceed state arena"
    );
    for (index, value) in values.iter_mut().take(src_state_len).enumerate() {
        *value = bf16::from_f32(((index % 29) as f32 - 14.0) * scale).to_bits();
    }
    values
}

fn flat_token_indices(num_tokens_per_req: &[u32], existing_context_len: u32) -> Vec<u32> {
    num_tokens_per_req
        .iter()
        .flat_map(|&num_req_tokens| existing_context_len..existing_context_len + num_req_tokens)
        .collect()
}

fn identity_u32(num_values: u32) -> Vec<u32> {
    (0..num_values).collect()
}

fn tensor_bytes(model: GDNModelProfile, tensors: &SafeTensors<'_>, name: &str, dtype: safetensors::Dtype) -> Vec<u8> {
    let view = tensors
        .tensor(name)
        .unwrap_or_else(|err| panic!("missing safetensor {name}: {err:?}"));
    assert_eq!(view.dtype(), dtype, "unexpected dtype for tensor {name}");
    validate_tensor_shape(model, name, &view);
    view.data().to_vec()
}

fn validate_tensor_shape(model: GDNModelProfile, name: &str, view: &TensorView<'_>) {
    let shape = view.shape();
    if name.ends_with("linear_attn.in_proj_qkv.weight") {
        assert_eq!(shape, &[model.conv_dim(), packed_k_words(model.hidden_dim)]);
    } else if name.ends_with("linear_attn.in_proj_qkv.scales") || name.ends_with("linear_attn.in_proj_qkv.biases") {
        assert_eq!(shape, &[model.conv_dim(), groups(model.hidden_dim)]);
    } else if name.ends_with("linear_attn.in_proj_a.weight") || name.ends_with("linear_attn.in_proj_b.weight") {
        assert_eq!(shape, &[model.num_v_heads, packed_k_words(model.hidden_dim)]);
    } else if name.ends_with("linear_attn.in_proj_a.scales")
        || name.ends_with("linear_attn.in_proj_a.biases")
        || name.ends_with("linear_attn.in_proj_b.scales")
        || name.ends_with("linear_attn.in_proj_b.biases")
    {
        assert_eq!(shape, &[model.num_v_heads, groups(model.hidden_dim)]);
    } else if name.ends_with("linear_attn.in_proj_z.weight") {
        assert_eq!(shape, &[model.v_dim(), packed_k_words(model.hidden_dim)]);
    } else if name.ends_with("linear_attn.in_proj_z.scales") || name.ends_with("linear_attn.in_proj_z.biases") {
        assert_eq!(shape, &[model.v_dim(), groups(model.hidden_dim)]);
    } else if name.ends_with("linear_attn.conv1d.weight") {
        assert_eq!(shape, &[model.conv_dim(), GDN_CONV_KERNEL_SIZE, 1]);
    } else if name.ends_with("linear_attn.norm.weight") {
        assert_eq!(shape, &[GDN_V_HEAD_DIM]);
    } else if name.ends_with("linear_attn.A_log") || name.ends_with("linear_attn.dt_bias") {
        assert_eq!(shape, &[model.num_v_heads]);
    } else if name.ends_with("linear_attn.out_proj.weight") {
        assert_eq!(shape, &[model.hidden_dim, packed_k_words(model.v_dim())]);
    } else if name.ends_with("linear_attn.out_proj.scales") || name.ends_with("linear_attn.out_proj.biases") {
        assert_eq!(shape, &[model.hidden_dim, groups(model.v_dim())]);
    } else {
        panic!("unexpected GDN tensor name {name}");
    }
}

fn validate_qkvabz_sizes(model: GDNModelProfile, weight: &[u8], scales: &[u8], biases: &[u8]) {
    assert_eq!(
        weight.len(),
        model.qkvabz_dim() * packed_k_words(model.hidden_dim) * size_of::<u32>()
    );
    assert_eq!(
        scales.len(),
        model.qkvabz_dim() * groups(model.hidden_dim) * Dtype::Bfloat16.item_size()
    );
    assert_eq!(biases.len(), scales.len());
}

fn concat_parts(parts: &[&[u8]]) -> Vec<u8> {
    let len = parts.iter().map(|part| part.len()).sum();
    let mut out = Vec::with_capacity(len);
    for part in parts {
        out.extend_from_slice(part);
    }
    out
}

fn packed_k_words(k: usize) -> usize {
    k * BITS as usize / 32
}

fn groups(k: usize) -> usize {
    k / GROUP_SIZE as usize
}

fn measure_runs(runs: usize, warmup_iters: usize, iters: usize, mut run: impl FnMut()) -> Vec<f64> {
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        for _ in 0..warmup_iters {
            run();
        }
        let mut duration = Duration::ZERO;
        for _ in 0..iters {
            let start = Instant::now();
            run();
            duration += start.elapsed();
        }
        samples.push(duration.as_secs_f64() * 1_000_000.0 / iters as f64);
    }
    samples
}

fn print_perf(
    num_tokens: u32,
    num_reqs: u32,
    existing_context_len: Option<u32>,
    path: Option<&str>,
    iters: usize,
    samples: &[f64],
) {
    let median_us = median(samples);
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    let context_text = existing_context_len
        .map(|value| format!(" ctx={value}"))
        .unwrap_or_default();
    let path_text = path.map(|value| format!(" path={value}")).unwrap_or_default();
    println!(
        "perf component=gdn impl=full-forward-replay tokens={num_tokens} num_reqs={num_reqs}{context_text}{path_text} \
         iters={iters} runs={} median_us={median_us:.3} samples_us=[{sample_text}]",
        samples.len()
    );
}

fn print_skip(num_tokens: u32, num_reqs: u32, existing_context_len: Option<u32>, path: Option<&str>, reason: &str) {
    let context_text = existing_context_len
        .map(|value| format!(" ctx={value}"))
        .unwrap_or_default();
    let path_text = path.map(|value| format!(" path={value}")).unwrap_or_default();
    println!("skip component=gdn tokens={num_tokens} num_reqs={num_reqs}{context_text}{path_text} reason={reason}",);
}

fn print_named_perf(
    component: &str,
    num_tokens: u32,
    num_reqs: u32,
    existing_context_len: Option<u32>,
    iters: usize,
    samples: &[f64],
) {
    let median_us = median(samples);
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    let context_text = existing_context_len
        .map(|value| format!(" ctx={value}"))
        .unwrap_or_default();
    println!(
        "perf component={component} impl=subcomponent-replay tokens={num_tokens} num_reqs={num_reqs}{context_text} \
         iters={iters} runs={} median_us={median_us:.3} samples_us=[{sample_text}]",
        samples.len()
    );
}

fn build_single_invocation_replay<I>(stream: &Stream, invocation: I) -> ReplayProgram
where
    I: Operator,
{
    let mut recorder = ReplayRecorder::new(stream.create_replay_program());
    recorder.record(ReplayOp::opaque(invocation));
    recorder.build()
}

fn gdn_qkvabz_affine_config(model: GDNModelProfile) -> affine_quantized::Config {
    affine_quantized::Config {
        n: model.qkvabz_dim().try_into().expect("GDN qkvabz n must fit i32"),
        k: model.hidden_dim.try_into().expect("GDN qkvabz k must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("GDN group size must fit i32"),
        bits: BITS.try_into().expect("GDN bits must fit i32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    }
}

fn gdn_output_affine_config(model: GDNModelProfile) -> affine_quantized::Config {
    affine_quantized::Config {
        n: model.hidden_dim.try_into().expect("GDN output n must fit i32"),
        k: model.v_dim().try_into().expect("GDN output k must fit i32"),
        group_size: GROUP_SIZE.try_into().expect("GDN group size must fit i32"),
        bits: BITS.try_into().expect("GDN bits must fit i32"),
        input_dtype: Dtype::Bfloat16,
        output_dtype: Dtype::Bfloat16,
        scale_bias_dtype: Dtype::Bfloat16,
    }
}

fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}
