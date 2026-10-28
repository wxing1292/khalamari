use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use inference_executor_core::model::ExecutionSubmission;
use inference_executor_core::model::qwen::v4_x::reference;
use inference_executor_core::ngram_embedding::core::HashParameters;
use inference_executor_core::ngram_embedding::core::NGramEmbeddingCore;
use inference_runtime_core::compute::DecoderSyncBlocks;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::config::SamplingConfig;
use safetensors::tensor::View;
use safetensors::tensor::serialize_to_file;

use super::*;

#[test]
fn test_unsupported_metal_geometry_fails_before_weight_io() {
    let missing_model = std::env::temp_dir().join(format!("khalamari-missing-qwen4x-{}", std::process::id()));
    assert!(!missing_model.exists());
    for unsupported in ["head_geometry", "expert_count", "shader_dimension", "residual_width"] {
        let mut model: Qwen4xModelConfig = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../inference-executor-core/src/model/qwen/v4_x/config_fixture.json"
        )))
        .unwrap();
        match unsupported {
            "head_geometry" => model.text_config.num_attention_heads = 12,
            "expert_count" => model.text_config.num_experts = 1024,
            "shader_dimension" => model.text_config.max_position_embeddings = i32::MAX as usize + 1,
            "residual_width" => {
                model.text_config.ple_layer_ids.clear();
                model.text_config.hidden_size = 1 << 30;
            },
            _ => unreachable!(),
        }
        model.normalize().unwrap();
        let error = Qwen4xExecutor::new(
            &missing_model,
            model,
            Qwen4xExecutorConfig {
                max_requests: 1,
                max_tokens: 1,
                max_tokens_per_request: 1,
                num_cache_pages: 1,
                num_tokens_per_block: 512,
            },
        )
        .err()
        .expect("Metal must reject unsupported geometry before opening the model directory");
        assert!(error.to_string().contains("Qwen4x Metal"), "{unsupported}: {error}");
    }
}

#[test]
fn test_first8_cpu_reference_fixed_random_and_mixed_batch() {
    for (seed, dense) in [
        (0, false),
        (0x4e45_5854_5245_4653, false),
        (0, true),
        (0x4e45_5854_5245_4653, true),
    ] {
        let fixture = Fixture::new(seed, dense);
        let mut model = fixture.executor();
        let mut next_page = 0;
        let mut a = Cache::new(0);
        let mut b = Cache::new(1);
        let tokens_a = [3, 7, 0, 11, 19, 23, 5];
        let tokens_b = [29, 31, 37];
        let req_a = a.request(&model, 0, &tokens_a, false, &mut next_page);
        let req_b = b.request(&model, 0, &tokens_b, false, &mut next_page);
        let outputs = execute(&mut model, 1, vec![req_a, req_b]);
        for (slot, actual) in outputs {
            let tokens = if slot == 0 {
                tokens_a.as_slice()
            } else {
                tokens_b.as_slice()
            };
            let expected = reference::forward(&fixture.config, &model.bindings, &fixture.decoded, tokens);
            assert_close(&actual, &expected, 0.025);
        }
        model.state.finish_commit();
        model.gdn.finish_commit();
    }
}

#[test]
fn test_first8_chunked_decode_prefix_restore_and_hibernation() {
    let fixture = Fixture::new(0x4341_4348_455f_4e58, false);
    let mut model = fixture.executor();
    let mut next_page = 0;
    let mut full = Cache::new(0);
    let mut chunked = Cache::new(1);
    let tokens: Vec<u32> = (0..512).map(|i| (i * 17 % 251 + 1) as u32).collect();
    let request = full.request(&model, 0, &tokens, false, &mut next_page);
    let expected = execute(&mut model, 1, vec![request]).remove(0).1;
    let request = chunked.request(&model, 0, &tokens[..509], false, &mut next_page);
    execute(&mut model, 2, vec![request]);
    let request = chunked.request(&model, 509, &tokens[509..], true, &mut next_page);
    let actual = execute(&mut model, 3, vec![request]).remove(0).1;
    assert_close(&actual, &expected[509 * 32..], 0.025);
    let request = full.request(&model, 512, &[43], true, &mut next_page);
    let expected = execute(&mut model, 4, vec![request]).remove(0).1;
    model.state.finish_commit();
    model.gdn.finish_commit();
    model.reset_req_slots(&[1]);
    let mut restored = full.clone();
    restored.slot = 1;
    restored.kv.truncate(1);
    restored.state.truncate(1);
    let request = restored.request(&model, 512, &[43], true, &mut next_page);
    let actual = execute(&mut model, 5, vec![request]).remove(0).1;
    assert_close(&actual, &expected, 0.025);
    let request = restored.request(&model, 513, &[47, 53], false, &mut next_page);
    let expected = execute(&mut model, 6, vec![request]).remove(0).1;
    let snapshot = fixture.path.join("state");
    let plan =
        ExecutorHibernationPlan::selected(std::iter::once(0..1).collect(), std::iter::once(0..next_page).collect());
    model.clear_replay_cache();
    model.unload_state(&snapshot, &plan).unwrap();
    model.unload_weights();
    model.load_weights().unwrap();
    model.load_state(&snapshot, &plan).unwrap();
    let request = full.request(&model, 513, &[47, 53], false, &mut next_page);
    let after = execute(&mut model, 7, vec![request]).remove(0).1;
    assert_close(&after, &expected, 0.025);
    model.state.finish_commit();
    model.gdn.finish_commit();
}

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
#[derive(Clone)]
struct Tensor {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
impl View for &Tensor {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}
struct Fixture {
    path: PathBuf,
    config: Qwen4xModelConfig,
    decoded: HashMap<String, Vec<f32>>,
    tensors: BTreeMap<String, Tensor>,
    random: u64,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
impl Fixture {
    fn new(seed: u64, dense: bool) -> Self {
        let path = std::env::temp_dir().join(format!(
            "khalamari-qwen4x-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let mut config: Qwen4xModelConfig = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../inference-executor-core/src/model/qwen/v4_x/config_fixture.json"
        )))
        .unwrap();
        if dense {
            config.text_config.num_experts = 0;
            config.text_config.num_experts_per_tok = 0;
            config.text_config.intermediate_size = 64;
        }
        config.normalize().unwrap();
        let mut f = Self {
            path,
            config,
            decoded: HashMap::new(),
            tensors: BTreeMap::new(),
            random: seed,
        };
        let t = f.config.text_config.clone();
        let h = t.hidden_size;
        let wide = h * t.hc_count;
        f.affine("model.embed_tokens", &[t.vocab_size, h], 0.03125);
        f.affine("lm_head", &[t.vocab_size, h], 0.00390625);
        f.gr("model.hyper_connection_mixer", false);
        for i in 0..t.num_hidden_layers {
            let p = format!("model.layers.{i}");
            f.gr(&format!("{p}.attn_hyper_connection"), true);
            f.gr(&format!("{p}.mlp_hyper_connection"), true);
            if f.config.layer_type_at(i) == inference_executor_core::model::qwen::v4_x::LayerType::GDN {
                let p = format!("{p}.linear_attn");
                let qkv = 2 * t.linear_num_key_heads * t.linear_key_head_dim
                    + t.linear_num_value_heads * t.linear_value_head_dim;
                for (name, n, k) in [
                    ("in_proj_qkv", qkv, h),
                    ("in_proj_a", t.linear_num_value_heads, h),
                    ("in_proj_b", t.linear_num_value_heads, h),
                    ("in_proj_z", t.linear_num_value_heads * t.linear_value_head_dim, h),
                    ("out_proj", h, t.linear_num_value_heads * t.linear_value_head_dim),
                ] {
                    f.affine(&format!("{p}.{name}"), &[n, k], 0.001953125);
                }
                f.raw(
                    &format!("{p}.conv1d.weight"),
                    &[qkv, 1, t.linear_conv_kernel_dim],
                    0.0625,
                );
                f.raw_const(&format!("{p}.norm.weight"), &[t.linear_value_head_dim], 1.0);
                f.raw_const(&format!("{p}.A_log"), &[t.linear_num_value_heads], -1.0);
                f.raw(&format!("{p}.dt_bias"), &[t.linear_num_value_heads], 0.03125);
            } else {
                let p = format!("{p}.self_attn");
                for (name, n, k) in [
                    ("q_proj", t.num_attention_heads * t.head_dim * 2, h),
                    ("k_proj", t.num_key_value_heads * t.head_dim, h),
                    ("v_proj", t.num_key_value_heads * t.head_dim, h),
                    ("o_proj", h, t.num_attention_heads * t.head_dim),
                    ("indexer.index_qk_proj", (t.indexer_n_heads + 1) * t.indexer_head_dim, h),
                ] {
                    f.affine(&format!("{p}.{name}"), &[n, k], 0.001953125);
                }
                for (name, width) in [
                    ("q_norm", t.head_dim),
                    ("k_norm", t.head_dim),
                    ("indexer.q_layernorm", t.indexer_head_dim),
                    ("indexer.k_layernorm", t.indexer_head_dim),
                ] {
                    f.raw(&format!("{p}.{name}.weight"), &[width], 0.0625);
                }
            }
            let mlp = format!("{p}.mlp");
            if dense {
                for (name, n, k) in [
                    ("gate_proj", t.intermediate_size, h),
                    ("up_proj", t.intermediate_size, h),
                    ("down_proj", h, t.intermediate_size),
                ] {
                    f.affine(&format!("{mlp}.{name}"), &[n, k], 0.00390625);
                }
            } else {
                f.affine(&format!("{mlp}.gate"), &[t.num_experts, h], 0.00390625);
                for (name, n, k) in [
                    ("gate_proj", t.moe_intermediate_size, h),
                    ("up_proj", t.moe_intermediate_size, h),
                    ("down_proj", h, t.moe_intermediate_size),
                ] {
                    f.affine(&format!("{mlp}.switch_mlp.{name}"), &[t.num_experts, n, k], 0.00390625);
                }
                f.affine(&format!("{mlp}.shared_expert_gate"), &[1, h], 0.00390625);
                for (name, n, k) in [
                    ("gate_proj", t.shared_expert_intermediate_size, h),
                    ("up_proj", t.shared_expert_intermediate_size, h),
                    ("down_proj", h, t.shared_expert_intermediate_size),
                ] {
                    f.affine(&format!("{mlp}.shared_expert.{name}"), &[n, k], 0.00390625);
                }
            }
            if let Some(ordinal) = t.ple_layer_ids.iter().position(|&id| id == i + 1) {
                let p = format!("{p}.ple");
                f.affine(&format!("{p}.key_proj"), &[wide, t.ple_embed_dim], 0.00390625);
                f.affine(&format!("{p}.value_proj"), &[h, t.ple_embed_dim], 0.00390625);
                for name in ["norm_key", "norm_query", "norm_conv"] {
                    f.raw(&format!("{p}.{name}.weight"), &[wide], 0.0625);
                }
                f.raw(
                    &format!("{p}.conv1d.weight"),
                    &[wide, t.ple_conv_kernel_size, 1],
                    0.0625,
                );
                let ngram = NGramEmbeddingCore {
                    vocab_size: t.vocab_size as u32,
                    eos_token_id: 0,
                    ngram_size: t.ngram_size as u32,
                    heads_per_ngram: t.heads_per_ngram as u32,
                    embedding_dim: t.ple_embed_dim as u32,
                };
                let parameters = HashParameters::new(ngram, t.ngram_vocab_size_base, ordinal as u32, t.seed);
                let rows = (parameters.num_rows() as usize).div_ceil(t.make_ngram_vocab_size_divisible_by)
                    * t.make_ngram_vocab_size_divisible_by;
                for shard in 0..t.split_ngram_parts {
                    let rows = rows / t.split_ngram_parts + usize::from(shard < rows % t.split_ngram_parts);
                    f.affine(
                        &format!("{p}.ple_embedding.ngram_embedding.shards.{shard}"),
                        &[rows, ngram.head_dim() as usize],
                        0.03125,
                    );
                }
            }
        }
        f.write();
        f
    }
    fn write(&self) {
        std::fs::write(self.path.join("config.json"), serde_json::to_vec(&self.config).unwrap()).unwrap();
        std::fs::write(self.path.join("generation_config.json"), r#"{"eos_token_id":[0]}"#).unwrap();
        serialize_to_file(
            self.tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
            None,
            &self.path.join("model.safetensors"),
        )
        .unwrap();
        std::fs::write(
            self.path.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": self.tensors.keys().map(|name| (name, "model.safetensors")).collect::<BTreeMap<_, _>>()
            }))
            .unwrap(),
        )
        .unwrap();
    }
    fn add_mtp(&mut self) {
        if self.config.text_config.num_experts > 0 {
            self.config.text_config.mtp_num_experts_per_tok = Some(1);
        }
        let sources = [
            ("model.layers.3.", "mtp.layers.0."),
            ("model.hyper_connection_mixer.", "mtp.hyper_connection_mixer."),
        ];
        for (source, target) in sources {
            let tensors: Vec<_> = self
                .tensors
                .iter()
                .filter_map(|(name, tensor)| {
                    name.strip_prefix(source)
                        .map(|suffix| (format!("{target}{suffix}"), tensor.clone(), self.decoded[name].clone()))
                })
                .collect();
            for (name, tensor, decoded) in tensors {
                self.tensors.insert(name.clone(), tensor);
                self.decoded.insert(name, decoded);
            }
        }
        let h = self.config.text_config.hidden_size;
        let wide = h * self.config.text_config.hc_count;
        self.raw("mtp.pre_fc_norm_hidden.weight", &[wide], 0.0625);
        self.raw("mtp.pre_fc_norm_embedding.weight", &[h], 0.0625);
        self.affine("mtp.fc_hidden", &[h, h], 0.015625);
        self.affine("mtp.fc_embedding", &[h, h], 0.015625);
        self.config.text_config.mtp = Some(inference_executor_core::model::qwen::v4_x::Qwen4xMTPConfig {
            num_hidden_layers: 1,
            layer_types: vec!["full_attention".to_owned()],
            rope_theta: Some(10_000_000.0),
            ..Default::default()
        });
        self.write();
    }
    fn next(&mut self) -> u32 {
        if self.random == 0 {
            return 11;
        }
        self.random ^= self.random << 13;
        self.random ^= self.random >> 7;
        self.random ^= self.random << 17;
        self.random as u32
    }
    fn affine(&mut self, prefix: &str, shape: &[usize], scale: f32) {
        let k = *shape.last().unwrap();
        let count = shape.iter().product::<usize>();
        assert!(k.is_multiple_of(32));
        let values: Vec<_> = (0..count)
            .map(|i| {
                if self.random == 0 {
                    ((i * 7 + i / k * 3) % 16) as u32
                } else {
                    self.next() % 16
                }
            })
            .collect();
        let words: Vec<u32> = values
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| chunk.iter().enumerate().fold(0, |word, (i, &v)| word | (v << (i * 4))))
            .collect();
        let mut packed = shape.to_vec();
        *packed.last_mut().unwrap() = k / 8;
        self.tensors.insert(
            format!("{prefix}.weight"),
            Tensor {
                dtype: safetensors::Dtype::U32,
                shape: packed,
                bytes: words.iter().flat_map(|w| w.to_ne_bytes()).collect(),
            },
        );
        let decoded = values.iter().map(|&v| (v as f32 - 8.0) * scale).collect();
        self.decoded.insert(format!("{prefix}.weight"), decoded);
        let mut params = shape.to_vec();
        *params.last_mut().unwrap() = k / 32;
        self.raw_const(&format!("{prefix}.scales"), &params, scale);
        self.raw_const(&format!("{prefix}.biases"), &params, -8.0 * scale);
    }
    fn raw_const(&mut self, name: &str, shape: &[usize], value: f32) {
        let values = vec![half::bf16::from_f32(value).to_f32(); shape.iter().product()];
        self.insert_raw(name, shape, values);
    }
    fn raw(&mut self, name: &str, shape: &[usize], amplitude: f32) {
        let values = (0..shape.iter().product())
            .map(|i| {
                let v = if self.random == 0 {
                    ((i * 5) % 17) as f32 / 8.0 - 1.0
                } else {
                    (self.next() % 257) as f32 / 128.0 - 1.0
                };
                half::bf16::from_f32(v * amplitude).to_f32()
            })
            .collect();
        self.insert_raw(name, shape, values);
    }
    fn insert_raw(&mut self, name: &str, shape: &[usize], values: Vec<f32>) {
        self.tensors.insert(
            name.to_owned(),
            Tensor {
                dtype: safetensors::Dtype::BF16,
                shape: shape.to_vec(),
                bytes: values
                    .iter()
                    .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_ne_bytes())
                    .collect(),
            },
        );
        self.decoded.insert(name.to_owned(), values);
    }
    fn gr(&mut self, prefix: &str, write: bool) {
        let t = &self.config.text_config;
        let (wide, rank, channels) = (t.hidden_size * t.hc_count, t.hc_lowrank, t.hc_count);
        self.raw(&format!("{prefix}.hc_norm.weight"), &[wide], 0.0625);
        self.raw(
            &format!("{prefix}.input_mix_weight_down.weight"),
            &[rank, wide],
            0.015625,
        );
        self.raw(&format!("{prefix}.input_mix_weight_up.weight"), &[wide, rank], 0.015625);
        if write {
            self.raw(
                &format!("{prefix}.block_inject_weight.weight"),
                &[channels, wide],
                0.015625,
            );
        }
    }
    fn executor(&self) -> Qwen4xExecutor {
        init_qwen4x_model(
            &self.path,
            Qwen4xExecutorConfig {
                max_requests: 2,
                max_tokens: 512,
                max_tokens_per_request: 512,
                num_cache_pages: 512,
                num_tokens_per_block: 512,
            },
        )
        .unwrap()
    }
}
#[derive(Clone)]
struct Cache {
    slot: u32,
    kv: Vec<Vec<u32>>,
    state: Vec<Vec<u32>>,
    mtp_kv: Vec<Vec<Vec<u32>>>,
    mtp_state: Vec<Vec<Vec<u32>>>,
}
impl Cache {
    fn new(slot: u32) -> Self {
        Self {
            slot,
            kv: Vec::new(),
            state: Vec::new(),
            mtp_kv: Vec::new(),
            mtp_state: Vec::new(),
        }
    }
    fn request(
        &mut self,
        model: &Qwen4xExecutor,
        position: usize,
        tokens: &[u32],
        decode: bool,
        next_page: &mut u32,
    ) -> DeviceRequest {
        let blocks = (position + tokens.len()).div_ceil(model.config.num_tokens_per_block);
        let mut allocate = |n: usize| {
            let out = (*next_page..*next_page + n as u32).collect();
            *next_page += n as u32;
            out
        };
        while self.kv.len() < blocks {
            self.kv.push(allocate(model.num_kv_page_ids_per_block()));
            self.state.push(allocate(model.num_state_page_ids_per_block()));
        }
        for (lane, (kv, state)) in model.num_mtp_page_ids_per_block().into_iter().enumerate() {
            if self.mtp_kv.len() <= lane {
                self.mtp_kv.push(Vec::new());
                self.mtp_state.push(Vec::new());
            }
            while self.mtp_kv[lane].len() < blocks {
                self.mtp_kv[lane].push(allocate(kv));
                self.mtp_state[lane].push(allocate(state));
            }
        }
        assert!(*next_page as usize <= model.config.num_cache_pages);
        let ids = tokens.iter().map(|&id| Token::new(id)).collect();
        let query = if decode {
            QueryTokens::Decode {
                epoch: 1,
                token_index: position,
                tokens: ids,
                spec_tokens: vec![],
            }
        } else {
            QueryTokens::Prefill {
                epoch: 1,
                token_index: position,
                tokens: ids,
                window: tokens.len() - model.num_spec_tokens(),
            }
        };
        DeviceRequest::new(
            usize::try_from(self.slot).unwrap() + 1,
            self.slot,
            query,
            DecoderSyncBlocks::new(
                0,
                std::iter::once(self.kv.clone())
                    .chain(self.mtp_kv.iter().cloned())
                    .collect(),
                std::iter::once(self.state.clone())
                    .chain(self.mtp_state.iter().cloned())
                    .collect(),
            ),
            vec![],
            SamplingConfig {
                seed: Some(1),
                ..Default::default()
            },
        )
    }
}
fn execute(model: &mut Qwen4xExecutor, seq: u64, requests: Vec<DeviceRequest>) -> Vec<(u32, Vec<f32>)> {
    let mut request = BatchDeviceRequest {
        seq,
        dev_reqs: requests,
    };
    let batch = model.prepare_batch(&mut request);
    let mut recorder = model.begin_ops_recording(&batch);
    let hidden = model.embed_main(&mut recorder, &batch);
    let hidden = model.forward_main(&mut recorder, &batch, hidden);
    model.unembed_main(&mut recorder, &batch, &hidden);
    model.sample_main(&mut recorder, &batch, &());
    let start = Instant::now();
    let submission = model.submit_main(&recorder);
    submission.wait();
    let output = model.read_main(&recorder, &batch, start.elapsed(), None);
    let width = model.model_config.text_config.hidden_size;
    let micro = batch.microbatch();
    let result = micro
        .req_slots()
        .iter()
        .enumerate()
        .map(|(i, &slot)| {
            let begin = micro.cu_tokens()[i] as usize;
            let count = (micro.cu_tokens()[i + 1] - micro.cu_tokens()[i]) as usize;
            (
                slot,
                hidden
                    .read_typed::<u16>(begin * width, count * width)
                    .into_iter()
                    .map(|v| half::bf16::from_bits(v).to_f32())
                    .collect(),
            )
        })
        .collect();
    model.commit_batch(request, output);
    result
}
fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "index={i} actual={a} expected={e}"
        );
    }
}

#[path = "mtp/test.rs"]
mod mtp;
