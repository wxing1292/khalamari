use std::rc::Rc;

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_backend_metal::operators::bf16_concat_rows;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MTPEmbedWeightBindings;
use inference_executor_core::replay::ReplayBucketPolicy;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedInput;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::qwen::v3_x::weight::validate_len;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

const QWEN35_MTP_EMBED_NUM_ACTIVE_TOKENS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3.5.mtp_embed.num_active_tokens");

pub struct Qwen35MTPEmbed {
    embed: Option<Rc<Embed>>,
    hidden_norm: RMSNorm,
    embedding_norm: RMSNorm,
    concat: bf16_concat_rows::Kernel,
    fc: affine_quantized::Matmul,
    projection_weights: Option<Qwen35MTPProjectionWeights>,
    normed_hidden: Buffer,
    normed_embedding: Buffer,
    fused_input: Buffer,
    replay_bucket_policy: ReplayBucketPolicy,
}

#[derive(Clone, Copy)]
pub struct Qwen35MTPEmbedArgs<'a> {
    pub num_tokens: u32,
    pub prev_hidden_input: &'a Buffer,
    pub token_ids: &'a Buffer,
    pub token_hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
}

impl Qwen35MTPEmbed {
    pub fn new(
        device: &Device,
        config: &Qwen35ModelConfig,
        embed: Rc<Embed>,
        max_tokens: usize,
    ) -> Result<Self, ModelExecutorError> {
        let hidden_dim = config.text_config.hidden_size;
        let quant = config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("qwen3.5 MTP requires quantized checkpoint weights"))?;
        let fused_hidden_dim = hidden_dim
            .checked_mul(2)
            .expect("qwen3.5 MTP fused hidden dimension must fit usize");
        let fc_config = affine_quantized::Config {
            n: hidden_dim
                .try_into()
                .expect("qwen3.5 MTP hidden dimension must fit i32"),
            k: fused_hidden_dim
                .try_into()
                .expect("qwen3.5 MTP fused hidden dimension must fit i32"),
            group_size: quant
                .group_size
                .try_into()
                .expect("qwen3.5 MTP group size must fit i32"),
            bits: quant.bits.try_into().expect("qwen3.5 MTP bits must fit i32"),
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let hidden_elements = max_tokens
            .checked_mul(hidden_dim)
            .expect("qwen3.5 MTP hidden capacity must fit usize");
        u32::try_from(hidden_elements).expect("qwen3.5 MTP hidden capacity must fit shader u32 count");
        let fused_elements = hidden_elements
            .checked_mul(2)
            .expect("qwen3.5 MTP fused input capacity must fit usize");
        u32::try_from(fused_elements).expect("qwen3.5 MTP fused capacity must fit shader u32 count");
        let max_tokens_u32 = u32::try_from(max_tokens).expect("qwen3.5 MTP token capacity must fit u32");
        assert_eq!(
            embed.max_tokens(),
            max_tokens_u32,
            "qwen3.5 MTPEmbed and shared embedding token capacities must match"
        );
        let fc = affine_quantized::Matmul::new(device, fc_config);
        let replay_bucket_policy =
            ReplayBucketPolicy::with_topology_boundaries(max_tokens_u32, &fc.topology_boundaries());
        Ok(Self {
            embed: Some(embed),
            hidden_norm: RMSNorm::new(device, hidden_dim, config.text_config.rms_norm_eps),
            embedding_norm: RMSNorm::new(device, hidden_dim, config.text_config.rms_norm_eps),
            concat: bf16_concat_rows::Kernel::new(
                device,
                bf16_concat_rows::Config {
                    num_columns: hidden_dim
                        .try_into()
                        .expect("qwen3.5 MTP hidden dimension must fit u32"),
                },
            ),
            fc,
            projection_weights: None,
            normed_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            normed_embedding: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            fused_input: Buffer::new_zeroed_elements(device, fused_elements, Dtype::Bfloat16),
            replay_bucket_policy,
        })
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        bindings: Qwen35MTPEmbedWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(
            self.projection_weights.is_none(),
            "qwen3.5 MTP embed weights are already loaded"
        );
        let hidden_dim = config.text_config.hidden_size;
        let quant = config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("qwen3.5 MTP requires quantized checkpoint weights"))?;
        let weights = Qwen35MTPEmbedWeights::load(device, store, &bindings, hidden_dim, quant.group_size, quant.bits)?;
        self.hidden_norm.load_weights(weights.prev_hidden_norm_weight);
        self.embedding_norm.load_weights(weights.token_hidden_norm_weight);
        self.projection_weights = Some(Qwen35MTPProjectionWeights {
            weight: weights.fc_weight,
            scales: weights.fc_scales,
            biases: weights.fc_biases,
        });
        Ok(())
    }

    pub fn load_shared_weights(&mut self, embed: Rc<Embed>) {
        assert!(
            self.embed.is_none(),
            "qwen3.5 MTP shared embed weights are already loaded"
        );
        self.embed = Some(embed);
    }

    pub fn unload_weights(&mut self) -> Rc<Embed> {
        assert!(
            self.projection_weights.is_some(),
            "qwen3.5 MTP embed weights are not loaded"
        );
        let embed = self
            .embed
            .take()
            .expect("qwen3.5 MTP shared embed weights are not loaded");
        self.projection_weights.take();
        self.embedding_norm.unload_weights();
        self.hidden_norm.unload_weights();
        embed
    }

    fn projection_weights(&self) -> &Qwen35MTPProjectionWeights {
        self.projection_weights
            .as_ref()
            .expect("qwen3.5 MTP embed weights must be loaded before execution")
    }

    fn loaded_embed(&self) -> &Embed {
        self.embed
            .as_deref()
            .expect("qwen3.5 MTP shared embed weights must be loaded before execution")
    }

    fn record_projection<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        previous_hidden: &'a Buffer,
        shifted_embeddings: &'a Buffer,
        output: &'a Buffer,
    ) -> &'a Buffer {
        self.hidden_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            previous_hidden,
            &self.normed_hidden,
        );
        self.embedding_norm.record(
            recorder,
            num_total_tokens,
            num_active_tokens,
            shifted_embeddings,
            &self.normed_embedding,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(self.concat.invoke(
            num_total_tokens,
            num_active_tokens,
            bf16_concat_rows::Buffers {
                lhs: &self.normed_embedding,
                rhs: &self.normed_hidden,
                output: &self.fused_input,
            },
        )));
        let projection_weights = self.projection_weights();
        let invocation = self.fc.invoke(
            num_total_tokens,
            num_active_tokens,
            output,
            0,
            &self.fused_input,
            0,
            &projection_weights.weight,
            0,
            &projection_weights.scales,
            0,
            &projection_weights.biases,
            0,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(invocation));
        output
    }

    pub fn prepare_replay(&self, num_active_tokens: u32) -> (Qwen35MTPEmbedReplayKey, ReplayArguments) {
        let key = self.replay_key_for_active_tokens(num_active_tokens);
        let arguments = ReplayArguments::new().with_u32(QWEN35_MTP_EMBED_NUM_ACTIVE_TOKENS, num_active_tokens);
        (key, arguments)
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        args: Qwen35MTPEmbedArgs<'a>,
    ) -> &'a Buffer {
        match num_active_tokens {
            ReplayU32::Fixed(value) => {
                assert_eq!(value, args.num_tokens);
                assert_eq!(value, num_total_tokens);
            },
            ReplayU32::Parameter(_) => {
                assert_eq!(
                    self.replay_bucket_policy.capacity(args.num_tokens),
                    num_total_tokens,
                    "qwen3.5 MTPEmbed total token count must match its selected capacity"
                );
                assert_eq!(
                    self.fc.topology(args.num_tokens),
                    self.fc.topology(num_total_tokens),
                    "qwen3.5 MTPEmbed capacity must preserve FC topology"
                );
            },
        }
        let _ = <Embed as ReplayLayer>::record(
            self.loaded_embed(),
            recorder,
            EmbedInput {
                num_total_tokens,
                num_active_tokens,
                token_ids: args.token_ids,
                output_hidden: args.token_hidden_input,
            },
        );
        self.record_projection(
            recorder,
            num_total_tokens,
            num_active_tokens,
            args.prev_hidden_input,
            args.token_hidden_input,
            args.hidden_output,
        )
    }

    fn replay_key_for_active_tokens(&self, num_active_tokens: u32) -> Qwen35MTPEmbedReplayKey {
        let num_total_tokens = self.replay_bucket_policy.capacity(num_active_tokens);
        Qwen35MTPEmbedReplayKey::for_capacity(num_total_tokens, self.fc.topology(num_total_tokens))
    }
}

struct Qwen35MTPEmbedWeights {
    token_hidden_norm_weight: Buffer,
    prev_hidden_norm_weight: Buffer,
    fc_weight: Buffer,
    fc_scales: Buffer,
    fc_biases: Buffer,
}

struct Qwen35MTPProjectionWeights {
    weight: Buffer,
    scales: Buffer,
    biases: Buffer,
}

impl Qwen35MTPEmbedWeights {
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: &Qwen35MTPEmbedWeightBindings,
        hidden_dim: usize,
        group_size: usize,
        bits: usize,
    ) -> Result<Self, ModelExecutorError> {
        let mut tensor_names = Vec::new();
        bindings.push_tensor_names(&mut tensor_names);
        let mut tensors = store.load_tensors(tensor_names)?;
        let fused_hidden_dim = hidden_dim
            .checked_mul(2)
            .ok_or_else(|| ModelExecutorError::custom("qwen3.5 MTP fused hidden dimension overflow"))?;
        let fc_config = affine_quantized::Config {
            n: hidden_dim
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 MTP hidden_dim must fit i32"))?,
            k: fused_hidden_dim
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 MTP fused hidden dimension must fit i32"))?,
            group_size: group_size
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 MTP group_size must fit i32"))?,
            bits: bits
                .try_into()
                .map_err(|_| ModelExecutorError::custom("qwen3.5 MTP bits must fit i32"))?,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let fc_weight = remove_quant_weight(&mut tensors, &bindings.projection.weight)?;
        let fc_scales = remove_tensor(&mut tensors, &bindings.projection.scales, safetensors::Dtype::BF16)?.into_data();
        let fc_biases = remove_tensor(&mut tensors, &bindings.projection.biases, safetensors::Dtype::BF16)?.into_data();
        validate_len("MTP fc weight", fc_weight.len(), fc_config.weight_bytes())?;
        validate_len("MTP fc scales", fc_scales.len(), fc_config.scale_or_bias_bytes())?;
        validate_len("MTP fc biases", fc_biases.len(), fc_config.scale_or_bias_bytes())?;
        let weights = Self {
            token_hidden_norm_weight: remove_qwen3x_norm_weight(
                device,
                &mut tensors,
                &bindings.token_hidden_norm_weight,
                &[hidden_dim],
            )?,
            prev_hidden_norm_weight: remove_qwen3x_norm_weight(
                device,
                &mut tensors,
                &bindings.prev_hidden_norm_weight,
                &[hidden_dim],
            )?,
            fc_weight: Buffer::from_slice(device, &fc_weight),
            fc_scales: Buffer::from_slice(device, &fc_scales),
            fc_biases: Buffer::from_slice(device, &fc_biases),
        };
        assert!(tensors.is_empty(), "qwen3.5 MTP embed must consume its tensor map");
        Ok(weights)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MTPEmbedReplayKey {
    num_total_tokens: u32,
    fc_topology: affine_quantized::KernelKind,
}

impl Qwen35MTPEmbedReplayKey {
    fn for_capacity(num_total_tokens: u32, fc_topology: affine_quantized::KernelKind) -> Self {
        Self {
            num_total_tokens,
            fc_topology,
        }
    }
}

impl ReplayComponent for Qwen35MTPEmbed {
    type Key = Qwen35MTPEmbedReplayKey;
    type Input<'a> = Qwen35MTPEmbedArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.replay_key_for_active_tokens(input.num_tokens)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        Qwen35MTPEmbed::record(
            self,
            recorder,
            key.num_total_tokens,
            ReplayU32::Parameter(QWEN35_MTP_EMBED_NUM_ACTIVE_TOKENS),
            *input,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    use half::bf16;
    use inference_backend_metal::metal::Stream;
    use inference_executor_core::checkpoint::QuantizedTensorBindings;
    use inference_executor_core::checkpoint::SafeTensorIndex;
    use inference_executor_core::mlp::dense::reference::QuantizedAffineReferenceShape;
    use inference_executor_core::mlp::dense::reference::quantized_affine_reference;
    use inference_executor_core::mlp::dense::reference::quantized_affine_weight_row_reference;
    use inference_executor_core::reference::rms_norm_reference;
    use safetensors::Dtype as SafeTensorDtype;
    use safetensors::tensor::View;
    use safetensors::tensor::serialize_to_file;

    use super::*;
    use crate::model::embedding::EmbedConfig;
    use crate::replay::Replay;

    const MAX_TOKENS: u32 = 32;
    const VOCAB_SIZE: u32 = 32;
    const HIDDEN_DIM: u32 = 32;
    const GROUP_SIZE: u32 = 32;
    const NUM_TOTAL_TOKENS: u32 = 8;
    const EPS: f32 = 1.0e-6;

    const EMBED_WEIGHT: &str = "embed.weight";
    const EMBED_SCALES: &str = "embed.scales";
    const EMBED_BIASES: &str = "embed.biases";

    struct OwnedTensor {
        dtype: SafeTensorDtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for &OwnedTensor {
        fn dtype(&self) -> SafeTensorDtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    struct TempModelDir(PathBuf);

    impl TempModelDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "khalamari-qwen35-mtp-embed-test-{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempModelDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn test_replay_matches_cpu_reference_across_active_counts() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = &stream;
        let (component, weights) = fixture_component(&device);
        let buffers = TestBuffers::new(&device);
        let input = buffers.input(NUM_TOTAL_TOKENS);
        let mut replay = Replay::new("qwen3.5 MTPEmbed component test", component);
        let (recorded_key, cache_hit) = replay.record(runtime, &input);
        assert!(!cache_hit);

        for num_active_tokens in [1_usize, 8, 3, 7, 2, 6, 4, 5] {
            assert_eq!(replay.record(runtime, &input), (recorded_key.clone(), true));
            runtime
                .submit_replay_with_arguments(
                    replay.replay(&recorded_key),
                    &ReplayArguments::new().with_u32(QWEN35_MTP_EMBED_NUM_ACTIVE_TOKENS, num_active_tokens as u32),
                )
                .wait();

            let gathered = buffers.gathered_reference(num_active_tokens);
            assert_bf16_close(
                &buffers.prev_hidden_input,
                &gathered,
                num_active_tokens * HIDDEN_DIM as usize,
                0.0,
            );
            let embeddings = buffers.embedding_reference(num_active_tokens, &weights);
            assert_bf16_close(
                &buffers.token_hidden_input,
                &embeddings,
                num_active_tokens * HIDDEN_DIM as usize,
                0.0,
            );
            let expected = projection_reference(num_active_tokens, &gathered, &embeddings, &weights);
            assert_bf16_close(
                &buffers.hidden_output,
                &expected,
                num_active_tokens * HIDDEN_DIM as usize,
                0.125,
            );
        }
    }

    struct TestBuffers {
        prev_hidden_input: Buffer,
        token_ids: Buffer,
        token_hidden_input: Buffer,
        hidden_output: Buffer,
        prev_hidden_values: Vec<f32>,
        prev_hidden_index_values: Vec<u32>,
        token_id_values: Vec<u32>,
    }

    impl TestBuffers {
        fn new(device: &Device) -> Self {
            let hidden_elements = MAX_TOKENS as usize * HIDDEN_DIM as usize;
            let prev_hidden_values = (0..hidden_elements)
                .map(|index| bf16::from_f32(((index * 19 + 7) % 101) as f32 * 0.015_625 - 0.75).to_f32())
                .collect::<Vec<_>>();
            let prev_hidden_bits = prev_hidden_values
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>();
            let prev_hidden_index_values = vec![7, 1, 15, 3, 20, 0, 31, 9];
            let token_id_values = vec![4, 17, 2, 29, 11, 0, 23, 8];
            let selected_prev_hidden_bits = prev_hidden_index_values
                .iter()
                .flat_map(|&row_index| {
                    let begin = row_index as usize * HIDDEN_DIM as usize;
                    prev_hidden_bits[begin..begin + HIDDEN_DIM as usize].iter().copied()
                })
                .collect::<Vec<_>>();
            Self {
                prev_hidden_input: Buffer::from_slice(device, &selected_prev_hidden_bits),
                token_ids: Buffer::from_slice(device, &token_id_values),
                token_hidden_input: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
                hidden_output: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
                prev_hidden_values,
                prev_hidden_index_values,
                token_id_values,
            }
        }

        fn input(&self, num_tokens: u32) -> Qwen35MTPEmbedArgs<'_> {
            Qwen35MTPEmbedArgs {
                num_tokens,
                prev_hidden_input: &self.prev_hidden_input,
                token_ids: &self.token_ids,
                token_hidden_input: &self.token_hidden_input,
                hidden_output: &self.hidden_output,
            }
        }

        fn gathered_reference(&self, num_active_tokens: usize) -> Vec<f32> {
            let mut output = Vec::with_capacity(num_active_tokens * HIDDEN_DIM as usize);
            for &row_index in &self.prev_hidden_index_values[..num_active_tokens] {
                let begin = row_index as usize * HIDDEN_DIM as usize;
                output.extend_from_slice(&self.prev_hidden_values[begin..begin + HIDDEN_DIM as usize]);
            }
            output
        }

        fn embedding_reference(&self, num_active_tokens: usize, weights: &TestWeights) -> Vec<f32> {
            let shape = QuantizedAffineReferenceShape {
                num_rows: 1,
                output_dim: VOCAB_SIZE as usize,
                input_dim: HIDDEN_DIM as usize,
                group_size: GROUP_SIZE as usize,
                bits: 8,
            };
            let mut output = Vec::with_capacity(num_active_tokens * HIDDEN_DIM as usize);
            for &token_id in &self.token_id_values[..num_active_tokens] {
                output.extend(
                    quantized_affine_weight_row_reference(
                        shape,
                        token_id as usize,
                        &weights.embed_weight,
                        &weights.embed_scales,
                        &weights.embed_biases,
                    )
                    .into_iter()
                    .map(|value| bf16::from_f32(value).to_f32()),
                );
            }
            output
        }
    }

    struct TestWeights {
        embed_weight: Vec<u8>,
        embed_scales: Vec<f32>,
        embed_biases: Vec<f32>,
        hidden_norm: Vec<f32>,
        embedding_norm: Vec<f32>,
        projection_weight: Vec<u8>,
        projection_scales: Vec<f32>,
        projection_biases: Vec<f32>,
    }

    fn fixture_component(device: &Device) -> (Qwen35MTPEmbed, TestWeights) {
        const FILE_NAME: &str = "model.safetensors";
        let fc_config = affine_quantized::Config {
            n: HIDDEN_DIM as i32,
            k: (HIDDEN_DIM * 2) as i32,
            group_size: GROUP_SIZE as i32,
            bits: 8,
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype: Dtype::Bfloat16,
        };
        let embed_weight_bytes = VOCAB_SIZE as usize * HIDDEN_DIM as usize;
        let embed_affine_elements = VOCAB_SIZE as usize * HIDDEN_DIM as usize / GROUP_SIZE as usize;
        let embed_weight = (0..embed_weight_bytes)
            .map(|index| ((index * 17 + 3) % 251) as u8)
            .collect::<Vec<_>>();
        let embed_scales = (0..embed_affine_elements)
            .map(|index| bf16::from_f32(0.003_906_25 + (index % 7) as f32 * 0.000_244_140_63).to_f32())
            .collect::<Vec<_>>();
        let embed_biases = (0..embed_affine_elements)
            .map(|index| bf16::from_f32((index as f32 - 16.0) * 0.000_976_562_5).to_f32())
            .collect::<Vec<_>>();
        let tensors = HashMap::from([
            owned_tensor(EMBED_WEIGHT, SafeTensorDtype::U32, embed_weight.clone()),
            owned_tensor(EMBED_SCALES, SafeTensorDtype::BF16, bf16_bytes(&embed_scales)),
            owned_tensor(EMBED_BIASES, SafeTensorDtype::BF16, bf16_bytes(&embed_biases)),
        ]);
        let model_dir = TempModelDir::new();
        serialize_to_file(
            tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
            None,
            &model_dir.0.join(FILE_NAME),
        )
        .unwrap();
        let index = SafeTensorIndex::new(
            tensors
                .keys()
                .map(|name| (name.clone(), FILE_NAME.to_string()))
                .collect(),
        )
        .unwrap();
        let mut store = SafeTensorStore::new(&model_dir.0, index);
        let mut embed = Embed::new(
            device,
            EmbedConfig {
                max_tokens: MAX_TOKENS,
                vocab_size: VOCAB_SIZE,
                hidden_dim: HIDDEN_DIM,
                group_size: GROUP_SIZE,
                bits: 8,
                scale_bias_dtype: Dtype::Bfloat16,
                output_dtype: Dtype::Bfloat16,
            },
        );
        embed
            .load_weights(
                device,
                &mut store,
                QuantizedTensorBindings {
                    weight: EMBED_WEIGHT.to_string(),
                    scales: EMBED_SCALES.to_string(),
                    biases: EMBED_BIASES.to_string(),
                },
            )
            .unwrap();
        let embed = Rc::new(embed);
        let fc = affine_quantized::Matmul::new(device, fc_config);
        let replay_bucket_policy = ReplayBucketPolicy::with_topology_boundaries(MAX_TOKENS, &fc.topology_boundaries());
        let hidden_elements = MAX_TOKENS as usize * HIDDEN_DIM as usize;
        let hidden_norm_values = (0..HIDDEN_DIM)
            .map(|index| bf16::from_f32(0.75 + index as f32 * 0.007_812_5).to_f32())
            .collect::<Vec<_>>();
        let embedding_norm_values = (0..HIDDEN_DIM)
            .map(|index| bf16::from_f32(1.125 - index as f32 * 0.003_906_25).to_f32())
            .collect::<Vec<_>>();
        let mut hidden_norm = RMSNorm::new(device, HIDDEN_DIM as usize, EPS);
        hidden_norm.load_weights(bf16_buffer(device, &hidden_norm_values));
        let mut embedding_norm = RMSNorm::new(device, HIDDEN_DIM as usize, EPS);
        embedding_norm.load_weights(bf16_buffer(device, &embedding_norm_values));
        let projection_weight = (0..fc_config.weight_bytes())
            .map(|index| ((index * 29 + 11) % 251) as u8)
            .collect::<Vec<_>>();
        let num_projection_affine_values = fc_config.scale_or_bias_bytes() / size_of::<u16>();
        let projection_scales = (0..num_projection_affine_values)
            .map(|index| bf16::from_f32(0.000_244_140_63 + (index % 5) as f32 * 0.000_030_517_578).to_f32())
            .collect::<Vec<_>>();
        let projection_biases = (0..num_projection_affine_values)
            .map(|index| bf16::from_f32(-0.000_122_070_31 + (index % 3) as f32 * 0.000_030_517_578).to_f32())
            .collect::<Vec<_>>();
        let component = Qwen35MTPEmbed {
            embed: Some(embed),
            hidden_norm,
            embedding_norm,
            concat: bf16_concat_rows::Kernel::new(
                device,
                bf16_concat_rows::Config {
                    num_columns: HIDDEN_DIM,
                },
            ),
            fc,
            projection_weights: Some(Qwen35MTPProjectionWeights {
                weight: Buffer::from_slice(device, &projection_weight),
                scales: bf16_buffer(device, &projection_scales),
                biases: bf16_buffer(device, &projection_biases),
            }),
            normed_hidden: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            normed_embedding: Buffer::new_zeroed_elements(device, hidden_elements, Dtype::Bfloat16),
            fused_input: Buffer::new_zeroed_elements(device, hidden_elements * 2, Dtype::Bfloat16),
            replay_bucket_policy,
        };
        (
            component,
            TestWeights {
                embed_weight,
                embed_scales,
                embed_biases,
                hidden_norm: hidden_norm_values,
                embedding_norm: embedding_norm_values,
                projection_weight,
                projection_scales,
                projection_biases,
            },
        )
    }

    fn projection_reference(
        num_active_tokens: usize,
        gathered: &[f32],
        embeddings: &[f32],
        weights: &TestWeights,
    ) -> Vec<f32> {
        let normed_hidden = rms_norm_reference(
            gathered,
            &weights.hidden_norm,
            None,
            num_active_tokens,
            HIDDEN_DIM as usize,
            EPS,
        )
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
        let normed_embeddings = rms_norm_reference(
            embeddings,
            &weights.embedding_norm,
            None,
            num_active_tokens,
            HIDDEN_DIM as usize,
            EPS,
        )
        .into_iter()
        .map(|value| bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
        let mut fused = Vec::with_capacity(num_active_tokens * HIDDEN_DIM as usize * 2);
        for row_index in 0..num_active_tokens {
            let begin = row_index * HIDDEN_DIM as usize;
            let end = begin + HIDDEN_DIM as usize;
            fused.extend_from_slice(&normed_embeddings[begin..end]);
            fused.extend_from_slice(&normed_hidden[begin..end]);
        }
        quantized_affine_reference(
            QuantizedAffineReferenceShape {
                num_rows: num_active_tokens,
                output_dim: HIDDEN_DIM as usize,
                input_dim: HIDDEN_DIM as usize * 2,
                group_size: GROUP_SIZE as usize,
                bits: 8,
            },
            &fused,
            &weights.projection_weight,
            &weights.projection_scales,
            &weights.projection_biases,
        )
    }

    fn assert_bf16_close(buffer: &Buffer, expected: &[f32], num_values: usize, tolerance: f32) {
        let actual = buffer
            .read_typed::<u16>(0, num_values)
            .into_iter()
            .map(|bits| bf16::from_bits(bits).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value mismatch at index={index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        Buffer::from_slice(
            device,
            &values
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>(),
        )
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn owned_tensor(name: &str, dtype: SafeTensorDtype, data: Vec<u8>) -> (String, OwnedTensor) {
        let item_size = match dtype {
            SafeTensorDtype::U32 => size_of::<u32>(),
            SafeTensorDtype::BF16 => size_of::<u16>(),
            _ => unreachable!(),
        };
        (
            name.to_string(),
            OwnedTensor {
                dtype,
                shape: vec![data.len() / item_size],
                data,
            },
        )
    }
}
