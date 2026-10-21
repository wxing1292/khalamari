use std::rc::Rc;

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::sampling::dflash2_selector;
use inference_backend_metal::components::sampling::top_k;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::checkpoint::remove_tensor;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2Config;
use inference_executor_core::model::qwen::v3_x::dflash2::Qwen3xDFlash2SelectorWeightBindings;

use crate::checkpoint::SafeTensorStore;
use crate::def::layer::ReplayLayer;
use crate::model::embedding::Embed;
use crate::model::embedding::EmbedConfig;
use crate::model::embedding::EmbedInput;
use crate::model::gather::Gather;
use crate::model::qwen::v3_x::weight::affine_parameter_safetensors_dtype;
use crate::model::qwen::v3_x::weight::remove_quant_weight;
use crate::model::qwen::v3_x::weight::to_u32;
use crate::model::qwen::v3_x::weight::validate_len;
use crate::model::unembedding::Unembed;
use crate::model::unembedding::UnembedInput;
use crate::replay::ReplayComponent;
use crate::sampling::sampling_params::SamplingParamsStore;
use crate::sampling::spec_probs::SpecProbsStore;

const DFLASH2_OUTPUT_NUM_ACTIVE_PROPOSAL_ROWS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3x.dflash2.output.num_active_proposal_rows");
const DFLASH2_OUTPUT_NUM_ACTIVE_CANDIDATES: ReplayParameterKey =
    ReplayParameterKey::new("qwen3x.dflash2.output.num_active_candidates");
const DFLASH2_OUTPUT_NUM_ACTIVE_REQUESTS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3x.dflash2.output.num_active_requests");

pub struct Qwen3xDFlash2Proposal {
    pub token_ids: Vec<Vec<u32>>,
    pub token_probs: Vec<Vec<f32>>,
}

struct Qwen3xDFlash2SelectorWeights {
    hidden_projection_weight: Buffer,
    hidden_projection_scales: Buffer,
    hidden_projection_biases: Buffer,
}

pub struct Qwen3xDFlash2Output {
    num_spec_tokens: u32,
    spec_block_size: u32,
    max_requests: u32,
    hidden_dim: u32,
    vocab_size: u32,
    selector_config: dflash2_selector::Config,
    params: Rc<SamplingParamsStore>,
    gather: Gather,
    unembed: Option<Rc<Unembed>>,
    hidden_projection_config: affine_quantized::Config,
    hidden_projection: affine_quantized::Matmul,
    predecessor_codebook: Embed,
    successor_codebook: Embed,
    top_k_map: top_k::MapCompute,
    top_k_reduce: top_k::ReduceCompute,
    selector: dflash2_selector::Compute,
    weights: Option<Qwen3xDFlash2SelectorWeights>,
    row_indices: Buffer,
    proposal_hidden: Buffer,
    logits: Buffer,
    partial_candidate_token_ids: Buffer,
    partial_candidate_logits: Buffer,
    candidate_token_ids: Buffer,
    candidate_logits: Buffer,
    projected_hidden: Buffer,
    anchor_token_ids: Buffer,
    predecessor_token_ids: Buffer,
    predecessor_embeddings: Buffer,
    successor_embeddings: Buffer,
    scores: Buffer,
    req_slots: Buffer,
    output_distribution_indices: Buffer,
    proposal_token_ids: Buffer,
    proposal_probs: Buffer,
}

#[derive(Clone, Copy)]
pub struct Qwen3xDFlash2OutputArgs<'a> {
    pub num_requests: u32,
    pub hidden: &'a Buffer,
    pub sample_positions: &'a Buffer,
    pub distribution_store: &'a SpecProbsStore,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen3xDFlash2OutputReplayKey {
    num_total_requests: u32,
    unembed_topology: affine_quantized::KernelKind,
    hidden_projection_topology: affine_quantized::KernelKind,
}

impl Qwen3xDFlash2Output {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        config: &Qwen3xDFlash2Config,
        num_spec_tokens: usize,
        max_requests: usize,
        unembed: Rc<Unembed>,
        bindings: &Qwen3xDFlash2SelectorWeightBindings,
        params: Rc<SamplingParamsStore>,
        scale_bias_dtype: Dtype,
    ) -> Result<Self, ModelExecutorError> {
        let num_spec_tokens = to_u32("Qwen3x DFlash2 proposal count", num_spec_tokens)?;
        let spec_block_size = num_spec_tokens
            .checked_add(1)
            .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 query block size must fit u32"))?;
        let max_requests = to_u32("Qwen3x DFlash2 request capacity", max_requests)?;
        let hidden_dim = to_u32("Qwen3x DFlash2 hidden dimension", config.hidden_size)?;
        let vocab_size = to_u32("Qwen3x DFlash2 vocabulary", config.vocab_size)?;
        let selector_config = dflash2_selector::Config {
            rank: to_u32("Qwen3x DFlash2 selector rank", config.selector_rank)?,
            top_k: to_u32("Qwen3x DFlash2 selector top K", config.selector_top_k)?,
            embedding_dtype: Dtype::Bfloat16,
        };
        selector_config.validate();
        let sampler_bounds = params.bounds();
        if selector_config.top_k > sampler_bounds.top_k {
            return Err(ModelExecutorError::custom(format!(
                "Qwen3x DFlash2 selector_top_k={} exceeds executor sparse-distribution width {}",
                selector_config.top_k, sampler_bounds.top_k
            )));
        }
        let num_proposal_rows = max_requests
            .checked_mul(num_spec_tokens)
            .expect("Qwen3x DFlash2 proposal row capacity must fit u32");
        let top_k_shape = top_k::Shape {
            num_total_sampling_inputs: num_proposal_rows,
            vocab_size,
            top_k: selector_config.top_k,
        };
        top_k_shape.validate();
        let selector_shape = dflash2_selector::Shape {
            num_total_requests: max_requests,
            num_steps: num_spec_tokens,
        };
        selector_shape.validate();
        let top_k_map = top_k::MapCompute::new(device);
        let partial_candidates = top_k_map.candidate_count(top_k_shape);
        let quantization = config
            .quantization
            .as_ref()
            .ok_or_else(|| ModelExecutorError::custom("Qwen3x DFlash2 selector requires quantization config"))?;
        let hidden_projection_quantization = quantization.resolve_for_tensor(&bindings.hidden_projection.weight);
        require_affine_quantization(&hidden_projection_quantization, &bindings.hidden_projection.weight)?;
        let hidden_projection_config = affine_quantized::Config {
            n: selector_config
                .rank
                .try_into()
                .expect("Qwen3x DFlash2 selector rank must fit i32"),
            k: hidden_dim
                .try_into()
                .expect("Qwen3x DFlash2 hidden dimension must fit i32"),
            group_size: hidden_projection_quantization
                .group_size
                .try_into()
                .expect("Qwen3x DFlash2 selector affine group size must fit i32"),
            bits: hidden_projection_quantization
                .bits
                .try_into()
                .expect("Qwen3x DFlash2 selector affine bits must fit i32"),
            input_dtype: Dtype::Bfloat16,
            output_dtype: Dtype::Bfloat16,
            scale_bias_dtype,
        };
        hidden_projection_config.validate();
        let codebook = |binding: &inference_executor_core::checkpoint::QuantizedTensorBindings| {
            let resolved = quantization.resolve_for_tensor(&binding.weight);
            require_affine_quantization(&resolved, &binding.weight)?;
            Ok::<_, ModelExecutorError>(EmbedConfig {
                max_tokens: num_proposal_rows
                    .checked_mul(selector_config.top_k)
                    .expect("Qwen3x DFlash2 selector codebook row capacity must fit u32"),
                vocab_size,
                hidden_dim: selector_config.rank,
                group_size: resolved
                    .group_size
                    .try_into()
                    .expect("Qwen3x DFlash2 selector codebook group size must fit u32"),
                bits: resolved
                    .bits
                    .try_into()
                    .expect("Qwen3x DFlash2 selector codebook bits must fit u32"),
                scale_bias_dtype,
                output_dtype: Dtype::Bfloat16,
            })
        };
        let candidate_count = selector_config.candidate_count(selector_shape);
        let proposal_hidden_elements = num_proposal_rows as usize * hidden_dim as usize;
        Ok(Self {
            num_spec_tokens,
            spec_block_size,
            max_requests,
            hidden_dim,
            vocab_size,
            selector_config,
            params,
            gather: Gather::new(device, hidden_dim),
            unembed: Some(unembed),
            hidden_projection_config,
            hidden_projection: affine_quantized::Matmul::new(device, hidden_projection_config),
            predecessor_codebook: Embed::new(device, codebook(&bindings.predecessor_codebook)?),
            successor_codebook: Embed::new(device, codebook(&bindings.successor_codebook)?),
            top_k_map,
            top_k_reduce: top_k::ReduceCompute::new(device),
            selector: dflash2_selector::Compute::new(device, selector_config),
            weights: None,
            row_indices: Buffer::new_zeroed_elements(device, num_proposal_rows as usize, Dtype::Uint32),
            proposal_hidden: Buffer::new_zeroed_elements(device, proposal_hidden_elements, Dtype::Bfloat16),
            logits: Buffer::new_zeroed_elements(
                device,
                num_proposal_rows as usize * vocab_size as usize,
                Dtype::Bfloat16,
            ),
            partial_candidate_token_ids: Buffer::new_zeroed_elements(device, partial_candidates, Dtype::Int32),
            partial_candidate_logits: Buffer::new_zeroed_elements(device, partial_candidates, Dtype::Float32),
            candidate_token_ids: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Int32),
            candidate_logits: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Float32),
            projected_hidden: Buffer::new_zeroed(device, selector_config.projected_hidden_bytes(selector_shape)),
            anchor_token_ids: Buffer::new_zeroed_elements(device, max_requests as usize, Dtype::Int32),
            predecessor_token_ids: Buffer::new_zeroed_elements(device, candidate_count, Dtype::Int32),
            predecessor_embeddings: Buffer::new_zeroed(device, selector_config.embedding_bytes(selector_shape)),
            successor_embeddings: Buffer::new_zeroed(device, selector_config.embedding_bytes(selector_shape)),
            scores: Buffer::new_zeroed_elements(device, selector_config.score_count(selector_shape), Dtype::Float32),
            req_slots: Buffer::new_zeroed_elements(device, max_requests as usize, Dtype::Uint32),
            output_distribution_indices: Buffer::new_zeroed_elements(device, num_proposal_rows as usize, Dtype::Uint32),
            proposal_token_ids: Buffer::new_zeroed_elements(device, num_proposal_rows as usize, Dtype::Int32),
            proposal_probs: Buffer::new_zeroed_elements(device, num_proposal_rows as usize, Dtype::Float32),
        })
    }

    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        bindings: Qwen3xDFlash2SelectorWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        assert!(
            self.weights.is_none(),
            "Qwen3x DFlash2 selector weights are already loaded"
        );
        let mut tensors = store.load_tensors([
            bindings.hidden_projection.weight.as_str(),
            bindings.hidden_projection.scales.as_str(),
            bindings.hidden_projection.biases.as_str(),
        ])?;
        let weight = remove_quant_weight(&mut tensors, &bindings.hidden_projection.weight)?;
        let scale_bias_dtype = affine_parameter_safetensors_dtype(self.hidden_projection_config.scale_bias_dtype);
        let scales = remove_tensor(&mut tensors, &bindings.hidden_projection.scales, scale_bias_dtype)?.into_data();
        let biases = remove_tensor(&mut tensors, &bindings.hidden_projection.biases, scale_bias_dtype)?.into_data();
        validate_len(
            "Qwen3x DFlash2 selector hidden-projection weight",
            weight.len(),
            self.hidden_projection_config.weight_bytes(),
        )?;
        validate_len(
            "Qwen3x DFlash2 selector hidden-projection scales",
            scales.len(),
            self.hidden_projection_config.scale_or_bias_bytes(),
        )?;
        validate_len(
            "Qwen3x DFlash2 selector hidden-projection biases",
            biases.len(),
            self.hidden_projection_config.scale_or_bias_bytes(),
        )?;
        assert!(tensors.is_empty());
        self.predecessor_codebook
            .load_weights(device, store, bindings.predecessor_codebook)?;
        self.successor_codebook
            .load_weights(device, store, bindings.successor_codebook)?;
        self.weights = Some(Qwen3xDFlash2SelectorWeights {
            hidden_projection_weight: Buffer::from_slice(device, &weight),
            hidden_projection_scales: Buffer::from_slice(device, &scales),
            hidden_projection_biases: Buffer::from_slice(device, &biases),
        });
        Ok(())
    }

    pub fn unload_weights(&mut self) -> Rc<Unembed> {
        assert!(self.weights.is_some(), "Qwen3x DFlash2 selector weights are not loaded");
        self.weights.take();
        self.successor_codebook.unload_weights();
        self.predecessor_codebook.unload_weights();
        self.unembed.take().expect("Qwen3x DFlash2 Main unembed is not loaded")
    }

    pub fn load_unembed(&mut self, unembed: Rc<Unembed>) {
        assert!(self.unembed.is_none(), "Qwen3x DFlash2 Main unembed is already loaded");
        self.unembed = Some(unembed);
    }

    pub fn prepare_static(&self, req_slots: &[u32], distribution_store: &SpecProbsStore) -> u32 {
        let num_requests = req_slots.len();
        assert!(num_requests > 0 && num_requests <= self.max_requests as usize);
        self.req_slots.write_typed(0, req_slots);
        let proposal_rows = num_requests * self.num_spec_tokens as usize;
        let mut row_indices = Vec::with_capacity(proposal_rows);
        let mut distribution_indices = Vec::with_capacity(proposal_rows);
        for (request_index, &req_slot) in req_slots.iter().enumerate() {
            for step_index in 0..self.num_spec_tokens as usize {
                row_indices.push((request_index * self.spec_block_size as usize + step_index + 1) as u32);
                let distribution_index = distribution_store.draft_distribution_index(req_slot, step_index);
                assert!(
                    distribution_index < distribution_store.num_draft_distributions(),
                    "Qwen3x DFlash2 draft distribution index exceeds the distribution store"
                );
                distribution_indices.push(distribution_index);
            }
        }
        self.row_indices.write_typed(0, &row_indices);
        self.output_distribution_indices.write_typed(0, &distribution_indices);
        num_requests as u32
    }

    pub fn anchor_token_ids(&self) -> &Buffer {
        &self.anchor_token_ids
    }

    pub fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }

    pub fn add_replay_arguments(&self, num_requests: u32, arguments: &mut ReplayArguments) {
        let top_k_shape = self.top_k_shape(num_requests);
        let selector_shape = self.selector_shape(num_requests);
        let num_proposal_rows = top_k_shape.num_total_sampling_inputs;
        let candidate_count = self
            .selector_config
            .candidate_count(selector_shape)
            .try_into()
            .expect("Qwen3x DFlash2 active candidate count must fit u32");
        arguments.set_u32(DFLASH2_OUTPUT_NUM_ACTIVE_PROPOSAL_ROWS, num_proposal_rows);
        arguments.set_u32(DFLASH2_OUTPUT_NUM_ACTIVE_CANDIDATES, candidate_count);
        arguments.set_u32(DFLASH2_OUTPUT_NUM_ACTIVE_REQUESTS, num_requests);
        self.top_k_map
            .add_replay_arguments(top_k_shape, top_k_shape.num_total_sampling_inputs, arguments);
        self.top_k_reduce
            .add_replay_arguments(top_k_shape, top_k_shape.num_total_sampling_inputs, arguments);
    }

    pub fn read_proposal(&self, req_slots: &[u32], distribution_store: &mut SpecProbsStore) -> Qwen3xDFlash2Proposal {
        let count = req_slots.len() * self.num_spec_tokens as usize;
        let flat_token_ids = self.proposal_token_ids.read_typed::<i32>(0, count);
        let flat_probs = self.proposal_probs.read_typed::<f32>(0, count);
        let mut token_ids = Vec::with_capacity(req_slots.len());
        let mut token_probs = Vec::with_capacity(req_slots.len());
        for (request_index, &req_slot) in req_slots.iter().enumerate() {
            let begin = request_index * self.num_spec_tokens as usize;
            let end = begin + self.num_spec_tokens as usize;
            let ids = flat_token_ids[begin..end]
                .iter()
                .enumerate()
                .map(|(step_index, &token_id)| {
                    let token_id =
                        u32::try_from(token_id).expect("Qwen3x DFlash2 selector returned a negative token ID");
                    distribution_store.set_expected_draft_token(req_slot, step_index, token_id);
                    token_id
                })
                .collect();
            token_ids.push(ids);
            token_probs.push(flat_probs[begin..end].to_vec());
        }
        Qwen3xDFlash2Proposal { token_ids, token_probs }
    }

    fn top_k_shape(&self, num_requests: u32) -> top_k::Shape {
        assert!(num_requests > 0 && num_requests <= self.max_requests);
        top_k::Shape {
            num_total_sampling_inputs: num_requests * self.num_spec_tokens,
            vocab_size: self.vocab_size,
            top_k: self.selector_config.top_k,
        }
    }

    fn selector_shape(&self, num_requests: u32) -> dflash2_selector::Shape {
        dflash2_selector::Shape {
            num_total_requests: num_requests,
            num_steps: self.num_spec_tokens,
        }
    }

    fn unembed(&self) -> &Unembed {
        self.unembed
            .as_deref()
            .expect("Qwen3x DFlash2 Main unembed must be loaded before execution")
    }
}

impl ReplayComponent for Qwen3xDFlash2Output {
    type Key = Qwen3xDFlash2OutputReplayKey;
    type Input<'a> = Qwen3xDFlash2OutputArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        Qwen3xDFlash2OutputReplayKey {
            num_total_requests: input.num_requests,
            unembed_topology: self
                .unembed()
                .replay_topology(input.num_requests * self.num_spec_tokens),
            hidden_projection_topology: self
                .hidden_projection
                .topology(input.num_requests * self.num_spec_tokens),
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let num_proposal_rows = input.num_requests * self.num_spec_tokens;
        self.gather.record(
            recorder,
            num_proposal_rows,
            ReplayU32::Parameter(DFLASH2_OUTPUT_NUM_ACTIVE_PROPOSAL_ROWS),
            input.hidden,
            &self.row_indices,
            &self.proposal_hidden,
        );
        let _ = <Unembed as ReplayLayer>::record(
            self.unembed(),
            recorder,
            UnembedInput {
                num_total_rows: num_proposal_rows,
                num_active_rows: ReplayU32::Parameter(DFLASH2_OUTPUT_NUM_ACTIVE_PROPOSAL_ROWS),
                hidden: &self.proposal_hidden,
                logits: &self.logits,
            },
        );
        let top_k_shape = self.top_k_shape(input.num_requests);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.top_k_map.invoke_replay(
            top_k_shape,
            Dtype::Bfloat16,
            top_k::Operation::Merge,
            top_k::MapBuffers {
                logits: &self.logits,
                logits_offset_bytes: 0,
                tile_token_ids: &self.partial_candidate_token_ids,
                tile_logits: &self.partial_candidate_logits,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.top_k_reduce.invoke_merge(
            top_k_shape,
            top_k::MergeBuffers {
                tile_token_ids: &self.partial_candidate_token_ids,
                tile_logits: &self.partial_candidate_logits,
                token_ids: &self.candidate_token_ids,
                logits: &self.candidate_logits,
            },
        )));
        let weights = self
            .weights
            .as_ref()
            .expect("Qwen3x DFlash2 selector weights must be loaded before execution");
        recorder.record_with_barrier_before(ReplayOp::opaque(self.hidden_projection.invoke(
            num_proposal_rows,
            ReplayU32::Parameter(DFLASH2_OUTPUT_NUM_ACTIVE_PROPOSAL_ROWS),
            &self.projected_hidden,
            0,
            &self.proposal_hidden,
            0,
            &weights.hidden_projection_weight,
            0,
            &weights.hidden_projection_scales,
            0,
            &weights.hidden_projection_biases,
            0,
        )));
        let selector_shape = self.selector_shape(input.num_requests);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.selector.invoke_predecessor_ids(
            selector_shape,
            ReplayU32::Parameter(DFLASH2_OUTPUT_NUM_ACTIVE_REQUESTS),
            dflash2_selector::PredecessorIdBuffers {
                anchor_token_ids: &self.anchor_token_ids,
                candidate_token_ids: &self.candidate_token_ids,
                predecessor_token_ids: &self.predecessor_token_ids,
            },
        )));
        let candidate_count = selector_shape.num_total_requests * self.num_spec_tokens * self.selector_config.top_k;
        let _ = <Embed as ReplayLayer>::record(
            &self.predecessor_codebook,
            recorder,
            EmbedInput {
                num_total_tokens: candidate_count,
                num_active_tokens: ReplayU32::Parameter(DFLASH2_OUTPUT_NUM_ACTIVE_CANDIDATES),
                token_ids: &self.predecessor_token_ids,
                output_hidden: &self.predecessor_embeddings,
            },
        );
        let _ = <Embed as ReplayLayer>::record(
            &self.successor_codebook,
            recorder,
            EmbedInput {
                num_total_tokens: candidate_count,
                num_active_tokens: ReplayU32::Parameter(DFLASH2_OUTPUT_NUM_ACTIVE_CANDIDATES),
                token_ids: &self.candidate_token_ids,
                output_hidden: &self.successor_embeddings,
            },
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(self.selector.invoke_scores(
            selector_shape,
            ReplayU32::Parameter(DFLASH2_OUTPUT_NUM_ACTIVE_REQUESTS),
            dflash2_selector::ScoreBuffers {
                candidate_logits: &self.candidate_logits,
                projected_hidden: &self.projected_hidden,
                predecessor_embeddings: &self.predecessor_embeddings,
                successor_embeddings: &self.successor_embeddings,
                scores: &self.scores,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.selector.invoke_walk(
            selector_shape,
            ReplayU32::Parameter(DFLASH2_OUTPUT_NUM_ACTIVE_REQUESTS),
            dflash2_selector::WalkBuffers {
                candidate_token_ids: &self.candidate_token_ids,
                scores: &self.scores,
                params: self.params.buffer(),
                req_slots: &self.req_slots,
                sample_positions: input.sample_positions,
                sampling_domain: u32::from(inference_executor_core::sampling::SamplingDomain::Draft),
                output_distribution_indices: &self.output_distribution_indices,
                proposal_token_ids: &self.proposal_token_ids,
                proposal_probs: &self.proposal_probs,
                distribution_token_ids: input.distribution_store.draft_token_ids(),
                distribution_probs: input.distribution_store.draft_probs(),
                max_distribution_k: input.distribution_store.max_k() as u32,
                num_output_distributions: input.distribution_store.num_draft_distributions(),
            },
        )));
    }
}

fn require_affine_quantization(
    quantization: &inference_executor_core::model::qwen::v3_x::ResolvedQuantizationConfig,
    tensor_name: &str,
) -> Result<(), ModelExecutorError> {
    if !matches!(quantization.mode.as_deref(), None | Some("affine")) {
        return Err(ModelExecutorError::custom(format!(
            "Qwen3x DFlash2 selector tensor {tensor_name:?} requires affine quantization, got mode={:?}",
            quantization.mode
        )));
    }
    Ok(())
}
