//! Qwen3.5 MTP combined Spec invocation.
//!
//! MTP consumes the Main final hidden state. It does not use the DFlash2 selected-residual capture or the
//! independent Spec Prefill and Decode lifecycle. One outer invocation performs sequential proposal steps because
//! each sampled draft token is an input to the next step.
//!
//! ```text
//! Main final hidden + Main decision
//!                    |
//!                    v
//!           prepare MTP requests
//!                    |
//!                    v
//! +---------- for proposal step i = 0..P-1 ----------+
//! |                                                    |
//! | gathered previous hidden + token embedding         |
//! |                    |                               |
//! |                    v                               |
//! |                 MTP Embed                          |
//! |                    |                               |
//! |                    v                               |
//! |                 MTP Body                           |
//! |                    |                               |
//! |                    v                               |
//! |            Gather + Unembed                        |
//! |                    |                               |
//! |                    v                               |
//! |          sample one draft token                    |
//! |                    |                               |
//! |                    +----------> input for step i+1  |
//! |                                                    |
//! +----------------------------------------------------+
//!                    |
//!                    v
//!       proposal tokens/probabilities
//!                    |
//!                    v
//!              SpecProbsStore
//! ```
//!
//! The outer lifecycle is combined even when the data dependency requires an internal submit, wait, and read between
//! proposal steps.

use std::rc::Rc;

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::check_finite;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;
use inference_executor_core::replay::ReplayBucketPolicy;
use inference_runtime_core::compute::BatchDeviceRequest;

use crate::attn::gqa::backend::GQAReplayTopology;
use crate::attn::gqa::backend::add_gqa_private_replay_arguments;
use crate::attn::gqa::batch_metadata::GQAReplayShape;
use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::checkpoint::SafeTensorStore;
use crate::mlp::dense::scratch::DenseMLPScratch;
use crate::mlp::moe::scratch::MoEScratch;
use crate::model::qwen::v3_5::Qwen35GQAReplayKey;
use crate::model::qwen::v3_5::component_config::Qwen35MetalDefaults;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPLayer;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPLayerInput;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPLayerScratch;
use crate::model::qwen::v3_5::mtp::layer::Qwen35MTPMLPReplayTopology;
use crate::model::qwen::v3_x::state::Qwen3xGQAState;
use crate::model::qwen::v3_x::weight::remove_qwen3x_norm_weight;
use crate::model::rms_norm::RMSNorm;
use crate::replay::ReplayComponent;

pub mod decode_plan;
pub mod embed;
pub mod hidden_state_cache;
pub mod hidden_state_transfer;
pub mod layer;

pub const QWEN35_MTP_GQA_LAYER_INDEX: ReplayParameterKey = ReplayParameterKey::new("qwen3.5.mtp.gqa_layer_index");
const QWEN35_MTP_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("qwen3.5.mtp.num_active_tokens");
const QWEN35_MTP_FIRST_CACHE_LANE: usize = 1;

pub struct Qwen35MTP {
    finite_check: Option<check_finite::Compute>,
    layer: Qwen35MTPLayer,
    output_norm: RMSNorm,
    request_page_table: Option<Rc<GQARequestPageTable>>,
    num_cache_pages: usize,
    replay_bucket_policy: ReplayBucketPolicy,
}

#[derive(Clone, Copy)]
pub struct Qwen35MTPArgs<'a> {
    pub num_tokens: u32,
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub gqa: &'a crate::attn::gqa::batch_metadata::GQAMetadataBuffers,
    pub gqa_replay_topology: GQAReplayTopology,
    pub pages: &'a Buffer,
}

impl Qwen35MTP {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        main_config: &Qwen35ModelConfig,
        config: &Qwen35ModelConfig,
        max_tokens: usize,
        defaults: Qwen35MetalDefaults,
        gqa_state: &Qwen3xGQAState,
        num_cache_pages: usize,
        layer_scratch: Rc<Qwen35MTPLayerScratch>,
        dense_scratch: Option<&Rc<DenseMLPScratch>>,
        moe_scratch: Option<&Rc<MoEScratch>>,
    ) -> Result<Self, ModelExecutorError> {
        let hidden_dim = config.text_config.hidden_size;
        let layer = Qwen35MTPLayer::new(
            device,
            config,
            defaults,
            main_config.text_config.num_hidden_layers,
            gqa_state,
            layer_scratch,
            dense_scratch,
            moe_scratch,
        )?;
        let max_tokens = max_tokens
            .try_into()
            .expect("qwen3.5 MTP replay token capacity must fit u32");
        let mut topology_boundaries = gqa_state.replay_token_topology_boundaries().into_vec();
        topology_boundaries.extend(layer.mlp_replay_topology_boundaries());
        Ok(Self {
            finite_check: crate::trace::qwen35_check_finite().then(|| {
                check_finite::Compute::new(
                    device,
                    Dtype::Bfloat16,
                    hidden_dim.try_into().expect("hidden dim must fit u32"),
                )
            }),
            layer,
            output_norm: RMSNorm::new(device, hidden_dim, config.text_config.rms_norm_eps),
            request_page_table: Some(Rc::clone(gqa_state.request_page_table())),
            num_cache_pages,
            replay_bucket_policy: ReplayBucketPolicy::with_topology_boundaries(max_tokens, &topology_boundaries),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        bindings: Qwen35LayerWeightBindings,
        final_norm_weight: String,
    ) -> Result<(), ModelExecutorError> {
        self.layer.load_weights(device, store, config, bindings)?;
        let hidden_dim = config.text_config.hidden_size;
        let mut tensors = store.load_tensors([final_norm_weight.as_str()])?;
        self.output_norm.load_weights(remove_qwen3x_norm_weight(
            device,
            &mut tensors,
            &final_norm_weight,
            &[hidden_dim],
        )?);
        assert!(tensors.is_empty(), "qwen3.5 MTP must consume its final norm tensor map");
        Ok(())
    }

    pub fn unload_weights(&mut self) {
        self.output_norm.unload_weights();
        self.layer.unload_weights();
    }

    pub fn unload_state(&mut self) {
        self.layer.unload_state();
        self.request_page_table
            .take()
            .expect("qwen3.5 MTP request page-table state must be loaded");
    }

    pub fn load_state(&mut self, state: &Qwen3xGQAState) {
        assert!(
            self.request_page_table.is_none(),
            "qwen3.5 MTP request page-table state is already loaded"
        );
        self.request_page_table = Some(Rc::clone(state.request_page_table()));
        self.layer.load_state(state);
    }

    pub fn capacity(&self, num_active_tokens: u32) -> u32 {
        self.replay_bucket_policy.capacity(num_active_tokens)
    }

    pub fn prepare_replay(
        &self,
        num_active_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
    ) -> Qwen35MTPReplayKey {
        self.replay_key_for(num_active_tokens, gqa_shape, gqa_topology)
    }

    pub fn replay_arguments(
        &self,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        gqa_layer_index: u32,
    ) -> ReplayArguments {
        self.validate_capacity(gqa_shape.num_tokens, gqa_shape.num_total_tokens);
        mtp_replay_arguments(gqa_shape, gqa_topology, gqa_layer_index)
    }

    pub fn prepare_pages(&self, batch: &BatchDeviceRequest) {
        let request_page_table = self.request_page_table();
        assert!(
            request_page_table.num_layers() > 0,
            "qwen3.5 MTP page table requires logical layers"
        );
        for request in &batch.dev_reqs {
            let page_ids_by_lane_and_block = request.decoder_sync_blocks.kv_page_ids();
            for gqa_layer_index in 0..request_page_table.num_layers() {
                let cache_lane = QWEN35_MTP_FIRST_CACHE_LANE
                    .checked_add(gqa_layer_index)
                    .expect("qwen3.5 MTP cache lane must fit usize");
                let page_ids_by_block = page_ids_by_lane_and_block
                    .get(cache_lane)
                    .unwrap_or_else(|| panic!("qwen3.5 MTP request page table missing cache lane {cache_lane}"));
                for (block_offset, page_ids) in page_ids_by_block.iter().enumerate() {
                    assert_eq!(
                        page_ids.len(),
                        request_page_table.num_page_ids_per_block(),
                        "qwen3.5 MTP cache lane {cache_lane} must contain one physical layer's page IDs"
                    );
                    self.write_page_ids(
                        request.req_slot,
                        gqa_layer_index,
                        request
                            .decoder_sync_blocks
                            .block_index()
                            .checked_add(block_offset)
                            .expect("qwen3.5 MTP cache-block index must fit usize"),
                        page_ids,
                    );
                }
            }
        }
    }

    pub fn write_page_ids(&self, req_slot: u32, layer_index: usize, block_index: usize, page_ids: &[u32]) {
        assert!(
            page_ids
                .iter()
                .all(|&page_id| (page_id as usize) < self.num_cache_pages),
            "runtime supplied a qwen3.5 MTP page ID outside the cache-page buffer"
        );
        self.request_page_table()
            .write_page_ids(req_slot, layer_index, block_index, page_ids);
    }

    pub fn read_page_ids(&self, req_slot: u32, layer_index: usize, block_index: usize) -> Vec<u32> {
        self.request_page_table()
            .read_page_ids(req_slot, layer_index, block_index)
    }

    pub fn validate_batch(&self, microbatch: &Qwen35Microbatch) {
        let max_context_tokens = (0..microbatch.num_reqs())
            .map(|req_index| {
                microbatch.token_indices()[req_index]
                    .checked_add(microbatch.num_total_tokens(req_index))
                    .expect("qwen3.5 MTP GQA request context length overflow")
            })
            .max()
            .expect("qwen3.5 MTP batch requires requests") as usize;
        let page_capacity = self
            .request_page_table()
            .num_blocks()
            .checked_mul(self.request_page_table().num_page_ids_per_block())
            .expect("qwen3.5 MTP GQA page capacity must fit usize");
        let tokens_per_page = self.layer.gqa_tokens_per_page();
        assert!(
            max_context_tokens.div_ceil(tokens_per_page.max(1)) <= page_capacity,
            "qwen3.5 MTP GQA request context exceeds page-table capacity"
        );
    }

    fn request_page_table(&self) -> &GQARequestPageTable {
        self.request_page_table
            .as_deref()
            .expect("qwen3.5 MTP request page-table state must be loaded before execution")
    }
}

impl Qwen35MTP {
    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        args: Qwen35MTPArgs<'a>,
    ) -> &'a Buffer {
        match num_active_tokens {
            ReplayU32::Fixed(value) => {
                assert_eq!(value, args.num_tokens);
                assert_eq!(value, num_total_tokens);
            },
            ReplayU32::Parameter(_) => self.validate_capacity(args.num_tokens, num_total_tokens),
        }
        let hidden = self.layer.record(
            recorder,
            num_total_tokens,
            num_active_tokens,
            Qwen35MTPLayerInput {
                gqa: args.gqa,
                num_tokens: args.num_tokens,
                pages: args.pages,
                residual_input: args.hidden_input,
            },
        );
        self.output_norm.record_with_barrier(
            recorder,
            num_total_tokens,
            num_active_tokens,
            hidden,
            args.hidden_output,
        );
        if let Some(check) = &self.finite_check {
            recorder.record(ReplayOp::opaque(check.invoke(
                check_finite::Input {
                    buffer: args.hidden_output,
                    num_total_rows: num_total_tokens,
                    num_active_rows: num_active_tokens,
                },
                "MTP output_norm.output".to_owned(),
            )));
        }
        args.hidden_output
    }

    fn replay_key_for(
        &self,
        num_active_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
    ) -> Qwen35MTPReplayKey {
        gqa_shape.validate();
        assert_eq!(
            gqa_shape.num_tokens, num_active_tokens,
            "qwen3.5 MTP GQA active tokens must match the stage"
        );
        let num_total_tokens = gqa_shape.num_total_tokens;
        self.validate_capacity(num_active_tokens, num_total_tokens);
        Qwen35MTPReplayKey::new(
            num_total_tokens,
            gqa_shape,
            gqa_topology,
            self.layer.mlp_replay_topology(num_total_tokens),
        )
    }

    fn validate_capacity(&self, num_active_tokens: u32, num_total_tokens: u32) {
        assert_eq!(
            self.capacity(num_active_tokens),
            num_total_tokens,
            "qwen3.5 MTP metadata token capacity must match the stage policy"
        );
        assert_eq!(
            self.layer.mlp_replay_topology(num_active_tokens),
            self.layer.mlp_replay_topology(num_total_tokens),
            "qwen3.5 MTP replay token capacity must preserve the MLP topology"
        );
    }
}

fn mtp_replay_arguments(
    gqa_shape: GQAReplayShape,
    gqa_topology: GQAReplayTopology,
    gqa_layer_index: u32,
) -> ReplayArguments {
    let mut arguments = ReplayArguments::new().with_u32(QWEN35_MTP_NUM_ACTIVE_TOKENS, gqa_shape.num_tokens);
    add_gqa_private_replay_arguments(gqa_shape, gqa_topology, &mut arguments);
    arguments.set_u32(QWEN35_MTP_GQA_LAYER_INDEX, gqa_layer_index);
    arguments
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MTPReplayKey {
    num_total_tokens: u32,
    mlp_topology: Qwen35MTPMLPReplayTopology,
    gqa: Qwen35GQAReplayKey,
}

impl Qwen35MTPReplayKey {
    fn new(
        num_total_tokens: u32,
        gqa_shape: GQAReplayShape,
        gqa_topology: GQAReplayTopology,
        mlp_topology: Qwen35MTPMLPReplayTopology,
    ) -> Self {
        gqa_shape.validate();
        assert_eq!(
            gqa_shape.num_total_tokens, num_total_tokens,
            "qwen3.5 MTP GQA key capacity must match the stage"
        );
        Self {
            num_total_tokens,
            mlp_topology,
            gqa: Qwen35GQAReplayKey::new(gqa_shape, gqa_topology),
        }
    }

    fn num_total_tokens(&self) -> u32 {
        self.num_total_tokens
    }
}

impl ReplayComponent for Qwen35MTP {
    type Key = Qwen35MTPReplayKey;
    type Input<'a> = Qwen35MTPArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.replay_key_for(input.num_tokens, input.gqa.replay_shape(), input.gqa_replay_topology)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        Qwen35MTP::record(
            self,
            recorder,
            key.num_total_tokens(),
            ReplayU32::Parameter(QWEN35_MTP_NUM_ACTIVE_TOKENS),
            *input,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use inference_backend_metal::metal::Device;
    use inference_executor_core::model::qwen::v3_5::Qwen35TextConfig;
    use inference_runtime_core::compute::BatchDeviceRequest;
    use inference_runtime_core::compute::DecoderSyncBlocks;
    use inference_runtime_core::compute::DeviceRequest;
    use inference_runtime_core::compute::QueryTokens;
    use inference_runtime_core::config::SamplingConfig;
    use inference_runtime_core::runtime::Token;

    use super::*;
    use crate::attn::gqa::request_page_table::GQAPageTableLayout;
    use crate::model::qwen::v3_5::component_config::derive_qwen35_dense_mlp_configs;
    use crate::model::qwen::v3_5::component_config::derive_qwen35_gqa_configs;

    #[test]
    fn test_prepare_pages_maps_cache_lanes_to_gqa_layers() {
        let device = Device::system_default();
        let mtp = test_mtp(&device);
        let batch = BatchDeviceRequest::new(
            1,
            [DeviceRequest::new(
                7,
                1,
                QueryTokens::Decode {
                    epoch: 0,
                    token_index: 1,
                    tokens: vec![Token::new(42)],
                    spec_tokens: Vec::new(),
                },
                DecoderSyncBlocks::new(
                    1,
                    vec![
                        vec![vec![1, 2]],
                        vec![vec![10, 11], vec![12, 13]],
                        vec![vec![20, 21], vec![22, 23]],
                        vec![vec![30, 31], vec![32, 33]],
                    ],
                    vec![Vec::new(); 4],
                ),
                None,
                vec![],
                SamplingConfig::default(),
            )],
        );

        mtp.prepare_pages(&batch);

        assert_eq!(mtp.read_page_ids(1, 0, 1), vec![10, 11]);
        assert_eq!(mtp.read_page_ids(1, 0, 2), vec![12, 13]);
        assert_eq!(mtp.read_page_ids(1, 1, 1), vec![20, 21]);
        assert_eq!(mtp.read_page_ids(1, 1, 2), vec![22, 23]);
        assert_eq!(mtp.read_page_ids(1, 2, 1), vec![30, 31]);
        assert_eq!(mtp.read_page_ids(1, 2, 2), vec![32, 33]);
    }

    fn test_mtp(device: &Device) -> Qwen35MTP {
        let main_config = test_model_config();
        let mtp_config = test_model_config();
        let defaults = Qwen35MetalDefaults::default();
        let (gqa_core, gqa_metal) = derive_qwen35_gqa_configs(
            main_config.text_config.num_hidden_layers,
            &mtp_config.text_config,
            defaults,
        )
        .unwrap();
        let gqa_state = Qwen3xGQAState::new(
            device,
            gqa_core,
            gqa_metal,
            GQAPageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 3,
                num_blocks: 4,
                num_page_ids_per_block: 2,
            },
            2,
            64,
        );
        let (dense_core, dense_metal) = derive_qwen35_dense_mlp_configs(0, &mtp_config.text_config, defaults).unwrap();
        let dense_scratch = Rc::new(DenseMLPScratch::new(device, &dense_core, dense_metal.io_dtype, 2));
        Qwen35MTP::new(
            device,
            &main_config,
            &mtp_config,
            2,
            defaults,
            &gqa_state,
            64,
            Rc::new(Qwen35MTPLayerScratch::new(
                device,
                2,
                mtp_config.text_config.hidden_size,
            )),
            Some(&dense_scratch),
            None,
        )
        .unwrap()
    }

    fn test_model_config() -> Qwen35ModelConfig {
        Qwen35ModelConfig {
            model_type: "qwen3_5".to_string(),
            tie_word_embeddings: false,
            text_config: Qwen35TextConfig {
                model_type: "qwen3_5_text".to_string(),
                hidden_size: 128,
                hidden_act: "silu".to_string(),
                intermediate_size: 128,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 128,
                rms_norm_eps: 1e-6,
                vocab_size: 16,
                max_position_embeddings: 32,
                attention_bias: false,
                tie_word_embeddings: false,
                layer_types: vec!["full_attention".to_string()],
                full_attention_interval: 1,
                linear_num_value_heads: 1,
                linear_num_key_heads: 1,
                linear_key_head_dim: 128,
                linear_value_head_dim: 128,
                linear_conv_kernel_dim: 4,
                decoder_sparse_step: 1,
                num_experts: 0,
                num_experts_per_tok: 0,
                shared_expert_intermediate_size: 0,
                moe_intermediate_size: 0,
                norm_topk_prob: true,
                mtp_num_hidden_layers: 1,
                mtp_use_dedicated_embeddings: false,
                rope_theta: 10_000.0,
                partial_rotary_factor: 1.0,
                rope_parameters: None,
                use_cache: true,
                dtype: None,
                scale: 1.0 / 128.0_f32.sqrt(),
                rope_dim: 128,
            },
            quantization: None,
        }
    }
}
