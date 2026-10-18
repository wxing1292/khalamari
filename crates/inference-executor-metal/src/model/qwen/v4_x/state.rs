use std::collections::HashSet;

use inference_backend_metal::components::ple;
use inference_backend_metal::components::qsa_indexer;
use inference_backend_metal::components::state_pages;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v4_x::LayerType;
use inference_executor_core::model::qwen::v4_x::Qwen4xModelConfig;
use inference_executor_core::replay::ReplayBucketPolicy;
use inference_runtime_core::compute::BatchDeviceRequest;
use wincode::SchemaRead;
use wincode::SchemaWrite;

use crate::attn::gqa::request_page_table::GQARequestPageTable;
use crate::attn::qsa::backend::QSAActive;
use crate::attn::qsa::batch_metadata::QSASDPAMetadataBuffers;
use crate::attn::qsa::sdpa::QSASDPAConfig;
use crate::attn::qsa::sdpa::QSASDPARequest;
use crate::attn::qsa::sdpa::QSASDPAShape;
use crate::attn::qsa_indexer::batch_metadata::QSAIndexerMetadataBuffers;
use crate::attn::qsa_indexer::batch_metadata::QSAIndexerRequest;
use crate::def::replay_op::MetalReplayRuntime;
use crate::def::replay_op::ReplayOp;
use crate::def::replay_op::ReplayRecorder;
use crate::model::qwen::v4_x::executor::Qwen4xExecutorConfig;
use crate::model::qwen::v4_x::main::Qwen4xMain;
use crate::model::qwen::v4_x::main::ngram_core;
use crate::model::qwen::v4_x::main::ple_core;
use crate::ple::batch_metadata::PLEMetadataBuffers;
use crate::ple::batch_metadata::PLERequest;

const PAGE_BYTES: usize = 32 * 1024;

pub struct Qwen4xBatchState {
    pub sdpa: QSASDPAMetadataBuffers,
    pub indexer: QSAIndexerMetadataBuffers,
    pub indexer_shape: qsa_indexer::Shape,
    pub active: QSAActive,
}

#[derive(Clone, Debug, SchemaRead, SchemaWrite)]
pub struct Qwen4xRequestState {
    versions: Vec<u32>,
    history: Vec<Vec<u32>>,
    index_page_ids: Vec<Vec<u32>>,
}
struct Publish {
    req_index: usize,
    version: u32,
    page_ids: Vec<u32>,
}

pub struct Qwen4xState {
    layout: GQAPageTableLayout,
    config: Qwen4xExecutorConfig,
    index_layout: qsa_indexer::CacheLayout,
    carry_bytes: usize,
    conv_bytes: usize,
    history_len: usize,
    eos: u32,
    num_ple_layers: usize,
    aux_bytes: usize,
    batch: Qwen4xBatchState,
    resources: Option<Resources>,
    versions: Vec<u32>,
    history: Vec<Vec<u32>>,
    gdn_page_ids: Vec<Vec<Vec<u32>>>,
    publishes: Vec<Publish>,
    ple_core: Option<ple::Config>,
    max_groups: u32,
    group_policy: ReplayBucketPolicy,
    query_policy: ReplayBucketPolicy,
    pending_commit: Option<crate::def::replay_op::MetalReplaySubmission>,
}
struct Resources {
    pages: GQARequestPageTable,
    index_page_ids: Vec<Buffer>,
    index_states: Vec<Buffer>,
    ple_states: Vec<Buffer>,
    ple_metadata: Option<PLEMetadataBuffers>,
    packed: Buffer,
    copy: state_pages::Compute,
    ple_commit: Option<ple::Compute>,
}
impl Qwen4xState {
    pub fn new(
        device: &Device,
        model: &Qwen4xModelConfig,
        config: Qwen4xExecutorConfig,
    ) -> Result<Self, ModelExecutorError> {
        let text = &model.text_config;
        let qsa_layers = (0..text.num_hidden_layers)
            .filter(|&i| model.layer_type_at(i) == LayerType::QSA)
            .count();
        assert!(qsa_layers > 0, "Qwen4x requires QSA layers");
        let tokens_per_index_page = PAGE_BYTES / (text.indexer_head_dim * 2) * text.indexer_compress_ratio;
        assert!(
            config.num_tokens_per_block.is_multiple_of(tokens_per_index_page),
            "Qwen4x cache blocks must align to index-key pages"
        );
        assert!(
            config.num_tokens_per_block.is_multiple_of(32),
            "Qwen4x cache blocks must align to main KV pages"
        );
        let num_blocks = text.max_position_embeddings.div_ceil(config.num_tokens_per_block);
        let layout = GQAPageTableLayout {
            num_req_slots: config.max_requests as u32,
            num_gqa_layers: qsa_layers as u32,
            num_blocks: num_blocks.try_into().expect("Qwen4x cache block count must fit u32"),
            num_page_ids_per_block: (config.num_tokens_per_block / 32) as u32,
        };
        let index_layout = qsa_indexer::CacheLayout {
            num_req_slots: config.max_requests as u32,
            num_pages_per_request: (num_blocks * (config.num_tokens_per_block / tokens_per_index_page))
                .try_into()
                .expect("Qwen4x index pages must fit u32"),
            num_state_slots: (config.max_requests + config.max_tokens)
                .try_into()
                .expect("Qwen4x carry slots must fit u32"),
            page_bytes: PAGE_BYTES as u32,
        };
        let carry_bytes = (text.indexer_compress_ratio - 1)
            .checked_mul(text.indexer_head_dim)
            .and_then(|n| n.checked_mul(2))
            .expect("Qwen4x carry bytes must fit usize");
        let pc = (!text.ple_layer_ids.is_empty()).then(|| ple_core(model));
        let conv_bytes = pc.map_or(0, |c| {
            (c.state_len() as usize)
                .checked_mul(c.residual_dim() as usize)
                .and_then(|n| n.checked_mul(2))
                .expect("Qwen4x PLE state bytes must fit usize")
        });
        let history_len = if pc.is_some() { text.ngram_size - 1 } else { 0 };
        let eos = if pc.is_some() { model.eos_token_id()? } else { 0 };
        let num_ple_layers = text.ple_layer_ids.len();
        let aux_bytes = qsa_layers
            .checked_mul(carry_bytes)
            .and_then(|n| n.checked_add(num_ple_layers * conv_bytes))
            .and_then(|n| n.checked_add(history_len * 4))
            .expect("Qwen4x auxiliary state bytes must fit usize");
        assert!(aux_bytes.is_multiple_of(4));
        let max_groups = (num_blocks * config.num_tokens_per_block / text.indexer_compress_ratio)
            .try_into()
            .expect("Qwen4x index groups must fit u32");
        let max_sdpa = QSASDPAShape {
            num_total_tokens: config.max_tokens as u32,
            num_total_sparse_queries: config.max_tokens as u32,
        };
        let indexer_shape = qsa_indexer::Shape {
            num_total_tokens: config.max_tokens as u32,
            num_total_sparse_queries: config.max_tokens as u32,
            num_total_request_sparse_queries: config.max_tokens as u32,
            num_total_groups: max_groups,
            num_total_score_task_templates: config.max_tokens as u32,
        };
        let core = inference_executor_core::attn::qsa_indexer::core::QSAIndexerCore {
            hidden_dim: text.hidden_size,
            num_q_heads: text.indexer_n_heads,
            head_dim: text.indexer_head_dim,
            compress_ratio: text.indexer_compress_ratio,
            top_k: text.indexer_budget / text.indexer_compress_ratio,
            rope_dim: (text.head_dim as f32 * text.rope_parameters.partial_rotary_factor.unwrap()) as usize,
            rope_theta: text.rope_parameters.rope_theta.unwrap(),
            norm_eps: text.rms_norm_eps,
        };
        let batch = Qwen4xBatchState {
            sdpa: QSASDPAMetadataBuffers::new(
                device,
                QSASDPAConfig {
                    num_q_heads: text.num_attention_heads as u32,
                    num_kv_heads: text.num_key_value_heads as u32,
                    head_dim: text.head_dim as u32,
                    scale: (text.head_dim as f32).sqrt().recip(),
                    page_bytes: PAGE_BYTES as u32,
                    page_table_layout: layout,
                    compress_ratio: text.indexer_compress_ratio as u32,
                    top_k: (text.indexer_budget / text.indexer_compress_ratio) as u32,
                },
                max_sdpa,
            ),
            indexer: QSAIndexerMetadataBuffers::new(device, core, index_layout, config.max_tokens as u32),
            indexer_shape,
            active: QSAActive {
                sdpa: Default::default(),
                num_active_score_task_templates: 0,
            },
        };
        let ple_config = pc.map(|c| {
            ple::Config {
                num_channels: c.num_channels,
                hidden_dim: c.hidden_dim,
                conv_kernel_size: c.conv_kernel_size,
                conv_dilation: c.conv_dilation,
                norm_eps: c.norm_eps,
            }
        });
        let mut state = Self {
            layout,
            config,
            index_layout,
            carry_bytes,
            conv_bytes,
            history_len,
            eos,
            num_ple_layers,
            aux_bytes,
            batch,
            resources: None,
            versions: vec![0; config.max_requests],
            history: vec![vec![eos; history_len]; config.max_requests],
            gdn_page_ids: Vec::new(),
            publishes: Vec::new(),
            ple_core: ple_config,
            max_groups,
            group_policy: ReplayBucketPolicy::with_topology_boundaries(max_groups, &[(core.top_k + 1) as u32]),
            query_policy: ReplayBucketPolicy::with_topology_boundaries(
                config.max_tokens as u32,
                &inference_backend_metal::components::qsa_sparse_gqa::Compute::topology_boundaries(),
            ),
            pending_commit: None,
        };
        state.allocate_resources(device, model)?;
        Ok(state)
    }
    pub fn allocate_resources(&mut self, device: &Device, model: &Qwen4xModelConfig) -> Result<(), ModelExecutorError> {
        assert!(self.resources.is_none(), "Qwen4x auxiliary state is already loaded");
        let c = self.config;
        let qsa = self.layout.num_gqa_layers as usize;
        let ple_slots = 2 * c.max_requests + self.max_publish_jobs();
        let ple_state_slots = u32::try_from(ple_slots).expect("Qwen4x PLE state slots must fit u32");
        self.resources = Some(Resources {
            pages: GQARequestPageTable::new(device, self.layout),
            index_page_ids: (0..qsa)
                .map(|_| {
                    Buffer::new_zeroed_elements(
                        device,
                        c.max_requests * self.index_layout.num_pages_per_request as usize,
                        Dtype::Uint32,
                    )
                })
                .collect(),
            index_states: (0..qsa)
                .map(|_| {
                    Buffer::new_zeroed(
                        device,
                        (c.max_requests + c.max_tokens)
                            .checked_mul(self.carry_bytes)
                            .expect("Qwen4x index state allocation must fit usize"),
                    )
                })
                .collect(),
            ple_states: (0..self.num_ple_layers)
                .map(|_| {
                    Buffer::new_zeroed(
                        device,
                        ple_slots
                            .checked_mul(self.conv_bytes)
                            .expect("Qwen4x PLE state allocation must fit usize"),
                    )
                })
                .collect(),
            ple_metadata: if self.num_ple_layers > 0 {
                Some(PLEMetadataBuffers::new(
                    device,
                    ngram_core(model)?,
                    c.max_tokens as u32,
                    c.max_requests as u32,
                    ple_state_slots,
                ))
            } else {
                None
            },
            packed: Buffer::new_zeroed(
                device,
                (c.max_requests + self.max_publish_jobs())
                    .checked_mul(self.aux_bytes)
                    .expect("Qwen4x packed state allocation must fit usize"),
            ),
            copy: state_pages::Compute::new(device),
            ple_commit: self.ple_core.map(|core| ple::Compute::new(device, core)),
        });
        Ok(())
    }
    fn resources(&self) -> &Resources {
        self.resources.as_ref().expect("Qwen4x auxiliary state must be loaded")
    }
    pub fn release_resources(&mut self) {
        self.resources.take().expect("Qwen4x auxiliary state is not loaded");
    }
    pub fn batch(&self) -> &Qwen4xBatchState {
        &self.batch
    }
    pub fn page_table_layout(&self) -> GQAPageTableLayout {
        self.layout
    }
    pub fn page_ids(&self) -> &Buffer {
        self.resources().pages.page_ids_buffer()
    }
    pub fn page_table(&self) -> &GQARequestPageTable {
        &self.resources().pages
    }
    pub fn page_table_mut(&mut self) -> &mut GQARequestPageTable {
        &mut self.resources.as_mut().unwrap().pages
    }
    pub fn index_page_ids(&self, index: usize) -> &Buffer {
        &self.resources().index_page_ids[index]
    }
    pub fn index_states(&self, index: usize) -> &Buffer {
        &self.resources().index_states[index]
    }
    pub fn ple_states(&self, index: usize) -> &Buffer {
        &self.resources().ple_states[index]
    }
    pub fn ple_metadata(&self) -> &PLEMetadataBuffers {
        self.resources()
            .ple_metadata
            .as_ref()
            .expect("Qwen4x model has no PLE layers")
    }
    pub fn max_sdpa_shape(&self) -> QSASDPAShape {
        QSASDPAShape {
            num_total_tokens: self.config.max_tokens as u32,
            num_total_sparse_queries: self.config.max_tokens as u32,
        }
    }
    pub fn max_indexer_shape(&self) -> qsa_indexer::Shape {
        qsa_indexer::Shape {
            num_total_tokens: self.config.max_tokens as u32,
            num_total_sparse_queries: self.config.max_tokens as u32,
            num_total_request_sparse_queries: self.config.max_tokens as u32,
            num_total_groups: self.max_groups,
            num_total_score_task_templates: self.config.max_tokens as u32,
        }
    }
    fn max_publish_jobs(&self) -> usize {
        self.config.max_requests + self.config.max_tokens / self.config.num_tokens_per_block
    }
    pub fn aux_pages_per_block(&self) -> usize {
        self.aux_bytes.div_ceil(PAGE_BYTES)
    }
    pub fn kv_pages_per_block(&self) -> usize {
        self.layout.num_gqa_layers as usize
            * (self.layout.num_page_ids_per_block as usize + self.index_pages_per_block())
    }
    fn index_pages_per_block(&self) -> usize {
        self.index_layout.num_pages_per_request as usize / self.layout.num_blocks as usize
    }
    pub fn gdn_page_ids(&self) -> &[Vec<Vec<u32>>] {
        &self.gdn_page_ids
    }

    /// Splits model-owned page spans once and restores the latest synchronized auxiliary prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &Device,
        runtime: &MetalReplayRuntime<'_>,
        pages: &Buffer,
        request: &BatchDeviceRequest,
        batch: &Qwen35Microbatch,
        num_total_tokens: u32,
        gdn_pages: usize,
    ) {
        self.publishes.clear();
        self.gdn_page_ids.clear();
        let mut restore_jobs = Vec::new();
        let mut restore_slots = Vec::new();
        for (req_index, req) in request.dev_reqs.iter().enumerate() {
            let slot = req.req_slot as usize;
            let position = batch.token_indices()[req_index];
            let synchronized = req
                .decoder_sync_blocks
                .kv_page_ids()
                .first()
                .expect("Qwen4x Main requires cache lane 0");
            for (offset, ids) in synchronized.iter().enumerate() {
                let block = req.decoder_sync_blocks.block_index() + offset;
                assert_eq!(ids.len(), self.kv_pages_per_block());
                let per_layer = self.layout.num_page_ids_per_block as usize + self.index_pages_per_block();
                for (layer, ids) in ids.chunks_exact(per_layer).enumerate() {
                    let (kv, index) = ids.split_at(self.layout.num_page_ids_per_block as usize);
                    self.resources().pages.write_page_ids(req.req_slot, layer, block, kv);
                    let at =
                        slot * self.index_layout.num_pages_per_request as usize + block * self.index_pages_per_block();
                    self.resources().index_page_ids[layer].write_typed(at, index);
                }
            }
            assert!(
                self.versions[slot] <= position,
                "Qwen4x state version exceeds the runtime input position"
            );
            let mut gdn = Vec::new();
            let mut latest = None;
            let state_blocks = req
                .decoder_sync_blocks
                .state_page_ids()
                .first()
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            for (offset, ids) in state_blocks.iter().enumerate() {
                assert_eq!(ids.len(), gdn_pages + self.aux_pages_per_block());
                let (gdn_ids, aux_ids) = ids.split_at(gdn_pages);
                gdn.push(gdn_ids.to_vec());
                let version =
                    ((req.decoder_sync_blocks.block_index() + offset + 1) * self.config.num_tokens_per_block) as u32;
                if version <= self.versions[slot] {
                    continue;
                }
                if version <= position {
                    latest = Some((version, aux_ids));
                } else {
                    self.publishes.push(Publish {
                        req_index,
                        version,
                        page_ids: aux_ids.to_vec(),
                    });
                }
            }
            self.gdn_page_ids.push(gdn);
            if let Some((version, ids)) = latest {
                append_page_jobs(&mut restore_jobs, ids, slot * self.aux_bytes, self.aux_bytes, true);
                restore_slots.push(req.req_slot);
                self.versions[slot] = version;
            }
            assert_eq!(
                self.versions[slot], position,
                "Qwen4x state must match the runtime input position"
            );
        }
        if !restore_jobs.is_empty() {
            let mut recorder = runtime.create_recorder();
            self.record_copy(device, &mut recorder, pages, &self.resources().packed, &restore_jobs);
            self.unpack(device, &mut recorder, &restore_slots);
            runtime.submit_replay(&recorder.build()).wait();
            for &slot in &restore_slots {
                self.history[slot as usize] = self.resources().packed.read_typed::<u32>(
                    (slot as usize * self.aux_bytes + self.history_offset()) / 4,
                    self.history_len,
                );
            }
        }
        let mut requests = Vec::new();
        let mut sparse = 0;
        let mut max_sparse = 0;
        let mut groups = 1;
        let top_k = self.batch.sdpa.config().top_k;
        for (i, &slot) in batch.req_slots().iter().enumerate() {
            let position = batch.token_indices()[i];
            let count = batch.cu_tokens()[i + 1] - batch.cu_tokens()[i];
            let sparse_count = (position + count).saturating_sub((top_k + 1) * 4 - 1).min(count);
            sparse += sparse_count;
            max_sparse = max_sparse.max(sparse_count);
            groups = groups.max((position + count) / 4);
            requests.push(QSAIndexerRequest {
                req_slot: slot,
                token_index: position,
                num_tokens: count,
                group_start: 0,
                src_state_slot: slot,
            });
        }
        let sparse_capacity = self.query_policy.capacity(sparse.max(1)).min(num_total_tokens);
        let shape = QSASDPAShape {
            num_total_tokens,
            num_total_sparse_queries: sparse_capacity,
        };
        self.batch.indexer_shape = qsa_indexer::Shape {
            num_total_tokens,
            num_total_sparse_queries: sparse_capacity,
            num_total_request_sparse_queries: self.query_policy.capacity(max_sparse.max(1)).min(sparse_capacity),
            num_total_groups: self.group_policy.capacity(groups),
            num_total_score_task_templates: sparse_capacity,
        };
        let sdpa_requests: Vec<_> = requests
            .iter()
            .map(|r| {
                QSASDPARequest {
                    req_slot: r.req_slot,
                    token_index: r.token_index,
                    num_tokens: r.num_tokens,
                    group_start: r.group_start,
                }
            })
            .collect();
        let sdpa = self.batch.sdpa.update_with_shape(&sdpa_requests, shape);
        let destinations: Vec<_> = (0..batch.total_tokens())
            .map(|t| self.config.max_requests as u32 + t as u32)
            .collect();
        let indexer = self
            .batch
            .indexer
            .update(&requests, &destinations, self.batch.indexer_shape);
        self.batch.active = QSAActive {
            sdpa,
            num_active_score_task_templates: indexer.num_active_score_task_templates,
        };
        if self.num_ple_layers > 0 {
            let token_ids: Vec<u32> = batch.flat_token_ids().iter().map(|&id| id as u32).collect();
            let ple_requests: Vec<_> = batch
                .req_slots()
                .iter()
                .enumerate()
                .map(|(i, &slot)| {
                    PLERequest {
                        token_ids: &token_ids[batch.cu_tokens()[i] as usize..batch.cu_tokens()[i + 1] as usize],
                        previous_tokens: &self.history[slot as usize],
                        src_state_slot: slot,
                        dst_state_slot: self.config.max_requests as u32 + slot,
                    }
                })
                .collect();
            self.resources
                .as_mut()
                .unwrap()
                .ple_metadata
                .as_mut()
                .unwrap()
                .update(&ple_requests);
        }
    }
    fn history_offset(&self) -> usize {
        self.layout.num_gqa_layers as usize * self.carry_bytes + self.num_ple_layers * self.conv_bytes
    }
    fn record_copy<'a>(
        &'a self,
        device: &Device,
        recorder: &mut ReplayRecorder,
        src: &'a Buffer,
        dst: &'a Buffer,
        jobs: &[[u64; 3]],
    ) {
        if jobs.is_empty() {
            return;
        }
        let values: Vec<u64> = jobs.iter().flatten().copied().collect();
        let buffer = Buffer::from_slice(device, &values);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.resources().copy.copy(
            src,
            dst,
            buffer,
            jobs.len() as u32,
        )));
    }
    fn unpack(&self, device: &Device, recorder: &mut ReplayRecorder, slots: &[u32]) {
        for layer in 0..self.layout.num_gqa_layers as usize {
            let jobs: Vec<_> = slots
                .iter()
                .map(|&slot| {
                    [
                        (slot as usize * self.aux_bytes + layer * self.carry_bytes) as u64,
                        (slot as usize * self.carry_bytes) as u64,
                        self.carry_bytes as u64,
                    ]
                })
                .collect();
            self.record_copy(
                device,
                recorder,
                &self.resources().packed,
                &self.resources().index_states[layer],
                &jobs,
            );
        }
        for layer in 0..self.num_ple_layers {
            let ple_state_offset_bytes =
                self.layout.num_gqa_layers as usize * self.carry_bytes + layer * self.conv_bytes;
            let jobs: Vec<_> = slots
                .iter()
                .map(|&slot| {
                    [
                        (slot as usize * self.aux_bytes + ple_state_offset_bytes) as u64,
                        (slot as usize * self.conv_bytes) as u64,
                        self.conv_bytes as u64,
                    ]
                })
                .collect();
            self.record_copy(
                device,
                recorder,
                &self.resources().packed,
                &self.resources().ple_states[layer],
                &jobs,
            );
        }
    }
    pub fn finish_commit(&mut self) {
        if let Some(submission) = self.pending_commit.take() {
            submission.wait();
        }
    }
    pub fn reset_req_slots(&mut self, slots: &[u32]) {
        self.resources().pages.reset_req_slots(slots);
        for &slot in slots {
            self.versions[slot as usize] = 0;
            self.history[slot as usize].fill(self.eos);
            for index in &self.resources().index_page_ids {
                index.zero_bytes(
                    slot as usize * self.index_layout.num_pages_per_request as usize * 4,
                    self.index_layout.num_pages_per_request as usize * 4,
                );
            }
            for state in &self.resources().index_states {
                state.zero_bytes(slot as usize * self.carry_bytes, self.carry_bytes);
            }
            for state in &self.resources().ple_states {
                state.zero_bytes(slot as usize * self.conv_bytes, self.conv_bytes);
            }
        }
    }
    pub fn commit(
        &mut self,
        device: &Device,
        runtime: &MetalReplayRuntime<'_>,
        pages: &Buffer,
        main: &Qwen4xMain,
        batch: &Qwen35Microbatch,
        versions: &[u32],
    ) {
        let count = batch.num_reqs();
        assert_eq!(versions.len(), count);
        let mut recorder = runtime.create_recorder();
        let mut commits = Vec::new();
        let mut accepted = Vec::new();
        // Each publish gets one independent convolution destination. Final destinations are request-local.
        for (i, &version) in versions.iter().enumerate() {
            let slot = batch.req_slots()[i];
            let n = version - batch.token_indices()[i];
            assert!(n > 0 && n <= batch.cu_tokens()[i + 1] - batch.cu_tokens()[i]);
            commits.extend_from_slice(&[batch.cu_tokens()[i], slot, self.config.max_requests as u32 + slot]);
            accepted.push(n);
        }
        let publishes: Vec<_> = self
            .publishes
            .iter()
            .filter(|p| p.version <= versions[p.req_index])
            .collect();
        for (index, publish) in publishes.iter().enumerate() {
            let i = publish.req_index;
            let n = publish.version - batch.token_indices()[i];
            commits.extend_from_slice(&[
                batch.cu_tokens()[i],
                batch.req_slots()[i],
                (2 * self.config.max_requests + index) as u32,
            ]);
            accepted.push(n);
        }
        for layer in 0..self.num_ple_layers {
            let compute = self.resources().ple_commit.as_ref().unwrap();
            recorder.record_with_barrier_before(ReplayOp::opaque(PLECommitJobs {
                compute,
                normalized: &main.ple_scratch(layer).normalized_gated,
                states: &self.resources().ple_states[layer],
                requests: Buffer::from_slice(device, &commits),
                accepted: Buffer::from_slice(device, &accepted),
                num_tokens: batch.total_tokens() as u32,
                num_jobs: accepted.len() as u32,
                num_state_slots: (2 * self.config.max_requests + self.max_publish_jobs()) as u32,
            }));
        }
        // Publish from immutable candidate states before replacing current request slots.
        for layer in 0..self.layout.num_gqa_layers as usize {
            let jobs: Vec<_> = publishes
                .iter()
                .enumerate()
                .map(|(index, p)| {
                    let flat = batch.cu_tokens()[p.req_index] + p.version - batch.token_indices()[p.req_index] - 1;
                    [
                        ((self.config.max_requests + flat as usize) * self.carry_bytes) as u64,
                        ((self.config.max_requests + index) * self.aux_bytes + layer * self.carry_bytes) as u64,
                        self.carry_bytes as u64,
                    ]
                })
                .collect();
            self.record_copy(
                device,
                &mut recorder,
                &self.resources().index_states[layer],
                &self.resources().packed,
                &jobs,
            );
        }
        for layer in 0..self.num_ple_layers {
            let ple_state_offset_bytes =
                self.layout.num_gqa_layers as usize * self.carry_bytes + layer * self.conv_bytes;
            let jobs: Vec<_> = (0..publishes.len())
                .map(|index| {
                    [
                        ((2 * self.config.max_requests + index) * self.conv_bytes) as u64,
                        ((self.config.max_requests + index) * self.aux_bytes + ple_state_offset_bytes) as u64,
                        self.conv_bytes as u64,
                    ]
                })
                .collect();
            self.record_copy(
                device,
                &mut recorder,
                &self.resources().ple_states[layer],
                &self.resources().packed,
                &jobs,
            );
        }
        let history_at = self.history_offset();
        let mut page_jobs = Vec::new();
        for (index, publish) in publishes.iter().enumerate() {
            let i = publish.req_index;
            let n = publish.version - batch.token_indices()[i];
            let history = self.history_after(batch, i, n);
            self.resources().packed.write_typed(
                ((self.config.max_requests + index) * self.aux_bytes + history_at) / 4,
                &history,
            );
            append_page_jobs(
                &mut page_jobs,
                &publish.page_ids,
                (self.config.max_requests + index) * self.aux_bytes,
                self.aux_bytes,
                false,
            );
        }
        self.record_copy(device, &mut recorder, &self.resources().packed, pages, &page_jobs);
        for layer in 0..self.layout.num_gqa_layers as usize {
            let jobs: Vec<_> = (0..count)
                .map(|i| {
                    let flat = batch.cu_tokens()[i] + versions[i] - batch.token_indices()[i] - 1;
                    [
                        ((self.config.max_requests + flat as usize) * self.carry_bytes) as u64,
                        (batch.req_slots()[i] as usize * self.carry_bytes) as u64,
                        self.carry_bytes as u64,
                    ]
                })
                .collect();
            self.record_copy(
                device,
                &mut recorder,
                &self.resources().index_states[layer],
                &self.resources().index_states[layer],
                &jobs,
            );
        }
        for layer in 0..self.num_ple_layers {
            let jobs: Vec<_> = batch
                .req_slots()
                .iter()
                .map(|&slot| {
                    [
                        ((self.config.max_requests + slot as usize) * self.conv_bytes) as u64,
                        (slot as usize * self.conv_bytes) as u64,
                        self.conv_bytes as u64,
                    ]
                })
                .collect();
            self.record_copy(
                device,
                &mut recorder,
                &self.resources().ple_states[layer],
                &self.resources().ple_states[layer],
                &jobs,
            );
        }
        self.pending_commit = Some(runtime.submit_replay(&recorder.build()));
        for (i, &version) in versions.iter().enumerate() {
            let slot = batch.req_slots()[i] as usize;
            self.history[slot] = self.history_after(batch, i, version - batch.token_indices()[i]);
            self.versions[slot] = version;
        }
    }
    fn history_after(&self, batch: &Qwen35Microbatch, index: usize, accepted: u32) -> Vec<u32> {
        if self.history_len == 0 {
            return Vec::new();
        }
        let mut history = self.history[batch.req_slots()[index] as usize].clone();
        history.extend(
            batch.flat_token_ids()[batch.cu_tokens()[index] as usize..(batch.cu_tokens()[index] + accepted) as usize]
                .iter()
                .map(|&id| id as u32),
        );
        history[history.len() - self.history_len..].to_vec()
    }
    pub fn pack_snapshot(&self, device: &Device, runtime: &MetalReplayRuntime<'_>) {
        let mut recorder = runtime.create_recorder();
        for layer in 0..self.layout.num_gqa_layers as usize {
            let jobs: Vec<_> = (0..self.config.max_requests)
                .map(|slot| {
                    [
                        (slot * self.carry_bytes) as u64,
                        (slot * self.aux_bytes + layer * self.carry_bytes) as u64,
                        self.carry_bytes as u64,
                    ]
                })
                .collect();
            self.record_copy(
                device,
                &mut recorder,
                &self.resources().index_states[layer],
                &self.resources().packed,
                &jobs,
            );
        }
        for layer in 0..self.num_ple_layers {
            let ple_state_offset_bytes =
                self.layout.num_gqa_layers as usize * self.carry_bytes + layer * self.conv_bytes;
            let jobs: Vec<_> = (0..self.config.max_requests)
                .map(|slot| {
                    [
                        (slot * self.conv_bytes) as u64,
                        (slot * self.aux_bytes + ple_state_offset_bytes) as u64,
                        self.conv_bytes as u64,
                    ]
                })
                .collect();
            self.record_copy(
                device,
                &mut recorder,
                &self.resources().ple_states[layer],
                &self.resources().packed,
                &jobs,
            );
        }
        runtime.submit_replay(&recorder.build()).wait();
    }
    pub fn restore_snapshot(&self, device: &Device, runtime: &MetalReplayRuntime<'_>) {
        let mut recorder = runtime.create_recorder();
        let slots: Vec<_> = (0..self.config.max_requests as u32).collect();
        self.unpack(device, &mut recorder, &slots);
        runtime.submit_replay(&recorder.build()).wait();
    }
    pub fn packed(&self) -> &Buffer {
        &self.resources().packed
    }
    pub fn state_bytes(&self) -> usize {
        self.aux_bytes
    }
    pub fn snapshot_metadata(&self, slots: &[std::ops::Range<u32>]) -> Qwen4xRequestState {
        let selected: HashSet<_> = slots.iter().flat_map(|r| r.clone()).collect();
        let mut metadata = Qwen4xRequestState {
            versions: vec![0; self.config.max_requests],
            history: vec![vec![self.eos; self.history_len]; self.config.max_requests],
            index_page_ids: self
                .resources()
                .index_page_ids
                .iter()
                .map(|b| vec![0; b.len_bytes() / 4])
                .collect(),
        };
        for &slot in &selected {
            let i = slot as usize;
            metadata.versions[i] = self.versions[i];
            metadata.history[i] = self.history[i].clone();
            let start = i * self.index_layout.num_pages_per_request as usize;
            let len = self.index_layout.num_pages_per_request as usize;
            for (out, buffer) in metadata.index_page_ids.iter_mut().zip(&self.resources().index_page_ids) {
                out[start..start + len].copy_from_slice(&buffer.read_typed(start, len));
            }
        }
        metadata
    }
    pub fn load_metadata(
        &mut self,
        metadata: Qwen4xRequestState,
        model: &Qwen4xModelConfig,
    ) -> Result<(), ModelExecutorError> {
        if metadata.versions.len() != self.config.max_requests
            || metadata.history.len() != self.config.max_requests
            || metadata
                .versions
                .iter()
                .any(|&v| v as usize > model.text_config.max_position_embeddings)
            || metadata
                .history
                .iter()
                .any(|h| h.len() != self.history_len || h.iter().any(|&id| id as usize >= model.text_config.vocab_size))
            || metadata.index_page_ids.len() != self.resources().index_page_ids.len()
            || metadata
                .index_page_ids
                .iter()
                .zip(&self.resources().index_page_ids)
                .any(|(ids, b)| {
                    ids.len() != b.len_bytes() / 4 || ids.iter().any(|&id| id as usize >= self.config.num_cache_pages)
                })
        {
            return Err(ModelExecutorError::custom(
                "Qwen4x snapshot auxiliary topology does not match the executor",
            ));
        }
        for (buffer, ids) in self.resources().index_page_ids.iter().zip(&metadata.index_page_ids) {
            buffer.write_typed(0, ids);
        }
        self.versions = metadata.versions;
        self.history = metadata.history;
        Ok(())
    }
}

fn append_page_jobs(jobs: &mut Vec<[u64; 3]>, page_ids: &[u32], packed_start: usize, len: usize, read: bool) {
    assert_eq!(page_ids.len(), len.div_ceil(PAGE_BYTES));
    for (index, &page) in page_ids.iter().enumerate() {
        let packed = (packed_start + index * PAGE_BYTES) as u64;
        let physical = u64::from(page) * PAGE_BYTES as u64;
        let bytes = (len - index * PAGE_BYTES).min(PAGE_BYTES) as u64;
        jobs.push(if read {
            [physical, packed, bytes]
        } else {
            [packed, physical, bytes]
        });
    }
}

struct PLECommitJobs<'a> {
    compute: &'a ple::Compute,
    normalized: &'a Buffer,
    states: &'a Buffer,
    requests: Buffer,
    accepted: Buffer,
    num_tokens: u32,
    num_jobs: u32,
    num_state_slots: u32,
}
impl inference_backend_metal::metal::Operator for PLECommitJobs<'_> {
    fn record(self, recorder: &inference_backend_metal::metal::CommandRecorder<'_>) {
        self.compute
            .commit(
                self.num_tokens,
                self.num_jobs,
                ReplayU32::Fixed(self.num_jobs),
                self.num_state_slots,
                ple::CommitBuffers {
                    normalized: self.normalized,
                    requests: &self.requests,
                    accepted_tokens: &self.accepted,
                    states: self.states,
                },
            )
            .record(recorder);
    }
}
