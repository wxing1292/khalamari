use inference_backend_metal::components::gqa::sdpa::sliced_kv;
use inference_backend_metal::components::qsa_sdpa;
use inference_backend_metal::components::qsa_sparse_gqa;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_executor_core::attn::GQAPageTableLayout;
use inference_executor_core::backend::recorder::Recorder;

use crate::attn::qsa::batch_metadata::QSASDPAMetadataBuffers;

/// The supported QSA SDPA kernel facts. This config does not include
/// projection, norm, indexer, or gate dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QSASDPAConfig {
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub page_bytes: u32,
    pub page_table_layout: GQAPageTableLayout,
    pub compress_ratio: u32,
    pub top_k: u32,
}

impl QSASDPAConfig {
    pub fn validate(self) {
        assert_eq!(self.num_q_heads, 24, "QSA SDPA currently supports 24 query heads");
        assert_eq!(self.num_kv_heads, 2, "QSA SDPA currently supports 2 KV heads");
        assert_eq!(self.head_dim, 256, "QSA SDPA currently supports head_dim=256");
        assert_eq!(self.compress_ratio, 4, "QSA SDPA currently supports groups of 4 tokens");
        assert_eq!(self.top_k, 512, "QSA SDPA currently supports 512 selected groups");
        assert!(self.scale.is_finite() && self.scale > 0.0);
        self.page_table_layout.validate();
        assert_eq!(self.page_bytes, 32_768, "QSA SDPA currently supports 32 KiB KV pages");
        assert_eq!(self.num_tokens_per_page(), 32);
        self.page_table_token_capacity();
    }

    pub fn num_tokens_per_page(self) -> u32 {
        let kv_bytes_per_token = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .and_then(|bytes| bytes.checked_mul(2))
            .expect("QSA SDPA K/V bytes per token must fit u32");
        assert!(self.page_bytes.is_multiple_of(kv_bytes_per_token));
        self.page_bytes / kv_bytes_per_token
    }

    pub fn page_table_token_capacity(self) -> u32 {
        self.page_table_layout
            .num_blocks
            .checked_mul(self.page_table_layout.num_page_ids_per_block)
            .and_then(|pages| pages.checked_mul(self.num_tokens_per_page()))
            .expect("QSA SDPA page-table token capacity must fit u32")
    }
}

/// Static replay capacity. Sparse capacity is independent of total token
/// capacity because dense-prefix rows do not use SplitKV.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QSASDPAShape {
    pub num_total_tokens: u32,
    pub num_total_sparse_queries: u32,
}

impl QSASDPAShape {
    pub fn validate(self) {
        assert!(self.num_total_tokens > 0);
        assert!(self.num_total_sparse_queries > 0);
        assert!(self.num_total_sparse_queries <= self.num_total_tokens);
    }
}

/// One contiguous active query segment for one request slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QSASDPARequest {
    pub req_slot: u32,
    /// Absolute request position of the first active query.
    pub token_index: u32,
    pub num_tokens: u32,
    /// First logical QSA group visible to the query.
    pub group_start: u32,
}

/// CPU metadata counts produced by one `QSASDPAMetadataBuffers::update`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QSASDPAActive {
    pub num_active_tokens: u32,
    pub num_active_dense_queries: u32,
    pub num_active_dense_tasks: u32,
    pub num_active_sparse_queries: u32,
}

impl QSASDPAActive {
    pub fn validate(self, shape: QSASDPAShape) {
        shape.validate();
        assert!(self.num_active_tokens > 0 && self.num_active_tokens <= shape.num_total_tokens);
        assert!(self.num_active_dense_queries <= self.num_active_tokens);
        assert!(self.num_active_dense_tasks <= shape.num_total_tokens);
        assert!(self.num_active_sparse_queries <= shape.num_total_sparse_queries);
        assert_eq!(
            self.num_active_sparse_queries,
            self.num_active_tokens - self.num_active_dense_queries
        );
    }

    pub fn fixed_replay(self) -> QSASDPAReplay {
        QSASDPAReplay {
            num_active_tokens: ReplayU32::Fixed(self.num_active_tokens),
            num_active_dense_queries: ReplayU32::Fixed(self.num_active_dense_queries),
            num_active_dense_tasks: ReplayU32::Fixed(self.num_active_dense_tasks),
            num_active_sparse_queries: ReplayU32::Fixed(self.num_active_sparse_queries),
        }
    }

    pub fn add_replay_arguments(self, arguments: &mut ReplayArguments) {
        arguments.set_u32(QSA_NUM_ACTIVE_TOKENS, self.num_active_tokens);
        arguments.set_u32(QSA_NUM_ACTIVE_DENSE_QUERIES, self.num_active_dense_queries);
        arguments.set_u32(QSA_NUM_ACTIVE_DENSE_TASKS, self.num_active_dense_tasks);
        arguments.set_u32(QSA_NUM_ACTIVE_SPARSE_QUERIES, self.num_active_sparse_queries);
    }
}

/// Active replay values for a recorded QSA SDPA shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QSASDPAReplay {
    pub num_active_tokens: ReplayU32,
    pub num_active_dense_queries: ReplayU32,
    pub num_active_dense_tasks: ReplayU32,
    pub num_active_sparse_queries: ReplayU32,
}

impl QSASDPAReplay {
    pub fn parameters() -> Self {
        Self {
            num_active_tokens: ReplayU32::Parameter(QSA_NUM_ACTIVE_TOKENS),
            num_active_dense_queries: ReplayU32::Parameter(QSA_NUM_ACTIVE_DENSE_QUERIES),
            num_active_dense_tasks: ReplayU32::Parameter(QSA_NUM_ACTIVE_DENSE_TASKS),
            num_active_sparse_queries: ReplayU32::Parameter(QSA_NUM_ACTIVE_SPARSE_QUERIES),
        }
    }
}

pub const QSA_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("qsa.num_active_tokens");
pub const QSA_NUM_ACTIVE_DENSE_QUERIES: ReplayParameterKey = ReplayParameterKey::new("qsa.num_active_dense_queries");
pub const QSA_NUM_ACTIVE_DENSE_TASKS: ReplayParameterKey = ReplayParameterKey::new("qsa.num_active_dense_tasks");
pub const QSA_NUM_ACTIVE_SPARSE_QUERIES: ReplayParameterKey = ReplayParameterKey::new("qsa.num_active_sparse_queries");

/// Buffers owned by one recorded QSA SDPA shape.
pub struct QSASDPAScratch {
    shape: QSASDPAShape,
    dense: sliced_kv::Scratch,
    sparse_partials: Option<sliced_kv::Scratch>,
}

#[derive(Clone, Copy)]
pub struct QSASDPAInput<'a> {
    pub q: &'a Buffer,
    pub kv_pages: &'a Buffer,
    pub page_ids: &'a Buffer,
    pub selected_group_ids: &'a Buffer,
    pub output: &'a Buffer,
    pub metadata: &'a QSASDPAMetadataBuffers,
    pub scratch: &'a QSASDPAScratch,
    pub page_table_index: ReplayU32,
}

/// Unified dense-prefix and sparse-suffix SDPA for prepared QSA inputs.
pub struct QSASDPA {
    device: Device,
    config: QSASDPAConfig,
}

impl QSASDPA {
    pub fn new(device: &Device, config: QSASDPAConfig) -> Self {
        config.validate();
        Self {
            device: device.clone(),
            config,
        }
    }

    pub fn page_table_layout(&self) -> GQAPageTableLayout {
        self.config.page_table_layout
    }

    pub fn compress_ratio(&self) -> u32 {
        self.config.compress_ratio
    }

    pub fn top_k(&self) -> u32 {
        self.config.top_k
    }

    pub fn new_scratch(&self, shape: QSASDPAShape) -> QSASDPAScratch {
        shape.validate();
        let dense_shape = sliced_kv::Shape {
            num_total_tokens: shape.num_total_tokens,
            num_total_sdpa_map_task_templates: shape.num_total_tokens,
        };
        let sparse_splits = qsa_sparse_gqa::Compute::selected_split_count(shape.num_total_sparse_queries);
        let sparse_shape = sliced_kv::Shape {
            num_total_tokens: shape.num_total_sparse_queries,
            num_total_sdpa_map_task_templates: shape
                .num_total_sparse_queries
                .checked_mul(sparse_splits)
                .expect("QSA sparse task capacity must fit u32"),
        };
        let sparse_partials = (sparse_splits > 1).then(|| {
            let mut sparse_partial_config = dense_sliced_kv_config(self.config);
            sparse_partial_config.dtype = Dtype::Float32;
            sliced_kv::Scratch::new(&self.device, sparse_partial_config, sparse_shape)
        });
        QSASDPAScratch {
            shape,
            dense: sliced_kv::Scratch::new(&self.device, dense_sliced_kv_config(self.config), dense_shape),
            sparse_partials,
        }
    }

    pub fn replay_sparse_query_topology_boundaries(&self) -> Box<[u32]> {
        qsa_sparse_gqa::Compute::split_topology_boundaries()
    }

    pub fn record<'a, R>(&self, recorder: &mut R, input: QSASDPAInput<'a>, active: QSASDPAReplay)
    where
        R: Recorder<'a, Operator = crate::def::replay_op::ReplayOp<'a>>,
    {
        let shape = input.metadata.shape();
        assert_eq!(input.metadata.config(), self.config);
        assert_eq!(input.scratch.shape, shape);
        let sparse_split_count = input.metadata.sparse_split_count();
        let metadata_active = input.metadata.active();
        validate_replay(active, metadata_active);

        let dense_shape = sliced_kv::Shape {
            num_total_tokens: shape.num_total_tokens,
            num_total_sdpa_map_task_templates: shape.num_total_tokens,
        };
        let dense_execution = input.metadata.dense_variant();
        let dense_config = dense_sliced_kv_config(self.config);
        let dense_map = sliced_kv::Compute::new(&self.device, dense_config, dense_execution, dense_shape);
        let dense_reduce = sliced_kv::MappedReduce::new(
            &self.device,
            dense_config,
            dense_execution,
            dense_shape,
            Dtype::Bfloat16,
        );
        let sparse_shape = qsa_sparse_gqa::Shape {
            num_total_tokens: shape.num_total_tokens,
            num_total_sparse_queries: shape.num_total_sparse_queries,
        };
        let sparse_map = qsa_sparse_gqa::Compute::new(
            &self.device,
            sparse_config(self.config),
            sparse_shape,
            sparse_split_count,
        );
        let sparse_reduce = if sparse_split_count > 1 {
            let sparse_reduce_shape = sliced_kv::Shape {
                num_total_tokens: shape.num_total_sparse_queries,
                num_total_sdpa_map_task_templates: shape.num_total_sparse_queries * sparse_split_count,
            };
            Some(sliced_kv::MappedReduce::new(
                &self.device,
                dense_config,
                dense_execution,
                sparse_reduce_shape,
                Dtype::Float32,
            ))
        } else {
            None
        };
        let dense_map = dense_map.invoke_map(
            input.scratch.dense.map_buffers(
                input.q,
                input.kv_pages,
                input.metadata.req_slots(),
                input.page_ids,
                input.metadata.dense_task_templates(),
            ),
            input.page_table_index,
            active.num_active_tokens,
            active.num_active_dense_tasks,
        );
        let sparse_destination = if sparse_split_count == 1 {
            qsa_sparse_gqa::MapDestination::Direct(input.output)
        } else {
            let sparse_partials = input
                .scratch
                .sparse_partials
                .as_ref()
                .expect("QSA sparse split map requires partial scratch");
            qsa_sparse_gqa::MapDestination::Partials {
                exp_sums: &sparse_partials.partial_exp_sums,
                max_logits: &sparse_partials.partial_max_logits,
                output: &sparse_partials.partial_output,
            }
        };
        let sparse_map = sparse_map.map(
            qsa_sparse_gqa::MapBuffers {
                q: input.q,
                pages: input.kv_pages,
                req_slots: input.metadata.req_slots(),
                page_ids: input.page_ids,
                selected_groups: input.selected_group_ids,
                query_positions: input.metadata.query_positions(),
                sparse_query_rows: input.metadata.sparse_query_rows(),
                destination: sparse_destination,
            },
            active.num_active_sparse_queries,
            input.page_table_index,
        );
        let dense_reduce = dense_reduce.invoke(
            sliced_kv::MappedReduceBuffers {
                partial_exp_sums: &input.scratch.dense.partial_exp_sums,
                partial_max_logits: &input.scratch.dense.partial_max_logits,
                partial_output: &input.scratch.dense.partial_output,
                cu_sdpa_partial_outputs: input.metadata.dense_cu_partial_outputs(),
                output_query_indices: input.metadata.dense_query_rows(),
                output: input.output,
            },
            active.num_active_dense_queries,
        );
        let sparse_reduce = sparse_reduce.map(|sparse_reduce| {
            let sparse_partials = input
                .scratch
                .sparse_partials
                .as_ref()
                .expect("QSA sparse split reduce requires partial scratch");
            sparse_reduce.invoke(
                sliced_kv::MappedReduceBuffers {
                    partial_exp_sums: &sparse_partials.partial_exp_sums,
                    partial_max_logits: &sparse_partials.partial_max_logits,
                    partial_output: &sparse_partials.partial_output,
                    cu_sdpa_partial_outputs: input.metadata.sparse_cu_partial_outputs(),
                    output_query_indices: input.metadata.sparse_query_rows(),
                    output: input.output,
                },
                active.num_active_sparse_queries,
            )
        });
        recorder.record(crate::def::replay_op::ReplayOp::opaque(qsa_sdpa::Invocation::new(
            input.output,
            dense_map,
            sparse_map,
            dense_reduce,
            sparse_reduce,
        )));
    }
}

fn validate_replay(active: QSASDPAReplay, expected: QSASDPAActive) {
    assert_replay_fixed(active.num_active_tokens, expected.num_active_tokens);
    assert_replay_fixed(active.num_active_dense_queries, expected.num_active_dense_queries);
    assert_replay_fixed(active.num_active_dense_tasks, expected.num_active_dense_tasks);
    assert_replay_fixed(active.num_active_sparse_queries, expected.num_active_sparse_queries);
}

fn assert_replay_fixed(value: ReplayU32, expected: u32) {
    if let ReplayU32::Fixed(value) = value {
        assert_eq!(value, expected, "fixed QSA SDPA replay count must match metadata");
    }
}

fn dense_sliced_kv_config(config: QSASDPAConfig) -> sliced_kv::Config {
    sliced_kv::Config {
        num_q_heads: config.num_q_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        scale: config.scale,
        page_bytes: config.page_bytes,
        page_table_layout: inference_backend_metal::components::gqa::kv_page_write::PageTableLayout {
            num_req_slots: config.page_table_layout.num_req_slots,
            num_gqa_layers: config.page_table_layout.num_gqa_layers,
            num_blocks: config.page_table_layout.num_blocks,
            num_page_ids_per_block: config.page_table_layout.num_page_ids_per_block,
        },
        dtype: Dtype::Bfloat16,
    }
}

fn sparse_config(config: QSASDPAConfig) -> qsa_sparse_gqa::Config {
    qsa_sparse_gqa::Config {
        num_q_heads: config.num_q_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        scale: config.scale,
        page_bytes: config.page_bytes,
        page_table_layout: inference_backend_metal::components::gqa::kv_page_write::PageTableLayout {
            num_req_slots: config.page_table_layout.num_req_slots,
            num_gqa_layers: config.page_table_layout.num_gqa_layers,
            num_blocks: config.page_table_layout.num_blocks,
            num_page_ids_per_block: config.page_table_layout.num_page_ids_per_block,
        },
        compress_ratio: config.compress_ratio,
        top_k: config.top_k,
    }
}
