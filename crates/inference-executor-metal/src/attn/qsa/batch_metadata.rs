use std::cell::Cell;
use std::collections::HashSet;

use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;

use crate::attn::gqa::batch_metadata::GQAReplayBucketPolicy;
use crate::attn::gqa::sdpa::RequestShape;
use crate::attn::gqa::sdpa::Selection;
use crate::attn::gqa::sdpa::Selector;
use crate::attn::qsa::sdpa::QSASDPAActive;
use crate::attn::qsa::sdpa::QSASDPAConfig;
use crate::attn::qsa::sdpa::QSASDPARequest;
use crate::attn::qsa::sdpa::QSASDPAShape;

const TASK_TEMPLATE_FIELDS: usize = 3;

/// Capacity-sized metadata for one recorded QSA SDPA shape.
///
/// Row maps retain original flattened query indices. Dense task ranges use
/// request-local K/V positions and include the request's `group_start` lower
/// bound. Sparse selected IDs stay indexed by original query row.
pub struct QSASDPAMetadataBuffers {
    config: QSASDPAConfig,
    capacity: QSASDPAShape,
    shape: Cell<QSASDPAShape>,
    req_slots: Buffer,
    query_positions: Buffer,
    dense_query_rows: Buffer,
    dense_task_templates: Buffer,
    dense_cu_partial_outputs: Buffer,
    sparse_query_rows: Buffer,
    sparse_cu_partial_outputs: Buffer,
    active: Cell<Option<QSASDPAActive>>,
    dense_variant: backend_sdpa::ExecutionVariant,
    dense_selector: Selector,
    dense_policy: GQAReplayBucketPolicy,
    sparse_split_count: Cell<u32>,
}

impl QSASDPAMetadataBuffers {
    pub fn new(device: &Device, config: QSASDPAConfig, shape: QSASDPAShape) -> Self {
        config.validate();
        shape.validate();
        let dense_map_tasks = shape.num_total_tokens as usize;
        let sparse_split_count = inference_backend_metal::components::qsa_sparse_gqa::Compute::selected_split_count(
            shape.num_total_sparse_queries,
        );
        let registry = backend_sdpa::Registry::new(backend_config(config));
        let dense_variant = registry
            .variants()
            .iter()
            .copied()
            .find(|variant| variant.map.kind == backend_sdpa::MapKernelKind::SlicedKv)
            .expect("QSA SDPA requires the supported SlicedKv dense variant");
        assert_eq!(
            registry.variants().len(),
            1,
            "QSA SDPA's supported geometry must select only SlicedKv"
        );
        shape
            .num_total_sparse_queries
            .checked_mul(sparse_split_count)
            .expect("QSA sparse partial-output offset must fit u32");
        let sparse_offsets = (0..=shape.num_total_sparse_queries)
            .map(|query| query * sparse_split_count)
            .collect::<Vec<_>>();
        let buffers = Self {
            config,
            capacity: shape,
            shape: Cell::new(shape),
            req_slots: Buffer::new_zeroed_elements(device, shape.num_total_tokens as usize, Dtype::Uint32),
            query_positions: Buffer::new_zeroed_elements(device, shape.num_total_tokens as usize, Dtype::Uint32),
            dense_query_rows: Buffer::new_zeroed_elements(device, shape.num_total_tokens as usize, Dtype::Uint32),
            dense_task_templates: Buffer::new_zeroed_elements(
                device,
                dense_map_tasks
                    .checked_mul(TASK_TEMPLATE_FIELDS)
                    .expect("QSA dense task-template capacity must fit usize"),
                Dtype::Uint32,
            ),
            dense_cu_partial_outputs: Buffer::new_zeroed_elements(
                device,
                shape.num_total_tokens as usize + 1,
                Dtype::Uint32,
            ),
            sparse_query_rows: Buffer::new_zeroed_elements(
                device,
                shape.num_total_sparse_queries as usize,
                Dtype::Uint32,
            ),
            sparse_cu_partial_outputs: Buffer::new_zeroed_elements(
                device,
                shape.num_total_sparse_queries as usize + 1,
                Dtype::Uint32,
            ),
            active: Cell::new(None),
            dense_variant,
            dense_selector: Selector::new(
                registry,
                shape.num_total_tokens as usize,
                dense_map_tasks,
                dense_map_tasks,
            ),
            dense_policy: GQAReplayBucketPolicy::new(shape.num_total_tokens, shape.num_total_tokens),
            sparse_split_count: Cell::new(sparse_split_count),
        };
        buffers.sparse_cu_partial_outputs.write_typed(0, &sparse_offsets);
        buffers
    }

    pub fn shape(&self) -> QSASDPAShape {
        self.shape.get()
    }

    pub fn config(&self) -> QSASDPAConfig {
        self.config
    }

    pub fn sparse_split_count(&self) -> u32 {
        self.sparse_split_count.get()
    }

    pub fn req_slots(&self) -> &Buffer {
        &self.req_slots
    }

    pub fn query_positions(&self) -> &Buffer {
        &self.query_positions
    }

    pub fn dense_query_rows(&self) -> &Buffer {
        &self.dense_query_rows
    }

    pub fn dense_task_templates(&self) -> &Buffer {
        &self.dense_task_templates
    }

    pub fn dense_cu_partial_outputs(&self) -> &Buffer {
        &self.dense_cu_partial_outputs
    }

    pub fn sparse_query_rows(&self) -> &Buffer {
        &self.sparse_query_rows
    }

    pub fn sparse_cu_partial_outputs(&self) -> &Buffer {
        &self.sparse_cu_partial_outputs
    }

    pub fn active(&self) -> QSASDPAActive {
        self.active
            .get()
            .expect("QSA SDPA metadata must be updated before recording")
    }

    pub fn dense_variant(&self) -> backend_sdpa::ExecutionVariant {
        self.dense_variant
    }

    /// Rebuilds active row partitions and the existing dense GQA selection.
    pub fn update(&self, requests: &[QSASDPARequest]) -> QSASDPAActive {
        self.update_with_shape(requests, self.shape())
    }

    /// Reuses initialized buffers for a smaller record-time shape.
    pub fn update_with_shape(&self, requests: &[QSASDPARequest], shape: QSASDPAShape) -> QSASDPAActive {
        shape.validate();
        assert!(shape.num_total_tokens <= self.capacity.num_total_tokens);
        assert!(shape.num_total_sparse_queries <= self.capacity.num_total_sparse_queries);
        self.shape.set(shape);
        let splits = inference_backend_metal::components::qsa_sparse_gqa::Compute::selected_split_count(
            shape.num_total_sparse_queries,
        );
        self.sparse_split_count.set(splits);
        let offsets: Vec<u32> = (0..=shape.num_total_sparse_queries).map(|q| q * splits).collect();
        self.sparse_cu_partial_outputs.write_typed(0, &offsets);
        let plan = build_plan(
            self.config,
            shape,
            &self.dense_selector,
            &self.dense_policy,
            self.dense_variant,
            requests,
        );
        self.req_slots.write_typed(0, &plan.req_slots);
        self.query_positions.write_typed(0, &plan.query_positions);
        self.dense_query_rows.write_typed(0, &plan.dense_query_rows);
        self.dense_task_templates.write_typed(0, &plan.dense_task_templates);
        self.dense_cu_partial_outputs
            .write_typed(0, &plan.dense_cu_partial_outputs);
        self.sparse_query_rows.write_typed(0, &plan.sparse_query_rows);
        self.active.set(Some(plan.active));
        plan.active
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Plan {
    active: QSASDPAActive,
    req_slots: Vec<u32>,
    query_positions: Vec<u32>,
    dense_query_rows: Vec<u32>,
    dense_task_templates: Vec<u32>,
    dense_cu_partial_outputs: Vec<u32>,
    sparse_query_rows: Vec<u32>,
}

fn build_plan(
    config: QSASDPAConfig,
    shape: QSASDPAShape,
    dense_selector: &Selector,
    dense_policy: &GQAReplayBucketPolicy,
    dense_variant: backend_sdpa::ExecutionVariant,
    requests: &[QSASDPARequest],
) -> Plan {
    assert!(!requests.is_empty(), "QSA SDPA requires active requests");
    let kv_capacity = config.page_table_token_capacity();

    let mut req_slots = Vec::with_capacity(shape.num_total_tokens as usize);
    let mut query_positions = Vec::with_capacity(shape.num_total_tokens as usize);
    let mut dense_query_rows = Vec::with_capacity(shape.num_total_tokens as usize);
    let mut sparse_query_rows = Vec::with_capacity(shape.num_total_sparse_queries as usize);
    let mut dense_bases = Vec::with_capacity(shape.num_total_tokens as usize);
    let mut dense_request_shapes = Vec::with_capacity(requests.len());
    let mut seen_slots = HashSet::with_capacity(requests.len());
    let mut flat_begin = 0_u32;

    for request in requests {
        assert!(request.num_tokens > 0, "QSA SDPA requests must contain query tokens");
        assert!(request.req_slot < config.page_table_layout.num_req_slots);
        assert!(
            seen_slots.insert(request.req_slot),
            "QSA SDPA requires one segment per request slot"
        );
        let flat_end = flat_begin
            .checked_add(request.num_tokens)
            .expect("QSA SDPA active token count must fit u32");
        assert!(flat_end <= shape.num_total_tokens);
        let token_end = request
            .token_index
            .checked_add(request.num_tokens)
            .expect("QSA SDPA request token interval must fit u32");
        assert!(
            token_end <= kv_capacity,
            "QSA SDPA query exceeds the page-table capacity"
        );
        let group_start_token = request
            .group_start
            .checked_mul(config.compress_ratio)
            .expect("QSA SDPA group-start token must fit u32");
        assert!(group_start_token <= request.token_index / config.compress_ratio * config.compress_ratio);

        let mut dense_count = 0_u32;
        for local_query in 0..request.num_tokens {
            let flat_query = flat_begin + local_query;
            let position = request.token_index + local_query;
            let visible_complete_groups = (position + 1) / config.compress_ratio - request.group_start;
            req_slots.push(request.req_slot);
            query_positions.push(position);
            if visible_complete_groups <= config.top_k {
                dense_query_rows.push(flat_query);
                dense_bases.push(group_start_token);
                dense_count += 1;
            } else {
                sparse_query_rows.push(flat_query);
            }
        }
        if dense_count > 0 {
            dense_request_shapes.push(RequestShape {
                num_history_tokens: request.token_index - group_start_token,
                num_q_tokens: dense_count,
            });
        }
        flat_begin = flat_end;
    }

    let num_tokens = flat_begin;
    let num_dense_queries = dense_query_rows.len() as u32;
    let num_sparse_queries = sparse_query_rows.len() as u32;
    assert!(num_sparse_queries <= shape.num_total_sparse_queries);

    let (dense_tasks, dense_cumulative) = if num_dense_queries == 0 {
        (Vec::new(), vec![0])
    } else {
        let selection = dense_selector.select(&dense_request_shapes, dense_policy, shape.num_total_tokens);
        assert_eq!(selection.variant(), dense_variant);
        let (tasks, cumulative) = materialize_dense_selection(&selection, &dense_query_rows, &dense_bases);
        (tasks, cumulative)
    };

    let num_dense_tasks = (dense_tasks.len() / TASK_TEMPLATE_FIELDS) as u32;

    let active = QSASDPAActive {
        num_active_tokens: num_tokens,
        num_active_dense_queries: num_dense_queries,
        num_active_dense_tasks: num_dense_tasks,
        num_active_sparse_queries: num_sparse_queries,
    };
    Plan {
        active,
        req_slots,
        query_positions,
        dense_query_rows,
        dense_task_templates: dense_tasks,
        dense_cu_partial_outputs: dense_cumulative,
        sparse_query_rows,
    }
}

fn backend_config(config: QSASDPAConfig) -> backend_sdpa::Config {
    backend_sdpa::Config {
        io_dtype: Dtype::Bfloat16,
        num_q_heads: config.num_q_heads,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        tokens_per_page: config.num_tokens_per_page(),
    }
}

fn materialize_dense_selection(
    selection: &Selection,
    dense_query_rows: &[u32],
    dense_bases: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut tasks = Vec::with_capacity(selection.map_task_templates().len() * TASK_TEMPLATE_FIELDS);
    for template in selection.map_task_templates() {
        let compact_query = template.q_token_range_index as usize;
        let query_row = dense_query_rows[compact_query];
        let base = dense_bases[compact_query];
        tasks.extend_from_slice(&[
            query_row,
            base + template.request_local_kv_token_indices.start,
            base + template.request_local_kv_token_indices.end,
        ]);
    }
    (tasks, selection.cu_partial_outputs_by_q_token_range().to_vec())
}
