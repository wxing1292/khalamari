use std::collections::HashSet;

use inference_backend_metal::components::qsa_indexer::CacheLayout;
use inference_backend_metal::components::qsa_indexer::ScoreGeometry;
use inference_backend_metal::components::qsa_indexer::ScoreTaskTemplate;
use inference_backend_metal::components::qsa_indexer::Shape;
use inference_backend_metal::components::qsa_indexer::Token;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::qsa_indexer::core::QSAIndexerCore;

/// One contiguous query segment for one request. Complete groups use logical IDs.
pub struct QSAIndexerRequest {
    pub req_slot: u32,
    pub token_index: u32,
    pub num_tokens: u32,
    pub group_start: u32,
    pub src_state_slot: u32,
}

pub struct QSAIndexerMetadataBuffers {
    tokens: Buffer,
    score_task_templates: Buffer,
    compress_ratio: u32,
    num_q_heads: u32,
    head_dim: u32,
    top_k: u32,
    layout: CacheLayout,
    groups_per_page: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QSAIndexerMetadataActive {
    pub num_active_tokens: u32,
    pub num_active_score_task_templates: u32,
}

impl QSAIndexerMetadataBuffers {
    pub fn new(device: &Device, core: QSAIndexerCore, layout: CacheLayout, max_tokens: u32) -> Self {
        core.validate();
        assert!(max_tokens > 0);
        assert!(layout.num_req_slots > 0 && layout.num_pages_per_request > 0 && layout.num_state_slots > 0);
        let key_bytes = core
            .head_dim
            .checked_mul(2)
            .expect("QSA indexer BF16 key bytes must fit usize");
        assert!(layout.page_bytes as usize >= key_bytes && (layout.page_bytes as usize).is_multiple_of(key_bytes));
        Self {
            layout,
            groups_per_page: (layout.page_bytes as usize / key_bytes) as u32,
            tokens: Buffer::new_zeroed(
                device,
                (max_tokens as usize)
                    .checked_mul(size_of::<Token>())
                    .expect("QSA indexer metadata capacity must fit usize"),
            ),
            score_task_templates: Buffer::new_zeroed(
                device,
                (max_tokens as usize)
                    .checked_mul(size_of::<ScoreTaskTemplate>())
                    .expect("QSA indexer score-task capacity must fit usize"),
            ),
            compress_ratio: core
                .compress_ratio
                .try_into()
                .expect("QSA indexer compression ratio must fit u32"),
            num_q_heads: core
                .num_q_heads
                .try_into()
                .expect("QSA indexer head count must fit u32"),
            head_dim: core
                .head_dim
                .try_into()
                .expect("QSA indexer head dimension must fit u32"),
            top_k: core.top_k.try_into().expect("QSA indexer top-k must fit u32"),
        }
    }

    pub fn layout(&self) -> CacheLayout {
        self.layout
    }

    pub fn tokens(&self) -> &Buffer {
        &self.tokens
    }

    pub fn score_task_templates(&self) -> &Buffer {
        &self.score_task_templates
    }

    /// Establishes batch addressing and state-version invariants before GPU submission.
    /// The caller separately supplies valid physical pages for each visible logical group.
    pub fn update(
        &self,
        requests: &[QSAIndexerRequest],
        dst_state_slots: &[u32],
        shape: Shape,
    ) -> QSAIndexerMetadataActive {
        let layout = self.layout;
        assert!(shape.num_total_tokens > 0 && shape.num_total_groups > 0);
        assert!(
            shape.num_total_score_task_templates > 0 && shape.num_total_score_task_templates <= shape.num_total_tokens
        );
        assert!(shape.num_total_tokens as usize <= self.tokens.len_bytes() / size_of::<Token>());
        assert!(
            shape.num_total_score_task_templates as usize
                <= self.score_task_templates.len_bytes() / size_of::<ScoreTaskTemplate>()
        );
        let num_tokens: u32 = requests
            .iter()
            .try_fold(0_u32, |count, request| count.checked_add(request.num_tokens))
            .expect("QSA indexer token count must fit u32");
        assert!(num_tokens > 0 && num_tokens <= shape.num_total_tokens);
        assert_eq!(dst_state_slots.len(), num_tokens as usize);
        let sources: HashSet<_> = requests.iter().map(|request| request.src_state_slot).collect();
        let mut destinations = HashSet::new();
        for &slot in dst_state_slots {
            assert!(slot < layout.num_state_slots);
            assert!(
                !sources.contains(&slot),
                "QSA indexer candidate state must not overwrite a source state"
            );
            assert!(
                destinations.insert(slot),
                "QSA indexer candidate states require distinct slots"
            );
        }
        let geometry = ScoreGeometry::for_shape(self.num_q_heads, self.head_dim, Dtype::Bfloat16, shape, layout);
        let mut req_slots = HashSet::new();
        let mut tokens = Vec::with_capacity(num_tokens as usize * size_of::<Token>() / size_of::<u32>());
        let mut score_tasks =
            Vec::with_capacity(shape.num_total_score_task_templates as usize * size_of::<ScoreTaskTemplate>() / 4);
        let mut num_sparse_queries = 0;
        let mut flat_token = 0_u32;
        for request in requests {
            assert!(request.num_tokens > 0);
            assert!(request.req_slot < layout.num_req_slots && request.src_state_slot < layout.num_state_slots);
            assert!(
                req_slots.insert(request.req_slot),
                "QSA indexer batch requires one segment per request"
            );
            let token_end = request
                .token_index
                .checked_add(request.num_tokens)
                .expect("QSA indexer token interval must fit u32");
            let group_end = token_end / self.compress_ratio;
            assert!(request.group_start <= request.token_index / self.compress_ratio);
            let last_visible_groups = group_end - request.group_start;
            assert!(last_visible_groups <= shape.num_total_groups);
            assert!((group_end as u64) <= (layout.num_pages_per_request as u64) * (self.groups_per_page as u64));
            let batch_start = flat_token;
            let mut sparse_query_start = None;
            let mut sparse_query_count = 0_u32;
            for position in request.token_index..token_end {
                let visible_groups = (position + 1) / self.compress_ratio - request.group_start;
                if visible_groups > self.top_k {
                    sparse_query_start.get_or_insert(flat_token);
                    sparse_query_count += 1;
                }
                tokens.extend_from_slice(
                    &Token {
                        batch_start,
                        token_index: position,
                        req_slot: request.req_slot,
                        src_state_slot: request.src_state_slot,
                        dst_state_slot: dst_state_slots[flat_token as usize],
                        group_start: request.group_start,
                    }
                    .to_array(),
                );
                flat_token += 1;
            }
            assert!(sparse_query_count <= shape.num_total_request_sparse_queries);
            num_sparse_queries += sparse_query_count;
            if let Some(mut query_start) = sparse_query_start {
                let task_width = geometry.num_queries_per_task(sparse_query_count);
                while sparse_query_count > 0 {
                    let task_queries = sparse_query_count.min(task_width);
                    score_tasks.extend_from_slice(
                        &ScoreTaskTemplate {
                            query_start,
                            num_queries: task_queries,
                        }
                        .to_array(),
                    );
                    query_start += task_queries;
                    sparse_query_count -= task_queries;
                }
            }
        }
        assert!(num_sparse_queries <= shape.num_total_sparse_queries);
        let num_active_score_task_templates = (score_tasks.len() / 2) as u32;
        assert!(num_active_score_task_templates <= shape.num_total_score_task_templates);
        self.tokens.write_typed(0, &tokens);
        self.score_task_templates.write_typed(0, &score_tasks);
        QSAIndexerMetadataActive {
            num_active_tokens: num_tokens,
            num_active_score_task_templates,
        }
    }
}
