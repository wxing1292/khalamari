use std::collections::HashSet;

use inference_backend_metal::components::qsa_indexer::CacheLayout;
use inference_backend_metal::components::qsa_indexer::Shape;
use inference_backend_metal::components::qsa_indexer::Token;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
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
    compress_ratio: u32,
    layout: CacheLayout,
    groups_per_page: u32,
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
            compress_ratio: core
                .compress_ratio
                .try_into()
                .expect("QSA indexer compression ratio must fit u32"),
        }
    }

    pub fn layout(&self) -> CacheLayout {
        self.layout
    }

    pub fn tokens(&self) -> &Buffer {
        &self.tokens
    }

    /// Establishes batch addressing and state-version invariants before GPU submission.
    /// The caller separately supplies valid physical pages for each visible logical group.
    pub fn update(&self, requests: &[QSAIndexerRequest], dst_state_slots: &[u32], shape: Shape) -> u32 {
        let layout = self.layout;
        assert!(shape.num_total_tokens > 0 && shape.num_total_groups > 0);
        assert!(shape.num_total_tokens as usize <= self.tokens.len_bytes() / size_of::<Token>());
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
        let mut req_slots = HashSet::new();
        let mut tokens = Vec::with_capacity(num_tokens as usize * size_of::<Token>() / size_of::<u32>());
        let mut flat_token = 0;
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
            assert!(group_end - request.group_start <= shape.num_total_groups);
            assert!((group_end as u64) <= (layout.num_pages_per_request as u64) * (self.groups_per_page as u64));
            let batch_start = flat_token;
            for position in request.token_index..token_end {
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
        }
        self.tokens.write_typed(0, &tokens);
        num_tokens
    }
}
