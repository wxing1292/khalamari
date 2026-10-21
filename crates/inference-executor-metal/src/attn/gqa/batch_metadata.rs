use std::cell::Cell;

use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::replay::ReplayBucketPolicy;

use super::sdpa::Selection;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GQAReplayShape {
    /// Current valid flattened Q-token count (`Tq`).
    pub num_tokens: u32,
    /// Recorded flattened Q-token capacity.
    pub num_total_tokens: u32,
    /// Current valid request-local Q-token-tile count along `Tq`.
    pub num_q_token_tiles: u32,
    /// Recorded request-local Q-token-tile capacity.
    pub num_total_q_token_tiles: u32,
    /// Current valid `SDPAMapTaskTemplate` count.
    pub num_sdpa_map_task_templates: u32,
    /// Recorded `SDPAMapTaskTemplate` capacity.
    pub num_total_sdpa_map_task_templates: u32,
    /// Whether the unpadded batch plan semantically has partial outputs to merge.
    pub reduce_sdpa_partial_outputs: bool,
}

impl GQAReplayShape {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_tokens: u32,
        num_total_tokens: u32,
        num_q_token_tiles: u32,
        num_total_q_token_tiles: u32,
        num_sdpa_map_task_templates: u32,
        num_total_sdpa_map_task_templates: u32,
        reduce_sdpa_partial_outputs: bool,
    ) -> Self {
        let shape = Self {
            num_tokens,
            num_total_tokens,
            num_q_token_tiles,
            num_total_q_token_tiles,
            num_sdpa_map_task_templates,
            num_total_sdpa_map_task_templates,
            reduce_sdpa_partial_outputs,
        };
        shape.validate();
        shape
    }

    pub fn validate(self) {
        assert!(self.num_tokens > 0);
        assert!(self.num_tokens <= self.num_total_tokens);
        assert!(self.num_q_token_tiles > 0 && self.num_q_token_tiles <= self.num_tokens);
        assert!(self.num_q_token_tiles <= self.num_total_q_token_tiles);
        assert!(self.num_sdpa_map_task_templates >= self.num_q_token_tiles);
        assert!(self.num_sdpa_map_task_templates <= self.num_total_sdpa_map_task_templates);
    }
}

const NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS: usize = 3;
const NUM_Q_TOKEN_RANGE_FIELDS: usize = 2;
const NUM_VISIBLE_KV_TOKEN_RANGE_FIELDS: usize = 2;

/// Buckets SDPA metadata dimensions within a model-selected token capacity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GQAReplayBucketPolicy {
    q_token_ranges: ReplayBucketPolicy,
    map_task_templates: ReplayBucketPolicy,
}

impl GQAReplayBucketPolicy {
    pub fn new(max_tokens: u32, max_map_task_templates: u32) -> Self {
        Self {
            q_token_ranges: ReplayBucketPolicy::new(max_tokens),
            map_task_templates: ReplayBucketPolicy::new(max_map_task_templates),
        }
    }

    pub fn max_tokens(&self) -> u32 {
        self.q_token_ranges.max_capacity()
    }

    pub fn max_map_task_templates(&self) -> u32 {
        self.map_task_templates.max_capacity()
    }

    pub fn capacity(&self, num_active_q_token_ranges: u32, num_active_map_task_templates: u32) -> (u32, u32) {
        (
            self.q_token_ranges.capacity(num_active_q_token_ranges),
            self.map_task_templates.capacity(num_active_map_task_templates),
        )
    }
}

/// Capacity-sized GPU metadata refreshed from one complete executor-owned SDPA
/// selection. This type does not select an execution variant or divide KV work.
pub struct GQAMetadataBuffers {
    req_slots: Buffer,
    // Indexed by flat token order. Each value is request-absolute.
    flat_token_indices: Buffer,
    // Materialized ABI: one request-local [visible_kv_token_begin, visible_kv_token_end) per flat Q token.
    visible_kv_token_ranges: Buffer,
    // Materialized ABI: [flat_q_token_begin, flat_q_token_end].
    q_token_ranges: Buffer,
    // Materialized ABI: [q_token_range_index, request_local_kv_token_begin, request_local_kv_token_end].
    sdpa_map_task_templates: Buffer,
    cu_sdpa_partial_outputs: Buffer,
    replay_shape: Cell<Option<GQAReplayShape>>,
    variant: Cell<Option<backend_sdpa::ExecutionVariant>>,
}

impl GQAMetadataBuffers {
    pub fn new(device: &Device, max_tokens: usize, max_map_task_templates: usize) -> Self {
        assert!(max_tokens > 0, "GQA batch metadata requires tokens");
        assert!(u32::try_from(max_tokens).is_ok(), "GQA token capacity must fit u32");
        assert!(
            max_map_task_templates > 0,
            "GQA batch metadata requires Map task templates"
        );
        assert!(
            u32::try_from(max_map_task_templates).is_ok(),
            "GQA Map task-template capacity must fit u32"
        );
        Self {
            req_slots: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            flat_token_indices: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            visible_kv_token_ranges: Buffer::new_zeroed_elements(
                device,
                max_tokens
                    .checked_mul(NUM_VISIBLE_KV_TOKEN_RANGE_FIELDS)
                    .expect("GQA visible K/V-token-range metadata capacity must fit usize"),
                Dtype::Uint32,
            ),
            q_token_ranges: Buffer::new_zeroed_elements(
                device,
                max_tokens
                    .checked_mul(NUM_Q_TOKEN_RANGE_FIELDS)
                    .expect("GQA token-range metadata capacity must fit usize"),
                Dtype::Uint32,
            ),
            sdpa_map_task_templates: Buffer::new_zeroed_elements(
                device,
                max_map_task_templates
                    .checked_mul(NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                    .expect("GQA SDPA Map task-template metadata capacity must fit usize"),
                Dtype::Uint32,
            ),
            cu_sdpa_partial_outputs: Buffer::new_zeroed_elements(
                device,
                max_tokens
                    .checked_add(1)
                    .expect("GQA SDPA partial-output cumulative-count capacity must fit usize"),
                Dtype::Uint32,
            ),
            replay_shape: Cell::new(None),
            variant: Cell::new(None),
        }
    }

    pub fn max_tokens(&self) -> usize {
        self.req_slots.len_bytes() / size_of::<u32>()
    }

    pub fn max_map_task_templates(&self) -> usize {
        self.sdpa_map_task_templates.len_bytes() / (NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS * size_of::<u32>())
    }

    pub fn update(
        &self,
        req_slots: &[u32],
        token_indices: &[u32],
        cu_tokens: &[u32],
        selection: &Selection,
    ) -> GQAReplayShape {
        assert_eq!(req_slots.len(), token_indices.len());
        assert_eq!(cu_tokens.len(), req_slots.len() + 1);
        assert_eq!(cu_tokens.first().copied(), Some(0));
        let replay_shape = selection.replay_shape();
        assert_eq!(cu_tokens.last().copied(), Some(replay_shape.num_tokens));
        assert!(replay_shape.num_total_tokens as usize <= self.max_tokens());
        assert!(replay_shape.num_total_q_token_tiles as usize <= self.max_tokens());
        assert!(replay_shape.num_total_sdpa_map_task_templates as usize <= self.max_map_task_templates());

        let mut req_slots_by_token = Vec::with_capacity(replay_shape.num_tokens as usize);
        let mut flat_token_indices = Vec::with_capacity(replay_shape.num_tokens as usize);
        let mut visible_kv_token_ranges =
            Vec::with_capacity(replay_shape.num_tokens as usize * NUM_VISIBLE_KV_TOKEN_RANGE_FIELDS);
        for request_index in 0..req_slots.len() {
            let flat_q_token_begin = cu_tokens[request_index];
            let flat_q_token_end = cu_tokens[request_index + 1];
            assert!(flat_q_token_begin <= flat_q_token_end);
            token_indices[request_index]
                .checked_add(flat_q_token_end - flat_q_token_begin)
                .expect("GQA request context length must fit u32");
            for q_token_index_in_request in 0..flat_q_token_end - flat_q_token_begin {
                let flat_token_index = token_indices[request_index] + q_token_index_in_request;
                req_slots_by_token.push(req_slots[request_index]);
                flat_token_indices.push(flat_token_index);
                visible_kv_token_ranges.extend_from_slice(&[0, flat_token_index + 1]);
            }
        }
        assert_eq!(req_slots_by_token.len(), replay_shape.num_tokens as usize);

        let q_token_ranges = selection
            .q_token_ranges()
            .iter()
            .flat_map(|range| [range.flat_q_token_indices.start, range.flat_q_token_indices.end])
            .collect::<Vec<_>>();
        assert_eq!(
            q_token_ranges.len() / NUM_Q_TOKEN_RANGE_FIELDS,
            replay_shape.num_q_token_tiles as usize
        );

        let mut sdpa_map_task_templates = selection
            .map_task_templates()
            .iter()
            .flat_map(|template| {
                [
                    template.q_token_range_index,
                    template.request_local_kv_token_indices.start,
                    template.request_local_kv_token_indices.end,
                ]
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS,
            replay_shape.num_sdpa_map_task_templates as usize
        );
        sdpa_map_task_templates.resize(
            replay_shape.num_total_sdpa_map_task_templates as usize * NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS,
            u32::MAX,
        );
        assert_eq!(
            selection.cu_partial_outputs_by_q_token_range().len(),
            replay_shape.num_q_token_tiles as usize + 1
        );

        self.req_slots.write_typed(0, &req_slots_by_token);
        self.flat_token_indices.write_typed(0, &flat_token_indices);
        self.visible_kv_token_ranges.write_typed(0, &visible_kv_token_ranges);
        if selection.variant().map.kind != backend_sdpa::MapKernelKind::SlicedKv {
            self.q_token_ranges.write_typed(0, &q_token_ranges);
        }
        self.sdpa_map_task_templates.write_typed(0, &sdpa_map_task_templates);
        self.cu_sdpa_partial_outputs
            .write_typed(0, selection.cu_partial_outputs_by_q_token_range());
        self.replay_shape.set(Some(replay_shape));
        self.variant.set(Some(selection.variant()));
        replay_shape
    }

    pub fn req_slots(&self) -> &Buffer {
        &self.req_slots
    }

    pub fn flat_token_indices(&self) -> &Buffer {
        &self.flat_token_indices
    }

    pub fn visible_kv_token_ranges(&self) -> &Buffer {
        &self.visible_kv_token_ranges
    }

    pub fn q_token_ranges(&self) -> &Buffer {
        &self.q_token_ranges
    }

    pub fn sdpa_map_task_templates(&self) -> &Buffer {
        &self.sdpa_map_task_templates
    }

    pub fn cu_sdpa_partial_outputs(&self) -> &Buffer {
        &self.cu_sdpa_partial_outputs
    }

    pub fn replay_shape(&self) -> GQAReplayShape {
        self.replay_shape
            .get()
            .expect("GQA batch metadata must be updated before recording")
    }

    pub fn direct_output(&self) -> bool {
        self.variant().map.kind != backend_sdpa::MapKernelKind::SlicedKv
            && !self.replay_shape().reduce_sdpa_partial_outputs
    }

    pub fn variant(&self) -> backend_sdpa::ExecutionVariant {
        self.variant
            .get()
            .expect("GQA batch metadata must be updated before recording")
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
    use inference_backend_metal::metal::Device;
    use inference_backend_metal::metal::Dtype;

    use super::GQAMetadataBuffers;
    use super::GQAReplayBucketPolicy;
    use crate::attn::gqa::batch_metadata::GQAReplayShape;
    use crate::attn::gqa::sdpa::RequestShape;
    use crate::attn::gqa::sdpa::Selector;

    fn config() -> backend_sdpa::Config {
        backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 2,
            num_kv_heads: 2,
            head_dim: 128,
            tokens_per_page: 8,
        }
    }

    fn selector(max_tokens: usize, variant: backend_sdpa::ExecutionVariant) -> Selector {
        Selector::new(
            backend_sdpa::Registry::from_variants(config(), vec![variant]),
            max_tokens,
            max_tokens,
            max_tokens * variant.map.thread_block.max_q_tokens as usize,
        )
    }

    #[test]
    fn test_map_metadata_capacity_is_independent_of_token_capacity() {
        let device = Device::system_default();
        let metadata = GQAMetadataBuffers::new(&device, 4, 32);
        let variant = backend_sdpa::ExecutionVariant::sliced_kv(config(), 64, 128, 1);
        let selector = Selector::new(
            backend_sdpa::Registry::from_variants(config(), vec![variant]),
            4,
            32,
            32,
        );
        let policy = GQAReplayBucketPolicy::new(4, 32);
        let selection = selector.select(&RequestShape::from_batch(&[1088], &[0, 1]), &policy, 2);
        let shape = metadata.update(&[1], &[1088], &[0, 1], &selection);
        assert_eq!(metadata.max_tokens(), 4);
        assert_eq!(metadata.max_map_task_templates(), 32);
        assert_eq!(shape.num_tokens, 1);
        assert_eq!(shape.num_total_tokens, 2);
        assert_eq!(shape.num_sdpa_map_task_templates, 18);
        assert_eq!(shape.num_total_sdpa_map_task_templates, 20);
        assert_eq!(metadata.req_slots().len_bytes(), 4 * size_of::<u32>());
        assert_eq!(
            metadata.sdpa_map_task_templates().len_bytes(),
            32 * 3 * size_of::<u32>()
        );
        assert_eq!(metadata.flat_token_indices().read_typed::<u32>(0, 1), vec![1088]);
        assert_eq!(
            metadata.visible_kv_token_ranges().read_typed::<u32>(0, 2),
            vec![0, 1089]
        );
        assert_eq!(
            metadata.sdpa_map_task_templates().read_typed::<u32>(18 * 3, 6),
            vec![u32::MAX; 6]
        );
        assert_eq!(metadata.cu_sdpa_partial_outputs().read_typed::<u32>(0, 2), vec![0, 18]);
    }

    #[test]
    fn test_metadata_upload_preserves_current_abi_for_complete_selections() {
        let device = Device::system_default();
        let metadata = GQAMetadataBuffers::new(&device, 8, 8);
        let shapes = RequestShape::from_batch(&[7, 20], &[0, 2, 5]);
        let policy = GQAReplayBucketPolicy::new(8, 8);
        let single = backend_sdpa::ExecutionVariant::sliced_kv(config(), 8, 32, 1);
        let single_selection = selector(8, single).select(&shapes, &policy, 5);

        let single_shape = metadata.update(&[2, 5], &[7, 20], &[0, 2, 5], &single_selection);
        assert_eq!(metadata.req_slots().read_typed::<u32>(0, 5), vec![2, 2, 5, 5, 5]);
        assert_eq!(
            metadata.flat_token_indices().read_typed::<u32>(0, 5),
            vec![7, 8, 20, 21, 22]
        );
        assert_eq!(
            metadata.visible_kv_token_ranges().read_typed::<u32>(0, 10),
            vec![0, 8, 0, 9, 0, 21, 0, 22, 0, 23]
        );
        assert_eq!(
            metadata.sdpa_map_task_templates().read_typed::<u32>(0, 24),
            vec![
                0, 0, 8, 1, 0, 9, 2, 0, 8, 2, 8, 21, 3, 0, 8, 3, 8, 22, 4, 0, 8, 4, 8, 23
            ]
        );
        assert_eq!(
            metadata.cu_sdpa_partial_outputs().read_typed::<u32>(0, 6),
            vec![0, 1, 2, 4, 6, 8]
        );
        assert_eq!(single_shape, GQAReplayShape::new(5, 5, 5, 6, 8, 8, true));

        let sliced_q = backend_sdpa::ExecutionVariant::sliced_q_per_head(config(), 8, 8, 1);
        let sliced_q_selection = selector(8, sliced_q).select(&shapes, &policy, 5);
        let sliced_q_shape = metadata.update(&[2, 5], &[7, 20], &[0, 2, 5], &sliced_q_selection);
        assert_eq!(metadata.q_token_ranges().read_typed::<u32>(0, 4), vec![0, 2, 2, 5]);
        assert_eq!(
            metadata.sdpa_map_task_templates().read_typed::<u32>(0, 18),
            vec![
                0,
                0,
                8,
                0,
                8,
                9,
                1,
                0,
                8,
                1,
                8,
                16,
                1,
                16,
                23,
                u32::MAX,
                u32::MAX,
                u32::MAX
            ]
        );
        assert_eq!(
            metadata.cu_sdpa_partial_outputs().read_typed::<u32>(0, 3),
            vec![0, 2, 5]
        );
        assert_eq!(sliced_q_shape, GQAReplayShape::new(5, 5, 2, 2, 5, 6, true));
        assert_eq!(metadata.variant(), sliced_q);
    }

    #[test]
    fn test_selection_preserves_non_kv_metadata_tail() {
        let device = Device::system_default();
        let metadata = GQAMetadataBuffers::new(&device, 12, 12);
        metadata
            .visible_kv_token_ranges()
            .write_typed(0, &[0x9494_9494_u32; 24]);
        metadata.q_token_ranges().write_typed(0, &[0xA5A5_A5A5_u32; 24]);
        metadata
            .sdpa_map_task_templates()
            .write_typed(0, &[0xB6B6_B6B6_u32; 36]);
        metadata
            .cu_sdpa_partial_outputs()
            .write_typed(0, &[0xC7C7_C7C7_u32; 13]);
        let variant = backend_sdpa::ExecutionVariant::sliced_q_per_head(config(), 8, 8, 1);
        let selector = selector(12, variant);
        let policy = GQAReplayBucketPolicy::new(12, 12);
        let selection = selector.select(&RequestShape::from_batch(&[0, 0, 0], &[0, 4, 8, 9]), &policy, 12);

        let shape = metadata.update(&[2, 5, 7], &[0, 0, 0], &[0, 4, 8, 9], &selection);
        assert_eq!(shape, GQAReplayShape::new(9, 12, 3, 4, 3, 4, true));
        assert_eq!(
            metadata.visible_kv_token_ranges().read_typed::<u32>(18, 2),
            [0x9494_9494_u32; 2]
        );
        assert_eq!(metadata.q_token_ranges().read_typed::<u32>(6, 2), [0xA5A5_A5A5_u32; 2]);
        assert_eq!(
            metadata.sdpa_map_task_templates().read_typed::<u32>(9, 3),
            [u32::MAX; 3]
        );
        assert_eq!(
            metadata.cu_sdpa_partial_outputs().read_typed::<u32>(4, 1),
            [0xC7C7_C7C7_u32]
        );
    }
}
