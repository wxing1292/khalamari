//! GPU metadata for history and block attention partials.

use std::cell::Cell;
use std::collections::BinaryHeap;

use inference_backend_metal::components::gqa::sdpa::ExecutionVariant;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::BiDiBlockGQAMetadata;

use crate::attn::bidi_block_gqa::capacity::BiDiBlockGQACapacity;
use crate::attn::gqa::batch_metadata::GQAReplayShape;

const NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS: usize = 3;
const NUM_VISIBLE_KV_TOKEN_RANGE_FIELDS: usize = 2;

struct BiDiBlockGQAMetadataUpload {
    visible_kv_token_ranges: Vec<u32>,
    q_token_ranges: Vec<u32>,
    sdpa_map_task_templates: Vec<u32>,
    cu_sdpa_partial_outputs: Vec<u32>,
    replay_shape: GQAReplayShape,
}

pub struct BiDiBlockGQAMetadataBuffers {
    capacity: BiDiBlockGQACapacity,
    req_slots: Buffer,
    flat_token_indices: Buffer,
    visible_kv_token_ranges: Buffer,
    q_token_ranges: Buffer,
    sdpa_map_task_templates: Buffer,
    cu_sdpa_partial_outputs: Buffer,
    replay_shape: Cell<Option<GQAReplayShape>>,
    variant: ExecutionVariant,
}

impl BiDiBlockGQAMetadataBuffers {
    pub fn new(device: &Device, capacity: BiDiBlockGQACapacity, variant: ExecutionVariant) -> Self {
        assert_eq!(
            variant.map.thread_block.max_q_tokens as usize, capacity.max_q_tokens,
            "BiDiBlockGQA history attention execution must match metadata capacity"
        );
        let task_template_values = capacity
            .max_sdpa_map_task_templates
            .checked_mul(NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
            .expect("BiDiBlockGQA TaskTemplate metadata capacity must fit usize");
        Self {
            capacity,
            req_slots: Buffer::new_zeroed_elements(device, capacity.block.max_tokens, Dtype::Uint32),
            flat_token_indices: Buffer::new_zeroed_elements(device, capacity.block.max_tokens, Dtype::Uint32),
            visible_kv_token_ranges: Buffer::new_zeroed_elements(
                device,
                capacity
                    .block
                    .max_tokens
                    .checked_mul(NUM_VISIBLE_KV_TOKEN_RANGE_FIELDS)
                    .expect("BiDiBlockGQA visible K/V-token-range capacity must fit usize"),
                Dtype::Uint32,
            ),
            q_token_ranges: Buffer::new_zeroed_elements(
                device,
                capacity
                    .max_q_token_ranges
                    .checked_mul(2)
                    .expect("BiDiBlockGQA Q-token-range capacity must fit usize"),
                Dtype::Uint32,
            ),
            sdpa_map_task_templates: Buffer::new_zeroed_elements(device, task_template_values, Dtype::Uint32),
            cu_sdpa_partial_outputs: Buffer::new_zeroed_elements(
                device,
                capacity
                    .max_q_token_ranges
                    .checked_add(1)
                    .expect("BiDiBlockGQA cumulative partial-output capacity must fit usize"),
                Dtype::Uint32,
            ),
            replay_shape: Cell::new(None),
            variant,
        }
    }

    pub fn update(&self, block: &BiDiBlockGQAMetadata) -> GQAReplayShape {
        self.update_with_active_requests(block, block.num_requests())
    }

    pub fn update_with_active_requests(
        &self,
        block: &BiDiBlockGQAMetadata,
        num_active_requests: usize,
    ) -> GQAReplayShape {
        assert!(
            block.num_requests() <= self.capacity.block.max_requests,
            "BiDiBlockGQA request count exceeds capacity"
        );
        assert!(
            num_active_requests > 0 && num_active_requests <= block.num_requests(),
            "active BiDiBlockGQA requests must fit the materialized request capacity"
        );
        assert_eq!(
            block.block_size(),
            self.capacity.block.block_size,
            "BiDiBlockGQA block size must match the static capacity"
        );
        assert!(
            block.num_tokens() <= self.capacity.block.max_tokens,
            "BiDiBlockGQA token count exceeds capacity"
        );
        self.req_slots.write_typed(0, block.req_slots());
        self.flat_token_indices.write_typed(0, block.flat_token_indices());

        let metadata = build_metal_metadata(block, self.variant, self.capacity.max_sdpa_map_task_templates);
        let num_q_token_ranges = metadata.replay_shape.num_q_token_tiles as usize;
        assert!(
            num_q_token_ranges <= self.capacity.max_q_token_ranges,
            "BiDiBlockGQA Q-token-range count exceeds capacity"
        );
        self.visible_kv_token_ranges
            .write_typed(0, &metadata.visible_kv_token_ranges);
        self.q_token_ranges.write_typed(0, &metadata.q_token_ranges);
        self.sdpa_map_task_templates
            .write_typed(0, &metadata.sdpa_map_task_templates);
        self.cu_sdpa_partial_outputs
            .write_typed(0, &metadata.cu_sdpa_partial_outputs);
        let num_q_ranges_per_request = block
            .block_size()
            .div_ceil(self.variant.map.thread_block.max_q_tokens as usize);
        let num_active_tokens = num_active_requests * block.block_size();
        let num_active_q_token_ranges = num_active_requests * num_q_ranges_per_request;
        let num_active_task_templates = metadata.cu_sdpa_partial_outputs[num_active_q_token_ranges];
        let replay_shape = GQAReplayShape {
            num_tokens: num_active_tokens as u32,
            num_q_token_tiles: num_active_q_token_ranges as u32,
            num_sdpa_map_task_templates: num_active_task_templates,
            ..metadata.replay_shape
        };
        self.replay_shape.set(Some(replay_shape));
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
            .expect("BiDiBlockGQA metadata must be updated before recording")
    }

    pub fn variant(&self) -> ExecutionVariant {
        self.variant
    }
}

fn build_metal_metadata(
    block: &BiDiBlockGQAMetadata,
    variant: ExecutionVariant,
    max_sdpa_map_task_templates: usize,
) -> BiDiBlockGQAMetadataUpload {
    let map = variant.map.thread_block;
    let kv_tokens_per_iteration = map.kv_tokens_per_iteration;
    debug_assert!(
        kv_tokens_per_iteration > 0,
        "BiDiBlockGQA KV-token iteration size must be positive"
    );
    debug_assert!(map.max_q_tokens > 0, "BiDiBlockGQA Q-token range must contain tokens");
    let num_tokens = block.num_tokens();
    let visible_kv_token_ranges = block
        .history_token_begins()
        .iter()
        .zip(block.history_token_ends())
        .flat_map(|(&begin, &end)| [begin, end])
        .collect::<Vec<_>>();
    debug_assert_eq!(
        visible_kv_token_ranges.len(),
        num_tokens * NUM_VISIBLE_KV_TOKEN_RANGE_FIELDS
    );
    let max_q_tokens = map.max_q_tokens as usize;
    let mut q_token_range_indices = Vec::new();
    for request_index in 0..block.num_requests() {
        let request_begin = request_index * block.block_size();
        let request_end = request_begin + block.block_size();
        let mut range_begin = request_begin;
        while range_begin < request_end {
            let range_end = (range_begin + max_q_tokens).min(request_end);
            q_token_range_indices.push(range_begin..range_end);
            range_begin = range_end;
        }
    }
    debug_assert!(
        max_sdpa_map_task_templates
            >= q_token_range_indices
                .len()
                .checked_mul(2)
                .expect("BiDiBlockGQA minimum TaskTemplate count must fit usize")
    );
    debug_assert!(
        max_sdpa_map_task_templates.is_power_of_two(),
        "BiDiBlockGQA TaskTemplate capacity must be a power of two"
    );

    let history_token_ranges = q_token_range_indices
        .iter()
        .map(|range| {
            let begin = block.history_token_begins()[range.clone()]
                .iter()
                .copied()
                .min()
                .expect("BiDiBlockGQA Q-token range must not be empty");
            let end = block.history_token_ends()[range.clone()]
                .iter()
                .copied()
                .max()
                .expect("BiDiBlockGQA Q-token range must not be empty");
            debug_assert!(begin < end, "BiDiBlockGQA history range must not be empty");
            begin..end
        })
        .collect::<Vec<_>>();
    let num_kv_iterations = history_token_ranges
        .iter()
        .map(|range| {
            let num_kv_iterations = (range.end - range.start).div_ceil(kv_tokens_per_iteration);
            // History can start above zero. Its padded endpoint must fit the GPU metadata.
            assert!(
                num_kv_iterations <= (u32::MAX - range.start) / kv_tokens_per_iteration,
                "BiDiBlockGQA padded history-token range must fit u32"
            );
            num_kv_iterations as usize
        })
        .collect::<Vec<_>>();
    let max_history_map_task_templates = max_sdpa_map_task_templates - q_token_range_indices.len();
    let num_history_kv_splits = allocate_history_kv_splits(&num_kv_iterations, max_history_map_task_templates);

    let mut q_token_ranges = Vec::with_capacity(q_token_range_indices.len() * 2);
    let mut sdpa_map_task_templates = Vec::new();
    let mut cu_sdpa_partial_outputs = Vec::with_capacity(q_token_range_indices.len() + 1);
    cu_sdpa_partial_outputs.push(0);
    for (q_token_range_index, q_token_range) in q_token_range_indices.iter().enumerate() {
        q_token_ranges.extend_from_slice(&[
            q_token_range
                .start
                .try_into()
                .expect("BiDiBlockGQA Q-token-range begin must fit u32"),
            q_token_range
                .end
                .try_into()
                .expect("BiDiBlockGQA Q-token-range end must fit u32"),
        ]);
        let history_range = &history_token_ranges[q_token_range_index];
        let num_iterations = num_kv_iterations[q_token_range_index];
        let num_kv_splits = num_history_kv_splits[q_token_range_index];
        for kv_split_index in 0..num_kv_splits {
            let iteration_begin = num_iterations * kv_split_index / num_kv_splits;
            let iteration_end = num_iterations * (kv_split_index + 1) / num_kv_splits;
            let kv_token_begin = history_range.start + iteration_begin as u32 * kv_tokens_per_iteration;
            let kv_token_end = history_range
                .end
                .min(history_range.start + iteration_end as u32 * kv_tokens_per_iteration);
            sdpa_map_task_templates.extend_from_slice(&[
                q_token_range_index
                    .try_into()
                    .expect("BiDiBlockGQA Q-token-range index must fit u32"),
                kv_token_begin,
                kv_token_end,
            ]);
        }

        sdpa_map_task_templates.extend_from_slice(&[u32::MAX; NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS]);
        cu_sdpa_partial_outputs.push(
            (sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS)
                .try_into()
                .expect("BiDiBlockGQA cumulative partial-output count must fit u32"),
        );
    }

    let num_task_templates = sdpa_map_task_templates.len() / NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS;
    let num_total_sdpa_map_task_templates = num_task_templates
        .checked_next_power_of_two()
        .expect("BiDiBlockGQA replay TaskTemplate count must fit usize");
    assert!(
        num_total_sdpa_map_task_templates <= max_sdpa_map_task_templates,
        "BiDiBlockGQA replay TaskTemplate count exceeds capacity"
    );
    sdpa_map_task_templates.resize(
        num_total_sdpa_map_task_templates * NUM_SDPA_MAP_TASK_TEMPLATE_FIELDS,
        u32::MAX,
    );
    let num_tokens = num_tokens.try_into().expect("BiDiBlockGQA token count must fit u32");
    let replay_shape = GQAReplayShape {
        num_tokens,
        num_total_tokens: num_tokens,
        num_q_token_tiles: q_token_range_indices
            .len()
            .try_into()
            .expect("BiDiBlockGQA Q-token-range count must fit u32"),
        num_total_q_token_tiles: q_token_range_indices
            .len()
            .try_into()
            .expect("BiDiBlockGQA Q-token-range count must fit u32"),
        num_sdpa_map_task_templates: num_task_templates
            .try_into()
            .expect("BiDiBlockGQA active TaskTemplate count must fit u32"),
        num_total_sdpa_map_task_templates: num_total_sdpa_map_task_templates
            .try_into()
            .expect("BiDiBlockGQA TaskTemplate count must fit u32"),
        reduce_sdpa_partial_outputs: true,
    };
    replay_shape.validate();
    BiDiBlockGQAMetadataUpload {
        visible_kv_token_ranges,
        q_token_ranges,
        sdpa_map_task_templates,
        cu_sdpa_partial_outputs,
        replay_shape,
    }
}

fn allocate_history_kv_splits(num_kv_iterations: &[usize], max_map_task_templates: usize) -> Vec<usize> {
    debug_assert!(num_kv_iterations.len() <= max_map_task_templates);
    if let &[iterations] = num_kv_iterations {
        return vec![iterations.min(max_map_task_templates)];
    }
    let mut num_kv_splits = vec![1; num_kv_iterations.len()];
    if num_kv_splits.len() == max_map_task_templates {
        return num_kv_splits;
    }

    // Preserve the history allocator's tie order: later Q ranges win.
    let mut candidates = num_kv_iterations
        .iter()
        .enumerate()
        .filter(|&(_, &iterations)| iterations > 1)
        .map(|(index, &iterations)| (iterations, index))
        .collect::<BinaryHeap<_>>();
    for _ in num_kv_splits.len()..max_map_task_templates {
        let Some((_, index)) = candidates.pop() else {
            break;
        };
        num_kv_splits[index] += 1;
        if num_kv_splits[index] < num_kv_iterations[index] {
            candidates.push((num_kv_iterations[index].div_ceil(num_kv_splits[index]), index));
        }
    }
    num_kv_splits
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::gqa::sdpa as backend_sdpa;
    use inference_backend_metal::metal::Dtype;
    use inference_executor_core::attn::BiDiBlockCapacity;

    use super::*;
    use crate::attn::bidi_block_gqa::sdpa::Selector;

    #[test]
    fn test_metadata_materializes_tiled_history_and_block_partials() {
        let device = Device::system_default();
        let capacity = BiDiBlockGQACapacity::new(BiDiBlockCapacity::new(2, 3), 8);
        let variant = backend_sdpa::ExecutionVariant::sliced_q_per_head(config(), 8, 16, 2);
        let buffers = BiDiBlockGQAMetadataBuffers::new(&device, capacity, variant);
        let block = BiDiBlockGQAMetadata::new(
            &[3, 7],
            &[11, 12, 13, 20, 21, 22],
            &[0..11, 0..11, 0..11, 3..20, 3..20, 3..20],
            3,
        );

        let shape = buffers.update(&block);

        assert_eq!(shape.num_tokens, 6);
        assert_eq!(shape.num_q_token_tiles, 2);
        assert_eq!(shape.num_total_sdpa_map_task_templates, 8);
        assert_eq!(
            buffers.visible_kv_token_ranges().read_typed::<u32>(0, 12),
            [0, 11, 0, 11, 0, 11, 3, 20, 3, 20, 3, 20]
        );
        assert_eq!(buffers.q_token_ranges().read_typed::<u32>(0, 4), [0, 3, 3, 6]);
        assert_eq!(buffers.cu_sdpa_partial_outputs().read_typed::<u32>(0, 3), [0, 2, 5]);
        assert_eq!(variant, buffers.variant());
    }

    #[test]
    fn test_long_history_is_partitioned_without_growing_static_scratch() {
        let anchor = 262_144;
        let block = BiDiBlockGQAMetadata::new(&[1], &(anchor..anchor + 6).collect::<Vec<_>>(), &vec![0..anchor; 6], 6);
        let metadata = build_metal_metadata(
            &block,
            backend_sdpa::ExecutionVariant::sliced_q_per_head(config(), 8, 16, 2),
            16,
        );

        let block_task = metadata.cu_sdpa_partial_outputs[1] as usize - 1;
        let history = metadata.sdpa_map_task_templates[..block_task * 3].as_chunks::<3>().0;
        assert_eq!(history.first().expect("history task")[1], 0);
        assert_eq!(history.last().expect("history task")[2], anchor);
        assert!(history.windows(2).all(|pair| pair[0][2] == pair[1][1]));
        assert_eq!(metadata.cu_sdpa_partial_outputs, [0, 16]);
        assert_eq!(metadata.replay_shape.num_total_sdpa_map_task_templates, 16);
    }

    #[test]
    fn test_q_tile_uses_history_union_and_preserves_per_query_masks() {
        let device = Device::system_default();
        let capacity = BiDiBlockGQACapacity::new(BiDiBlockCapacity::new(1, 3), 8);
        let variant = backend_sdpa::ExecutionVariant::sliced_q_per_head(config(), 8, 16, 2);
        let buffers = BiDiBlockGQAMetadataBuffers::new(&device, capacity, variant);
        let block = BiDiBlockGQAMetadata::new(&[3], &[20, 21, 22], &[0..20, 1..20, 2..20], 3);

        let shape = buffers.update(&block);

        assert_eq!(shape.num_q_token_tiles, 1);
        assert_eq!(
            buffers.visible_kv_token_ranges().read_typed::<u32>(0, 6),
            [0, 20, 1, 20, 2, 20]
        );
        let first_task = buffers.sdpa_map_task_templates().read_typed::<u32>(0, 3);
        assert_eq!(first_task[1], 0);
        assert!(first_task[2] <= 20);
    }

    #[test]
    fn test_spec_decode_success() {
        let full_history_variant = selected_variant(4, 7);
        for accepted in 0..=7 {
            assert_fixed_quota_rewrite_matches_cpu(&[10], &[7], &[accepted], 7, u32::MAX, full_history_variant);
        }
        assert_fixed_quota_rewrite_matches_cpu(&[10, 31], &[7, 7], &[0, 5], 7, u32::MAX, full_history_variant);
        assert_fixed_quota_rewrite_matches_cpu(
            &[10, 31, 65, 96],
            &[7, 7, 7, 7],
            &[7, 2, 0, 6],
            7,
            u32::MAX,
            full_history_variant,
        );

        let dflash2_variant = selected_variant(4, 8);
        assert_fixed_quota_rewrite_matches_cpu(&[10, 31, 65, 96], &[7, 7, 7, 7], &[0, 4, 7, 2], 8, 16, dflash2_variant);
    }

    fn assert_fixed_quota_rewrite_matches_cpu(
        anchor_indices: &[u32],
        num_spec_tokens: &[u32],
        num_accepted_tokens: &[u32],
        block_size: usize,
        history_window: u32,
        variant: ExecutionVariant,
    ) {
        assert_eq!(anchor_indices.len(), num_spec_tokens.len());
        assert_eq!(anchor_indices.len(), num_accepted_tokens.len());
        let maximum_accepted_anchor_indices = anchor_indices
            .iter()
            .zip(num_spec_tokens)
            .map(|(&anchor_index, &num_spec_tokens)| anchor_index + num_spec_tokens)
            .collect::<Vec<_>>();
        let accepted_anchor_indices = anchor_indices
            .iter()
            .zip(num_accepted_tokens)
            .map(|(&anchor_index, &num_accepted_tokens)| anchor_index + num_accepted_tokens)
            .collect::<Vec<_>>();
        let req_slots = (0..anchor_indices.len() as u32).collect::<Vec<_>>();
        let static_block = spec_decode_block(&req_slots, &maximum_accepted_anchor_indices, block_size, history_window);
        let cpu_block = spec_decode_block(&req_slots, &accepted_anchor_indices, block_size, history_window);
        let max_templates = BiDiBlockGQACapacity::new(
            BiDiBlockCapacity::new(anchor_indices.len(), block_size),
            variant.map.thread_block.max_q_tokens,
        )
        .max_sdpa_map_task_templates;
        let mut rewritten = build_metal_metadata(&static_block, variant, max_templates);
        rewrite_fixed_quota_for_test(
            &mut rewritten,
            &accepted_anchor_indices,
            block_size,
            history_window,
            variant,
        );
        let cpu = build_metal_metadata(&cpu_block, variant, max_templates);

        assert_eq!(rewritten.visible_kv_token_ranges, cpu.visible_kv_token_ranges);
        assert_eq!(rewritten.q_token_ranges, cpu.q_token_ranges);
        let rewritten_tasks = rewritten.sdpa_map_task_templates.as_chunks::<3>().0;
        let cpu_tasks = cpu.sdpa_map_task_templates.as_chunks::<3>().0;
        for q_token_range_index in 0..cpu.replay_shape.num_q_token_tiles as usize {
            let rewritten_begin = rewritten.cu_sdpa_partial_outputs[q_token_range_index] as usize;
            let rewritten_end = rewritten.cu_sdpa_partial_outputs[q_token_range_index + 1] as usize - 1;
            let cpu_begin = cpu.cu_sdpa_partial_outputs[q_token_range_index] as usize;
            let cpu_end = cpu.cu_sdpa_partial_outputs[q_token_range_index + 1] as usize - 1;
            let expected_begin = cpu_tasks[cpu_begin][1];
            let expected_end = cpu_tasks[cpu_end - 1][2];
            let history_tasks = &rewritten_tasks[rewritten_begin..rewritten_end];
            assert_eq!(
                history_tasks.first().expect("fixed quota history Task")[1],
                expected_begin
            );
            assert_eq!(history_tasks.last().expect("fixed quota history Task")[2], expected_end);
            assert!(history_tasks.iter().all(|task| task[0] == q_token_range_index as u32));
            assert!(history_tasks.windows(2).all(|pair| pair[0][2] == pair[1][1]));
            assert_eq!(rewritten_tasks[rewritten_end], [u32::MAX; 3]);
        }
    }

    fn spec_decode_block(
        req_slots: &[u32],
        accepted_anchor_indices: &[u32],
        block_size: usize,
        history_window: u32,
    ) -> BiDiBlockGQAMetadata {
        let mut flat_query_token_indices = Vec::with_capacity(req_slots.len() * block_size);
        let mut visible_history_token_ranges = Vec::with_capacity(req_slots.len() * block_size);
        for &anchor_index in accepted_anchor_indices {
            for block_offset in 0..block_size as u32 {
                let query_position = anchor_index + block_offset;
                flat_query_token_indices.push(query_position);
                visible_history_token_ranges.push((query_position + 1).saturating_sub(history_window)..anchor_index);
            }
        }
        BiDiBlockGQAMetadata::new(
            req_slots,
            &flat_query_token_indices,
            &visible_history_token_ranges,
            block_size,
        )
    }

    fn rewrite_fixed_quota_for_test(
        metadata: &mut BiDiBlockGQAMetadataUpload,
        accepted_anchor_indices: &[u32],
        block_size: usize,
        history_window: u32,
        variant: ExecutionVariant,
    ) {
        let num_q_ranges_per_request = block_size.div_ceil(variant.map.thread_block.max_q_tokens as usize);
        for (request_index, &anchor_index) in accepted_anchor_indices.iter().enumerate() {
            let spec_block_begin = request_index * block_size;
            for block_offset in 0..block_size {
                let query_position = anchor_index + block_offset as u32;
                let query_index = spec_block_begin + block_offset;
                metadata.visible_kv_token_ranges[query_index * 2] = (query_position + 1).saturating_sub(history_window);
                metadata.visible_kv_token_ranges[query_index * 2 + 1] = anchor_index;
            }
            for local_range_index in 0..num_q_ranges_per_request {
                let q_range_index = request_index * num_q_ranges_per_request + local_range_index;
                let first_query_position =
                    anchor_index + metadata.q_token_ranges[q_range_index * 2] - spec_block_begin as u32;
                let history_begin = (first_query_position + 1).saturating_sub(history_window);
                let task_begin = metadata.cu_sdpa_partial_outputs[q_range_index] as usize;
                let task_end = metadata.cu_sdpa_partial_outputs[q_range_index + 1] as usize - 1;
                let num_tasks = task_end - task_begin;
                let num_iterations =
                    (anchor_index - history_begin).div_ceil(variant.map.thread_block.kv_tokens_per_iteration) as usize;
                let tasks = metadata.sdpa_map_task_templates.as_chunks_mut::<3>().0;
                for task_offset in 0..num_tasks {
                    let iteration_begin = num_iterations * task_offset / num_tasks;
                    let iteration_end = num_iterations * (task_offset + 1) / num_tasks;
                    tasks[task_begin + task_offset] = [
                        q_range_index as u32,
                        history_begin + iteration_begin as u32 * variant.map.thread_block.kv_tokens_per_iteration,
                        anchor_index.min(
                            history_begin + iteration_end as u32 * variant.map.thread_block.kv_tokens_per_iteration,
                        ),
                    ];
                }
            }
        }
    }

    fn selected_variant(max_requests: usize, block_size: usize) -> ExecutionVariant {
        Selector::new(
            backend_sdpa::Registry::new(config()),
            BiDiBlockCapacity::new(max_requests, block_size),
        )
        .select()
        .variant()
    }

    fn config() -> backend_sdpa::Config {
        backend_sdpa::Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 128,
            tokens_per_page: 8,
        }
    }
}
