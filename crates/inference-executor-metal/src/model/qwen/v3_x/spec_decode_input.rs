use std::ops::Range;

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::gqa::sdpa::MapThreadBlockConstants;
use inference_backend_metal::components::sampling::spec_decode_input as backend;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_executor_core::attn::BiDiBlockGQAMetadata;
use inference_executor_core::replay::ReplayBucketPolicy;

use crate::attn::bidi_block_gqa::metadata::BiDiBlockGQAMetadataBuffers;
use crate::replay::ReplayComponent;
use crate::sampling::rejection_sampling::SparseRejectionSamplingOutput;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpecDecodeInputConfig {
    max_requests: u32,
    spec_block_size: u32,
    sdpa: MapThreadBlockConstants,
    history_window: u32,
    max_anchor_position: u32,
    max_task_templates: u32,
    mask_token_id: i32,
}

impl SpecDecodeInputConfig {
    pub fn new(
        max_requests: u32,
        spec_block_size: u32,
        sdpa: MapThreadBlockConstants,
        history_window: u32,
        max_anchor_position: u32,
        max_task_templates: u32,
        mask_token_id: i32,
    ) -> Self {
        assert!(max_requests > 0, "Spec Decode input requires request capacity");
        assert!(sdpa.max_q_tokens > 0, "Spec Decode input requires a Q-token tile size");
        let config = Self {
            max_requests,
            spec_block_size,
            sdpa,
            history_window,
            max_anchor_position,
            max_task_templates,
            mask_token_id,
        };
        let backend = config.backend();
        backend.validate();
        assert!(
            u64::from(max_requests) * u64::from(spec_block_size) <= u64::from(u32::MAX),
            "Spec Decode maximum query-token count must fit the shader u32 domain"
        );
        assert!(
            u64::from(max_requests) * u64::from(backend.num_q_ranges_per_request) <= u64::from(u32::MAX),
            "Spec Decode maximum Q-token-range count must fit the shader u32 domain"
        );
        config
    }

    fn backend(self) -> backend::Config {
        backend::Config {
            spec_block_size: self.spec_block_size,
            num_q_ranges_per_request: self.spec_block_size.div_ceil(self.sdpa.max_q_tokens),
            kv_tokens_per_iteration: self.sdpa.kv_tokens_per_iteration,
            history_window: self.history_window,
            max_anchor_position: self.max_anchor_position,
            max_task_templates: self.max_task_templates,
            mask_token_id: self.mask_token_id,
        }
    }
}

pub struct SpecDecodeInput {
    config: SpecDecodeInputConfig,
    request_bucket_policy: ReplayBucketPolicy,
    prepare: backend::Compute,
    anchor_indices: Buffer,
    sample_positions: Buffer,
}

pub struct SpecDecodeInputArgs<'a> {
    pub num_active_requests: u32,
    pub num_total_requests: u32,
    pub rejection_sampling: SparseRejectionSamplingOutput<'a>,
    pub metadata: &'a BiDiBlockGQAMetadataBuffers,
    pub block_token_ids: &'a Buffer,
    pub anchor_token_ids: &'a Buffer,
}

pub struct SpecDecodeInputRecording {
    pub key: SpecDecodeInputReplayKey,
    pub arguments: ReplayArguments,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SpecDecodeInputReplayKey {
    num_total_requests: u32,
    num_total_q_token_ranges: u32,
    num_total_task_templates: u32,
}

impl SpecDecodeInput {
    pub fn new(device: &Device, config: SpecDecodeInputConfig) -> Self {
        let backend = config.backend();
        Self {
            config,
            request_bucket_policy: ReplayBucketPolicy::new(config.max_requests),
            prepare: backend::Compute::new(device, backend),
            anchor_indices: Buffer::new_zeroed_elements(device, config.max_requests as usize, Dtype::Uint32),
            sample_positions: Buffer::new_zeroed_elements(device, config.max_requests as usize, Dtype::Uint32),
        }
    }

    pub fn prepare(&self, req_slots: &[u32], anchor_indices: &[u32], num_spec_tokens: &[u32]) -> BiDiBlockGQAMetadata {
        let num_active_requests = req_slots.len();
        let num_total_requests = self.request_bucket_policy.capacity(num_active_requests as u32) as usize;
        self.anchor_indices.write_typed(0, anchor_indices);

        let max_requests = self.config.max_requests as usize;
        let mut used = vec![false; max_requests];
        let mut materialized_req_slots = Vec::with_capacity(num_total_requests);
        // Active row order aligns rejection outputs, anchors, token buffers,
        // metadata, and sampling destinations. `num_total_requests` is cached
        // replay/grid capacity; active rows are its prefix, and inactive GPU
        // rows return before access. Unused valid slots only materialize a
        // structurally valid BiDiBlockGQA tail.
        for &req_slot in req_slots {
            let req_slot = req_slot as usize;
            used[req_slot] = true;
            materialized_req_slots.push(req_slot as u32);
        }
        let num_inactive_requests = num_total_requests - num_active_requests;
        materialized_req_slots.extend(
            used.iter()
                .enumerate()
                .filter(|(_, used)| !**used)
                .take(num_inactive_requests)
                .map(|(req_slot, _)| req_slot as u32),
        );
        let mut flat_query_token_indices =
            Vec::with_capacity(num_total_requests * self.config.spec_block_size as usize);
        let mut visible_history_token_ranges = Vec::with_capacity(flat_query_token_indices.capacity());
        // A history TaskTemplate is {q_token_range_index, kv_token_begin,
        // kv_token_end}. It assigns one GQA Map threadblock one half-open
        // persistent-history range, which can contain multiple K/V iterations.
        // Maximum acceptance retains the allocator's worst-case recorded split
        // count and parallelism. Prepare rewrites active coordinates and
        // endpoints, but it cannot change the recorded replay key/grid,
        // Q-range/CU structure, or TaskTemplate count. Lower acceptance maps
        // excess splits to empty ranges; fewer seed splits could remain correct
        // but under-partition a later longer history range.
        for request_index in 0..num_total_requests {
            let active = request_index < num_active_requests;
            let anchor = if active {
                anchor_indices[request_index] + num_spec_tokens[request_index]
            } else {
                1
            };
            for block_offset in 0..self.config.spec_block_size {
                let query_position = anchor + block_offset;
                flat_query_token_indices.push(query_position);
                visible_history_token_ranges.push(self.visible_history_range(query_position, anchor));
            }
        }
        BiDiBlockGQAMetadata::new(
            &materialized_req_slots,
            &flat_query_token_indices,
            &visible_history_token_ranges,
            self.config.spec_block_size as usize,
        )
    }

    pub fn replay_arguments(&self, key: &SpecDecodeInputReplayKey, num_active_requests: u32) -> ReplayArguments {
        let mut arguments = ReplayArguments::new();
        self.prepare
            .add_replay_arguments(self.backend_shape(key), num_active_requests, &mut arguments);
        arguments
    }

    fn visible_history_range(&self, query_position: u32, anchor_position: u32) -> Range<u32> {
        (query_position + 1).saturating_sub(self.config.history_window)..anchor_position
    }

    pub fn sample_positions(&self) -> &Buffer {
        &self.sample_positions
    }

    fn backend_shape(&self, key: &SpecDecodeInputReplayKey) -> backend::Shape {
        backend::Shape {
            num_total_requests: key.num_total_requests,
            num_total_q_token_ranges: key.num_total_q_token_ranges,
            num_total_task_templates: key.num_total_task_templates,
        }
    }
}

impl ReplayComponent for SpecDecodeInput {
    type Key = SpecDecodeInputReplayKey;
    type Input<'a> = SpecDecodeInputArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        let replay_shape = input.metadata.replay_shape();
        SpecDecodeInputReplayKey {
            num_total_requests: input.num_total_requests,
            num_total_q_token_ranges: replay_shape.num_total_q_token_tiles,
            num_total_task_templates: replay_shape.num_total_sdpa_map_task_templates,
        }
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        let shape = self.backend_shape(&key);
        recorder.record_with_barrier_before(ReplayOp::opaque(self.prepare.invoke_replay(
            shape,
            backend::Buffers {
                num_accepted_tokens: input.rejection_sampling.num_accepted_tokens,
                sampled_token_ids: input.rejection_sampling.sampled_token_ids,
                anchor_indices: &self.anchor_indices,
                anchor_token_ids: input.anchor_token_ids,
                sample_positions: &self.sample_positions,
                block_token_ids: input.block_token_ids,
                flat_query_token_indices: input.metadata.flat_token_indices(),
                visible_history_token_ranges: input.metadata.visible_kv_token_ranges(),
                q_token_ranges: input.metadata.q_token_ranges(),
                cu_sdpa_partial_outputs: input.metadata.cu_sdpa_partial_outputs(),
                sdpa_map_task_templates: input.metadata.sdpa_map_task_templates(),
            },
        )));
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::components::gqa::sdpa::MapThreadBlockConstants;
    use inference_backend_metal::metal::Device;

    use super::SpecDecodeInput;
    use super::SpecDecodeInputConfig;

    #[test]
    fn test_prepare_query_block_size() {
        let device = Device::system_default();
        for block_size in [1, 2, 3, 4, 5, 7, 8] {
            let input = SpecDecodeInput::new(
                &device,
                SpecDecodeInputConfig::new(
                    2,
                    block_size,
                    MapThreadBlockConstants {
                        max_q_tokens: 2,
                        max_q_heads: 4,
                        kv_tokens_per_iteration: 4,
                        required_threads: 32,
                    },
                    u32::MAX,
                    128,
                    32,
                    0,
                ),
            );
            // Previous verification widths differ from the new proposal block.
            let metadata = input.prepare(&[1, 0], &[10, 20], &[7, 0]);
            let expected = (17..17 + block_size).chain(20..20 + block_size).collect::<Vec<_>>();
            assert_eq!(metadata.flat_token_indices(), expected);
            assert_eq!(metadata.req_slots().len(), 2 * block_size as usize);
            assert_eq!(metadata.block_size(), block_size as usize);
        }
    }

    #[test]
    fn test_prepare_bucketing() {
        let input = SpecDecodeInput::new(
            &Device::system_default(),
            SpecDecodeInputConfig::new(
                4,
                3,
                MapThreadBlockConstants {
                    max_q_tokens: 2,
                    max_q_heads: 4,
                    kv_tokens_per_iteration: 4,
                    required_threads: 32,
                },
                u32::MAX,
                128,
                32,
                0,
            ),
        );

        let metadata = input.prepare(&[0, 2, 3], &[10, 20, 30], &[3, 0, 5]);

        assert_eq!(metadata.num_requests(), 4);
        assert_eq!(metadata.block_size(), 3);
        assert_eq!(metadata.req_slots(), [0, 0, 0, 2, 2, 2, 3, 3, 3, 1, 1, 1]);
        assert_eq!(
            metadata.flat_token_indices(),
            [13, 14, 15, 20, 21, 22, 35, 36, 37, 1, 2, 3]
        );
        assert_eq!(
            metadata.history_token_ends(),
            [13, 13, 13, 20, 20, 20, 35, 35, 35, 1, 1, 1]
        );
    }
}
