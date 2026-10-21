use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::components::resource_embed::Config;
use inference_backend_metal::components::resource_embed::Mapping;
use inference_backend_metal::components::resource_embed::MappingTable;
use inference_backend_metal::components::resource_embed::Shape;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_runtime_core::compute::DeviceRequest;

use crate::replay::ReplayComponent;

const RESOURCE_EMBED_NUM_ACTIVE_MAPPINGS: ReplayParameterKey =
    ReplayParameterKey::new("resource_embed.num_active_mappings");

pub struct ResourceEmbed {
    compute: inference_backend_metal::components::resource_embed::Compute,
}

#[derive(Clone, Copy)]
pub struct ResourceEmbedInput<'a> {
    pub table: &'a MappingTable,
    pub arena: &'a Buffer,
    pub mappings: &'a Buffer,
    pub hidden: &'a Buffer,
}

impl ResourceEmbed {
    pub fn new(device: &Device, config: Config) -> Self {
        Self {
            compute: inference_backend_metal::components::resource_embed::Compute::new(device, config),
        }
    }

    pub fn prepare_replay(&self, table: &MappingTable) -> (Shape, ReplayArguments) {
        let shape = table.shape();
        let arguments = ReplayArguments::new().with_u32(RESOURCE_EMBED_NUM_ACTIVE_MAPPINGS, shape.num_total_mappings);
        (shape, arguments)
    }
}

impl ReplayComponent for ResourceEmbed {
    type Input<'a> = ResourceEmbedInput<'a>;
    type Key = Shape;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        input.table.shape()
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.invoke(
            input.table,
            ReplayU32::Parameter(RESOURCE_EMBED_NUM_ACTIVE_MAPPINGS),
            inference_backend_metal::components::resource_embed::Buffers {
                resource_arena: input.arena,
                mappings: input.mappings,
                hidden: input.hidden,
            },
        )));
    }
}

pub fn build_mapping_table(config: Config, requests: &[DeviceRequest]) -> Option<MappingTable> {
    let hidden_dim_bytes = config.hidden_dim_bytes();
    let mut flat_token_start = 0usize;
    let mut mappings = vec![];
    for request in requests {
        let query_token_start = request.decoder_query_tokens.token_index();
        let num_query_tokens = request.decoder_query_tokens.token_consumption();
        let query_token_end = query_token_start + num_query_tokens;
        for resource in &request.resource_placements {
            for &(placement_token_start, resource_index, num_resource_tokens) in resource.placements() {
                let placement_token_end = placement_token_start + num_resource_tokens;
                let active_token_start = query_token_start.max(placement_token_start);
                let active_token_end = query_token_end.min(placement_token_end);
                if active_token_start >= active_token_end {
                    continue;
                }

                let active_resource_start = resource_index + active_token_start - placement_token_start;
                let num_active_resource_tokens = active_token_end - active_token_start;
                let active_resource_end = active_resource_start + num_active_resource_tokens;
                let relative_source_start_bytes = active_resource_start as u64 * hidden_dim_bytes;
                let relative_source_end_bytes = active_resource_end as u64 * hidden_dim_bytes;
                debug_assert!(
                    relative_source_end_bytes <= resource.arena_len_bytes(),
                    "resource embedding source range exceeds its allocation"
                );
                let mut source_offset_bytes = resource.arena_offset_bytes() + relative_source_start_bytes;
                let destination_row_start = flat_token_start + active_token_start - query_token_start;
                for token_offset in 0..num_active_resource_tokens {
                    let destination_row = destination_row_start + token_offset;
                    debug_assert!(destination_row <= u32::MAX as usize);
                    mappings.push(Mapping {
                        destination_row: destination_row as u32,
                        source_offset_bytes,
                    });
                    source_offset_bytes += hidden_dim_bytes;
                }
            }
        }
        flat_token_start += num_query_tokens;
    }

    (!mappings.is_empty()).then(|| MappingTable::new(config, mappings))
}

#[cfg(test)]
#[path = "resource_embed_test.rs"]
mod tests;
