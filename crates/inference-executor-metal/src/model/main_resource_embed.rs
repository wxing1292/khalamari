use std::sync::Arc;

use inference_backend_metal::components::resource_embed;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::Stream;
use inference_runtime_core::compute::DeviceRequest;

use crate::model::resource_arena::MetalResourceArena;
use crate::model::resource_embed::ResourceEmbed;
use crate::model::resource_embed::ResourceEmbedInput;
use crate::model::resource_embed::build_mapping_table;
use crate::replay::Replay;

pub struct MainResourceEmbed {
    config: resource_embed::Config,
    arena: Arc<MetalResourceArena>,
    mappings: Buffer,
    replay: Replay<ResourceEmbed>,
    prepared_table: Option<resource_embed::MappingTable>,
}

impl MainResourceEmbed {
    pub fn new(
        device: &Device,
        config: resource_embed::Config,
        arena: Arc<MetalResourceArena>,
        max_tokens: usize,
    ) -> Self {
        assert!(max_tokens > 0, "MainResourceEmbed requires token capacity");
        assert!(
            u32::try_from(max_tokens).is_ok(),
            "MainResourceEmbed token capacity must fit u32"
        );
        let num_mapping_values = max_tokens * 3;
        Self {
            config,
            arena,
            mappings: Buffer::new_zeroed_elements(
                device,
                num_mapping_values,
                inference_backend_metal::metal::Dtype::Uint32,
            ),
            replay: Replay::new("ResourceEmbed", ResourceEmbed::new(device, config)),
            prepared_table: None,
        }
    }

    pub fn prepare(&mut self, requests: &[DeviceRequest]) {
        self.prepared_table = build_mapping_table(self.config, requests);
        if let Some(table) = &self.prepared_table {
            self.mappings.write_typed(0, table.encoded_u32s());
        }
    }

    pub fn prepare_replay(&self) -> Option<(resource_embed::Shape, ReplayArguments)> {
        self.prepared_table
            .as_ref()
            .map(|table| self.replay.component().prepare_replay(table))
    }

    pub fn record(&mut self, runtime: &Stream, hidden: &Buffer) -> Option<resource_embed::Shape> {
        let table = self.prepared_table.as_ref()?;
        let input = ResourceEmbedInput {
            table,
            arena: self.arena.storage().buffer(),
            mappings: &self.mappings,
            hidden,
        };
        Some(self.replay.record(runtime, &input).0)
    }

    pub fn replay(&self, key: &resource_embed::Shape) -> &inference_backend_metal::metal::ReplayProgram {
        self.replay.replay(key)
    }

    pub fn clear(&mut self) {
        self.replay.clear();
    }
}
