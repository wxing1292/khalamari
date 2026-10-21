use crate::attn::gqa::backend::GQAReplayTopology;
use crate::attn::gqa::batch_metadata::GQAReplayShape;

pub mod executor;
pub mod main;
pub mod component_config;

mod mtp;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35GQAReplayKey {
    num_total_tokens: u32,
    num_total_q_token_tiles: u32,
    num_total_sdpa_map_task_templates: u32,
    topology: GQAReplayTopology,
}

impl Qwen35GQAReplayKey {
    pub fn new(shape: GQAReplayShape, topology: GQAReplayTopology) -> Self {
        shape.validate();
        Self {
            num_total_tokens: shape.num_total_tokens,
            num_total_q_token_tiles: shape.num_total_q_token_tiles,
            num_total_sdpa_map_task_templates: shape.num_total_sdpa_map_task_templates,
            topology,
        }
    }
}

pub const QWEN35_PAGE_SIZE_BYTES: usize = 32 * 1024;
