use crate::components::gdn::compute::Buffers;
use crate::components::gdn::compute::Compute;
use crate::components::gdn::compute::ComputeConstants;
use crate::components::gdn::compute::Config;
use crate::components::gdn::compute::Shape;
use crate::components::gdn::compute::recurrent;
use crate::components::gdn::compute::set_recurrent_count;
use crate::components::gdn::compute::set_replay_u32;
use crate::components::gdn::compute::validate_buffers;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Operator;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../../metal/gdn_compute_replay.metal");
const REPLAY_V_ROWS: u32 = 32;
const REPLAY_LANES_PER_ROW: u32 = 8;
const REPLAY_TOKEN_TILE: u32 = 16;

pub fn token_tile(config: Config, max_threadblock_memory_bytes: usize) -> u32 {
    let rows = [REPLAY_V_ROWS, 16, 8, 4]
        .into_iter()
        .find(|&rows| config.v_head_dim.is_multiple_of(rows))
        .unwrap();
    let row_bytes = 2 * (config.qk_head_dim as u64 + 4) * 4 + (rows as u64 + 8) * 2 + 8;
    let preferred = if config.qk_head_dim > 128 { 8 } else { REPLAY_TOKEN_TILE };
    [16, 8, 4, 2, 1]
        .into_iter()
        .find(|&tokens| tokens <= preferred && tokens as u64 * row_bytes <= max_threadblock_memory_bytes as u64)
        .expect("GDN replay token tile must fit device shared memory")
}

fn geometry(constants: ComputeConstants) -> (u32, u32, u32, u32) {
    let rows = [REPLAY_V_ROWS, 16, 8, 4]
        .into_iter()
        .find(|&rows| constants.model.v_head_dim.is_multiple_of(rows))
        .unwrap();
    let lanes = REPLAY_LANES_PER_ROW.min(constants.model.qk_head_dim / 4);
    let groups = (rows * lanes).div_ceil(32);
    let tokens = constants.replay_token_tile;
    (rows, groups, tokens, lanes)
}

pub fn required_threads(constants: ComputeConstants) -> usize {
    geometry(constants).1 as usize * 32
}

/// Per-layer inputs that survive forward until the accepted state is committed.
///
/// Alpha, normalized K, and U use F32. Raw pre-convolution QKV uses BF16.
/// `token_offset` selects this layer in the all-layer log.
/// `num_total_tokens` is the log capacity, independent of the full forward shape.
/// The caller must validate that the active replay suffix fits this capacity.
#[derive(Clone, Copy)]
pub struct ReplayBuffers<'a> {
    pub alpha: &'a Buffer,
    pub k: &'a Buffer,
    pub u: &'a Buffer,
    pub qkv: &'a Buffer,
    pub token_offset: u64,
    pub num_total_tokens: u32,
}

pub fn source(constants: ComputeConstants) -> String {
    let (rows, groups, tokens, lanes) = geometry(constants);
    format!(
        "constant uint replay_lanes_per_row = {lanes}u;\nconstant uint replay_num_v_rows = {rows}u;\nconstant uint \
         replay_num_simdgroups = {groups}u;\nconstant uint replay_token_tile = {tokens}u;\n{SOURCE}",
    )
}

impl Compute {
    /// Recurrent writes final/boundary states. The suffix writes a replay log.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_with_replay<'a>(
        &'a self,
        shape: Shape,
        buffers: Buffers<'a>,
        replay: ReplayBuffers<'a>,
        num_active_reqs: ReplayU32,
        num_active_tokens: ReplayU32,
        num_active_recurrent_requests: ReplayU32,
    ) -> ReplayInvocation<'a> {
        self.constants.validate_shape(shape);
        shape.validate_active(num_active_reqs, num_active_tokens, num_active_recurrent_requests);
        validate_buffers(self.constants, shape, &buffers);
        assert!(replay.num_total_tokens > 0);
        let model = self.constants.model;
        for (buffer, row_bytes) in [
            (replay.alpha, model.num_v_heads as u64 * size_of::<f32>() as u64),
            (
                replay.k,
                model.num_qk_heads as u64 * model.qk_head_dim as u64 * size_of::<f32>() as u64,
            ),
            (
                replay.u,
                model.num_v_heads as u64 * model.v_head_dim as u64 * size_of::<f32>() as u64,
            ),
            (replay.qkv, model.qkv_dim() as u64 * size_of::<u16>() as u64),
        ] {
            validate_log_range(
                buffer.len_bytes_u64(),
                row_bytes,
                replay.token_offset,
                replay.num_total_tokens,
            );
        }
        ReplayInvocation {
            compute: self,
            shape,
            buffers,
            replay,
            num_active_reqs,
            num_active_tokens,
            num_active_recurrent_requests,
        }
    }
}

pub struct ReplayInvocation<'a> {
    compute: &'a Compute,
    shape: Shape,
    buffers: Buffers<'a>,
    replay: ReplayBuffers<'a>,
    num_active_reqs: ReplayU32,
    num_active_tokens: ReplayU32,
    num_active_recurrent_requests: ReplayU32,
}

impl Operator for ReplayInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let compute = self.compute;
        recorder.set_kernel(&compute.short_conv_replay);
        compute.record_short_conv_bindings(
            recorder,
            self.shape,
            &self.buffers,
            self.num_active_reqs,
            self.num_active_tokens,
            false,
        );
        recorder.set_buffer_write(13, self.replay.qkv, 0);
        recorder.set_u64(14, self.replay.token_offset);
        set_recurrent_count(
            recorder,
            15,
            self.num_active_recurrent_requests,
            self.shape.num_total_reqs,
        );
        compute.dispatch_short_conv(recorder, self.shape);
        // Recurrent and replay requests have disjoint state slots and output rows.
        recorder.record_disjoint_buffers(
            &[self.buffers.recurrent_output, self.buffers.recurrent_state_arena],
            || {
                recurrent::record_recurrent_state(
                    compute,
                    recorder,
                    self.shape,
                    &self.buffers,
                    self.num_active_recurrent_requests,
                    true,
                );
                self.record_replay(recorder);
            },
        );
        compute.record_output_norm_gate(recorder, self.shape, &self.buffers, self.num_active_tokens);
    }
}

fn validate_log_range(buffer_bytes: u64, row_bytes: u64, token_offset: u64, num_tokens: u32) {
    // Row widths are positive and bounded by validated model geometry.
    // Compare row counts so neither an offset sum nor a byte product can wrap.
    let num_rows = buffer_bytes / row_bytes;
    assert!(
        token_offset <= num_rows,
        "GDN replay log offset exceeds buffer capacity"
    );
    assert!(
        num_tokens as u64 <= num_rows - token_offset,
        "GDN replay log buffer is too small"
    );
}

impl ReplayInvocation<'_> {
    fn record_replay(&self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(&self.compute.replay);
        recorder.set_buffer_write(0, self.buffers.recurrent_output, 0);
        recorder.set_buffer_read(1, self.buffers.recurrent_state_arena, 0);
        recorder.set_buffer_read(2, self.buffers.conv_qkv, 0);
        recorder.set_buffer_read(3, self.buffers.a, 0);
        recorder.set_buffer_read(4, self.buffers.b, 0);
        recorder.set_buffer_read(5, self.buffers.a_log, 0);
        recorder.set_buffer_read(6, self.buffers.dt_bias, 0);
        recorder.set_buffer_read(7, self.buffers.src_recurrent_state_slots, 0);
        recorder.set_buffer_write(8, self.replay.u, 0);
        recorder.set_buffer_read(9, self.buffers.cu_tokens, 0);
        recorder.set_f32(10, self.compute.q_scale);
        set_replay_u32(recorder, 11, self.num_active_reqs, self.shape.num_total_reqs);
        recorder.set_u64(12, self.buffers.recurrent_state_arena_offset_bytes);
        set_recurrent_count(
            recorder,
            13,
            self.num_active_recurrent_requests,
            self.shape.num_total_reqs,
        );
        recorder.set_buffer_write(14, self.replay.k, 0);
        recorder.set_buffer_write(15, self.replay.alpha, 0);
        recorder.set_u64(16, self.replay.token_offset);
        let (rows, groups, ..) = geometry(self.compute.constants);
        recorder.dispatch_threadblocks(
            (
                (self.compute.constants.model.v_head_dim / rows) as usize,
                self.shape.num_total_reqs as usize * self.compute.constants.model.num_v_heads as usize,
                1,
            ),
            (32, groups as usize, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::validate_log_range;

    #[test]
    fn test_log_range() {
        validate_log_range(64, 4, 12, 4);
        for (bytes, row_bytes, offset, tokens) in [(64, 4, 13, 4), (64, 4, u64::MAX, 1), (64, 4, 1 << 62, 1)] {
            assert!(std::panic::catch_unwind(|| validate_log_range(bytes, row_bytes, offset, tokens)).is_err());
        }
    }
}
