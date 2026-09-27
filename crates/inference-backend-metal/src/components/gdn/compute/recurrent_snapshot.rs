use crate::components::gdn::compute::Buffers;
use crate::components::gdn::compute::Compute;
use crate::components::gdn::compute::ComputeConstants;
use crate::components::gdn::compute::Shape;
use crate::components::gdn::compute::set_recurrent_count;
use crate::components::gdn::compute::set_replay_u32;
use crate::metal::CommandRecorder;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../../metal/gdn_compute_recurrent_snapshot.metal");

pub fn source(constants: ComputeConstants) -> String {
    let final_state = constants.kernels.final_recurrent_state.thread_block;
    let candidate_state = constants.kernels.candidate_recurrent_state.thread_block;
    format!(
        "constant uint final_recurrent_state_num_v_rows = {final_num_v_rows}u;\nconstant uint \
         final_recurrent_state_num_qk_dim_threads = {final_num_qk_dim_threads}u;\nconstant uint \
         candidate_recurrent_state_num_qk_dim_threads = {candidate_num_qk_dim_threads}u;\nconstant uint \
         candidate_recurrent_state_num_v_rows_per_simdgroup = {candidate_num_v_rows_per_simdgroup}u;\nconstant uint \
         candidate_recurrent_state_num_simdgroups = {candidate_num_simdgroups}u;\n{SOURCE}",
        final_num_v_rows = final_state.num_v_rows,
        final_num_qk_dim_threads = final_state.num_qk_dim_threads,
        candidate_num_qk_dim_threads = candidate_state.num_qk_dim_threads,
        candidate_num_v_rows_per_simdgroup = candidate_state.simdgroup.num_v_rows,
        candidate_num_simdgroups = candidate_state.num_simdgroups,
    )
}

/// Current final-state recurrent execution (`R = num_reqs`):
///
/// ```text
/// recurrent_state: [S, Hv, Dv, Dqk]  (Dqk contiguous)
/// grid:             (Dv / num_v_rows, R * Hv, 1)
/// threadblock:      (num_qk_dim_threads, num_v_rows, 1)
/// FinalRecurrentStateThreadBlockTask / threadblock
///   -> owns recurrent_state[slot, v_head_index, v_dim_indices, 0..Dqk]
///   -> advances it over flat_token_indices in order
/// task from grid: request_index, v_head_index, v_dim_indices
/// task from metadata: flat_token_indices
/// parallel: requests, V heads, V-row ranges, Dqk lanes
/// ordered:  tokens within one request
/// produces: recurrent_output; updates: destination recurrent_state slice
/// ```
///
/// The kernel derives the task from its arguments, thread-block index, and
/// constants. It does not require a materialized task buffer.
pub fn record_final_recurrent_state(
    compute: &Compute,
    recorder: &CommandRecorder,
    shape: Shape,
    buffers: &Buffers<'_>,
    num_active_reqs: ReplayU32,
    num_active_recurrent_requests: ReplayU32,
) {
    recorder.set_kernel(&compute.final_recurrent_state);
    recorder.set_buffer_write(0, buffers.recurrent_output, 0);
    recorder.set_buffer_read_write(1, buffers.recurrent_state_arena, 0);
    recorder.set_buffer_read(2, buffers.conv_qkv, 0);
    recorder.set_buffer_read(3, buffers.a, 0);
    recorder.set_buffer_read(4, buffers.b, 0);
    recorder.set_buffer_read(5, buffers.a_log, 0);
    recorder.set_buffer_read(6, buffers.dt_bias, 0);
    recorder.set_buffer_read(7, buffers.src_recurrent_state_slots, 0);
    recorder.set_buffer_read(8, buffers.flat_recurrent_state_write_slots, 0);
    recorder.set_buffer_read(9, buffers.cu_tokens, 0);
    recorder.set_f32(10, compute.q_scale);
    set_replay_u32(
        recorder,
        11,
        num_active_reqs,
        shape.num_total_reqs,
        "GDN active request count",
    );
    recorder.set_u64(12, buffers.recurrent_state_arena_offset_bytes);
    set_recurrent_count(recorder, 13, num_active_recurrent_requests, shape.num_total_reqs);
    let thread_block = compute.constants.kernels.final_recurrent_state.thread_block;
    let num_v_row_ranges = compute.constants.model.v_head_dim / thread_block.num_v_rows;
    recorder.dispatch_threadblocks(
        (
            num_v_row_ranges as usize,
            shape.num_total_reqs as usize * compute.constants.model.num_v_heads as usize,
            1,
        ),
        (
            thread_block.num_qk_dim_threads as usize,
            thread_block.num_v_rows as usize,
            1,
        ),
    );
}

/// Current candidate-state recurrent execution:
///
/// ```text
/// grid:        (Dv / num_v_rows, R * Hv, 1)
/// threadblock: (num_qk_dim_threads, num_simdgroups, 1)
/// CandidateRecurrentStateThreadBlockTask / threadblock
///   -> owns recurrent_state[slot, v_head_index, v_dim_indices, 0..Dqk]
///   -> advances flat_token_indices in order
///   -> can materialize the state after each token
/// ```
///
/// Each SIMDgroup owns `simdgroup.num_v_rows` rows. The full thread block
/// owns `thread_block.num_v_rows()` rows. The kernel derives the task from
/// its arguments, thread-block index, and constants.
pub fn record_candidate_recurrent_state(
    compute: &Compute,
    recorder: &CommandRecorder,
    shape: Shape,
    buffers: &Buffers<'_>,
    num_active_reqs: ReplayU32,
    num_active_recurrent_requests: ReplayU32,
) {
    recorder.set_kernel(&compute.candidate_recurrent_state);
    recorder.set_buffer_write(0, buffers.recurrent_output, 0);
    recorder.set_buffer_read_write(1, buffers.recurrent_state_arena, 0);
    recorder.set_buffer_read(2, buffers.conv_qkv, 0);
    recorder.set_buffer_read(3, buffers.a, 0);
    recorder.set_buffer_read(4, buffers.b, 0);
    recorder.set_buffer_read(5, buffers.a_log, 0);
    recorder.set_buffer_read(6, buffers.dt_bias, 0);
    recorder.set_buffer_read(7, buffers.src_recurrent_state_slots, 0);
    recorder.set_buffer_read(8, buffers.flat_recurrent_state_write_slots, 0);
    recorder.set_buffer_read(9, buffers.cu_tokens, 0);
    recorder.set_f32(10, compute.q_scale);
    set_replay_u32(
        recorder,
        11,
        num_active_reqs,
        shape.num_total_reqs,
        "GDN active request count",
    );
    recorder.set_u64(12, buffers.recurrent_state_arena_offset_bytes);
    set_recurrent_count(recorder, 13, num_active_recurrent_requests, shape.num_total_reqs);
    let thread_block = compute.constants.kernels.candidate_recurrent_state.thread_block;
    let num_threadblocks = compute.constants.model.v_head_dim / thread_block.num_v_rows();
    recorder.dispatch_threadblocks(
        (
            num_threadblocks as usize,
            shape.num_total_reqs as usize * compute.constants.model.num_v_heads as usize,
            1,
        ),
        (
            thread_block.num_qk_dim_threads as usize,
            thread_block.num_simdgroups as usize,
            1,
        ),
    );
}
