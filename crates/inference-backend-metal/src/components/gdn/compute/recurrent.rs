use crate::components::gdn::compute::Buffers;
use crate::components::gdn::compute::Compute;
use crate::components::gdn::compute::ComputeConstants;
use crate::components::gdn::compute::RECURRENT_NUM_QK_DIM_THREADS;
use crate::components::gdn::compute::Shape;
use crate::components::gdn::compute::set_recurrent_count;
use crate::metal::CommandRecorder;
use crate::metal::ReplayU32;

const SOURCE: &str = include_str!("../../metal/gdn_compute_recurrent.metal");

pub fn source(constants: ComputeConstants) -> String {
    let thread_block = constants.kernels.recurrent_state.thread_block;
    format!(
        "constant uint recurrent_token_chunk_size = {num_tokens}u;\nconstant uint recurrent_state_num_v_rows = \
         {num_v_rows}u;\nconstant uint recurrent_state_num_simdgroups = {num_simdgroups}u;\nconstant uint \
         recurrent_state_num_qk_dim_threads = {RECURRENT_NUM_QK_DIM_THREADS}u;\n{SOURCE}",
        num_tokens = thread_block.num_tokens(),
        num_v_rows = thread_block.num_v_rows(),
        num_simdgroups = thread_block.num_simdgroups(),
    )
}

pub fn record_recurrent_state(
    compute: &Compute,
    recorder: &CommandRecorder,
    shape: Shape,
    buffers: &Buffers<'_>,
    num_active_recurrent_requests: ReplayU32,
    write_candidate_states: bool,
) {
    recorder.set_kernel(&compute.recurrent_state);
    recorder.set_barrier_before();
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
    set_recurrent_count(recorder, 11, num_active_recurrent_requests, shape.num_total_reqs);
    recorder.set_u64(12, buffers.recurrent_state_arena_offset_bytes);
    recorder.set_u32(13, u32::from(write_candidate_states));
    let thread_block = compute.constants.kernels.recurrent_state.thread_block;
    recorder.dispatch_threadblocks(
        (
            (compute.constants.model.v_head_dim / thread_block.num_v_rows()) as usize,
            shape.num_total_reqs as usize * compute.constants.model.num_v_heads as usize,
            1,
        ),
        (
            RECURRENT_NUM_QK_DIM_THREADS as usize,
            thread_block.num_simdgroups() as usize,
            1,
        ),
    );
}
