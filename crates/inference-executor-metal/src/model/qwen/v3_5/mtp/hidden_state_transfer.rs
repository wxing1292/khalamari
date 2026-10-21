//! Qwen3.5 MTP hidden-state routing and retained-tail writeback.
//!
//! One flat BF16 buffer stores the persistent state for all request slots and
//! non-final MTP modules. Before module `m` runs, this component gathers its
//! previous-hidden input. The source is Main, the previous module output, or
//! one old boundary row from the persistent buffer. The component then writes
//! the retained tail from module `m - 1` back to that buffer.
//!
//! The gather must complete before the in-place write. This order preserves an
//! old boundary row when the read and write use the same module range. A second
//! buffer is not necessary.

use std::ops::Range;

use inference_backend_metal::components::ReplayOp;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayParameterKey;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::row_route;
use inference_backend_metal::operators::row_scatter;
use inference_executor_core::replay::ReplayBucketPolicy;

use crate::replay::ReplayComponent;

const QWEN35_MTP_HIDDEN_STATE_TRANSFER_NUM_ACTIVE_ROWS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3.5.mtp_hidden_state_transfer.num_active_rows");
const QWEN35_MTP_HIDDEN_STATE_TRANSFER_NUM_ACTIVE_WRITE_ROWS: ReplayParameterKey =
    ReplayParameterKey::new("qwen3.5.mtp_hidden_state_transfer.num_active_write_rows");

const MAIN_HIDDEN_SOURCE: u32 = 0;
const PREVIOUS_MODULE_HIDDEN_SOURCE: u32 = 1;
const HIDDEN_STATE_CACHE_SOURCE: u32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35MTPPreviousHiddenSource {
    Main { output_row_offset: usize },
    HiddenStateCache { cache_row_offset: usize },
    PreviousModule { output_row_offset: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35MTPHiddenStateTransferPlan {
    module_index: usize,
    num_input_rows: usize,
    num_reused_tokens: usize,
}

impl Qwen35MTPHiddenStateTransferPlan {
    pub fn new(module_index: usize, num_input_rows: usize, num_reused_tokens: usize) -> Self {
        debug_assert!(num_input_rows > 0);
        debug_assert!(num_reused_tokens <= module_index);
        Self {
            module_index,
            num_input_rows,
            num_reused_tokens,
        }
    }

    pub fn num_input_rows(self) -> usize {
        self.num_input_rows
    }

    pub fn previous_hidden_source(self, input_row_offset: usize) -> Qwen35MTPPreviousHiddenSource {
        debug_assert!(input_row_offset < self.num_input_rows());
        if self.module_index == 0 {
            return Qwen35MTPPreviousHiddenSource::Main {
                output_row_offset: input_row_offset,
            };
        }
        if self.module_index > self.num_reused_tokens && input_row_offset == 0 {
            return Qwen35MTPPreviousHiddenSource::HiddenStateCache {
                cache_row_offset: self.num_reused_tokens,
            };
        }
        Qwen35MTPPreviousHiddenSource::PreviousModule {
            output_row_offset: if self.module_index > self.num_reused_tokens {
                input_row_offset - 1
            } else {
                input_row_offset
            },
        }
    }

    pub fn append_routes(
        self,
        routes: &mut Vec<u32>,
        main_hidden_flat_start: usize,
        previous_module_flat_start: usize,
        hidden_state_cache_rows: Range<u32>,
    ) {
        for input_row_offset in 0..self.num_input_rows() {
            let (source, source_row) = match self.previous_hidden_source(input_row_offset) {
                Qwen35MTPPreviousHiddenSource::Main { output_row_offset } => {
                    (MAIN_HIDDEN_SOURCE, (main_hidden_flat_start + output_row_offset) as u32)
                },
                Qwen35MTPPreviousHiddenSource::HiddenStateCache { cache_row_offset } => {
                    debug_assert!(cache_row_offset < hidden_state_cache_rows.len());
                    (
                        HIDDEN_STATE_CACHE_SOURCE,
                        hidden_state_cache_rows.start + cache_row_offset as u32,
                    )
                },
                Qwen35MTPPreviousHiddenSource::PreviousModule { output_row_offset } => {
                    (
                        PREVIOUS_MODULE_HIDDEN_SOURCE,
                        (previous_module_flat_start + output_row_offset) as u32,
                    )
                },
            };
            routes.extend_from_slice(&[source, source_row]);
        }
    }
}

pub fn append_mtp_hidden_state_cache_write_routes(
    routes: &mut Vec<u32>,
    hidden_state_cache_rows: Range<u32>,
    input_flat_start: usize,
    num_input_rows: usize,
) {
    debug_assert!(num_input_rows > 0);
    let num_write_rows = hidden_state_cache_rows.len();
    debug_assert!(num_input_rows >= num_write_rows);
    let input_start = input_flat_start + num_input_rows - num_write_rows;
    let cache_start = hidden_state_cache_rows.start;
    for row_offset in 0..num_write_rows {
        routes.extend_from_slice(&[(input_start + row_offset) as u32, cache_start + row_offset as u32]);
    }
}

pub struct Qwen35MTPHiddenStateTransfer {
    route: row_route::Kernel,
    scatter: row_scatter::Kernel,
    replay_bucket_policy: ReplayBucketPolicy,
}

#[derive(Clone, Copy)]
pub struct Qwen35MTPHiddenStateTransferArgs<'a> {
    pub num_rows: u32,
    pub main_hidden_input: &'a Buffer,
    pub previous_module_hidden_input: &'a Buffer,
    pub hidden_state_cache_input: &'a Buffer,
    pub routes: &'a Buffer,
    pub previous_hidden_output: &'a Buffer,
    pub num_write_rows: u32,
    pub write_input: &'a Buffer,
    pub write_routes: &'a Buffer,
    pub hidden_state_cache_output: &'a Buffer,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen35MTPHiddenStateTransferReplayKey {
    num_total_rows: u32,
    num_total_write_rows: u32,
}

impl Qwen35MTPHiddenStateTransfer {
    pub fn new(device: &Device, hidden_dim: usize, max_tokens: usize) -> Self {
        let hidden_dim = u32::try_from(hidden_dim).expect("qwen3.5 MTP hidden dimension must fit u32");
        let max_tokens = u32::try_from(max_tokens).expect("qwen3.5 MTP hidden-state transfer capacity must fit u32");
        let config = row_route::Config {
            num_cols: hidden_dim,
            dtype: Dtype::Bfloat16,
        };
        Self {
            route: row_route::Kernel::new(device, config),
            scatter: row_scatter::Kernel::new(
                device,
                row_scatter::Config {
                    num_cols: hidden_dim,
                    dtype: Dtype::Bfloat16,
                },
            ),
            replay_bucket_policy: ReplayBucketPolicy::new(max_tokens),
        }
    }

    pub fn prepare_replay(
        &self,
        num_active_rows: u32,
        num_active_write_rows: u32,
    ) -> (Qwen35MTPHiddenStateTransferReplayKey, ReplayArguments) {
        let key = self.replay_key_for_active_rows(num_active_rows, num_active_write_rows);
        let mut arguments =
            ReplayArguments::new().with_u32(QWEN35_MTP_HIDDEN_STATE_TRANSFER_NUM_ACTIVE_ROWS, num_active_rows);
        if num_active_write_rows > 0 {
            arguments = arguments.with_u32(
                QWEN35_MTP_HIDDEN_STATE_TRANSFER_NUM_ACTIVE_WRITE_ROWS,
                num_active_write_rows,
            );
        }
        (key, arguments)
    }

    pub fn record<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        key: &Qwen35MTPHiddenStateTransferReplayKey,
        num_active_rows: ReplayU32,
        num_active_write_rows: ReplayU32,
        args: Qwen35MTPHiddenStateTransferArgs<'a>,
    ) {
        debug_assert!(args.num_rows > 0);
        validate_active_rows(
            num_active_rows,
            args.num_rows,
            key.num_total_rows,
            &self.replay_bucket_policy,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(self.route.invoke(
            row_route::Shape {
                num_total_rows: key.num_total_rows,
            },
            num_active_rows,
            row_route::Buffers {
                first_input: args.main_hidden_input,
                second_input: args.previous_module_hidden_input,
                third_input: args.hidden_state_cache_input,
                routes: args.routes,
                output: args.previous_hidden_output,
            },
        )));
        if key.num_total_write_rows == 0 {
            debug_assert_eq!(args.num_write_rows, 0);
            return;
        }
        validate_active_rows(
            num_active_write_rows,
            args.num_write_rows,
            key.num_total_write_rows,
            &self.replay_bucket_policy,
        );
        recorder.record_with_barrier_before(ReplayOp::opaque(self.scatter.invoke(
            row_scatter::Shape {
                num_total_rows: key.num_total_write_rows,
            },
            num_active_write_rows,
            row_scatter::Buffers {
                input: args.write_input,
                routes: args.write_routes,
                output: args.hidden_state_cache_output,
            },
        )));
    }

    fn replay_key_for_active_rows(
        &self,
        num_active_rows: u32,
        num_active_write_rows: u32,
    ) -> Qwen35MTPHiddenStateTransferReplayKey {
        let num_total_write_rows = if num_active_write_rows > 0 {
            self.replay_bucket_policy.capacity(num_active_write_rows)
        } else {
            0
        };
        Qwen35MTPHiddenStateTransferReplayKey {
            num_total_rows: self.replay_bucket_policy.capacity(num_active_rows),
            num_total_write_rows,
        }
    }
}

impl ReplayComponent for Qwen35MTPHiddenStateTransfer {
    type Key = Qwen35MTPHiddenStateTransferReplayKey;
    type Input<'a> = Qwen35MTPHiddenStateTransferArgs<'a>;

    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        self.replay_key_for_active_rows(input.num_rows, input.num_write_rows)
    }

    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let key = self.replay_key(input);
        let num_active_write_rows = if input.num_write_rows == 0 {
            ReplayU32::Fixed(0)
        } else {
            ReplayU32::Parameter(QWEN35_MTP_HIDDEN_STATE_TRANSFER_NUM_ACTIVE_WRITE_ROWS)
        };
        Qwen35MTPHiddenStateTransfer::record(
            self,
            recorder,
            &key,
            ReplayU32::Parameter(QWEN35_MTP_HIDDEN_STATE_TRANSFER_NUM_ACTIVE_ROWS),
            num_active_write_rows,
            *input,
        );
    }
}

fn validate_active_rows(active: ReplayU32, input_rows: u32, total_rows: u32, policy: &ReplayBucketPolicy) {
    match active {
        ReplayU32::Fixed(value) => {
            debug_assert_eq!(value, input_rows);
            debug_assert_eq!(value, total_rows);
        },
        ReplayU32::Parameter(_) => {
            debug_assert_eq!(policy.capacity(input_rows), total_rows);
        },
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::ReplayExecution;
    use inference_backend_metal::metal::Stream;

    use super::*;
    use crate::replay::Replay;

    #[test]
    fn test_decode_routes_cover_reject_partial_and_full_accept() {
        assert_sources(0, 0, &[Qwen35MTPPreviousHiddenSource::Main { output_row_offset: 0 }]);
        assert_sources(
            0,
            2,
            &[
                Qwen35MTPPreviousHiddenSource::HiddenStateCache { cache_row_offset: 0 },
                Qwen35MTPPreviousHiddenSource::PreviousModule { output_row_offset: 0 },
                Qwen35MTPPreviousHiddenSource::PreviousModule { output_row_offset: 1 },
            ],
        );
        assert_sources(
            1,
            2,
            &[
                Qwen35MTPPreviousHiddenSource::HiddenStateCache { cache_row_offset: 1 },
                Qwen35MTPPreviousHiddenSource::PreviousModule { output_row_offset: 0 },
                Qwen35MTPPreviousHiddenSource::PreviousModule { output_row_offset: 1 },
            ],
        );
        assert_sources(
            3,
            2,
            &[
                Qwen35MTPPreviousHiddenSource::PreviousModule { output_row_offset: 0 },
                Qwen35MTPPreviousHiddenSource::PreviousModule { output_row_offset: 1 },
                Qwen35MTPPreviousHiddenSource::PreviousModule { output_row_offset: 2 },
                Qwen35MTPPreviousHiddenSource::PreviousModule { output_row_offset: 3 },
            ],
        );
    }

    #[test]
    fn test_mtp0_has_no_cache_reads_and_final_module_has_no_writes() {
        for num_continuation_tokens in [0, 1, 3] {
            let plan = Qwen35MTPHiddenStateTransferPlan::new(0, num_continuation_tokens + 1, 0);
            assert!((0..plan.num_input_rows()).all(|row| {
                matches!(
                    plan.previous_hidden_source(row),
                    Qwen35MTPPreviousHiddenSource::Main { .. }
                )
            }));
        }
        let mut routes = Vec::new();
        append_mtp_hidden_state_cache_write_routes(&mut routes, 0..0, 7, 4);
        assert!(routes.is_empty());
    }

    #[test]
    fn test_decode_write_routes_replace_the_entire_retained_tail() {
        let mut routes = Vec::new();
        append_mtp_hidden_state_cache_write_routes(&mut routes, 10..13, 5, 4);
        assert_eq!(routes, vec![6, 10, 7, 11, 8, 12]);
    }

    #[test]
    fn test_transfer_reads_old_boundary_before_in_place_writeback() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = &stream;
        for num_continuation_tokens in [0, 1, 3] {
            run_transfer_case(&device, runtime, num_continuation_tokens);
        }
    }

    #[test]
    fn test_replay_repairs_hidden_inputs_with_and_without_writeback() {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let runtime = &stream;
        let mut replay = Replay::new(
            "qwen3.5 MTP tail repair test",
            Qwen35MTPHiddenStateTransfer::new(&device, 1, 8),
        );
        let main_hidden = Buffer::from_slice(&device, &[30_u16; 8]);
        let previous_values = [100_u16, 101, 102, 103, 104, 105, 106, 107];
        let previous_hidden = Buffer::from_slice(&device, &previous_values);
        let cache = Buffer::from_slice(&device, &[9_u16, 10]);
        let routes = Buffer::new_zeroed_elements(&device, 16, Dtype::Uint32);
        let write_routes = Buffer::new_zeroed_elements(&device, 4, Dtype::Uint32);
        let output = Buffer::from_slice(&device, &[999_u16; 8]);
        let mut seen = std::collections::HashSet::new();
        for module_index in [1, 2] {
            // Normal Prefill, repaired Prefill, and repaired Decode share routing.
            for (repair_tail, write_tail) in [(false, false), (true, false), (true, true)] {
                for window in 1..=5 {
                    cache.write_typed(0, &[9_u16, 10]);
                    output.write_typed(0, &[999_u16; 8]);
                    let num_rows = window + if repair_tail { module_index } else { 0 };
                    let plan = Qwen35MTPHiddenStateTransferPlan::new(
                        module_index,
                        num_rows,
                        if repair_tail { 0 } else { module_index },
                    );
                    let mut row_routes = Vec::new();
                    plan.append_routes(&mut row_routes, 0, 0, 0..module_index as u32);
                    routes.write_typed(0, &row_routes);
                    let mut writes = Vec::new();
                    if write_tail {
                        append_mtp_hidden_state_cache_write_routes(
                            &mut writes,
                            0..module_index as u32,
                            0,
                            num_rows - 1,
                        );
                        write_routes.write_typed(0, &writes);
                    }
                    let num_write_rows = (writes.len() / 2) as u32;
                    let input = Qwen35MTPHiddenStateTransferArgs {
                        num_rows: num_rows as u32,
                        main_hidden_input: &main_hidden,
                        previous_module_hidden_input: &previous_hidden,
                        hidden_state_cache_input: &cache,
                        routes: &routes,
                        previous_hidden_output: &output,
                        num_write_rows,
                        write_input: &previous_hidden,
                        write_routes: &write_routes,
                        hidden_state_cache_output: &cache,
                    };
                    let (key, hit) = replay.record(runtime, &input);
                    assert_eq!(hit, !seen.insert(key.clone()));
                    let (_, arguments) = replay.component().prepare_replay(num_rows as u32, num_write_rows);
                    runtime
                        .submit_replay_sequence(&[ReplayExecution::new(replay.replay(&key), &arguments)])
                        .wait();
                    let mut expected = Vec::new();
                    if repair_tail {
                        expected.push(9);
                    }
                    expected.extend_from_slice(&previous_values[..num_rows - expected.len()]);
                    expected.resize(8, 999);
                    assert_eq!(output.read_typed::<u16>(0, 8), expected);
                    let mut expected_cache = vec![9, 10];
                    if write_tail {
                        expected_cache[..module_index].copy_from_slice(&previous_values[window - 1..num_rows - 1]);
                    }
                    assert_eq!(cache.read_typed::<u16>(0, 2), expected_cache);
                }
            }
        }
    }

    fn assert_sources(num_continuation_tokens: usize, module_index: usize, expected: &[Qwen35MTPPreviousHiddenSource]) {
        let plan = Qwen35MTPHiddenStateTransferPlan::new(
            module_index,
            num_continuation_tokens.max(module_index) + 1,
            num_continuation_tokens.min(module_index),
        );
        assert_eq!(plan.num_input_rows(), expected.len());
        for (row_offset, &source) in expected.iter().enumerate() {
            assert_eq!(plan.previous_hidden_source(row_offset), source);
        }
    }

    fn run_transfer_case(device: &Device, runtime: &Stream, num_continuation_tokens: usize) {
        let hidden_state_cache = Buffer::from_slice(device, &[9_u16, 10, 11]);
        let module0_values = (100_u16..100 + num_continuation_tokens as u16 + 1).collect::<Vec<_>>();
        let module0_hidden = Buffer::from_slice(device, &module0_values);
        let module1_previous_hidden = run_transfer(
            device,
            runtime,
            num_continuation_tokens,
            1,
            0..1,
            &module0_hidden,
            &hidden_state_cache,
        );
        let expected_module1_previous_hidden = if num_continuation_tokens == 0 {
            vec![9, 100]
        } else {
            module0_values.clone()
        };
        assert_eq!(module1_previous_hidden, expected_module1_previous_hidden);

        let num_module1_rows = num_continuation_tokens.max(1) + 1;
        let module1_values = (200_u16..200 + num_module1_rows as u16).collect::<Vec<_>>();
        let module1_hidden = Buffer::from_slice(device, &module1_values);
        let module2_previous_hidden = run_transfer(
            device,
            runtime,
            num_continuation_tokens,
            2,
            1..3,
            &module1_hidden,
            &hidden_state_cache,
        );
        let mut expected_module2_previous_hidden = Vec::new();
        if num_continuation_tokens < 2 {
            expected_module2_previous_hidden.push(10 + num_continuation_tokens as u16);
        }
        expected_module2_previous_hidden.extend_from_slice(
            &module1_values[..module2_previous_hidden.len() - expected_module2_previous_hidden.len()],
        );
        assert_eq!(module2_previous_hidden, expected_module2_previous_hidden);

        let expected_cache = [
            module0_values[module0_values.len() - 1],
            module1_values[module1_values.len() - 2],
            module1_values[module1_values.len() - 1],
        ];
        assert_eq!(
            hidden_state_cache.read_typed::<u16>(0, 3),
            expected_cache,
            "continuation tokens={num_continuation_tokens}"
        );
    }

    fn run_transfer(
        device: &Device,
        runtime: &Stream,
        num_continuation_tokens: usize,
        module_index: usize,
        hidden_state_cache_rows: Range<u32>,
        previous_module_hidden: &Buffer,
        hidden_state_cache: &Buffer,
    ) -> Vec<u16> {
        let plan = Qwen35MTPHiddenStateTransferPlan::new(
            module_index,
            num_continuation_tokens.max(module_index) + 1,
            num_continuation_tokens.min(module_index),
        );
        let num_rows = plan.num_input_rows();
        let main_hidden = Buffer::from_slice(device, &[30_u16; 4]);
        let mut routes = Vec::new();
        plan.append_routes(&mut routes, 0, 0, hidden_state_cache_rows.clone());
        routes.resize(8, 0);
        let routes = Buffer::from_slice(device, &routes);
        let previous_hidden_output = Buffer::new_zeroed_elements(device, 4, Dtype::Bfloat16);
        let mut write_routes = Vec::new();
        let num_previous_module_rows = num_continuation_tokens.max(module_index - 1) + 1;
        append_mtp_hidden_state_cache_write_routes(
            &mut write_routes,
            hidden_state_cache_rows,
            0,
            num_previous_module_rows,
        );
        let num_write_rows = (write_routes.len() / 2) as u32;
        let write_routes = Buffer::from_slice(device, &write_routes);
        let transfer = Qwen35MTPHiddenStateTransfer::new(device, 1, 4);
        let input = Qwen35MTPHiddenStateTransferArgs {
            num_rows: num_rows as u32,
            main_hidden_input: &main_hidden,
            previous_module_hidden_input: previous_module_hidden,
            hidden_state_cache_input: hidden_state_cache,
            routes: &routes,
            previous_hidden_output: &previous_hidden_output,
            num_write_rows,
            write_input: previous_module_hidden,
            write_routes: &write_routes,
            hidden_state_cache_output: hidden_state_cache,
        };
        let mut replay = Replay::new("qwen3.5 MTP hidden-state transfer test", transfer);
        let (key, _) = replay.record(runtime, &input);
        let (_, arguments) = replay.component().prepare_replay(num_rows as u32, num_write_rows);
        runtime
            .submit_replay_sequence(&[ReplayExecution::new(replay.replay(&key), &arguments)])
            .wait();

        previous_hidden_output.read_typed::<u16>(0, num_rows)
    }
}
