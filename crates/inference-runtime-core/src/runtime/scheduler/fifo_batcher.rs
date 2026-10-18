use std::cmp::min;
use std::marker::PhantomData;

use ahash::AHashMap;

use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::CompletionReason;
use crate::runtime::RawRequestID;
use crate::runtime::scheduler::Batcher;
use crate::runtime::scheduler::CancelResult;
use crate::runtime::scheduler::CommitResult;
use crate::runtime::scheduler::ComputePhase;
use crate::runtime::scheduler::PrepareResult;
use crate::runtime::scheduler::ScheduleQueue;
use crate::runtime::scheduler::UserRequest;

pub struct FIFOBatcher<UserReq, DeviceReq, DeviceResp> {
    running_reqs: Vec<UserReq>,

    phantom_data_dev_req: PhantomData<DeviceReq>,
    phantom_data_dev_resp: PhantomData<DeviceResp>,
}

impl<UserReq, DeviceReq, DeviceResp> FIFOBatcher<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    pub fn new() -> Self {
        Self {
            running_reqs: Vec::new(),
            phantom_data_dev_req: PhantomData,
            phantom_data_dev_resp: PhantomData,
        }
    }
}

impl<UserReq, DeviceReq, DeviceResp> Batcher<UserReq, DeviceReq, DeviceResp>
    for FIFOBatcher<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    fn prepare(
        &mut self,
        mut req_budget: usize,
        mut token_budget: usize,
        max_token_per_req: usize,
        mut sticky_token_budgets: AHashMap<RawRequestID, usize>,
        schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
    ) -> Vec<DeviceReq> {
        debug_assert!(0 < req_budget, "fifo batcher requires a positive request budget");
        debug_assert!(0 < token_budget, "fifo batcher requires a positive token budget");
        debug_assert!(
            0 < max_token_per_req,
            "fifo batcher requires a positive per-request token budget"
        );
        debug_assert!(self.running_reqs.is_empty(), "fifo batcher scratch must be empty");
        self.running_reqs.reserve(req_budget);

        let mut dev_reqs = Vec::with_capacity(req_budget);
        'prepare_loop: while let Some(mut user_req) = schedule_queue.pop_front() {
            if req_budget == 0 || token_budget == 0 {
                self.running_reqs.push(user_req);
                break 'prepare_loop;
            }
            let req_id = user_req.id();
            let token_estimate = match sticky_token_budgets.get(&req_id).copied() {
                Some(token_budget) => token_budget,
                None => {
                    // With PP > 1, a final Prefill can commit while a later Decode for the same request remains in
                    // flight. The request then stays in the ID map without a run-queue entry, so its sticky token
                    // budget remains unused.
                    user_req
                        .token_estimate()
                        .token_consumption(min(max_token_per_req, token_budget))
                },
            };
            debug_assert!(
                token_estimate <= token_budget,
                "planned request token budget exceeds remaining batch token budget"
            );
            if token_estimate == 0 {
                self.running_reqs.push(user_req);
                continue 'prepare_loop;
            }
            match user_req.prepare(token_estimate) {
                PrepareResult::ResourceLimitExceeded => {
                    if let Some(preempted_req) = pop_preemption_candidate(schedule_queue) {
                        // TODO uninit and put to async queue
                        tracing::warn!(
                            target: "inference-runtime-core::scheduler",
                            phase = "request.preempted",
                            request_id = preempted_req.id(),
                            "request aborted due to preemption"
                        );
                        drop(preempted_req);
                        schedule_queue.push_front(user_req);
                        continue 'prepare_loop;
                    } else {
                        // TODO uninit and put to async queue
                        tracing::warn!(
                            target: "inference-runtime-core::scheduler",
                            phase = "request.resource_limit",
                            request_id = user_req.id(),
                            "request aborted due to insufficient memory"
                        );
                        drop(user_req);
                        break 'prepare_loop;
                    }
                },
                PrepareResult::BlockingAsyncTask { req } => {
                    sticky_token_budgets.remove(&req_id);
                    schedule_queue.insert(user_req);
                    if schedule_queue.handle_async_task_req(req).is_err() {
                        panic!("async task request queue must have available capacity");
                    }
                    continue 'prepare_loop;
                },
                PrepareResult::NonblockingAsyncTask { req } => {
                    schedule_queue.push_front(user_req);
                    if schedule_queue.handle_async_task_req(req).is_err() {
                        panic!("async task request queue must have available capacity");
                    }
                    continue 'prepare_loop;
                },
                PrepareResult::Await { wait } => {
                    sticky_token_budgets.remove(&req_id);
                    schedule_queue.push_waiting_reqs(user_req, wait);
                    continue 'prepare_loop;
                },
                PrepareResult::Skip => {
                    sticky_token_budgets.remove(&req_id);
                    schedule_queue.insert(user_req);
                },
                PrepareResult::Continue { dev_req, compute_phase } => {
                    sticky_token_budgets.remove(&req_id);
                    let request_cost = dev_req.request_cost();
                    let token_cost = dev_req.token_cost();
                    debug_assert!(
                        request_cost <= req_budget,
                        "prepared request cost exceeds fifo batch request budget"
                    );
                    debug_assert!(
                        token_cost <= token_estimate,
                        "prepared request token cost exceeds assigned request token budget"
                    );
                    debug_assert!(
                        token_cost <= token_budget,
                        "prepared request token cost exceeds fifo batch token budget"
                    );
                    req_budget -= request_cost;
                    token_budget -= token_cost;
                    match compute_phase {
                        ComputePhase::Prefill { .. } => self.running_reqs.push(user_req),
                        ComputePhase::Decode { .. } => schedule_queue.insert(user_req),
                    }
                    dev_reqs.push(dev_req);
                },
                PrepareResult::Terminal => {
                    sticky_token_budgets.remove(&req_id);
                },
            }
        }
        for user_req in self.running_reqs.drain(..).rev() {
            schedule_queue.push_front(user_req);
        }

        debug_assert!(self.running_reqs.is_empty(), "fifo batcher scratch must be drained");
        dev_reqs
    }

    fn cancel(&mut self, schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>, dev_reqs: Vec<DeviceReq>) {
        debug_assert!(self.running_reqs.is_empty(), "fifo batcher scratch must be empty");
        for dev_req in dev_reqs {
            let req_id = dev_req.id();
            let mut user_req = schedule_queue
                .remove(&req_id)
                .expect("fifo batch cancellation requires a matching request");
            match user_req.cancel(dev_req) {
                CancelResult::Continue => schedule_queue.push_front(user_req),
                CancelResult::Pending => schedule_queue.insert(user_req),
                CancelResult::Terminal => { /* noop */ },
            }
        }
    }

    fn commit(
        &mut self,
        schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
        dev_resps: Vec<DeviceResp>,
    ) -> Vec<(UserReq, CompletionReason)> {
        debug_assert!(self.running_reqs.is_empty(), "fifo batcher scratch must be empty");
        let mut completed_turns = Vec::new();
        for dev_resp in dev_resps {
            let req_id = dev_resp.id();
            let mut user_req = schedule_queue
                .remove(&req_id)
                .expect("fifo batch commit requires a matching request");
            match user_req.commit(dev_resp) {
                CommitResult::Continue => schedule_queue.push_front(user_req),
                CommitResult::Pending => schedule_queue.insert(user_req),
                CommitResult::TurnCompleted(reason) => completed_turns.push((user_req, reason)),
                CommitResult::Terminal => { /* noop */ },
            }
        }
        completed_turns
    }
}

fn pop_preemption_candidate<UserReq, DeviceReq, DeviceResp>(
    schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
) -> Option<UserReq>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    let mut in_flight_reqs = Vec::new();
    let candidate = loop {
        match schedule_queue.pop_back() {
            Some(user_req) => {
                if user_req.num_in_flight_computes() == 0 {
                    break Some(user_req);
                } else {
                    in_flight_reqs.push(user_req);
                }
            },
            None => break None,
        }
    };
    for user_req in in_flight_reqs.into_iter().rev() {
        schedule_queue.push_back(user_req);
    }
    candidate
}

#[cfg(test)]
mod tests {
    use async_channel::bounded as async_bounded;
    use mockall::Sequence;
    use mockall::predicate::eq;

    use super::*;
    use crate::compute::MockDevReq;
    use crate::compute::MockDevResp;
    use crate::runtime::scheduler::MockUserRequest;
    use crate::runtime::scheduler::ReqTokenInventory;
    use crate::runtime::tasks::AsyncTaskReq;
    use crate::runtime::tasks::AsyncTaskResp;

    type TestUserReq = MockUserRequest<MockDevReq, MockDevResp>;
    type TestScheduleQueue = ScheduleQueue<TestUserReq, MockDevReq, MockDevResp>;

    #[test]
    fn test_pop_preemption_candidate_success() {
        let mut candidate = mock_user_req(1);
        candidate.expect_num_in_flight_computes().once().return_const(0usize);
        let mut in_flight_req_1 = mock_user_req(2);
        in_flight_req_1
            .expect_num_in_flight_computes()
            .once()
            .return_const(1usize);
        let mut in_flight_req_2 = mock_user_req(3);
        in_flight_req_2
            .expect_num_in_flight_computes()
            .once()
            .return_const(2usize);

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_back(candidate);
        schedule_queue.push_back(in_flight_req_1);
        schedule_queue.push_back(in_flight_req_2);

        let candidate = pop_preemption_candidate(&mut schedule_queue).unwrap();

        assert_eq!(candidate.id(), 1);
        assert_eq!(schedule_queue.pop_front().map(|req| req.id()), Some(2));
        assert_eq!(schedule_queue.pop_front().map(|req| req.id()), Some(3));
        assert!(schedule_queue.pop_front().is_none());
    }

    #[test]
    fn test_pop_preemption_candidate_fail() {
        let mut in_flight_req_1 = mock_user_req(1);
        in_flight_req_1
            .expect_num_in_flight_computes()
            .once()
            .return_const(1usize);
        let mut in_flight_req_2 = mock_user_req(2);
        in_flight_req_2
            .expect_num_in_flight_computes()
            .once()
            .return_const(2usize);

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_back(in_flight_req_1);
        schedule_queue.push_back(in_flight_req_2);

        assert!(pop_preemption_candidate(&mut schedule_queue).is_none());
        assert_eq!(schedule_queue.pop_front().map(|req| req.id()), Some(1));
        assert_eq!(schedule_queue.pop_front().map(|req| req.id()), Some(2));
        assert!(schedule_queue.pop_front().is_none());
    }

    #[test]
    fn test_prepare_cancel_schedules_all() {
        let req_budget = 3;
        let token_budget = 24;
        let max_token_per_req = 8;
        let scheduled_reqs = [
            new_test_scheduled_req(1, 8, 1, 8),
            new_test_scheduled_req(2, 8, 1, 8),
            new_test_scheduled_req(3, 8, 1, 8),
        ];
        let mut seq = Sequence::new();
        let mut user_reqs = Vec::with_capacity(scheduled_reqs.len());
        for scheduled_req in &scheduled_reqs {
            let mut user_req = mock_user_req(scheduled_req.req_id);
            expect_prefill_prepare(&mut seq, &mut user_req, scheduled_req);
            user_reqs.push(user_req);
        }
        for user_req in &mut user_reqs {
            expect_cancel(&mut seq, user_req);
        }

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        for user_req in user_reqs {
            schedule_queue.push_back(user_req);
        }

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(
            req_budget,
            token_budget,
            max_token_per_req,
            AHashMap::new(),
            &mut schedule_queue,
        );
        assert_eq!(dev_reqs.len(), scheduled_reqs.len());
        batcher.cancel(&mut schedule_queue, dev_reqs);
    }

    #[test]
    fn test_prepare_cancel_schedules_half_token_budget() {
        let req_budget = 3;
        let token_budget = 12;
        let max_token_per_req = 8;
        let scheduled_reqs = [new_test_scheduled_req(1, 8, 1, 8), new_test_scheduled_req(2, 4, 1, 4)];
        let mut seq = Sequence::new();
        let mut user_reqs = Vec::with_capacity(scheduled_reqs.len() + 1);
        for scheduled_req in &scheduled_reqs {
            let mut user_req = mock_user_req(scheduled_req.req_id);
            expect_prefill_prepare(&mut seq, &mut user_req, scheduled_req);
            user_reqs.push(user_req);
        }
        user_reqs.push(mock_user_req(3));
        for user_req in user_reqs.iter_mut().take(scheduled_reqs.len()) {
            expect_cancel(&mut seq, user_req);
        }

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        for user_req in user_reqs {
            schedule_queue.push_back(user_req);
        }

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(
            req_budget,
            token_budget,
            max_token_per_req,
            AHashMap::new(),
            &mut schedule_queue,
        );
        assert_eq!(dev_reqs.len(), scheduled_reqs.len());
        batcher.cancel(&mut schedule_queue, dev_reqs);
    }

    #[test]
    fn test_prepare_cancel_schedules_half_req_budget() {
        let req_budget = 2;
        let token_budget = 24;
        let max_token_per_req = 8;
        let scheduled_reqs = [new_test_scheduled_req(1, 8, 1, 8), new_test_scheduled_req(2, 8, 1, 8)];
        let mut seq = Sequence::new();
        let mut user_reqs = Vec::with_capacity(scheduled_reqs.len() + 1);
        for scheduled_req in &scheduled_reqs {
            let mut user_req = mock_user_req(scheduled_req.req_id);
            expect_prefill_prepare(&mut seq, &mut user_req, scheduled_req);
            user_reqs.push(user_req);
        }
        user_reqs.push(mock_user_req(3));
        for user_req in user_reqs.iter_mut().take(scheduled_reqs.len()) {
            expect_cancel(&mut seq, user_req);
        }

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        for user_req in user_reqs {
            schedule_queue.push_back(user_req);
        }

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(
            req_budget,
            token_budget,
            max_token_per_req,
            AHashMap::new(),
            &mut schedule_queue,
        );
        assert_eq!(dev_reqs.len(), scheduled_reqs.len());
        batcher.cancel(&mut schedule_queue, dev_reqs);
    }

    #[test]
    fn test_prepare_commit_schedules_all() {
        let req_budget = 3;
        let token_budget = 24;
        let max_token_per_req = 8;
        let scheduled_reqs = [
            new_test_scheduled_req(1, 8, 1, 8),
            new_test_scheduled_req(2, 8, 1, 8),
            new_test_scheduled_req(3, 8, 1, 8),
        ];
        let mut seq = Sequence::new();
        let mut user_reqs = Vec::with_capacity(scheduled_reqs.len());
        for scheduled_req in &scheduled_reqs {
            let mut user_req = mock_user_req(scheduled_req.req_id);
            expect_prefill_prepare(&mut seq, &mut user_req, scheduled_req);
            user_reqs.push(user_req);
        }
        for user_req in &mut user_reqs {
            expect_commit(&mut seq, user_req);
        }

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        for user_req in user_reqs {
            schedule_queue.push_back(user_req);
        }

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(
            req_budget,
            token_budget,
            max_token_per_req,
            AHashMap::new(),
            &mut schedule_queue,
        );
        assert_eq!(dev_reqs.len(), scheduled_reqs.len());
        let dev_resps = scheduled_reqs
            .iter()
            .map(|scheduled_req| mock_dev_resp(scheduled_req.req_id))
            .collect();
        batcher.commit(&mut schedule_queue, dev_resps);
    }

    #[test]
    fn test_prepare_commit_schedules_half_token_budget() {
        let req_budget = 3;
        let token_budget = 12;
        let max_token_per_req = 8;
        let scheduled_reqs = [new_test_scheduled_req(1, 8, 1, 8), new_test_scheduled_req(2, 4, 1, 4)];
        let mut seq = Sequence::new();
        let mut user_reqs = Vec::with_capacity(scheduled_reqs.len() + 1);
        for scheduled_req in &scheduled_reqs {
            let mut user_req = mock_user_req(scheduled_req.req_id);
            expect_prefill_prepare(&mut seq, &mut user_req, scheduled_req);
            user_reqs.push(user_req);
        }
        user_reqs.push(mock_user_req(3));
        for user_req in user_reqs.iter_mut().take(scheduled_reqs.len()) {
            expect_commit(&mut seq, user_req);
        }

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        for user_req in user_reqs {
            schedule_queue.push_back(user_req);
        }

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(
            req_budget,
            token_budget,
            max_token_per_req,
            AHashMap::new(),
            &mut schedule_queue,
        );
        assert_eq!(dev_reqs.len(), scheduled_reqs.len());
        let dev_resps = scheduled_reqs
            .iter()
            .map(|scheduled_req| mock_dev_resp(scheduled_req.req_id))
            .collect();
        batcher.commit(&mut schedule_queue, dev_resps);
    }

    #[test]
    fn test_prepare_commit_schedules_half_req_budget() {
        let req_budget = 2;
        let token_budget = 24;
        let max_token_per_req = 8;
        let scheduled_reqs = [new_test_scheduled_req(1, 8, 1, 8), new_test_scheduled_req(2, 8, 1, 8)];
        let mut seq = Sequence::new();
        let mut user_reqs = Vec::with_capacity(scheduled_reqs.len() + 1);
        for scheduled_req in &scheduled_reqs {
            let mut user_req = mock_user_req(scheduled_req.req_id);
            expect_prefill_prepare(&mut seq, &mut user_req, scheduled_req);
            user_reqs.push(user_req);
        }
        user_reqs.push(mock_user_req(3));
        for user_req in user_reqs.iter_mut().take(scheduled_reqs.len()) {
            expect_commit(&mut seq, user_req);
        }

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        for user_req in user_reqs {
            schedule_queue.push_back(user_req);
        }

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(
            req_budget,
            token_budget,
            max_token_per_req,
            AHashMap::new(),
            &mut schedule_queue,
        );
        assert_eq!(dev_reqs.len(), scheduled_reqs.len());
        let dev_resps = scheduled_reqs
            .iter()
            .map(|scheduled_req| mock_dev_resp(scheduled_req.req_id))
            .collect();
        batcher.commit(&mut schedule_queue, dev_resps);
    }

    #[test]
    fn test_prepare_cancel_req_state() {
        let continue_req_id = 1;
        let pending_req_id = 2;
        let terminal_req_id = 3;
        let skip_req_id = 4;
        let token_budget_per_req = 8;
        let mut seq = Sequence::new();

        let mut continue_req = mock_user_req(continue_req_id);
        continue_req
            .expect_token_estimate()
            .in_sequence(&mut seq)
            .returning(move || ReqTokenInventory::new::<1>(continue_req_id, token_budget_per_req, 0, 0, &[]));
        continue_req
            .expect_prepare()
            .once()
            .with(eq(token_budget_per_req))
            .in_sequence(&mut seq)
            .return_once(move |_| {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(continue_req_id);
                dev_req.expect_request_cost().once().return_const(1usize);
                dev_req.expect_token_cost().once().return_const(token_budget_per_req);
                PrepareResult::Continue {
                    dev_req,
                    compute_phase: ComputePhase::Prefill {
                        epoch: 0,
                        token_index: 0,
                    },
                }
            });

        let mut pending_req = mock_user_req(pending_req_id);
        pending_req
            .expect_token_estimate()
            .in_sequence(&mut seq)
            .returning(move || ReqTokenInventory::new::<1>(pending_req_id, token_budget_per_req, 0, 0, &[]));
        pending_req
            .expect_prepare()
            .once()
            .with(eq(token_budget_per_req))
            .in_sequence(&mut seq)
            .return_once(move |_| {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(pending_req_id);
                dev_req.expect_request_cost().once().return_const(1usize);
                dev_req.expect_token_cost().once().return_const(token_budget_per_req);
                PrepareResult::Continue {
                    dev_req,
                    compute_phase: ComputePhase::Decode {
                        epoch: 0,
                        token_index: 0,
                    },
                }
            });

        let mut terminal_req = mock_user_req(terminal_req_id);
        terminal_req
            .expect_token_estimate()
            .in_sequence(&mut seq)
            .returning(move || ReqTokenInventory::new::<1>(terminal_req_id, token_budget_per_req, 0, 0, &[]));
        terminal_req
            .expect_prepare()
            .once()
            .with(eq(token_budget_per_req))
            .in_sequence(&mut seq)
            .return_once(move |_| {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(terminal_req_id);
                dev_req.expect_request_cost().once().return_const(1usize);
                dev_req.expect_token_cost().once().return_const(token_budget_per_req);
                PrepareResult::Continue {
                    dev_req,
                    compute_phase: ComputePhase::Decode {
                        epoch: 0,
                        token_index: 0,
                    },
                }
            });

        let mut skip_req = mock_user_req(skip_req_id);
        skip_req
            .expect_token_estimate()
            .in_sequence(&mut seq)
            .returning(move || ReqTokenInventory::new::<1>(skip_req_id, token_budget_per_req, 0, 0, &[]));
        skip_req
            .expect_prepare()
            .once()
            .with(eq(token_budget_per_req))
            .in_sequence(&mut seq)
            .return_once(|_| PrepareResult::Skip);

        continue_req
            .expect_cancel()
            .once()
            .in_sequence(&mut seq)
            .return_once(|_| CancelResult::Continue);
        pending_req
            .expect_cancel()
            .once()
            .in_sequence(&mut seq)
            .return_once(|_| CancelResult::Pending);
        terminal_req
            .expect_cancel()
            .once()
            .in_sequence(&mut seq)
            .return_once(|_| CancelResult::Terminal);

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_back(continue_req);
        schedule_queue.push_back(pending_req);
        schedule_queue.push_back(terminal_req);
        schedule_queue.push_back(skip_req);

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(
            4,
            4 * token_budget_per_req,
            token_budget_per_req,
            AHashMap::new(),
            &mut schedule_queue,
        );
        assert_eq!(dev_reqs.iter().map(DevReq::id).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert!(schedule_queue.get_ref(&continue_req_id).is_some());
        assert!(schedule_queue.get_ref(&pending_req_id).is_some());
        assert!(schedule_queue.get_ref(&terminal_req_id).is_some());
        assert!(schedule_queue.get_ref(&skip_req_id).is_some());
        assert_eq!(schedule_queue.run_queue_size(), 1);
        let continue_req = schedule_queue.pop_front().unwrap();
        assert_eq!(continue_req.id(), continue_req_id);
        assert!(schedule_queue.pop_front().is_none());
        schedule_queue.push_back(continue_req);

        batcher.cancel(&mut schedule_queue, dev_reqs);
        assert!(schedule_queue.get_ref(&continue_req_id).is_some());
        assert!(schedule_queue.get_ref(&pending_req_id).is_some());
        assert!(schedule_queue.get_ref(&terminal_req_id).is_none());
        assert!(schedule_queue.get_ref(&skip_req_id).is_some());
        assert_eq!(schedule_queue.run_queue_size(), 1);
        assert_eq!(schedule_queue.pop_front().map(|req| req.id()), Some(continue_req_id));
        assert!(schedule_queue.pop_front().is_none());
    }

    #[test]
    fn test_prepare_commit_req_state() {
        let continue_req_id = 1;
        let pending_req_id = 2;
        let completed_turn_req_id = 3;
        let terminal_req_id = 4;
        let skip_req_id = 5;
        let token_budget_per_req = 8;
        let mut seq = Sequence::new();

        let mut continue_req = mock_user_req(continue_req_id);
        continue_req
            .expect_token_estimate()
            .in_sequence(&mut seq)
            .returning(move || ReqTokenInventory::new::<1>(continue_req_id, token_budget_per_req, 0, 0, &[]));
        continue_req
            .expect_prepare()
            .once()
            .with(eq(token_budget_per_req))
            .in_sequence(&mut seq)
            .return_once(move |_| {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(continue_req_id);
                dev_req.expect_request_cost().once().return_const(1usize);
                dev_req.expect_token_cost().once().return_const(token_budget_per_req);
                PrepareResult::Continue {
                    dev_req,
                    compute_phase: ComputePhase::Prefill {
                        epoch: 0,
                        token_index: 0,
                    },
                }
            });

        let mut pending_req = mock_user_req(pending_req_id);
        pending_req
            .expect_token_estimate()
            .in_sequence(&mut seq)
            .returning(move || ReqTokenInventory::new::<1>(pending_req_id, token_budget_per_req, 0, 0, &[]));
        pending_req
            .expect_prepare()
            .once()
            .with(eq(token_budget_per_req))
            .in_sequence(&mut seq)
            .return_once(move |_| {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(pending_req_id);
                dev_req.expect_request_cost().once().return_const(1usize);
                dev_req.expect_token_cost().once().return_const(token_budget_per_req);
                PrepareResult::Continue {
                    dev_req,
                    compute_phase: ComputePhase::Decode {
                        epoch: 0,
                        token_index: 0,
                    },
                }
            });

        let mut terminal_req = mock_user_req(terminal_req_id);
        terminal_req
            .expect_token_estimate()
            .in_sequence(&mut seq)
            .returning(move || ReqTokenInventory::new::<1>(terminal_req_id, token_budget_per_req, 0, 0, &[]));
        terminal_req
            .expect_prepare()
            .once()
            .with(eq(token_budget_per_req))
            .in_sequence(&mut seq)
            .return_once(move |_| {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(terminal_req_id);
                dev_req.expect_request_cost().once().return_const(1usize);
                dev_req.expect_token_cost().once().return_const(token_budget_per_req);
                PrepareResult::Continue {
                    dev_req,
                    compute_phase: ComputePhase::Decode {
                        epoch: 0,
                        token_index: 0,
                    },
                }
            });

        let mut completed_turn_req = mock_user_req(completed_turn_req_id);
        completed_turn_req
            .expect_token_estimate()
            .in_sequence(&mut seq)
            .returning(move || ReqTokenInventory::new::<1>(completed_turn_req_id, token_budget_per_req, 0, 0, &[]));
        completed_turn_req
            .expect_prepare()
            .once()
            .with(eq(token_budget_per_req))
            .in_sequence(&mut seq)
            .return_once(move |_| {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(completed_turn_req_id);
                dev_req.expect_request_cost().once().return_const(1usize);
                dev_req.expect_token_cost().once().return_const(token_budget_per_req);
                PrepareResult::Continue {
                    dev_req,
                    compute_phase: ComputePhase::Decode {
                        epoch: 0,
                        token_index: 0,
                    },
                }
            });

        let mut skip_req = mock_user_req(skip_req_id);
        skip_req
            .expect_token_estimate()
            .in_sequence(&mut seq)
            .returning(move || ReqTokenInventory::new::<1>(skip_req_id, token_budget_per_req, 0, 0, &[]));
        skip_req
            .expect_prepare()
            .once()
            .with(eq(token_budget_per_req))
            .in_sequence(&mut seq)
            .return_once(|_| PrepareResult::Skip);

        continue_req
            .expect_commit()
            .once()
            .in_sequence(&mut seq)
            .return_once(|_| CommitResult::Continue);
        pending_req
            .expect_commit()
            .once()
            .in_sequence(&mut seq)
            .return_once(|_| CommitResult::Pending);
        terminal_req
            .expect_commit()
            .once()
            .in_sequence(&mut seq)
            .return_once(|_| CommitResult::Terminal);
        completed_turn_req
            .expect_commit()
            .once()
            .in_sequence(&mut seq)
            .return_once(|_| CommitResult::TurnCompleted(CompletionReason::LengthLimit));

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_back(continue_req);
        schedule_queue.push_back(pending_req);
        schedule_queue.push_back(terminal_req);
        schedule_queue.push_back(completed_turn_req);
        schedule_queue.push_back(skip_req);

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(
            5,
            5 * token_budget_per_req,
            token_budget_per_req,
            AHashMap::new(),
            &mut schedule_queue,
        );
        assert_eq!(dev_reqs.iter().map(DevReq::id).collect::<Vec<_>>(), vec![1, 2, 4, 3]);
        assert!(schedule_queue.get_ref(&continue_req_id).is_some());
        assert!(schedule_queue.get_ref(&pending_req_id).is_some());
        assert!(schedule_queue.get_ref(&completed_turn_req_id).is_some());
        assert!(schedule_queue.get_ref(&terminal_req_id).is_some());
        assert!(schedule_queue.get_ref(&skip_req_id).is_some());
        assert_eq!(schedule_queue.run_queue_size(), 1);
        let continue_req = schedule_queue.pop_front().unwrap();
        assert_eq!(continue_req.id(), continue_req_id);
        assert!(schedule_queue.pop_front().is_none());
        schedule_queue.push_back(continue_req);

        let completed_turns = batcher.commit(
            &mut schedule_queue,
            vec![
                mock_dev_resp(continue_req_id),
                mock_dev_resp(pending_req_id),
                mock_dev_resp(terminal_req_id),
                mock_dev_resp(completed_turn_req_id),
            ],
        );

        assert_eq!(completed_turns.len(), 1);
        assert_eq!(completed_turns[0].0.id(), completed_turn_req_id);
        assert_eq!(completed_turns[0].1, CompletionReason::LengthLimit);
        assert!(schedule_queue.get_ref(&continue_req_id).is_some());
        assert!(schedule_queue.get_ref(&pending_req_id).is_some());
        assert!(schedule_queue.get_ref(&completed_turn_req_id).is_none());
        assert!(schedule_queue.get_ref(&terminal_req_id).is_none());
        assert!(schedule_queue.get_ref(&skip_req_id).is_some());
        assert_eq!(schedule_queue.run_queue_size(), 1);
        assert_eq!(schedule_queue.pop_front().map(|req| req.id()), Some(continue_req_id));
        assert!(schedule_queue.pop_front().is_none());
    }

    #[test]
    fn test_prepare_w_sticky_token_budgets() {
        let mut sticky_req_1 = mock_user_req(1);
        sticky_req_1.expect_prepare().once().with(eq(3)).return_once(|_| {
            let mut dev_req = MockDevReq::new();
            dev_req.expect_id().return_const(1usize);
            dev_req.expect_request_cost().once().return_const(1usize);
            dev_req.expect_token_cost().once().return_const(3usize);
            PrepareResult::Continue {
                dev_req,
                compute_phase: ComputePhase::Decode {
                    epoch: 0,
                    token_index: 0,
                },
            }
        });

        let mut sticky_req_2 = mock_user_req(2);
        sticky_req_2.expect_prepare().once().with(eq(2)).return_once(|_| {
            let mut dev_req = MockDevReq::new();
            dev_req.expect_id().return_const(2usize);
            dev_req.expect_request_cost().once().return_const(1usize);
            dev_req.expect_token_cost().once().return_const(2usize);
            PrepareResult::Continue {
                dev_req,
                compute_phase: ComputePhase::Decode {
                    epoch: 0,
                    token_index: 0,
                },
            }
        });

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_back(sticky_req_1);
        schedule_queue.push_back(sticky_req_2);

        let mut batcher = FIFOBatcher::new();
        let sticky_token_budgets = AHashMap::from([(1, 3), (2, 2)]);
        let dev_reqs = batcher.prepare(2, 5, 3, sticky_token_budgets, &mut schedule_queue);

        assert_eq!(dev_reqs.iter().map(DevReq::id).collect::<Vec<_>>(), vec![1, 2]);
        assert!(schedule_queue.get_ref(&1).is_some());
        assert!(schedule_queue.get_ref(&2).is_some());
        assert_eq!(schedule_queue.run_queue_size(), 0);
        assert!(schedule_queue.pop_front().is_none());
    }

    #[test]
    fn test_prepare_wo_sticky_token_budgets() {
        let mut req_1 = mock_user_req(1);
        req_1
            .expect_token_estimate()
            .once()
            .returning(|| ReqTokenInventory::new::<1>(1, 3, 0, 0, &[]));
        req_1.expect_prepare().once().with(eq(3)).return_once(|_| {
            let mut dev_req = MockDevReq::new();
            dev_req.expect_id().return_const(1usize);
            dev_req.expect_request_cost().once().return_const(1usize);
            dev_req.expect_token_cost().once().return_const(3usize);
            PrepareResult::Continue {
                dev_req,
                compute_phase: ComputePhase::Decode {
                    epoch: 0,
                    token_index: 0,
                },
            }
        });

        let mut req_2 = mock_user_req(2);
        req_2
            .expect_token_estimate()
            .once()
            .returning(|| ReqTokenInventory::new::<1>(2, 2, 0, 0, &[]));
        req_2.expect_prepare().once().with(eq(2)).return_once(|_| {
            let mut dev_req = MockDevReq::new();
            dev_req.expect_id().return_const(2usize);
            dev_req.expect_request_cost().once().return_const(1usize);
            dev_req.expect_token_cost().once().return_const(2usize);
            PrepareResult::Continue {
                dev_req,
                compute_phase: ComputePhase::Decode {
                    epoch: 0,
                    token_index: 0,
                },
            }
        });

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_back(req_1);
        schedule_queue.push_back(req_2);

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(2, 5, 3, AHashMap::new(), &mut schedule_queue);

        assert_eq!(dev_reqs.iter().map(DevReq::id).collect::<Vec<_>>(), vec![1, 2]);
        assert!(schedule_queue.get_ref(&1).is_some());
        assert!(schedule_queue.get_ref(&2).is_some());
        assert_eq!(schedule_queue.run_queue_size(), 0);
        assert!(schedule_queue.pop_front().is_none());
    }

    #[test]
    fn test_prepare_blocking_async_task() {
        let req_id = 1;
        let token_budget = 8;
        let mut user_req = mock_user_req(req_id);
        user_req
            .expect_token_estimate()
            .once()
            .returning(move || ReqTokenInventory::new::<1>(req_id, token_budget, 0, 0, &[]));
        user_req.expect_prepare().once().return_once(move |_| {
            PrepareResult::BlockingAsyncTask {
                req: Box::new(TestAsyncTaskReq { request_id: req_id }),
            }
        });

        let (async_task_req_tx, async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_back(user_req);

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(1, token_budget, token_budget, AHashMap::new(), &mut schedule_queue);

        assert!(dev_reqs.is_empty());
        assert_eq!(async_task_req_rx.try_recv().unwrap().request_id(), req_id);
        assert!(schedule_queue.get_ref(&req_id).is_some());
        assert_eq!(schedule_queue.run_queue_size(), 0);
    }

    #[test]
    fn test_prepare_nonblocking_async_task() {
        let req_id = 1;
        let token_budget = 8;
        let mut user_req = mock_user_req(req_id);
        user_req
            .expect_token_estimate()
            .times(2)
            .returning(move || ReqTokenInventory::new::<1>(req_id, token_budget, 0, 0, &[]));
        let mut prepare_count = 0;
        user_req.expect_prepare().times(2).returning(move |_| {
            prepare_count += 1;
            if prepare_count == 1 {
                PrepareResult::NonblockingAsyncTask {
                    req: Box::new(TestAsyncTaskReq { request_id: req_id }),
                }
            } else {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(req_id);
                dev_req.expect_request_cost().once().return_const(1usize);
                dev_req.expect_token_cost().once().return_const(token_budget);
                PrepareResult::Continue {
                    dev_req,
                    compute_phase: ComputePhase::Decode {
                        epoch: 0,
                        token_index: 0,
                    },
                }
            }
        });

        let (async_task_req_tx, async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_back(user_req);

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(1, token_budget, token_budget, AHashMap::new(), &mut schedule_queue);

        assert_eq!(dev_reqs.iter().map(DevReq::id).collect::<Vec<_>>(), vec![req_id]);
        assert_eq!(async_task_req_rx.try_recv().unwrap().request_id(), req_id);
        assert!(schedule_queue.get_ref(&req_id).is_some());
        assert_eq!(schedule_queue.run_queue_size(), 0);
    }

    #[test]
    fn test_prepare_await_success() {
        let req_budget = 1;
        let token_budget = 8;
        let max_token_per_req = 8;
        let req_id = 1;

        let mut seq = Sequence::new();
        let mut user_req = mock_user_req(req_id);
        user_req
            .expect_token_estimate()
            .in_sequence(&mut seq)
            .returning(move || ReqTokenInventory::new::<1>(req_id, token_budget, 0, 0, &[]));
        user_req
            .expect_prepare()
            .once()
            .with(eq(token_budget))
            .in_sequence(&mut seq)
            .return_once(|_| {
                PrepareResult::Await {
                    wait: Box::pin(async {}),
                }
            });
        let (async_task_req_tx, async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_back(user_req);

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(
            req_budget,
            token_budget,
            max_token_per_req,
            AHashMap::new(),
            &mut schedule_queue,
        );
        assert!(dev_reqs.is_empty());

        assert_eq!(schedule_queue.pop_ready_reqs().map(|req| req.id()), Some(req_id));
        assert!(async_task_req_rx.try_recv().is_err());
    }

    #[test]
    fn test_prepare_resource_limit_preempts_tail_and_retries_current() {
        let req_budget = 1;
        let token_budget = 8;
        let max_token_per_req = 8;
        let current_req_id = 1;
        let preempted_req_id = 2;

        let mut current_req = mock_user_req(current_req_id);
        current_req
            .expect_token_estimate()
            .times(2)
            .returning(move || ReqTokenInventory::new::<1>(current_req_id, token_budget, 0, 0, &[]));
        let mut prepare_count = 0;
        current_req
            .expect_prepare()
            .times(2)
            .with(eq(token_budget))
            .returning(move |_| {
                prepare_count += 1;
                if prepare_count == 1 {
                    PrepareResult::ResourceLimitExceeded
                } else {
                    PrepareResult::Terminal
                }
            });
        let mut preempted_req = mock_user_req(preempted_req_id);
        preempted_req
            .expect_num_in_flight_computes()
            .once()
            .return_const(0usize);

        let (async_task_req_tx, _async_task_req_rx) = async_bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_back(current_req);
        schedule_queue.push_back(preempted_req);

        let mut batcher = FIFOBatcher::new();
        let dev_reqs = batcher.prepare(
            req_budget,
            token_budget,
            max_token_per_req,
            AHashMap::new(),
            &mut schedule_queue,
        );

        assert!(dev_reqs.is_empty());
        assert_eq!(schedule_queue.run_queue_size(), 0);
    }

    #[derive(Clone, Copy)]
    struct TestScheduledReq {
        req_id: RawRequestID,
        prepare_token_budget: usize,
        request_cost: usize,
        token_cost: usize,
    }

    struct TestAsyncTaskReq {
        request_id: RawRequestID,
    }

    impl AsyncTaskReq for TestAsyncTaskReq {
        type Resp = dyn AsyncTaskResp;

        fn request_id(&self) -> RawRequestID {
            self.request_id
        }

        fn run(self: Box<Self>) -> futures_lite::future::Boxed<Box<Self::Resp>> {
            Box::pin(async { unreachable!("FIFO batcher tests do not execute async tasks") })
        }
    }

    fn new_test_scheduled_req(
        req_id: RawRequestID,
        prepare_token_budget: usize,
        request_cost: usize,
        token_cost: usize,
    ) -> TestScheduledReq {
        TestScheduledReq {
            req_id,
            prepare_token_budget,
            request_cost,
            token_cost,
        }
    }

    fn mock_user_req(req_id: RawRequestID) -> TestUserReq {
        let mut user_req = TestUserReq::new();
        user_req.expect_id().return_const(req_id);
        user_req.expect_is_terminal().return_const(false);
        user_req
    }

    fn expect_prefill_prepare(seq: &mut Sequence, user_req: &mut TestUserReq, scheduled_req: &TestScheduledReq) {
        let TestScheduledReq {
            req_id,
            prepare_token_budget,
            request_cost,
            token_cost,
        } = *scheduled_req;
        user_req
            .expect_token_estimate()
            .in_sequence(seq)
            .returning(move || ReqTokenInventory::new::<1>(req_id, prepare_token_budget, 0, 0, &[]));
        user_req
            .expect_prepare()
            .once()
            .with(eq(prepare_token_budget))
            .in_sequence(seq)
            .return_once(move |_| {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(req_id);
                dev_req.expect_request_cost().once().return_const(request_cost);
                dev_req.expect_token_cost().once().return_const(token_cost);
                PrepareResult::Continue {
                    dev_req,
                    compute_phase: ComputePhase::Prefill {
                        epoch: 0,
                        token_index: 0,
                    },
                }
            });
    }

    fn expect_cancel(seq: &mut Sequence, user_req: &mut TestUserReq) {
        user_req
            .expect_cancel()
            .once()
            .in_sequence(seq)
            .return_once(|_| CancelResult::Continue);
    }

    fn expect_commit(seq: &mut Sequence, user_req: &mut TestUserReq) {
        user_req
            .expect_commit()
            .once()
            .in_sequence(seq)
            .return_once(|_| CommitResult::Continue);
    }

    fn mock_dev_resp(req_id: RawRequestID) -> MockDevResp {
        let mut dev_resp = MockDevResp::new();
        dev_resp.expect_id().return_const(req_id);
        dev_resp
    }
}
