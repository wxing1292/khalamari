use std::collections::VecDeque;

use ahash::AHashMap;

use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::CompletionReason;
use crate::runtime::RawComputeSlotSeq;
use crate::runtime::RawRequestID;
use crate::runtime::scheduler::BatchBudget;
use crate::runtime::scheduler::Batcher;
use crate::runtime::scheduler::ComputeSlot;
use crate::runtime::scheduler::ComputeSlotSeq;
use crate::runtime::scheduler::ScheduleQueue;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::scheduler::UserRequest;
use crate::runtime::scheduler::allocate_sticky_token_budgets;
use crate::runtime::tasks::AsyncTaskResp;

pub struct SimpleScheduler<UserReq, DeviceReq, DeviceResp, B> {
    schedule_queue: ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
    batcher: B,
    max_req_budget: usize,
    max_token_budget: usize,
    max_token_per_req: usize,

    next_compute_slot_seq: RawComputeSlotSeq,
    free_compute_slots: VecDeque<ComputeSlot>,
    used_compute_slots: VecDeque<ComputeSlot>,
}

impl<UserReq, DeviceReq, DeviceResp, B> SimpleScheduler<UserReq, DeviceReq, DeviceResp, B>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    B: Batcher<UserReq, DeviceReq, DeviceResp>,
{
    pub fn new(
        schedule_queue: ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
        batcher: B,
        max_req_budget: usize,
        max_token_budget: usize,
        max_token_per_req: usize,
        num_compute_slots: usize,
    ) -> Self {
        assert!(
            max_req_budget > 0,
            "simple scheduler requires a positive request budget"
        );
        assert!(
            max_token_budget > 0,
            "simple scheduler requires a positive token budget"
        );
        assert!(
            max_token_per_req > 0,
            "simple scheduler requires a positive per-request token budget"
        );
        assert!(num_compute_slots > 0, "simple scheduler requires compute slots");

        Self {
            schedule_queue,
            batcher,
            max_req_budget,
            max_token_budget,
            max_token_per_req,

            next_compute_slot_seq: 1,
            free_compute_slots: (0..num_compute_slots).map(ComputeSlot::new).collect(),
            used_compute_slots: VecDeque::with_capacity(num_compute_slots),
        }
    }

    pub fn run_queue_size(&self) -> usize {
        self.schedule_queue.run_queue_size()
    }

    pub fn new_queue_size(&self) -> usize {
        self.schedule_queue.new_queue_size()
    }
}

impl<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, B>
    Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>
    for SimpleScheduler<UserReq, DeviceReq, DeviceResp, B>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
    B: Batcher<UserReq, DeviceReq, DeviceResp>,
{
    fn enqueue(&mut self, user_req: UserReq) {
        self.schedule_queue.enqueue(user_req);
    }

    fn handle_async_task_resp(&mut self, resp: Box<dyn AsyncTaskResp>) {
        self.schedule_queue.handle_async_task_resp(resp);
        self.schedule_queue.handle_ready_waits();
    }

    fn can_flush(&self) -> bool {
        let token_budget = self.max_token_budget.min(self.max_token_per_req);
        !self.free_compute_slots.is_empty() && self.schedule_queue.token_estimate(token_budget) > 0
    }

    fn next_compute_slot_seq(&self) -> ComputeSlotSeq {
        let seq_id = self.next_compute_slot_seq;
        if self.free_compute_slots.is_empty() {
            ComputeSlotSeq::Pending { seq_id }
        } else {
            ComputeSlotSeq::Ready { seq_id }
        }
    }

    fn num_in_flight_compute_slots(&self) -> usize {
        self.used_compute_slots.len()
    }

    fn prepare(&mut self) -> BatchDeviceReq {
        let mut compute_slot = self
            .free_compute_slots
            .pop_front()
            .expect("simple scheduler prepare requires a free compute slot");

        let compute_slot_seq = self.next_compute_slot_seq;
        compute_slot.prepare(compute_slot_seq);
        self.next_compute_slot_seq += 1;

        let sticky_token_budgets = allocate_slot_token_budgets(
            &compute_slot,
            &self.schedule_queue,
            BatchBudget {
                req_budget: self.max_req_budget,
                token_budget: self.max_token_budget,
                max_token_per_req: self.max_token_per_req,
            },
        );
        let dev_reqs = self.batcher.prepare(
            self.max_req_budget,
            self.max_token_budget,
            self.max_token_per_req,
            sticky_token_budgets,
            &mut self.schedule_queue,
        );
        let sticky_req_ids = compute_slot.sticky_req_ids_mut();
        sticky_req_ids.clear();
        sticky_req_ids.extend(dev_reqs.iter().map(DevReq::id));
        debug_assert!(
            sticky_req_ids
                .iter()
                .enumerate()
                .all(|(req_index, req_id)| { !sticky_req_ids[..req_index].contains(req_id) }),
            "compute slot batch request IDs must be unique"
        );
        self.used_compute_slots.push_back(compute_slot);
        BatchDeviceReq::from_parts(compute_slot_seq, dev_reqs)
    }

    fn cancel(&mut self, batch_dev_req: BatchDeviceReq) {
        let mut compute_slot = self
            .used_compute_slots
            .pop_back()
            .expect("simple scheduler cancellation requires a matching compute slot");
        let (compute_slot_seq, dev_reqs) = batch_dev_req.into_inner();
        debug_assert_eq!(
            compute_slot.seq(),
            Some(compute_slot_seq),
            "simple scheduler cancellation compute slot sequence mismatch"
        );
        compute_slot.reset();
        self.free_compute_slots.push_front(compute_slot);

        self.batcher.cancel(&mut self.schedule_queue, dev_reqs);
        self.schedule_queue.handle_ready_waits();
    }

    fn commit(&mut self, batch_dev_resp: BatchDeviceResp) -> Vec<(UserReq, CompletionReason)> {
        let mut compute_slot = self
            .used_compute_slots
            .pop_front()
            .expect("simple scheduler commit requires a matching compute slot");
        let (compute_slot_seq, dev_resps) = batch_dev_resp.into_inner();
        assert_eq!(
            compute_slot.seq(),
            Some(compute_slot_seq),
            "simple scheduler commit compute slot sequence mismatch"
        );
        compute_slot.reset();
        self.free_compute_slots.push_back(compute_slot);

        let completed_turns = self.batcher.commit(&mut self.schedule_queue, dev_resps);
        self.schedule_queue.handle_ready_waits();
        completed_turns
    }
}

fn allocate_slot_token_budgets<UserReq, DeviceReq, DeviceResp>(
    compute_slot: &ComputeSlot,
    schedule_queue: &ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
    batch_budget: BatchBudget,
) -> AHashMap<RawRequestID, usize>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    let sticky_token_inventories = compute_slot
        .sticky_req_ids_ref()
        .iter()
        .filter_map(|req_id| schedule_queue.get_ref(req_id).map(UserRequest::token_estimate))
        .collect::<Vec<_>>();
    allocate_sticky_token_budgets(batch_budget, &sticky_token_inventories)
}

#[cfg(test)]
mod tests {
    use async_channel::bounded;

    use super::*;
    use crate::compute::MockDevReq;
    use crate::compute::MockDevResp;
    use crate::runtime::scheduler::MockBatcher;
    use crate::runtime::scheduler::MockUserRequest;
    use crate::runtime::scheduler::ReqTokenInventory;

    type TestUserReq = MockUserRequest<MockDevReq, MockDevResp>;
    type TestBatcher = MockBatcher<TestUserReq, MockDevReq, MockDevResp>;

    struct TestBatchDevReq {
        seq: RawComputeSlotSeq,
        dev_reqs: Vec<MockDevReq>,
    }

    impl BatchDevReq<MockDevReq> for TestBatchDevReq {
        fn seq(&self) -> RawComputeSlotSeq {
            self.seq
        }

        fn request_cost(&self) -> usize {
            self.dev_reqs.len()
        }

        fn token_cost(&self) -> usize {
            self.dev_reqs.len()
        }

        fn from_parts(seq: RawComputeSlotSeq, mut dev_reqs: Vec<MockDevReq>) -> Self {
            // Model packing may change request order within a compute slot.
            dev_reqs.reverse();
            Self { seq, dev_reqs }
        }

        fn into_inner(self) -> (RawComputeSlotSeq, Vec<MockDevReq>) {
            (self.seq, self.dev_reqs)
        }
    }

    struct TestBatchDeviceResp {
        seq: RawComputeSlotSeq,
        dev_resps: Vec<MockDevResp>,
    }

    impl BatchDevResp<MockDevResp> for TestBatchDeviceResp {
        fn seq(&self) -> RawComputeSlotSeq {
            self.seq
        }

        fn from_parts(seq: RawComputeSlotSeq, dev_resps: Vec<MockDevResp>) -> Self {
            Self { seq, dev_resps }
        }

        fn into_inner(self) -> (RawComputeSlotSeq, Vec<MockDevResp>) {
            (self.seq, self.dev_resps)
        }
    }

    #[test]
    fn test_can_flush_tracks_runnable_work_and_compute_slot() {
        let max_req_budget = 1;
        let max_token_budget = 8;
        let max_token_per_req = 4;

        let mut batcher = TestBatcher::new();
        batcher
            .expect_prepare()
            .once()
            .with(
                mockall::predicate::eq(max_req_budget),
                mockall::predicate::eq(max_token_budget),
                mockall::predicate::eq(max_token_per_req),
                mockall::predicate::always(),
                mockall::predicate::always(),
            )
            .return_once(|_, _, _, _, _| {
                let mut dev_req = MockDevReq::new();
                dev_req.expect_id().return_const(1usize);
                vec![dev_req]
            });
        batcher.expect_cancel().once().return_once(|_, _| {});

        let (async_task_req_tx, _async_task_req_rx) = bounded(1);
        let mut scheduler = SimpleScheduler::new(
            ScheduleQueue::new(async_task_req_tx),
            batcher,
            max_req_budget,
            max_token_budget,
            max_token_per_req,
            1,
        );

        {
            let scheduler: &mut dyn Scheduler<TestUserReq, MockDevReq, MockDevResp, TestBatchDevReq, TestBatchDeviceResp> =
                &mut scheduler;
            assert_eq!(scheduler.next_compute_slot_seq(), ComputeSlotSeq::Ready { seq_id: 1 });
            assert_eq!(scheduler.num_in_flight_compute_slots(), 0);
            assert!(!scheduler.can_flush());
            let mut user_req = TestUserReq::new();
            user_req
                .expect_token_estimate()
                .times(2)
                .returning(|| ReqTokenInventory::new::<1>(1, 1, 0, 0, &[]));
            scheduler.enqueue(user_req);
            assert!(scheduler.can_flush());

            let batch_dev_req = scheduler.prepare();
            assert_eq!(batch_dev_req.seq(), 1);
            assert_eq!(scheduler.next_compute_slot_seq(), ComputeSlotSeq::Pending { seq_id: 2 });
            assert_eq!(scheduler.num_in_flight_compute_slots(), 1);
            assert!(!scheduler.can_flush());

            scheduler.cancel(batch_dev_req);
            assert_eq!(scheduler.next_compute_slot_seq(), ComputeSlotSeq::Ready { seq_id: 2 });
            assert_eq!(scheduler.num_in_flight_compute_slots(), 0);
            assert!(scheduler.can_flush());
        }
    }

    #[test]
    fn test_prepare_commit() {
        let mut batcher = TestBatcher::new();
        batcher.expect_prepare().times(3).returning(|_, _, _, _, _| {
            [1usize, 2]
                .into_iter()
                .map(|id| {
                    let mut request = MockDevReq::new();
                    request.expect_id().return_const(id);
                    request
                })
                .collect()
        });
        batcher
            .expect_commit()
            .times(3)
            .withf(|_, responses| responses.iter().map(DevResp::id).eq([2, 1]))
            .returning(|_, _| Vec::new());

        let (async_task_req_tx, _async_task_req_rx) = bounded(1);
        let mut scheduler = SimpleScheduler::new(ScheduleQueue::new(async_task_req_tx), batcher, 2, 8, 4, 3);
        let (first, second, third) = {
            let scheduler: &mut dyn Scheduler<TestUserReq, MockDevReq, MockDevResp, TestBatchDevReq, TestBatchDeviceResp> =
                &mut scheduler;
            let first = scheduler.prepare();
            let second = scheduler.prepare();
            let third = scheduler.prepare();
            assert_eq!(first.seq(), 1);
            assert_eq!(second.seq(), 2);
            assert_eq!(third.seq(), 3);
            assert_eq!(scheduler.next_compute_slot_seq(), ComputeSlotSeq::Pending { seq_id: 4 });
            assert_eq!(scheduler.num_in_flight_compute_slots(), 3);
            (first, second, third)
        };

        {
            let scheduler: &mut dyn Scheduler<TestUserReq, MockDevReq, MockDevResp, TestBatchDevReq, TestBatchDeviceResp> =
                &mut scheduler;
            for (batch_index, batch) in [first, second, third].into_iter().enumerate() {
                let responses = batch
                    .dev_reqs
                    .iter()
                    .map(|request| {
                        let mut response = MockDevResp::new();
                        response.expect_id().return_const(request.id());
                        response
                    })
                    .collect();
                scheduler.commit(TestBatchDeviceResp::from_parts(batch.seq(), responses));
                assert_eq!(scheduler.next_compute_slot_seq(), ComputeSlotSeq::Ready { seq_id: 4 });
                assert_eq!(scheduler.num_in_flight_compute_slots(), 2 - batch_index);
            }
        }
    }

    #[test]
    fn test_prepare_cancel() {
        let mut batcher = TestBatcher::new();
        batcher.expect_prepare().times(3).returning(|_, _, _, _, _| {
            [1usize, 2]
                .into_iter()
                .map(|id| {
                    let mut request = MockDevReq::new();
                    request.expect_id().return_const(id);
                    request
                })
                .collect()
        });
        batcher
            .expect_cancel()
            .times(3)
            .withf(|_, requests| requests.iter().map(DevReq::id).eq([2, 1]))
            .returning(|_, _| {});

        let (async_task_req_tx, _async_task_req_rx) = bounded(1);
        let mut scheduler = SimpleScheduler::new(ScheduleQueue::new(async_task_req_tx), batcher, 2, 8, 4, 3);
        let (first, second, third) = {
            let scheduler: &mut dyn Scheduler<TestUserReq, MockDevReq, MockDevResp, TestBatchDevReq, TestBatchDeviceResp> =
                &mut scheduler;
            let first = scheduler.prepare();
            let second = scheduler.prepare();
            let third = scheduler.prepare();
            assert_eq!(first.seq(), 1);
            assert_eq!(second.seq(), 2);
            assert_eq!(third.seq(), 3);
            assert_eq!(scheduler.next_compute_slot_seq(), ComputeSlotSeq::Pending { seq_id: 4 });
            assert_eq!(scheduler.num_in_flight_compute_slots(), 3);
            (first, second, third)
        };

        {
            let scheduler: &mut dyn Scheduler<TestUserReq, MockDevReq, MockDevResp, TestBatchDevReq, TestBatchDeviceResp> =
                &mut scheduler;
            scheduler.cancel(third);
            assert_eq!(scheduler.next_compute_slot_seq(), ComputeSlotSeq::Ready { seq_id: 4 });
            assert_eq!(scheduler.num_in_flight_compute_slots(), 2);
            scheduler.cancel(second);
            assert_eq!(scheduler.next_compute_slot_seq(), ComputeSlotSeq::Ready { seq_id: 4 });
            assert_eq!(scheduler.num_in_flight_compute_slots(), 1);
            scheduler.cancel(first);
            assert_eq!(scheduler.next_compute_slot_seq(), ComputeSlotSeq::Ready { seq_id: 4 });
            assert_eq!(scheduler.num_in_flight_compute_slots(), 0);
        }
    }

    #[test]
    fn test_allocate_slot_token_budgets() {
        let req_id = 7;
        let (async_task_req_tx, _async_task_req_rx) = bounded(1);
        let mut schedule_queue = ScheduleQueue::new(async_task_req_tx);
        let mut user_req = TestUserReq::new();
        user_req.expect_id().return_const(req_id);
        user_req
            .expect_token_estimate()
            .once()
            .returning(move || ReqTokenInventory::new::<1>(req_id, 1, 0, 0, &[]));
        schedule_queue.push_back(user_req);
        let mut compute_slot = ComputeSlot::new(0);
        compute_slot.sticky_req_ids_mut().push(req_id);

        assert_eq!(
            allocate_slot_token_budgets(
                &compute_slot,
                &schedule_queue,
                BatchBudget {
                    req_budget: 1,
                    token_budget: 8,
                    max_token_per_req: 4,
                },
            ),
            AHashMap::from([(req_id, 1)])
        );
    }
}
