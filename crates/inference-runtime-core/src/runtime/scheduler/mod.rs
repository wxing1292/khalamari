use ahash::AHashMap;
use futures_lite::future::Boxed;

use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::CompletionReason;
use crate::runtime::RawRequestID;
use crate::runtime::tasks::AsyncTaskReq;
use crate::runtime::tasks::AsyncTaskResp;

mod compute_slot;
pub use compute_slot::ComputeSlot;

mod dedup_vec_deque;

mod event_loop;
pub use event_loop::EventLoop;

mod executor_hibernate;

mod simple_scheduler;
pub use simple_scheduler::SimpleScheduler;

mod instrumented_scheduler;
pub use instrumented_scheduler::InstrumentedScheduler;

mod fifo_batcher;
pub use fifo_batcher::FIFOBatcher;

mod schedule_queue;
pub use schedule_queue::ScheduleQueue;

mod token_budget_allocator;
pub use token_budget_allocator::BatchBudget;
pub use token_budget_allocator::ReqTokenInventory;
pub use token_budget_allocator::allocate_sticky_token_budgets;

#[mockall::automock]
pub trait Scheduler<UserReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
{
    fn enqueue(&mut self, user_req: UserReq);
    fn handle_async_task_resp(&mut self, resp: Box<dyn AsyncTaskResp>);
    fn can_flush(&self) -> bool;

    fn prepare(&mut self) -> BatchDeviceReq;
    fn cancel(&mut self, batch_dev_req: BatchDeviceReq);
    fn commit(&mut self, batch_dev_resp: BatchDeviceResp) -> Vec<(UserReq, CompletionReason)>;
}

#[mockall::automock]
pub trait Batcher<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    fn prepare(
        &mut self,
        req_budget: usize,
        token_budget: usize,
        max_token_per_req: usize,
        sticky_token_budgets: AHashMap<RawRequestID, usize>,
        schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
    ) -> Vec<DeviceReq>;
    fn cancel(&mut self, schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>, dev_reqs: Vec<DeviceReq>);
    fn commit(
        &mut self,
        schedule_queue: &mut ScheduleQueue<UserReq, DeviceReq, DeviceResp>,
        dev_resps: Vec<DeviceResp>,
    ) -> Vec<(UserReq, CompletionReason)>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputePhase {
    Prefill { epoch: usize, token_index: usize },
    Decode { epoch: usize, token_index: usize },
}

pub enum PrepareResult<DeviceReq> {
    ResourceLimitExceeded,
    BlockingAsyncTask {
        req: Box<dyn AsyncTaskReq<Resp = dyn AsyncTaskResp>>,
    },
    NonblockingAsyncTask {
        req: Box<dyn AsyncTaskReq<Resp = dyn AsyncTaskResp>>,
    },
    Await {
        wait: Boxed<()>,
    },
    Skip,
    Continue {
        dev_req: DeviceReq,
        compute_phase: ComputePhase,
    },
    Terminal,
}

pub enum CommitResult {
    Continue,
    Pending,
    TurnCompleted(CompletionReason),
    Terminal,
}

pub enum CancelResult {
    Continue,
    Pending,
    Terminal,
}

#[mockall::automock]
pub trait UserRequest<DeviceReq, DeviceResp>: Send + Sized + 'static
where
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    fn id(&self) -> RawRequestID;

    fn store_swapped(&self) -> bool;
    fn is_terminal(&self) -> bool;
    fn num_in_flight_computes(&self) -> usize;
    fn num_in_flight_blocking_async_tasks(&self) -> usize;
    fn num_in_flight_nonblocking_async_tasks(&self) -> usize;

    fn token_estimate(&self) -> ReqTokenInventory<'_>;

    fn prepare(&mut self, token_budget: usize) -> PrepareResult<DeviceReq>;
    fn handle_async_task_resp(&mut self, resp: Box<dyn AsyncTaskResp>);
    fn cancel(&mut self, dev_req: DeviceReq) -> CancelResult;
    fn commit(&mut self, dev_resp: DeviceResp) -> CommitResult;
}
