use std::collections::HashMap;
use std::collections::VecDeque;
use std::collections::hash_map::Entry as HashEntry;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use async_channel::Sender;
use async_channel::TrySendError;
use futures_lite::future::Boxed;
use futures_util::Stream;
use futures_util::stream::FuturesUnordered;
use futures_util::task::noop_waker_ref;
use map_macro::hash_map;

use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::runtime::RawRequestID;
use crate::runtime::scheduler::UserRequest;
use crate::runtime::scheduler::dedup_vec_deque::DedupVecDeque;
use crate::runtime::tasks::AsyncTaskReq;
use crate::runtime::tasks::AsyncTaskResp;
use crate::runtime::tasks::AwaitReservation;

pub struct ScheduleQueue<UserReq, DeviceReq, DeviceResp> {
    id_requests: HashMap<RawRequestID, UserReq>,
    run_queue: DedupVecDeque<RawRequestID>,

    new_queue: VecDeque<UserReq>,
    waiting_reqs: FuturesUnordered<AwaitReservation<UserReq, DeviceReq, DeviceResp>>,
    async_task_req_tx: Sender<Box<dyn AsyncTaskReq<Resp = dyn AsyncTaskResp>>>,

    phantom_data_dev_req: PhantomData<DeviceReq>,
    phantom_data_dev_resp: PhantomData<DeviceResp>,
}

impl<UserReq, DeviceReq, DeviceResp> ScheduleQueue<UserReq, DeviceReq, DeviceResp>
where
    UserReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
{
    pub fn new(async_task_req_tx: Sender<Box<dyn AsyncTaskReq<Resp = dyn AsyncTaskResp>>>) -> Self {
        Self {
            id_requests: hash_map! {},

            new_queue: VecDeque::new(),
            run_queue: DedupVecDeque::new(),
            waiting_reqs: FuturesUnordered::new(),
            async_task_req_tx,

            phantom_data_dev_req: PhantomData,
            phantom_data_dev_resp: PhantomData,
        }
    }

    pub fn peek_front(&mut self) -> Option<&mut UserReq> {
        while let Some(req_id) = self.run_queue.front() {
            if self.id_requests.contains_key(req_id) {
                return self.id_requests.get_mut(req_id);
            } else {
                self.run_queue.pop_front();
            }
        }

        let user_req = self.new_queue.pop_front()?;
        let req_id = user_req.id();
        self.run_queue.push_back(req_id);
        match self.id_requests.entry(req_id) {
            HashEntry::Occupied(_) => unreachable!(),
            HashEntry::Vacant(entry) => Some(entry.insert(user_req)),
        }
    }

    pub fn pop_front(&mut self) -> Option<UserReq> {
        self.peek_front()?;
        let Some(req_id) = self.run_queue.pop_front() else {
            unreachable!()
        };
        self.id_requests.remove(&req_id)
    }

    pub fn push_front(&mut self, user_req: UserReq) {
        let req_id = user_req.id();
        self.run_queue.push_front(req_id);
        match self.id_requests.entry(req_id) {
            HashEntry::Occupied(_) => unreachable!(),
            HashEntry::Vacant(entry) => entry.insert(user_req),
        };
    }

    pub fn pop_back(&mut self) -> Option<UserReq> {
        while let Some(req_id) = self.run_queue.pop_back() {
            match self.id_requests.remove(&req_id) {
                Some(user_req) => return Some(user_req),
                None => continue,
            };
        }
        None
    }

    pub fn push_back(&mut self, user_req: UserReq) {
        let req_id = user_req.id();
        self.run_queue.push_back(req_id);
        match self.id_requests.entry(req_id) {
            HashEntry::Occupied(_) => unreachable!(),
            HashEntry::Vacant(entry) => entry.insert(user_req),
        };
    }

    pub fn enqueue(&mut self, user_req: UserReq) {
        self.new_queue.push_back(user_req);
    }

    pub fn push_waiting_reqs(&mut self, user_req: UserReq, wait: Boxed<()>) {
        self.waiting_reqs.push(AwaitReservation::new(user_req, wait));
    }

    pub fn pop_ready_reqs(&mut self) -> Option<UserReq> {
        let mut cx = Context::from_waker(noop_waker_ref());
        match Pin::new(&mut self.waiting_reqs).poll_next(&mut cx) {
            Poll::Ready(Some(user_req)) => Some(user_req),
            Poll::Ready(None) | Poll::Pending => None,
        }
    }

    pub fn handle_ready_waits(&mut self) {
        if self.waiting_reqs.is_empty() {
            return;
        }
        while let Some(user_req) = self.pop_ready_reqs() {
            if user_req.is_terminal() {
                tracing::debug!(
                    target: "inference-runtime-core::scheduler",
                    phase = "request.reservation_wait_terminal",
                    request_id = user_req.id(),
                    "terminal reservation-wait request dropped"
                );
            } else {
                self.push_back(user_req);
            }
        }
    }

    pub fn handle_async_task_req(
        &self,
        req: Box<dyn AsyncTaskReq<Resp = dyn AsyncTaskResp>>,
    ) -> Result<(), TrySendError<Box<dyn AsyncTaskReq<Resp = dyn AsyncTaskResp>>>> {
        self.async_task_req_tx.try_send(req)
    }

    pub fn handle_async_task_resp(&mut self, resp: Box<dyn AsyncTaskResp>) {
        let req_id = resp.request_id();
        let user_req = self
            .id_requests
            .get_mut(&req_id)
            .expect("async task response must reference a retained request");
        user_req.handle_async_task_resp(resp);
        let terminal = user_req.is_terminal();
        let has_in_flight_compute = user_req.num_in_flight_computes() != 0;
        let has_in_flight_blocking_async_task = user_req.num_in_flight_blocking_async_tasks() != 0;
        let has_in_flight_nonblocking_async_task = user_req.num_in_flight_nonblocking_async_tasks() != 0;
        if terminal
            && !has_in_flight_compute
            && !has_in_flight_blocking_async_task
            && !has_in_flight_nonblocking_async_task
        {
            self.id_requests.remove(&req_id);
        } else if !terminal && !has_in_flight_blocking_async_task {
            self.run_queue.push_back(req_id);
        }
    }

    pub fn insert(&mut self, user_req: UserReq) {
        let req_id = user_req.id();
        match self.id_requests.entry(req_id) {
            HashEntry::Occupied(_) => unreachable!(),
            HashEntry::Vacant(entry) => entry.insert(user_req),
        };
    }

    pub fn remove(&mut self, req_id: &RawRequestID) -> Option<UserReq> {
        self.run_queue.remove(req_id);
        self.id_requests.remove(req_id)
    }

    pub fn get_ref(&self, req_id: &RawRequestID) -> Option<&UserReq> {
        self.id_requests.get(req_id)
    }

    pub fn token_estimate(&self, max_token_per_req: usize) -> usize {
        self.run_queue.iter().fold(0, |sum, req_id| {
            sum + self
                .id_requests
                .get(req_id)
                .map(|user_req| user_req.token_estimate().token_consumption(max_token_per_req))
                .unwrap_or(0)
        }) + self.new_queue.iter().fold(0, |sum, user_req| {
            sum + user_req.token_estimate().token_consumption(max_token_per_req)
        })
    }

    pub fn run_queue_size(&self) -> usize {
        self.run_queue.len()
    }

    pub fn new_queue_size(&self) -> usize {
        self.new_queue.len()
    }
}

#[cfg(test)]
mod tests {
    use std::future;

    use async_channel::bounded;
    use event_listener::Event;

    use super::*;
    use crate::compute::MockDevReq;
    use crate::compute::MockDevResp;
    use crate::runtime::scheduler::MockUserRequest;

    type TestUserReq = MockUserRequest<MockDevReq, MockDevResp>;
    type TestScheduleQueue = ScheduleQueue<TestUserReq, MockDevReq, MockDevResp>;

    struct TestAsyncTaskResp {
        request_id: RawRequestID,
    }

    impl AsyncTaskResp for TestAsyncTaskResp {
        fn request_id(&self) -> RawRequestID {
            self.request_id
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn test_pop_ready_reqs_polls_notified_wait() {
        let req_id = 7;
        let event = Event::new();
        let wait = event.listen();

        let mut user_req = TestUserReq::new();
        user_req.expect_id().return_const(req_id);
        user_req.expect_is_terminal().once().return_const(false);

        let (async_task_req_tx, _async_task_req_rx) = bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_waiting_reqs(user_req, Box::pin(wait));

        event.notify(usize::MAX);

        assert_eq!(schedule_queue.pop_ready_reqs().map(|req| req.id()), Some(req_id));
    }

    #[test]
    fn test_pop_ready_reqs_returns_later_ready_req() {
        let mut pending_req = TestUserReq::new();
        pending_req.expect_is_terminal().once().return_const(false);

        let ready_req_id = 7;
        let mut ready_req = TestUserReq::new();
        ready_req.expect_id().return_const(ready_req_id);
        ready_req.expect_is_terminal().once().return_const(false);

        let (async_task_req_tx, _async_task_req_rx) = bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_waiting_reqs(pending_req, Box::pin(future::pending()));
        schedule_queue.push_waiting_reqs(ready_req, Box::pin(async {}));

        assert_eq!(schedule_queue.pop_ready_reqs().map(|req| req.id()), Some(ready_req_id));
        assert!(schedule_queue.pop_ready_reqs().is_none());
    }

    #[test]
    fn test_pop_ready_reqs_returns_terminal_req() {
        let req_id: RawRequestID = 7;
        let mut user_req = TestUserReq::new();
        user_req.expect_id().return_const(req_id);
        user_req.expect_is_terminal().once().return_const(true);

        let (async_task_req_tx, _async_task_req_rx) = bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.push_waiting_reqs(user_req, Box::pin(async {}));

        assert_eq!(schedule_queue.pop_ready_reqs().map(|req| req.id()), Some(req_id));
    }

    #[test]
    fn test_handle_async_task_resp_requeues_request_wo_in_flight_work() {
        let schedule_queue = schedule_queue_after_async_task_resp(false, 0, 0, 0);

        assert_eq!(1, schedule_queue.run_queue_size());
        assert!(schedule_queue.get_ref(&7).is_some());
    }

    #[test]
    fn test_handle_async_task_resp_requeues_request_w_in_flight_compute() {
        let schedule_queue = schedule_queue_after_async_task_resp(false, 1, 0, 0);

        assert_eq!(1, schedule_queue.run_queue_size());
        assert!(schedule_queue.get_ref(&7).is_some());
    }

    #[test]
    fn test_handle_async_task_resp_requeues_request_w_in_flight_nonblocking_async_task() {
        let schedule_queue = schedule_queue_after_async_task_resp(false, 0, 0, 1);

        assert_eq!(1, schedule_queue.run_queue_size());
        assert!(schedule_queue.get_ref(&7).is_some());
    }

    #[test]
    fn test_handle_async_task_resp_keeps_request_blocked_w_in_flight_blocking_async_task() {
        let schedule_queue = schedule_queue_after_async_task_resp(false, 0, 1, 0);

        assert_eq!(0, schedule_queue.run_queue_size());
        assert!(schedule_queue.get_ref(&7).is_some());
    }

    #[test]
    fn test_handle_async_task_resp_removes_terminal_request_wo_in_flight_work() {
        let schedule_queue = schedule_queue_after_async_task_resp(true, 0, 0, 0);

        assert_eq!(0, schedule_queue.run_queue_size());
        assert!(schedule_queue.get_ref(&7).is_none());
    }

    #[test]
    fn test_handle_async_task_resp_retains_terminal_request_w_in_flight_compute() {
        let schedule_queue = schedule_queue_after_async_task_resp(true, 1, 0, 0);

        assert_eq!(0, schedule_queue.run_queue_size());
        assert!(schedule_queue.get_ref(&7).is_some());
    }

    #[test]
    fn test_handle_async_task_resp_retains_terminal_request_w_in_flight_blocking_async_task() {
        let schedule_queue = schedule_queue_after_async_task_resp(true, 0, 1, 0);

        assert_eq!(0, schedule_queue.run_queue_size());
        assert!(schedule_queue.get_ref(&7).is_some());
    }

    #[test]
    fn test_handle_async_task_resp_retains_terminal_request_w_in_flight_nonblocking_async_task() {
        let schedule_queue = schedule_queue_after_async_task_resp(true, 0, 0, 1);

        assert_eq!(0, schedule_queue.run_queue_size());
        assert!(schedule_queue.get_ref(&7).is_some());
    }

    fn schedule_queue_after_async_task_resp(
        terminal: bool,
        num_in_flight_computes: usize,
        num_in_flight_blocking_async_tasks: usize,
        num_in_flight_nonblocking_async_tasks: usize,
    ) -> TestScheduleQueue {
        let req_id = 7;
        let mut user_req = TestUserReq::new();
        user_req.expect_id().once().return_const(req_id);
        user_req.expect_handle_async_task_resp().once().return_once(|_| {});
        user_req.expect_is_terminal().once().return_const(terminal);
        user_req
            .expect_num_in_flight_computes()
            .once()
            .return_const(num_in_flight_computes);
        user_req
            .expect_num_in_flight_blocking_async_tasks()
            .once()
            .return_const(num_in_flight_blocking_async_tasks);
        user_req
            .expect_num_in_flight_nonblocking_async_tasks()
            .once()
            .return_const(num_in_flight_nonblocking_async_tasks);

        let (async_task_req_tx, _async_task_req_rx) = bounded(1);
        let mut schedule_queue = TestScheduleQueue::new(async_task_req_tx);
        schedule_queue.insert(user_req);

        schedule_queue.handle_async_task_resp(Box::new(TestAsyncTaskResp { request_id: req_id }));
        schedule_queue
    }
}
