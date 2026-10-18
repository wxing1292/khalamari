use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use async_channel::Sender as AsyncSender;
use crossbeam_channel::Receiver;
use crossbeam_channel::Select;
use crossbeam_channel::Sender;
use crossbeam_channel::after;
use crossbeam_channel::never;

use super::executor_hibernate::collect_allocated_id_ranges;
use crate::Result;
use crate::channel::Shutdown;
use crate::compute::BatchDevReq;
use crate::compute::BatchDevResp;
use crate::compute::DevReq;
use crate::compute::DevResp;
use crate::compute::ExecutorHibernationPlan;
use crate::compute::ReplayableModelExecutorRequest;
use crate::compute::ReplayableModelExecutorResponse;
use crate::config::ExecutorHibernationMode;
use crate::log_err_internal;
use crate::memory::U32IDAllocator;
use crate::runtime::CompletionReason;
use crate::runtime::RequestSlotAllocator;
use crate::runtime::scheduler::InstrumentedScheduler;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::scheduler::UserRequest;
use crate::runtime::tasks::AsyncTaskResp;

const SCHEDULER_STATS_INTERVAL: Duration = Duration::from_secs(30);

pub struct EventLoop<UsrReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S> {
    new_req_rx: Receiver<UsrReq>,
    completed_turn_tx: AsyncSender<(UsrReq, CompletionReason)>,
    async_task_resp_rx: Receiver<Box<dyn AsyncTaskResp>>,
    model_executor_req_tx: Sender<ReplayableModelExecutorRequest<BatchDeviceReq>>,
    model_executor_resp_rx: Receiver<ReplayableModelExecutorResponse<BatchDeviceResp>>,

    scheduler: InstrumentedScheduler<S>,
    request_slot_allocator: RequestSlotAllocator,
    page_id_allocator: Arc<U32IDAllocator>,
    model_executor_state: ModelExecutorState,

    scheduler_stats_timer: Receiver<Instant>,

    executor_hibernation_mode: ExecutorHibernationMode,
    executor_hibernation_timeout: Duration,
    executor_hibernation_timer: Receiver<Instant>,
    num_in_flight_batches: usize,

    shutdown: Shutdown,

    phantom_data_device_req: PhantomData<DeviceReq>,
    phantom_data_device_resp: PhantomData<DeviceResp>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ModelExecutorState {
    Started,
    Stopped(ExecutorHibernationPlan),
}

impl<UsrReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S>
    EventLoop<UsrReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp, S>
where
    UsrReq: UserRequest<DeviceReq, DeviceResp>,
    DeviceReq: DevReq,
    DeviceResp: DevResp,
    BatchDeviceReq: BatchDevReq<DeviceReq>,
    BatchDeviceResp: BatchDevResp<DeviceResp>,
    S: Scheduler<UsrReq, DeviceReq, DeviceResp, BatchDeviceReq, BatchDeviceResp>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        new_req_rx: Receiver<UsrReq>,
        completed_turn_tx: AsyncSender<(UsrReq, CompletionReason)>,
        async_task_resp_rx: Receiver<Box<dyn AsyncTaskResp>>,
        model_executor_req_tx: Sender<ReplayableModelExecutorRequest<BatchDeviceReq>>,
        model_executor_resp_rx: Receiver<ReplayableModelExecutorResponse<BatchDeviceResp>>,
        scheduler: InstrumentedScheduler<S>,
        request_slot_allocator: RequestSlotAllocator,
        page_id_allocator: Arc<U32IDAllocator>,
        executor_hibernation_mode: ExecutorHibernationMode,
        executor_hibernation_timeout: Duration,
        shutdown: Shutdown,
    ) -> Self {
        assert!(
            !executor_hibernation_timeout.is_zero(),
            "runtime executor hibernation timeout must be positive"
        );
        Self {
            new_req_rx,
            completed_turn_tx,
            async_task_resp_rx,
            model_executor_req_tx,
            model_executor_resp_rx,
            scheduler,
            request_slot_allocator,
            page_id_allocator,
            model_executor_state: ModelExecutorState::Started,

            scheduler_stats_timer: after(SCHEDULER_STATS_INTERVAL),

            executor_hibernation_mode,
            executor_hibernation_timeout,
            executor_hibernation_timer: after(executor_hibernation_timeout),
            num_in_flight_batches: 0,

            shutdown,

            phantom_data_device_req: PhantomData,
            phantom_data_device_resp: PhantomData,
        }
    }

    pub fn event_loop(mut self) -> Result<()> {
        let _span = tracing::info_span!("runtime").entered();
        tracing::info!("started");

        let shutdown_rx = self.shutdown.sync_rx().clone();
        let mut result = Ok(());
        'event_loop: while !self.shutdown.is_shutdown() {
            let mut select = Select::new();
            let op_recv_new_req = select.recv(&self.new_req_rx);
            let op_recv_async_task_resp = select.recv(&self.async_task_resp_rx);
            let op_recv_model_executor_resp = select.recv(&self.model_executor_resp_rx);
            let op_scheduler_stats_timer = select.recv(&self.scheduler_stats_timer);
            let op_executor_hibernation_timer = match &self.model_executor_state {
                ModelExecutorState::Started => Some(select.recv(&self.executor_hibernation_timer)),
                ModelExecutorState::Stopped(_) => None,
            };
            let op_shutdown = select.recv(&shutdown_rx);
            let op = select.select();
            let op_index = op.index();
            match op_index {
                _ if op_index == op_recv_new_req => {
                    let Ok(new_req) = op.recv(&self.new_req_rx) else {
                        result = Err(log_err_internal!("new-request channel closed, stopping"));
                        break 'event_loop;
                    };
                    if !new_req.is_terminal() {
                        self.scheduler.enqueue(new_req);
                    }
                },
                _ if op_index == op_recv_async_task_resp => {
                    let Ok(resp) = op.recv(&self.async_task_resp_rx) else {
                        result = Err(log_err_internal!("async task response channel closed, stopping"));
                        break 'event_loop;
                    };
                    self.scheduler.handle_async_task_resp(resp);
                },
                _ if op_index == op_recv_model_executor_resp => {
                    let Ok(model_executor_resp) = op.recv(&self.model_executor_resp_rx) else {
                        result = Err(log_err_internal!("model executor response channel closed, stopping"));
                        break 'event_loop;
                    };
                    match model_executor_resp {
                        ReplayableModelExecutorResponse::Batch(batch_dev_resp) => {
                            for completed_turn in self.scheduler.commit(batch_dev_resp) {
                                let Ok(()) = self.completed_turn_tx.try_send(completed_turn) else {
                                    result = Err(log_err_internal!("completed-turn channel send failed, stopping"));
                                    break 'event_loop;
                                };
                            }
                            self.num_in_flight_batches -= 1;
                            if self.num_in_flight_batches == 0 {
                                self.executor_hibernation_timer = after(self.executor_hibernation_timeout);
                            }
                        },
                        ReplayableModelExecutorResponse::Started => {
                            if self.num_in_flight_batches == 0 {
                                self.executor_hibernation_timer = after(self.executor_hibernation_timeout);
                            }
                        },
                        ReplayableModelExecutorResponse::Stopped => {},
                    }
                },
                _ if op_index == op_scheduler_stats_timer => {
                    let _ = op
                        .recv(&self.scheduler_stats_timer)
                        .expect("selected scheduler stats timer must fire");
                    self.scheduler.print_periodical();
                    self.scheduler_stats_timer = after(SCHEDULER_STATS_INTERVAL);
                },
                _ if Some(op_index) == op_executor_hibernation_timer => {
                    let _ = op
                        .recv(&self.executor_hibernation_timer)
                        .expect("selected executor hibernation timer must fire");
                    if let Err(error) = self.stop_model_executor() {
                        result = Err(error);
                        break 'event_loop;
                    }
                },
                _ if op_index == op_shutdown => {
                    let _ = op.recv(&shutdown_rx);
                    break 'event_loop;
                },
                _ => unreachable!(),
            }
            if let Err(error) = self.do_flush() {
                result = Err(error);
                break 'event_loop;
            }
        }

        self.shutdown.shutdown();
        self.scheduler.print_lifetime();
        tracing::info!("stopped");
        result
    }

    fn do_flush(&mut self) -> Result<()> {
        match &self.model_executor_state {
            ModelExecutorState::Started => {
                while self.scheduler.can_flush() {
                    let batch_dev_req = self.scheduler.prepare();
                    match self
                        .model_executor_req_tx
                        .try_send(ReplayableModelExecutorRequest::Batch(batch_dev_req))
                    {
                        Ok(()) => {
                            self.num_in_flight_batches += 1;
                            self.executor_hibernation_timer = never();
                        },
                        Err(error) => {
                            let ReplayableModelExecutorRequest::Batch(batch_dev_req) = error.into_inner() else {
                                unreachable!("runtime scheduler only sends Batch model executor requests")
                            };
                            self.scheduler.cancel(batch_dev_req);
                            return Err(log_err_internal!(
                                "batch device request channel full / closed, stopping"
                            ));
                        },
                    }
                }
            },
            ModelExecutorState::Stopped(_) if self.scheduler.can_flush() => {
                self.start_model_executor()?;
            },
            ModelExecutorState::Stopped(_) => {},
        }
        Ok(())
    }

    fn start_model_executor(&mut self) -> Result<()> {
        let ModelExecutorState::Stopped(plan) = &self.model_executor_state else {
            panic!("runtime can start only a stopped model executor")
        };

        self.model_executor_req_tx
            .try_send(ReplayableModelExecutorRequest::Start(plan.clone()))
            .map_err(|error| {
                log_err_internal!("model executor request channel full / closed while starting, stopping: {error}")
            })?;
        self.model_executor_state = ModelExecutorState::Started;
        self.executor_hibernation_timer = never();
        Ok(())
    }

    fn stop_model_executor(&mut self) -> Result<()> {
        debug_assert_eq!(self.model_executor_state, ModelExecutorState::Started);
        debug_assert_eq!(self.num_in_flight_batches, 0);

        let plan = match self.executor_hibernation_mode {
            ExecutorHibernationMode::All => ExecutorHibernationPlan::All,
            ExecutorHibernationMode::Selected => {
                ExecutorHibernationPlan::selected(
                    collect_allocated_id_ranges(self.request_slot_allocator.allocated_ids_bitmap_iter()),
                    collect_allocated_id_ranges(self.page_id_allocator.allocated_ids_bitmap_iter()),
                )
            },
        };

        self.model_executor_req_tx
            .try_send(ReplayableModelExecutorRequest::Stop(plan.clone()))
            .map_err(|error| {
                log_err_internal!("model executor request channel full / closed while stopping, stopping: {error}")
            })?;
        self.model_executor_state = ModelExecutorState::Stopped(plan);
        Ok(())
    }
}
