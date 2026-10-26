use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::Select;
use crossbeam_channel::SelectedOperation;
use crossbeam_channel::Sender;
use inference_executor_core::model::EncoderExecutorLifecycle;
use inference_executor_core::model::ExecutionSubmission;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_runtime_core::channel::DedupNotifier;
use inference_runtime_core::channel::Shutdown;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::DeviceResponse;
use inference_runtime_core::compute::ExecutorHibernationPlan;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::compute::ReplayableModelExecutorRequest;
use inference_runtime_core::compute::ReplayableModelExecutorResponse;
use inference_runtime_core::compute::SampledTokens;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::Token;

use crate::executor::ReplayableModelExecutors;
use crate::perf_metrics::ExecutorBatchPerfMetrics;
use crate::perf_metrics::summarize_batch_device_request;
use crate::perf_metrics::summarize_batch_device_response;
use crate::profiling;
use crate::telemetry::emit_executor_batch_perf_metrics;

pub struct ReplayableDecoderModelEventLoop<M> {
    model_executor_req_rx: Receiver<ReplayableModelExecutorRequest>,
    model_executor_resp_tx: Sender<ReplayableModelExecutorResponse>,
    req_slot_reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
    req_slot_reset_rx: Receiver<()>,
    shutdown: Shutdown,
    model: M,
    encoder_executors: Vec<Arc<dyn EncoderExecutorLifecycle>>,
    model_state: ModelExecutorState,
    state_snapshot_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ModelExecutorState {
    Started,
    Stopped(ExecutorHibernationPlan),
}

impl<M> ReplayableDecoderModelEventLoop<M>
where
    M: ReplayableDecoderModel,
{
    pub fn new(
        model_executor_req_rx: Receiver<ReplayableModelExecutorRequest>,
        model_executor_resp_tx: Sender<ReplayableModelExecutorResponse>,
        req_slot_reset_notifier: Arc<DedupNotifier<RawRequestSlot>>,
        req_slot_reset_rx: Receiver<()>,
        shutdown: Shutdown,
        executors: ReplayableModelExecutors<M>,
        state_snapshot_path: PathBuf,
    ) -> Self {
        let (model, encoder_executors) = executors.into_parts();
        Self {
            model_executor_req_rx,
            model_executor_resp_tx,
            req_slot_reset_notifier,
            req_slot_reset_rx,
            shutdown,
            model,
            encoder_executors,
            model_state: ModelExecutorState::Started,
            state_snapshot_path,
        }
    }

    pub fn event_loop(mut self) {
        let _span = tracing::info_span!("replayable-executor", model = self.model.model_name()).entered();
        tracing::info!("started");

        let shutdown_rx = self.shutdown.sync_rx().clone();
        'event_loop: while !self.shutdown.is_shutdown() {
            let mut select = Select::new();
            let op_shutdown = select.recv(&shutdown_rx);
            let op_recv_req_slot_reset = select.recv(&self.req_slot_reset_rx);
            let op_recv_model_executor_req = select.recv(&self.model_executor_req_rx);

            let op = select.select();
            let op_index = op.index();
            match op_index {
                _ if op_index == op_shutdown => {
                    let _ = op.recv(&shutdown_rx);
                    break 'event_loop;
                },
                _ if op_index == op_recv_req_slot_reset => {
                    match op.recv(&self.req_slot_reset_rx) {
                        Ok(()) => {
                            if matches!(self.model_state, ModelExecutorState::Started) {
                                self.reset_req_slots();
                            } else {
                                tracing::debug!("model is stopped; deferring request slot resets until start");
                            }
                        },
                        Err(_) => {
                            tracing::debug!("request slot reset channel closed");
                            break 'event_loop;
                        },
                    }
                },
                _ if op_index == op_recv_model_executor_req => {
                    let Some(request) = self.recv_request(op) else {
                        break 'event_loop;
                    };
                    match request {
                        ReplayableModelExecutorRequest::Batch(batch_dev_req) => {
                            if !self.start(None) {
                                break 'event_loop;
                            }
                            self.reset_req_slots();
                            let batch_dev_resp = self.execute(batch_dev_req);
                            if !self.send_response(ReplayableModelExecutorResponse::Batch(batch_dev_resp)) {
                                break 'event_loop;
                            }
                        },
                        ReplayableModelExecutorRequest::Start(plan) => {
                            if !self.start(Some(plan)) {
                                break 'event_loop;
                            }
                            self.reset_req_slots();
                            if !self.send_response(ReplayableModelExecutorResponse::Started) {
                                break 'event_loop;
                            }
                        },
                        ReplayableModelExecutorRequest::Stop(plan) => {
                            if !self.stop(plan) {
                                break 'event_loop;
                            }
                            if !self.send_response(ReplayableModelExecutorResponse::Stopped) {
                                break 'event_loop;
                            }
                        },
                    }
                },
                _ => unreachable!(),
            }
        }

        self.remove_state_snapshot();
        self.shutdown.shutdown();
        tracing::info!("stopped");
    }

    fn start(&mut self, requested_plan: Option<ExecutorHibernationPlan>) -> bool {
        let ModelExecutorState::Stopped(snapshot_plan) = &self.model_state else {
            return true;
        };
        if let Some(requested_plan) = requested_plan {
            assert_eq!(
                requested_plan, *snapshot_plan,
                "model Start hibernation plan must match the preceding Stop plan"
            );
        }
        let snapshot_plan = snapshot_plan.clone();

        let start = Instant::now();
        tracing::info!(
            target: "inference-runtime-service::lifecycle",
            component = "model",
            phase = "start.begin",
            model = self.model.model_name(),
            snapshot_path = %self.state_snapshot_path.display(),
            "starting model"
        );
        if let Err(error) = self.model.load_weights() {
            tracing::error!(
                target: "inference-runtime-service::lifecycle",
                component = "model",
                phase = "start.failed",
                model = self.model.model_name(),
                error = %error,
                "unable to load model weights; shutting down"
            );
            self.shutdown.shutdown();
            return false;
        }
        if let Err(error) = self.model.load_state(&self.state_snapshot_path, &snapshot_plan) {
            tracing::error!(
                target: "inference-runtime-service::lifecycle",
                component = "model",
                phase = "start.failed",
                model = self.model.model_name(),
                error = %error,
                "unable to load model state; shutting down"
            );
            self.shutdown.shutdown();
            return false;
        }
        for encoder in &self.encoder_executors {
            if let Err(error) = encoder.start() {
                tracing::error!(
                    target: "inference-runtime-service::lifecycle",
                    component = "encoder",
                    phase = "start.failed",
                    model = encoder.model_name(),
                    error = %error,
                    "unable to load encoder weights; shutting down"
                );
                self.shutdown.shutdown();
                return false;
            }
        }
        self.model_state = ModelExecutorState::Started;
        self.remove_state_snapshot();
        tracing::info!(
            target: "inference-runtime-service::lifecycle",
            component = "model",
            phase = "start.complete",
            model = self.model.model_name(),
            elapsed_ms = start.elapsed().as_millis(),
            "model started"
        );
        true
    }

    fn stop(&mut self, plan: ExecutorHibernationPlan) -> bool {
        if let ModelExecutorState::Stopped(snapshot_plan) = &self.model_state {
            assert_eq!(
                plan, *snapshot_plan,
                "idempotent model Stop must use the existing hibernation plan"
            );
            return true;
        }
        plan.assert_valid();

        let start = Instant::now();
        tracing::info!(
            target: "inference-runtime-service::lifecycle",
            component = "model",
            phase = "stop.begin",
            model = self.model.model_name(),
            snapshot_path = %self.state_snapshot_path.display(),
            "stopping model"
        );
        for encoder in &self.encoder_executors {
            encoder.stop();
        }
        self.reset_req_slots();
        self.model.clear_replay_cache();
        if let Err(error) = self.model.unload_state(&self.state_snapshot_path, &plan) {
            tracing::error!(
                target: "inference-runtime-service::lifecycle",
                component = "model",
                phase = "stop.failed",
                model = self.model.model_name(),
                error = %error,
                "unable to unload model state; shutting down"
            );
            self.shutdown.shutdown();
            return false;
        }
        self.model.unload_weights();
        self.model_state = ModelExecutorState::Stopped(plan);
        tracing::info!(
            target: "inference-runtime-service::lifecycle",
            component = "model",
            phase = "stop.complete",
            model = self.model.model_name(),
            elapsed_ms = start.elapsed().as_millis(),
            "model stopped"
        );
        true
    }

    fn recv_request(&self, operation: SelectedOperation<'_>) -> Option<ReplayableModelExecutorRequest> {
        match operation.recv(&self.model_executor_req_rx) {
            Ok(request) => Some(request),
            Err(error) => {
                tracing::debug!("model executor request channel closed: {error}");
                None
            },
        }
    }

    fn send_response(&self, response: ReplayableModelExecutorResponse) -> bool {
        if let Err(error) = self.model_executor_resp_tx.send(response) {
            tracing::debug!("model executor response channel closed: {error}");
            return false;
        }
        true
    }

    fn remove_state_snapshot(&self) {
        let result = match std::fs::symlink_metadata(&self.state_snapshot_path) {
            Ok(metadata) if metadata.file_type().is_dir() => std::fs::remove_dir_all(&self.state_snapshot_path),
            Ok(_) => std::fs::remove_file(&self.state_snapshot_path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
            Err(error) => Err(error),
        };
        match result {
            Ok(()) => {},
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
            Err(error) => {
                tracing::warn!(
                    target: "inference-runtime-service::lifecycle",
                    model = self.model.model_name(),
                    snapshot_path = %self.state_snapshot_path.display(),
                    error = %error,
                    "unable to remove model state snapshot"
                );
            },
        }
    }

    fn reset_req_slots(&mut self) {
        self.req_slot_reset_rx.try_iter().for_each(drop);
        let req_slots = self.req_slot_reset_notifier.recv_many();
        if !req_slots.is_empty() {
            let req_slots = req_slots.into_iter().collect::<Vec<_>>();
            self.model.reset_req_slots(&req_slots);
        }
    }

    fn execute(&mut self, mut batch_req: BatchDeviceRequest) -> BatchDeviceResponse {
        let batch_seq = batch_req.seq;
        let batch_summary = summarize_batch_device_request(&batch_req);
        tracing::debug!(
            target: "inference-runtime-service::executor",
            component = "executor",
            phase = "executor.batch.request",
            model = self.model.model_name(),
            batch_seq,
            num_reqs = batch_req.dev_reqs.len(),
            requests = %batch_req
                .dev_reqs
                .iter()
                .map(summarize_device_request)
                .collect::<Vec<_>>()
                .join(" | "),
            "executor batch request"
        );

        let _executor_batch_span = profiling::span("executor.batch");

        if batch_req.dev_reqs.is_empty() {
            return BatchDeviceResponse::new(batch_seq, Vec::new());
        }

        let model_batch_req = {
            let _span = profiling::span("prepare_batch");
            self.model.prepare_batch(&mut batch_req)
        };

        let mut recorder = self.model.begin_ops_recording(&model_batch_req);

        let main_start = Instant::now();
        let model_batch_hidden_req = {
            let _span = profiling::span("model.embed_main.record");
            if self.model.first_pp_stage(&model_batch_req) {
                self.model.embed_main(&mut recorder, &model_batch_req)
            } else {
                todo!("pipeline stages after the first must read hidden states from the batch request")
            }
        };

        let model_batch_hidden_resp = {
            let _span = profiling::span("model.forward_main.record");
            self.model
                .forward_main(&mut recorder, &model_batch_req, model_batch_hidden_req)
        };

        let last_pp_stage = self.model.last_pp_stage(&model_batch_req);

        if last_pp_stage {
            let model_batch_resp = {
                let _span = profiling::span("model.unembed_main.record");
                self.model
                    .unembed_main(&mut recorder, &model_batch_req, &model_batch_hidden_resp)
            };
            {
                let _span = profiling::span("model.sample_main.record");
                self.model
                    .sample_main(&mut recorder, &model_batch_req, &model_batch_resp);
            }
        }

        let main_replay_start = Instant::now();
        let submission = {
            let _span = profiling::span("model.submit_main.wait");
            let submission = self.model.submit_main(&recorder);
            submission.wait();
            submission
        };
        let main_replay_elapsed = main_replay_start.elapsed();
        let gpu_timestamp_durations = submission.gpu_timestamp_durations();
        drop(submission);
        let mut sampled_output = if last_pp_stage {
            let _span = profiling::span("model.read_main");
            self.model.read_main(
                &recorder,
                &model_batch_req,
                main_replay_elapsed,
                gpu_timestamp_durations.as_deref(),
            )
        } else {
            self.model.empty_sampled_output()
        };
        let main_elapsed = main_start.elapsed();

        let run_spec = last_pp_stage && self.model.run_spec(&model_batch_req, &sampled_output);
        let run_spec_prefill = last_pp_stage && self.model.run_spec_prefill(&model_batch_req);
        let run_spec_decode = last_pp_stage && self.model.run_spec_decode(&model_batch_req, &sampled_output);
        debug_assert!(
            !run_spec || (!run_spec_prefill && !run_spec_decode),
            "model executor must not mix the combined Spec invocation with Spec Prefill or Spec Decode"
        );
        let mut spec_elapsed = Duration::ZERO;
        if run_spec || run_spec_prefill || run_spec_decode {
            let spec_start = Instant::now();
            if run_spec_prefill {
                let _span = profiling::span("model.prefill_spec.record");
                self.model
                    .prefill_spec(&mut recorder, &model_batch_req, &sampled_output);
            }
            if run_spec_decode {
                let _span = profiling::span("model.decode_spec.record");
                self.model.decode_spec(&mut recorder, &model_batch_req, &sampled_output);
            }
            if run_spec {
                let spec_batch_hidden_req = {
                    let _span = profiling::span("model.embed_spec.record");
                    self.model.embed_spec(
                        &mut recorder,
                        &model_batch_req,
                        &model_batch_hidden_resp,
                        &sampled_output,
                    )
                };
                let spec_batch_hidden_resp = {
                    let _span = profiling::span("model.forward_spec.record");
                    self.model
                        .forward_spec(&mut recorder, &model_batch_req, spec_batch_hidden_req)
                };
                let spec_batch_resp = {
                    let _span = profiling::span("model.unembed_spec.record");
                    self.model
                        .unembed_spec(&mut recorder, &model_batch_req, &spec_batch_hidden_resp)
                };
                {
                    let _span = profiling::span("model.sample_spec.record");
                    self.model
                        .sample_spec(&mut recorder, &model_batch_req, &spec_batch_resp);
                }
            }
            let spec_replay_start = Instant::now();
            let submission = {
                let _span = profiling::span("model.submit_spec.wait");
                let submission = self.model.submit_spec(&recorder);
                submission.wait();
                submission
            };
            let spec_replay_elapsed = spec_replay_start.elapsed();
            let spec_gpu_timestamp_durations = submission.gpu_timestamp_durations();
            drop(submission);
            if run_spec || run_spec_decode {
                let _span = profiling::span("model.read_spec");
                sampled_output = self.model.read_spec(
                    &recorder,
                    &model_batch_req,
                    sampled_output,
                    spec_replay_elapsed,
                    spec_gpu_timestamp_durations.as_deref(),
                );
            }
            spec_elapsed = spec_start.elapsed();
        }
        drop(recorder);
        let model_output_timing = self.model.sampled_output_timing(&sampled_output).unwrap_or_default();

        let mut batch_resp = {
            let _span = profiling::span("commit_batch");
            self.model.commit_batch(batch_req, sampled_output)
        };
        let mode = self.model.model_mode();
        let main_gpu = model_output_timing
            .main_gpu_elapsed
            .map(|main| main + model_output_timing.rejection_gpu_elapsed.unwrap_or_default());
        batch_resp.execution_timings = vec![("main".into(), main_gpu)];
        if mode != "vanilla" {
            batch_resp
                .execution_timings
                .push((mode.into(), model_output_timing.spec_gpu_elapsed()));
        }
        let response_summary = summarize_batch_device_response(&batch_resp);
        drop(_executor_batch_span);

        profiling::maybe_emit_tree_profile_summary("executor.batch", batch_seq);
        emit_executor_batch_perf_metrics(
            self.model.model_name(),
            self.model.model_mode(),
            batch_seq,
            batch_summary,
            response_summary,
            ExecutorBatchPerfMetrics {
                main_elapsed,
                spec_elapsed,
                spec_passes: model_output_timing.spec_passes,
                main_gpu_elapsed: model_output_timing.main_gpu_elapsed,
                rejection_gpu_elapsed: model_output_timing.rejection_gpu_elapsed,
                spec_prepare_gpu_elapsed: model_output_timing.spec_prepare_gpu_elapsed,
                spec_prefill_gpu_elapsed: model_output_timing.spec_prefill_gpu_elapsed,
                spec_decode_gpu_elapsed: model_output_timing.spec_decode_gpu_elapsed,
                spec_gpu_elapsed: model_output_timing.spec_gpu_elapsed(),
            },
        );
        tracing::debug!(
            target: "inference-runtime-service::executor",
            component = "executor",
            phase = "executor.batch.response",
            model = self.model.model_name(),
            batch_seq,
            num_responses = batch_resp.dev_resps.len(),
            responses = %batch_resp
                .dev_resps
                .iter()
                .map(summarize_device_response)
                .collect::<Vec<_>>()
                .join(" | "),
            "executor batch response"
        );

        batch_resp
    }
}

fn summarize_device_request(dev_req: &DeviceRequest) -> String {
    let lane_kv_page_ids = dev_req.decoder_sync_blocks.kv_page_ids();
    let lane0 = lane_kv_page_ids.first().map(Vec::as_slice).unwrap_or(&[]);
    let pages_per_kv_block = lane0.first().map(Vec::len).unwrap_or(0);
    let query_tokens = summarize_query_tokens(&dev_req.decoder_query_tokens);

    format!(
        "req_id={} req_slot={} kind={} epoch={} token_index={} {} kv_block_index={} kv_blocks={} pages_per_kv_block={}",
        dev_req.req_id,
        dev_req.req_slot,
        query_kind(&dev_req.decoder_query_tokens),
        dev_req.decoder_query_tokens.epoch(),
        dev_req.decoder_query_tokens.token_index(),
        query_tokens,
        dev_req.decoder_sync_blocks.block_index(),
        lane0.len(),
        pages_per_kv_block
    )
}

fn summarize_device_response(dev_resp: &DeviceResponse) -> String {
    match &dev_resp.sampled_tokens {
        SampledTokens::Prefill { epoch } => format!("req_id={} prefill epoch={epoch}", dev_resp.req_id),
        SampledTokens::Decode {
            epoch,
            validated_tokens,
            sampled_token,
            sampled_prob,
            spec_tokens,
            spec_probs,
            ..
        } => {
            format!(
                "req_id={} decode epoch={} validated={} sampled={} prob={:.6} spec_out={} spec_probs={}",
                dev_resp.req_id,
                epoch,
                summarize_tokens(validated_tokens),
                sampled_token.value(),
                sampled_prob.into_inner(),
                summarize_tokens(spec_tokens),
                summarize_f32_slice(&spec_probs.iter().map(|prob| prob.into_inner()).collect::<Vec<_>>())
            )
        },
    }
}

fn summarize_query_tokens(query_tokens: &QueryTokens) -> String {
    match query_tokens {
        QueryTokens::Prefill { tokens, window, .. } => {
            format!("tokens={} window={window}", summarize_tokens(tokens))
        },
        QueryTokens::Decode {
            tokens, spec_tokens, ..
        } => {
            format!(
                "tokens={} spec={}",
                summarize_tokens(tokens),
                summarize_tokens(spec_tokens)
            )
        },
    }
}

fn summarize_tokens(tokens: &[Token]) -> String {
    summarize_u32_slice(&tokens.iter().map(|token| token.value()).collect::<Vec<_>>())
}

fn query_kind(query_tokens: &QueryTokens) -> &'static str {
    match query_tokens {
        QueryTokens::Prefill { .. } => "prefill",
        QueryTokens::Decode { spec_tokens, .. } if spec_tokens.is_empty() => "decode",
        QueryTokens::Decode { .. } => "spec-decode",
    }
}

fn summarize_u32_slice(values: &[u32]) -> String {
    const MAX_VALUES: usize = 8;
    if values.len() <= MAX_VALUES {
        return format!("{values:?}");
    }
    format!("{:?}..(+{})", &values[..MAX_VALUES], values.len() - MAX_VALUES)
}

fn summarize_f32_slice(values: &[f32]) -> String {
    const MAX_VALUES: usize = 8;
    if values.len() <= MAX_VALUES {
        return format!(
            "[{}]",
            values
                .iter()
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    format!(
        "[{}]..(+{})",
        values[..MAX_VALUES]
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(", "),
        values.len() - MAX_VALUES
    )
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::path::Path;
    use std::rc::Rc;
    use std::time::Duration;

    use crossbeam_channel::Sender;
    use crossbeam_channel::bounded;
    use crossbeam_channel::unbounded;
    use inference_executor_core::def::ModelExecutorError;
    use inference_runtime_core::compute::DecoderSyncBlocks;

    use super::*;

    #[derive(Debug, Eq, PartialEq)]
    enum ModelEvent {
        ResetRequestSlots(Vec<RawRequestSlot>),
        ClearReplayCache,
        UnloadState(ExecutorHibernationPlan),
        UnloadWeights,
        LoadWeights,
        LoadState(ExecutorHibernationPlan),
        StartEncoder,
        StopEncoder,
    }

    struct TestModel {
        event_tx: Sender<ModelEvent>,
        weights_loaded: bool,
        state_loaded: bool,
    }

    struct TestSubmission;

    struct TestEncoder {
        event_tx: Sender<ModelEvent>,
    }

    impl EncoderExecutorLifecycle for TestEncoder {
        fn model_name(&self) -> &str {
            "test encoder"
        }

        fn start(&self) -> Result<(), ModelExecutorError> {
            self.event_tx.send(ModelEvent::StartEncoder).unwrap();
            Ok(())
        }

        fn stop(&self) {
            self.event_tx.send(ModelEvent::StopEncoder).unwrap();
        }
    }

    impl ExecutionSubmission for TestSubmission {
        fn wait(&self) {
            panic!("test model must not wait for a submission")
        }
    }

    impl ReplayableDecoderModel for TestModel {
        type ModelBatchRequest = ();
        type ModelBatchHidden = ();
        type ModelBatchResponse = ();
        type SampledOutput = ();
        type ModelOpsRecorder = ();
        type Submission = TestSubmission;

        fn model_name(&self) -> &str {
            "test"
        }

        fn vocab_size(&self) -> usize {
            1024
        }

        fn model_mode(&self) -> &'static str {
            "test"
        }

        fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]) {
            assert!(self.weights_loaded);
            assert!(self.state_loaded);
            self.event_tx
                .send(ModelEvent::ResetRequestSlots(request_slots.to_vec()))
                .unwrap();
        }

        fn clear_replay_cache(&mut self) {
            assert!(self.weights_loaded);
            assert!(self.state_loaded);
            self.event_tx.send(ModelEvent::ClearReplayCache).unwrap();
        }

        fn unload_state(
            &mut self,
            snapshot_path: &Path,
            plan: &ExecutorHibernationPlan,
        ) -> Result<(), ModelExecutorError> {
            assert!(self.weights_loaded);
            assert!(self.state_loaded);
            std::fs::create_dir(snapshot_path)
                .and_then(|()| std::fs::write(snapshot_path.join("state"), b"test model state"))
                .map_err(|error| ModelExecutorError::custom(error.to_string()))?;
            self.state_loaded = false;
            self.event_tx.send(ModelEvent::UnloadState(plan.clone())).unwrap();
            Ok(())
        }

        fn unload_weights(&mut self) {
            assert!(self.weights_loaded);
            assert!(!self.state_loaded);
            self.weights_loaded = false;
            self.event_tx.send(ModelEvent::UnloadWeights).unwrap();
        }

        fn load_weights(&mut self) -> Result<(), ModelExecutorError> {
            assert!(!self.weights_loaded);
            assert!(!self.state_loaded);
            self.weights_loaded = true;
            self.event_tx.send(ModelEvent::LoadWeights).unwrap();
            Ok(())
        }

        fn load_state(
            &mut self,
            snapshot_path: &Path,
            plan: &ExecutorHibernationPlan,
        ) -> Result<(), ModelExecutorError> {
            assert!(self.weights_loaded);
            assert!(!self.state_loaded);
            let state = std::fs::read(snapshot_path.join("state"))
                .map_err(|error| ModelExecutorError::custom(error.to_string()))?;
            assert_eq!(state, b"test model state");
            self.state_loaded = true;
            self.event_tx.send(ModelEvent::LoadState(plan.clone())).unwrap();
            Ok(())
        }

        fn prepare_batch(&mut self, _core_batch_req: &mut BatchDeviceRequest) -> Self::ModelBatchRequest {
            panic!("test model must not execute a non-empty batch")
        }

        fn commit_batch(
            &mut self,
            _core_batch_req: BatchDeviceRequest,
            _sampled_output: Self::SampledOutput,
        ) -> BatchDeviceResponse {
            panic!("test model must not commit a non-empty batch")
        }

        fn begin_ops_recording(&mut self, _batch_req: &Self::ModelBatchRequest) -> Self::ModelOpsRecorder {}

        fn embed_main(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _batch_req: &Self::ModelBatchRequest,
        ) -> Self::ModelBatchHidden {
        }

        fn unembed_main(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _model_batch_hidden: &Self::ModelBatchHidden,
        ) -> Self::ModelBatchResponse {
        }

        fn forward_main(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _model_batch_hidden: Self::ModelBatchHidden,
        ) -> Self::ModelBatchHidden {
        }

        fn submit_main(&mut self, _recorder: &Self::ModelOpsRecorder) -> Self::Submission {
            panic!("test model must not submit a non-empty batch")
        }

        fn read_main(
            &mut self,
            _recorder: &Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _replay_elapsed: Duration,
            _gpu_timestamp_durations: Option<&[Duration]>,
        ) -> Self::SampledOutput {
        }

        fn sample_main(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _model_batch_resp: &Self::ModelBatchResponse,
        ) {
        }

        fn empty_sampled_output(&self) -> Self::SampledOutput {}

        fn sampled_output_len(&self, _sampled_output: &Self::SampledOutput) -> usize {
            0
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum SpecLifecycleEvent {
        PrepareBatch,
        BeginRecording,
        EmbedMain,
        ForwardMain,
        UnembedMain,
        SampleMain,
        SubmitMain,
        WaitMain,
        ReadMain,
        RunSpec,
        RunSpecPrefill,
        RunSpecDecode,
        EmbedSpec,
        ForwardSpec,
        UnembedSpec,
        SampleSpec,
        PrefillSpec,
        DecodeSpec,
        SubmitSpec,
        WaitSpec,
        ReadSpec,
        CommitBatch,
    }

    struct SpecLifecycleSubmission {
        events: Rc<RefCell<Vec<SpecLifecycleEvent>>>,
        wait_event: SpecLifecycleEvent,
    }

    impl ExecutionSubmission for SpecLifecycleSubmission {
        fn wait(&self) {
            self.events.borrow_mut().push(self.wait_event);
        }

        fn gpu_timestamp_durations(&self) -> Option<Vec<Duration>> {
            Some(vec![Duration::from_micros(123)])
        }
    }

    struct SpecLifecycleModel {
        events: Rc<RefCell<Vec<SpecLifecycleEvent>>>,
        run_spec: bool,
        run_prefill: bool,
        run_decode: bool,
    }

    impl SpecLifecycleModel {
        fn push(&self, event: SpecLifecycleEvent) {
            self.events.borrow_mut().push(event);
        }
    }

    impl ReplayableDecoderModel for SpecLifecycleModel {
        type ModelBatchRequest = ();
        type ModelBatchHidden = ();
        type ModelBatchResponse = ();
        type SampledOutput = ();
        type ModelOpsRecorder = ();
        type Submission = SpecLifecycleSubmission;

        fn model_name(&self) -> &str {
            "spec-lifecycle-test"
        }

        fn vocab_size(&self) -> usize {
            1024
        }

        fn model_mode(&self) -> &'static str {
            "test"
        }

        fn reset_req_slots(&mut self, _request_slots: &[RawRequestSlot]) {
            unreachable!()
        }

        fn clear_replay_cache(&mut self) {
            unreachable!()
        }

        fn unload_state(
            &mut self,
            _snapshot_path: &Path,
            _plan: &ExecutorHibernationPlan,
        ) -> Result<(), ModelExecutorError> {
            unreachable!()
        }

        fn unload_weights(&mut self) {
            unreachable!()
        }

        fn load_weights(&mut self) -> Result<(), ModelExecutorError> {
            unreachable!()
        }

        fn load_state(
            &mut self,
            _snapshot_path: &Path,
            _plan: &ExecutorHibernationPlan,
        ) -> Result<(), ModelExecutorError> {
            unreachable!()
        }

        fn prepare_batch(&mut self, _core_batch_req: &mut BatchDeviceRequest) -> Self::ModelBatchRequest {
            self.push(SpecLifecycleEvent::PrepareBatch);
        }

        fn commit_batch(
            &mut self,
            core_batch_req: BatchDeviceRequest,
            _sampled_output: Self::SampledOutput,
        ) -> BatchDeviceResponse {
            self.push(SpecLifecycleEvent::CommitBatch);
            BatchDeviceResponse::new(core_batch_req.seq, Vec::new())
        }

        fn begin_ops_recording(&mut self, _batch_req: &Self::ModelBatchRequest) -> Self::ModelOpsRecorder {
            self.push(SpecLifecycleEvent::BeginRecording);
        }

        fn embed_main(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _batch_req: &Self::ModelBatchRequest,
        ) -> Self::ModelBatchHidden {
            self.push(SpecLifecycleEvent::EmbedMain);
        }

        fn forward_main(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _model_batch_hidden: Self::ModelBatchHidden,
        ) -> Self::ModelBatchHidden {
            self.push(SpecLifecycleEvent::ForwardMain);
        }

        fn unembed_main(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _model_batch_hidden: &Self::ModelBatchHidden,
        ) -> Self::ModelBatchResponse {
            self.push(SpecLifecycleEvent::UnembedMain);
        }

        fn sample_main(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _model_batch_resp: &Self::ModelBatchResponse,
        ) {
            self.push(SpecLifecycleEvent::SampleMain);
        }

        fn submit_main(&mut self, _recorder: &Self::ModelOpsRecorder) -> Self::Submission {
            self.push(SpecLifecycleEvent::SubmitMain);
            SpecLifecycleSubmission {
                events: Rc::clone(&self.events),
                wait_event: SpecLifecycleEvent::WaitMain,
            }
        }

        fn read_main(
            &mut self,
            _recorder: &Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _replay_elapsed: Duration,
            _gpu_timestamp_durations: Option<&[Duration]>,
        ) -> Self::SampledOutput {
            self.push(SpecLifecycleEvent::ReadMain);
        }

        fn run_spec(&self, _model_batch_req: &Self::ModelBatchRequest, _sampled_output: &Self::SampledOutput) -> bool {
            self.push(SpecLifecycleEvent::RunSpec);
            self.run_spec
        }

        fn embed_spec(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _model_batch_hidden: &Self::ModelBatchHidden,
            _sampled_output: &Self::SampledOutput,
        ) -> Self::ModelBatchHidden {
            self.push(SpecLifecycleEvent::EmbedSpec);
        }

        fn forward_spec(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _model_batch_hidden: Self::ModelBatchHidden,
        ) -> Self::ModelBatchHidden {
            self.push(SpecLifecycleEvent::ForwardSpec);
        }

        fn unembed_spec(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _model_batch_hidden: &Self::ModelBatchHidden,
        ) -> Self::ModelBatchResponse {
            self.push(SpecLifecycleEvent::UnembedSpec);
        }

        fn sample_spec(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _model_batch_resp: &Self::ModelBatchResponse,
        ) {
            self.push(SpecLifecycleEvent::SampleSpec);
        }

        fn run_spec_prefill(&self, _model_batch_req: &Self::ModelBatchRequest) -> bool {
            self.push(SpecLifecycleEvent::RunSpecPrefill);
            self.run_prefill
        }

        fn prefill_spec(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _sampled_output: &Self::SampledOutput,
        ) {
            self.push(SpecLifecycleEvent::PrefillSpec);
        }

        fn run_spec_decode(
            &self,
            _model_batch_req: &Self::ModelBatchRequest,
            _sampled_output: &Self::SampledOutput,
        ) -> bool {
            self.push(SpecLifecycleEvent::RunSpecDecode);
            self.run_decode
        }

        fn decode_spec(
            &mut self,
            _recorder: &mut Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            _sampled_output: &Self::SampledOutput,
        ) {
            self.push(SpecLifecycleEvent::DecodeSpec);
        }

        fn submit_spec(&mut self, _recorder: &Self::ModelOpsRecorder) -> Self::Submission {
            self.push(SpecLifecycleEvent::SubmitSpec);
            SpecLifecycleSubmission {
                events: Rc::clone(&self.events),
                wait_event: SpecLifecycleEvent::WaitSpec,
            }
        }

        fn read_spec(
            &mut self,
            _recorder: &Self::ModelOpsRecorder,
            _model_batch_req: &Self::ModelBatchRequest,
            sampled_output: Self::SampledOutput,
            _replay_elapsed: Duration,
            gpu_timestamp_durations: Option<&[Duration]>,
        ) -> Self::SampledOutput {
            assert_eq!(gpu_timestamp_durations, Some([Duration::from_micros(123)].as_slice()));
            self.push(SpecLifecycleEvent::ReadSpec);
            sampled_output
        }

        fn empty_sampled_output(&self) -> Self::SampledOutput {}

        fn sampled_output_len(&self, _sampled_output: &Self::SampledOutput) -> usize {
            0
        }

        fn sampled_output_timing(
            &self,
            _sampled_output: &Self::SampledOutput,
        ) -> Option<inference_executor_core::model::ModelOutputTiming> {
            Some(inference_executor_core::model::ModelOutputTiming {
                main_gpu_elapsed: Some(Duration::from_micros(2)),
                rejection_gpu_elapsed: Some(Duration::from_micros(3)),
                spec_replay_gpu_elapsed: Some(Duration::from_micros(7)),
                ..Default::default()
            })
        }
    }

    fn test_snapshot_path(test_name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "psi-dec-{test_name}-{}-{:?}.state",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn test_executor_hibernation_plan() -> ExecutorHibernationPlan {
        ExecutorHibernationPlan::selected(vec![2..3, 5..6], vec![7..9, 12..13])
    }

    #[test]
    fn test_spec_prefill_runs_without_decode() {
        assert_eq!(
            execute_spec_lifecycle(false, true, false),
            vec![
                SpecLifecycleEvent::PrepareBatch,
                SpecLifecycleEvent::BeginRecording,
                SpecLifecycleEvent::EmbedMain,
                SpecLifecycleEvent::ForwardMain,
                SpecLifecycleEvent::UnembedMain,
                SpecLifecycleEvent::SampleMain,
                SpecLifecycleEvent::SubmitMain,
                SpecLifecycleEvent::WaitMain,
                SpecLifecycleEvent::ReadMain,
                SpecLifecycleEvent::RunSpec,
                SpecLifecycleEvent::RunSpecPrefill,
                SpecLifecycleEvent::RunSpecDecode,
                SpecLifecycleEvent::PrefillSpec,
                SpecLifecycleEvent::SubmitSpec,
                SpecLifecycleEvent::WaitSpec,
                SpecLifecycleEvent::CommitBatch,
            ]
        );
    }

    #[test]
    fn test_spec_prefill_precedes_decode() {
        assert_eq!(
            execute_spec_lifecycle(false, true, true),
            vec![
                SpecLifecycleEvent::PrepareBatch,
                SpecLifecycleEvent::BeginRecording,
                SpecLifecycleEvent::EmbedMain,
                SpecLifecycleEvent::ForwardMain,
                SpecLifecycleEvent::UnembedMain,
                SpecLifecycleEvent::SampleMain,
                SpecLifecycleEvent::SubmitMain,
                SpecLifecycleEvent::WaitMain,
                SpecLifecycleEvent::ReadMain,
                SpecLifecycleEvent::RunSpec,
                SpecLifecycleEvent::RunSpecPrefill,
                SpecLifecycleEvent::RunSpecDecode,
                SpecLifecycleEvent::PrefillSpec,
                SpecLifecycleEvent::DecodeSpec,
                SpecLifecycleEvent::SubmitSpec,
                SpecLifecycleEvent::WaitSpec,
                SpecLifecycleEvent::ReadSpec,
                SpecLifecycleEvent::CommitBatch,
            ]
        );
    }

    #[test]
    fn test_combined_spec_invocation_stays_combined() {
        assert_eq!(
            execute_spec_lifecycle(true, false, false),
            vec![
                SpecLifecycleEvent::PrepareBatch,
                SpecLifecycleEvent::BeginRecording,
                SpecLifecycleEvent::EmbedMain,
                SpecLifecycleEvent::ForwardMain,
                SpecLifecycleEvent::UnembedMain,
                SpecLifecycleEvent::SampleMain,
                SpecLifecycleEvent::SubmitMain,
                SpecLifecycleEvent::WaitMain,
                SpecLifecycleEvent::ReadMain,
                SpecLifecycleEvent::RunSpec,
                SpecLifecycleEvent::RunSpecPrefill,
                SpecLifecycleEvent::RunSpecDecode,
                SpecLifecycleEvent::EmbedSpec,
                SpecLifecycleEvent::ForwardSpec,
                SpecLifecycleEvent::UnembedSpec,
                SpecLifecycleEvent::SampleSpec,
                SpecLifecycleEvent::SubmitSpec,
                SpecLifecycleEvent::WaitSpec,
                SpecLifecycleEvent::ReadSpec,
                SpecLifecycleEvent::CommitBatch,
            ]
        );
    }

    #[test]
    fn test_request_slot_resets() {
        let (_model_executor_req_tx, model_executor_req_rx) = bounded(1);
        let (model_executor_resp_tx, _model_executor_resp_rx) = bounded(1);
        let shutdown = Shutdown::new();
        let (req_slot_reset_notifier, req_slot_reset_rx) = DedupNotifier::new(shutdown.clone());
        let (event_tx, event_rx) = unbounded();
        let snapshot_path = test_snapshot_path("request-slot-resets");
        let executor = ReplayableDecoderModelEventLoop::new(
            model_executor_req_rx,
            model_executor_resp_tx,
            req_slot_reset_notifier.clone(),
            req_slot_reset_rx,
            shutdown.clone(),
            ReplayableModelExecutors::new(
                TestModel {
                    event_tx,
                    weights_loaded: true,
                    state_loaded: true,
                },
                vec![],
            ),
            snapshot_path,
        );
        let executor_thread = std::thread::spawn(move || executor.event_loop());

        req_slot_reset_notifier.send_one(3);
        assert_eq!(
            event_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            ModelEvent::ResetRequestSlots(vec![3])
        );
        req_slot_reset_notifier.send_one(7);
        assert_eq!(
            event_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            ModelEvent::ResetRequestSlots(vec![7])
        );

        shutdown.shutdown();
        executor_thread.join().unwrap();
    }

    #[test]
    fn test_start_stop_and_batch_start_are_idempotent() {
        let (model_executor_req_tx, model_executor_req_rx) = bounded(1);
        let (model_executor_resp_tx, model_executor_resp_rx) = bounded(1);
        let shutdown = Shutdown::new();
        let (req_slot_reset_notifier, req_slot_reset_rx) = DedupNotifier::new(shutdown.clone());
        let (event_tx, event_rx) = unbounded();
        let snapshot_path = test_snapshot_path("start-stop");
        let executor = ReplayableDecoderModelEventLoop::new(
            model_executor_req_rx,
            model_executor_resp_tx,
            req_slot_reset_notifier.clone(),
            req_slot_reset_rx,
            shutdown.clone(),
            ReplayableModelExecutors::new(
                TestModel {
                    event_tx: event_tx.clone(),
                    weights_loaded: true,
                    state_loaded: true,
                },
                vec![Arc::new(TestEncoder { event_tx })],
            ),
            snapshot_path.clone(),
        );
        let executor_thread = std::thread::spawn(move || executor.event_loop());
        let plan = test_executor_hibernation_plan();

        model_executor_req_tx
            .send(ReplayableModelExecutorRequest::Start(plan.clone()))
            .unwrap();
        assert!(matches!(
            model_executor_resp_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            ReplayableModelExecutorResponse::Started
        ));
        assert!(event_rx.try_recv().is_err());

        model_executor_req_tx
            .send(ReplayableModelExecutorRequest::Stop(plan.clone()))
            .unwrap();
        assert!(matches!(
            model_executor_resp_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            ReplayableModelExecutorResponse::Stopped
        ));
        assert_eq!(event_rx.recv().unwrap(), ModelEvent::StopEncoder);
        assert_eq!(event_rx.recv().unwrap(), ModelEvent::ClearReplayCache);
        assert_eq!(event_rx.recv().unwrap(), ModelEvent::UnloadState(plan.clone()));
        assert_eq!(event_rx.recv().unwrap(), ModelEvent::UnloadWeights);

        model_executor_req_tx
            .send(ReplayableModelExecutorRequest::Stop(plan.clone()))
            .unwrap();
        assert!(matches!(
            model_executor_resp_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            ReplayableModelExecutorResponse::Stopped
        ));
        assert!(event_rx.try_recv().is_err());

        req_slot_reset_notifier.send_one(11);
        assert!(event_rx.recv_timeout(Duration::from_millis(50)).is_err());
        model_executor_req_tx
            .send(ReplayableModelExecutorRequest::Batch(BatchDeviceRequest::new(
                7,
                Vec::new(),
            )))
            .unwrap();
        let ReplayableModelExecutorResponse::Batch(batch_response) =
            model_executor_resp_rx.recv_timeout(Duration::from_secs(1)).unwrap()
        else {
            panic!("empty batch must return a batch response")
        };
        assert_eq!(batch_response.seq, 7);
        assert_eq!(event_rx.recv().unwrap(), ModelEvent::LoadWeights);
        assert_eq!(event_rx.recv().unwrap(), ModelEvent::LoadState(plan.clone()));
        assert_eq!(event_rx.recv().unwrap(), ModelEvent::StartEncoder);
        assert_eq!(event_rx.recv().unwrap(), ModelEvent::ResetRequestSlots(vec![11]));
        assert!(!snapshot_path.exists());

        model_executor_req_tx
            .send(ReplayableModelExecutorRequest::Start(plan))
            .unwrap();
        assert!(matches!(
            model_executor_resp_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            ReplayableModelExecutorResponse::Started
        ));
        assert!(event_rx.try_recv().is_err());

        shutdown.shutdown();
        executor_thread.join().unwrap();
    }

    fn execute_spec_lifecycle(run_spec: bool, run_prefill: bool, run_decode: bool) -> Vec<SpecLifecycleEvent> {
        let (_model_executor_req_tx, model_executor_req_rx) = bounded(1);
        let (model_executor_resp_tx, _model_executor_resp_rx) = bounded(1);
        let shutdown = Shutdown::new();
        let (req_slot_reset_notifier, req_slot_reset_rx) = DedupNotifier::new(shutdown.clone());
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut executor = ReplayableDecoderModelEventLoop::new(
            model_executor_req_rx,
            model_executor_resp_tx,
            req_slot_reset_notifier,
            req_slot_reset_rx,
            shutdown,
            ReplayableModelExecutors::new(
                SpecLifecycleModel {
                    events: Rc::clone(&events),
                    run_spec,
                    run_prefill,
                    run_decode,
                },
                vec![],
            ),
            std::env::temp_dir().join("psi-dec-spec-lifecycle-test.state"),
        );
        let response = executor.execute(BatchDeviceRequest::new(
            1,
            [DeviceRequest::new(
                1,
                0,
                QueryTokens::Prefill {
                    epoch: 0,
                    token_index: 0,
                    tokens: vec![Token::new(1)],
                    window: 1,
                },
                DecoderSyncBlocks::new(0, Vec::new(), Vec::new()),
                vec![],
                Default::default(),
            )],
        ));
        assert_eq!(
            response.execution_timings,
            vec![
                ("main".into(), Some(Duration::from_micros(5))),
                ("test".into(), Some(Duration::from_micros(7)))
            ]
        );
        let recorded = events.borrow().clone();
        drop(executor);
        recorded
    }
}
