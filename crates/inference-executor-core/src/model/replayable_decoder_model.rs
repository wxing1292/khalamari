//! Shared Main and Spec replay lifecycle.
//!
//! The model executor records the Main invocation first. Main sampling or rejection produces the decision that can
//! enable Spec work. A model uses either the combined Spec hooks or the independent Spec Prefill and Decode hooks.
//!
//! ```text
//! previous Spec proposal
//!   {draft tokens, draft probabilities}
//!                 |
//!                 v
//! request microbatch
//!   {committed tokens + previous speculative suffix}
//!                 |
//!                 v
//! +------------------------------ Main module --------------------------------+
//! |                                                                            |
//! | token IDs -> Main Embed -> Main Body -> Gather + Unembed -> target logits   |
//! |                              |                                |             |
//! |                              +-> optional model-owned capture |             |
//! |                                                               v             |
//! |                                      normal sampling or rejection sampling  |
//! |                                                               |             |
//! |                                                               v             |
//! |                                      validated prefix + sampled anchor       |
//! +---------------------------------------------------------------+-------------+
//!                                                                 |
//!                                                   submit + wait + read
//!                                                                 |
//!                         +-------------------+--------------------+
//!                         |                   |                    |
//!                         v                   v                    v
//!                      Vanilla       independent Spec       combined Spec
//!                        return       Prefill -> Decode        invocation
//!                                           |                    |
//!                                           +---------+----------+
//!                                                     |
//!                                                     v
//!                                           next Spec proposal
//!                                                     |
//!                                                     +----> next Main invocation
//! ```
//!
//! Independent Prefill and Decode recordings can share one ordered submission. Independence does not require one
//! submission and wait for each recording.

use std::path::Path;
use std::time::Duration;

use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::BatchDeviceResponse;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::ExecutorHibernationPlan;
use inference_runtime_core::runtime::RawRequestSlot;
use inference_runtime_core::runtime::Token;

use crate::def::ModelExecutorError;

pub trait ExecutionSubmission {
    fn wait(&self);

    fn gpu_timestamp_durations(&self) -> Option<Vec<Duration>> {
        None
    }
}

pub trait ReplayableDecoderModel {
    type ModelBatchRequest;
    type ModelBatchHidden;
    type ModelBatchResponse;
    type SampledOutput;
    type ModelOpsRecorder;
    type Submission: ExecutionSubmission;

    fn model_name(&self) -> &str;
    fn model_mode(&self) -> &'static str;

    /// Returns the exclusive upper bound for input token IDs.
    fn vocab_size(&self) -> usize;

    /// Returns the maximum number of speculative tokens in one request step.
    fn num_spec_tokens(&self) -> usize {
        0
    }

    fn default_stop_sequences(&self) -> Vec<Vec<Token>> {
        vec![]
    }

    fn reset_req_slots(&mut self, request_slots: &[RawRequestSlot]);

    fn clear_replay_cache(&mut self);
    fn unload_state(&mut self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError>;
    fn unload_weights(&mut self);
    fn load_weights(&mut self) -> Result<(), ModelExecutorError>;
    fn load_state(&mut self, snapshot_path: &Path, plan: &ExecutorHibernationPlan) -> Result<(), ModelExecutorError>;

    /// Prepares model metadata. The executor may reorder complete requests before
    /// packing them. Commit receives the same order and preserves request identity.
    fn prepare_batch(&mut self, core_batch_req: &mut BatchDeviceRequest) -> Self::ModelBatchRequest;
    fn commit_batch(
        &mut self,
        core_batch_req: BatchDeviceRequest,
        sampled_output: Self::SampledOutput,
    ) -> BatchDeviceResponse;

    fn begin_ops_recording(&mut self, batch_req: &Self::ModelBatchRequest) -> Self::ModelOpsRecorder;

    fn embed_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        batch_req: &Self::ModelBatchRequest,
    ) -> Self::ModelBatchHidden;
    fn forward_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: Self::ModelBatchHidden,
    ) -> Self::ModelBatchHidden;
    fn unembed_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        model_batch_hidden: &Self::ModelBatchHidden,
    ) -> Self::ModelBatchResponse;
    fn sample_main(
        &mut self,
        recorder: &mut Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        model_batch_resp: &Self::ModelBatchResponse,
    );

    fn submit_main(&mut self, recorder: &Self::ModelOpsRecorder) -> Self::Submission;

    fn read_main(
        &mut self,
        recorder: &Self::ModelOpsRecorder,
        model_batch_req: &Self::ModelBatchRequest,
        replay_elapsed: Duration,
        gpu_timestamp_durations: Option<&[Duration]>,
    ) -> Self::SampledOutput;

    fn run_spec(&self, _model_batch_req: &Self::ModelBatchRequest, _sampled_output: &Self::SampledOutput) -> bool {
        false
    }

    fn embed_spec(
        &mut self,
        _recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _model_batch_hidden: &Self::ModelBatchHidden,
        _sampled_output: &Self::SampledOutput,
    ) -> Self::ModelBatchHidden {
        panic!("model executor does not have a speculator")
    }

    fn forward_spec(
        &mut self,
        _recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _model_batch_hidden: Self::ModelBatchHidden,
    ) -> Self::ModelBatchHidden {
        panic!("model executor does not have a speculator")
    }

    fn unembed_spec(
        &mut self,
        _recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _model_batch_hidden: &Self::ModelBatchHidden,
    ) -> Self::ModelBatchResponse {
        panic!("model executor does not have a speculator")
    }

    fn sample_spec(
        &mut self,
        _recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _model_batch_resp: &Self::ModelBatchResponse,
    ) {
        panic!("model executor does not have a speculator")
    }

    fn run_spec_prefill(&self, _model_batch_req: &Self::ModelBatchRequest) -> bool {
        false
    }

    fn prefill_spec(
        &mut self,
        _recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _sampled_output: &Self::SampledOutput,
    ) {
        panic!("model executor does not support Spec Prefill")
    }

    fn run_spec_decode(
        &self,
        _model_batch_req: &Self::ModelBatchRequest,
        _sampled_output: &Self::SampledOutput,
    ) -> bool {
        false
    }

    fn decode_spec(
        &mut self,
        _recorder: &mut Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        _sampled_output: &Self::SampledOutput,
    ) {
        panic!("model executor does not support Spec Decode")
    }

    fn submit_spec(&mut self, _recorder: &Self::ModelOpsRecorder) -> Self::Submission {
        panic!("model executor does not have a speculator")
    }

    fn read_spec(
        &mut self,
        _recorder: &Self::ModelOpsRecorder,
        _model_batch_req: &Self::ModelBatchRequest,
        sampled_output: Self::SampledOutput,
        _replay_elapsed: Duration,
        _gpu_timestamp_durations: Option<&[Duration]>,
    ) -> Self::SampledOutput {
        sampled_output
    }

    fn empty_sampled_output(&self) -> Self::SampledOutput;
    fn sampled_output_len(&self, sampled_output: &Self::SampledOutput) -> usize;
    fn sampled_output_timing(&self, _sampled_output: &Self::SampledOutput) -> Option<ModelOutputTiming> {
        None
    }

    fn first_pp_stage(&self, _batch_req: &Self::ModelBatchRequest) -> bool {
        true
    }
    fn last_pp_stage(&self, _batch_req: &Self::ModelBatchRequest) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelOutputTiming {
    pub main_replay_elapsed: Duration,
    pub main_sample_replay_elapsed: Duration,
    pub main_spec_replay_elapsed: Duration,
    pub sample_read_elapsed: Duration,
    pub rejection_build_elapsed: Duration,
    pub rejection_read_elapsed: Duration,
    pub spec_build_elapsed: Duration,
    pub spec_replay_elapsed: Duration,
    pub spec_read_elapsed: Duration,
    pub spec_passes: usize,
    /// Total GPU duration across separate Spec submissions.
    pub spec_replay_gpu_elapsed: Option<Duration>,
    pub main_gpu_elapsed: Option<Duration>,
    pub rejection_gpu_elapsed: Option<Duration>,
    pub spec_prepare_gpu_elapsed: Option<Duration>,
    pub spec_prefill_gpu_elapsed: Option<Duration>,
    pub spec_decode_gpu_elapsed: Option<Duration>,
}

impl ModelOutputTiming {
    pub fn add_assign(&mut self, other: Self) {
        self.main_replay_elapsed += other.main_replay_elapsed;
        self.main_sample_replay_elapsed += other.main_sample_replay_elapsed;
        self.main_spec_replay_elapsed += other.main_spec_replay_elapsed;
        self.sample_read_elapsed += other.sample_read_elapsed;
        self.rejection_build_elapsed += other.rejection_build_elapsed;
        self.rejection_read_elapsed += other.rejection_read_elapsed;
        self.spec_build_elapsed += other.spec_build_elapsed;
        self.spec_replay_elapsed += other.spec_replay_elapsed;
        self.spec_read_elapsed += other.spec_read_elapsed;
        self.spec_passes += other.spec_passes;
        self.spec_replay_gpu_elapsed =
            add_optional_duration(self.spec_replay_gpu_elapsed, other.spec_replay_gpu_elapsed);
        self.main_gpu_elapsed = add_optional_duration(self.main_gpu_elapsed, other.main_gpu_elapsed);
        self.rejection_gpu_elapsed = add_optional_duration(self.rejection_gpu_elapsed, other.rejection_gpu_elapsed);
        self.spec_prepare_gpu_elapsed =
            add_optional_duration(self.spec_prepare_gpu_elapsed, other.spec_prepare_gpu_elapsed);
        self.spec_prefill_gpu_elapsed =
            add_optional_duration(self.spec_prefill_gpu_elapsed, other.spec_prefill_gpu_elapsed);
        self.spec_decode_gpu_elapsed =
            add_optional_duration(self.spec_decode_gpu_elapsed, other.spec_decode_gpu_elapsed);
    }

    pub fn is_zero(self) -> bool {
        self == Self::default()
    }

    pub fn spec_gpu_elapsed(self) -> Option<Duration> {
        self.spec_replay_gpu_elapsed.or_else(|| {
            Some(self.spec_prepare_gpu_elapsed? + self.spec_prefill_gpu_elapsed? + self.spec_decode_gpu_elapsed?)
        })
    }
}

fn add_optional_duration(lhs: Option<Duration>, rhs: Option<Duration>) -> Option<Duration> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(lhs + rhs),
        (Some(elapsed), None) | (None, Some(elapsed)) => Some(elapsed),
        (None, None) => None,
    }
}

pub fn page_ids_by_layer_for_lane(
    request: &DeviceRequest,
    cache_lane: usize,
    num_gqa_layers: usize,
    num_page_ids_per_block: usize,
    model_name: &str,
) -> Vec<Vec<Vec<u32>>> {
    let page_ids_by_lane_and_block = request.decoder_sync_blocks.kv_page_ids();
    let page_ids_by_block = page_ids_by_lane_and_block
        .get(cache_lane)
        .unwrap_or_else(|| panic!("{model_name} missing cache lane {cache_lane} for kv page ids"));
    let mut page_ids_by_layer = (0..num_gqa_layers)
        .map(|_| Vec::with_capacity(page_ids_by_block.len()))
        .collect::<Vec<_>>();
    for page_ids_for_one_block in page_ids_by_block {
        assert_eq!(
            num_gqa_layers * num_page_ids_per_block,
            page_ids_for_one_block.len(),
            "{model_name} expects {} page ids for each synced kv block in cache lane {cache_lane}, got {}",
            num_gqa_layers * num_page_ids_per_block,
            page_ids_for_one_block.len()
        );
        for (gqa_layer_index, page_ids_by_block) in page_ids_by_layer.iter_mut().enumerate() {
            let page_id_start = gqa_layer_index * num_page_ids_per_block;
            page_ids_by_block
                .push(page_ids_for_one_block[page_id_start..page_id_start + num_page_ids_per_block].to_vec());
        }
    }

    page_ids_by_layer
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::ModelOutputTiming;

    #[test]
    fn test_spec_gpu_elapsed_separate_submissions() {
        let mut timing = ModelOutputTiming::default();
        timing.add_assign(ModelOutputTiming {
            spec_replay_gpu_elapsed: Some(Duration::from_millis(7)),
            ..ModelOutputTiming::default()
        });
        assert_eq!(timing.spec_gpu_elapsed(), Some(Duration::from_millis(7)));
        assert_eq!(timing.spec_prepare_gpu_elapsed, None);
        assert_eq!(timing.spec_prefill_gpu_elapsed, None);
        assert_eq!(timing.spec_decode_gpu_elapsed, None);
    }

    #[test]
    fn test_spec_gpu_elapsed_incomplete() {
        let mut timing = ModelOutputTiming {
            spec_prepare_gpu_elapsed: Some(Duration::from_millis(2)),
            spec_prefill_gpu_elapsed: Some(Duration::from_millis(3)),
            ..ModelOutputTiming::default()
        };
        assert_eq!(timing.spec_gpu_elapsed(), None);

        timing.spec_decode_gpu_elapsed = Some(Duration::from_millis(5));
        assert_eq!(timing.spec_gpu_elapsed(), Some(Duration::from_millis(10)));
    }
}
