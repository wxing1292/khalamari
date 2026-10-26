use std::num::NonZeroUsize;

use inference_executor_core::model::ExecutionSubmission;
use inference_executor_core::model::qwen::v3_5::Qwen35Microbatch;
use inference_executor_core::model::qwen::v4_x::weight_layout::Qwen4xMTPWeightBindings;
use inference_executor_core::model::qwen::v4_x::weight_layout::resolve_qwen4x_mtp_weight_bindings;
use inference_runtime_core::compute::DecoderSyncBlocks;

use crate::model::qwen::v4_x::mtp::Qwen4xMTP;
use crate::model::qwen::v4_x::mtp::Qwen4xMTPArgs;
use crate::model::qwen::v4_x::mtp::Qwen4xMTPReplayKey;
use crate::model::qwen::v4_x::mtp::plan;
use crate::sampling::rejection_replay::PreparedRejection;
use crate::sampling::rejection_replay::RejectionReplayKey;
use crate::sampling::rejection_replay::RejectionSampler;
use crate::sampling::rejection_replay::RejectionSamplerInput;
use crate::sampling::rejection_replay::RejectionSampling;
use crate::sampling::rejection_replay::RejectionSamplingInput;
use crate::sampling::rejection_sampling::SparseRejectionSamplingReqParams;
use crate::sampling::spec_probs::SpecProbsStore;
use crate::sampling::top_k_replay::DraftSampling;
use crate::sampling::top_k_replay::DraftSamplingInput;
use crate::sampling::top_k_sampling::TopKSamplingWriteDistributionOutput;

struct Qwen4xMTPSpeculator {
    model_dir: PathBuf,
    config: Qwen4xModelConfig,
    bindings: Qwen4xMTPWeightBindings,
    num_spec_tokens: usize,
    body: Replay<Qwen4xMTP>,
    states: Vec<Qwen4xState>,
    hidden_cache: Option<Buffer>,
    hidden_routes: Buffer,
    write_routes: Buffer,
    draft_indices: Buffer,
    target_indices: Buffer,
    sampling: Replay<DraftSampling>,
    rejection: Replay<RejectionSampling>,
    probabilities: SpecProbsStore,
    history: Vec<plan::History>,
    sync_blocks: Vec<DecoderSyncBlocks>,
    requests: Vec<plan::Request>,
    drafts: Vec<Vec<i32>>,
    draft_probs: Vec<Vec<f32>>,
    final_batch: Option<Qwen35Microbatch>,
    timing: ModelOutputTiming,
}
struct Qwen4xMTPStep {
    module_index: usize,
    batch: plan::Batch,
    embed: (Qwen35MainTextEmbedReplayKey, ReplayArguments),
    body: Option<(Qwen4xMTPReplayKey, ReplayArguments)>,
    output: Option<(Qwen35GatherUnembedReplayKey, ReplayArguments)>,
    sampling: Option<(TopKSamplingReplayKey, ReplayArguments)>,
}
impl Qwen4xMTPSpeculator {
    fn new(model: &Qwen4xExecutor, model_dir: &Path, depth: usize) -> Result<Self, ModelExecutorError> {
        let config = model.model_config.mtp_config()?;
        let device = model.runtime.device();
        let mut store = SafeTensorStore::from_model_dir(model_dir)?;
        let bindings = resolve_qwen4x_mtp_weight_bindings(&model.model_config, store.index().tensor_names())?;
        let states = (0..depth)
            .map(|_| {
                Qwen4xState::new_with_carry_history(
                    device,
                    model.runtime.stream().new_on_same_queue(),
                    &config,
                    model.config,
                    depth + 1,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut body = Qwen4xMTP::new(device, &config, &bindings, model.config.max_tokens as u32, &states[0])?;
        body.load_weights(device, &mut store, &config, &bindings)?;
        let rows = model
            .config
            .max_requests
            .checked_mul(depth)
            .and_then(|n| n.checked_mul(depth - 1))
            .map(|n| n / 2)
            .expect("MTP retained rows must fit usize");
        u32::try_from(rows).expect("MTP retained row indices must fit u32");
        let width = config.text_config.hidden_size * config.text_config.hc_count;
        let rejector = Rc::new(RejectionSampler::new(
            device,
            depth,
            model.config.max_requests,
            model.sampler_bounds.max_sampling_inputs as usize,
            MAX_TOP_K as u32,
        ));
        Ok(Self {
            model_dir: model_dir.to_owned(),
            config,
            bindings,
            num_spec_tokens: depth,
            body: Replay::new("Qwen4x MTP", body),
            states,
            hidden_cache: (rows > 0).then(|| {
                Buffer::new_zeroed_elements(
                    device,
                    rows.checked_mul(width)
                        .expect("MTP retained hidden elements must fit usize"),
                    Dtype::Bfloat16,
                )
            }),
            hidden_routes: Buffer::new_zeroed_elements(device, model.config.max_tokens * 2, Dtype::Uint32),
            write_routes: Buffer::new_zeroed_elements(device, model.config.max_tokens * 2, Dtype::Uint32),
            draft_indices: Buffer::new_zeroed_elements(device, model.config.max_requests, Dtype::Uint32),
            target_indices: Buffer::from_slice(
                device,
                &(0..model.sampler_bounds.max_sampling_inputs).collect::<Vec<_>>(),
            ),
            sampling: Replay::new(
                "Qwen4x DraftSampling",
                DraftSampling::new(Rc::clone(&model.sampler), model.config.max_requests as u32),
            ),
            rejection: Replay::new(
                "Qwen4x RejectionSampling",
                RejectionSampling::new(Rc::clone(&model.sampler), rejector),
            ),
            probabilities: SpecProbsStore::new(
                device,
                depth,
                model.config.max_requests,
                model.sampler_bounds.max_sampling_inputs as usize,
                MAX_TOP_K,
            ),
            history: vec![plan::History::default(); model.config.max_requests],
            sync_blocks: Vec::new(),
            requests: Vec::new(),
            drafts: Vec::new(),
            draft_probs: Vec::new(),
            final_batch: None,
            timing: ModelOutputTiming::default(),
        })
    }
}

/// Use a native Qwen4x MTP head with Main's geometry and affine layout.
pub fn init_qwen4x_model_with_mtp(
    model_dir: impl AsRef<Path>,
    mtp_dir: impl AsRef<Path>,
    num_spec_tokens: NonZeroUsize,
    config: Qwen4xExecutorConfig,
) -> Result<Qwen4xExecutor, ModelExecutorError> {
    let model_config = init_qwen4x_model_config(&model_dir)?;
    Qwen4xExecutor::new_with_mtp(model_dir, model_config, mtp_dir, num_spec_tokens, config)
}

impl Qwen4xExecutor {
    pub fn new_with_mtp(
        model_dir: impl AsRef<Path>,
        mut model_config: Qwen4xModelConfig,
        mtp_dir: impl AsRef<Path>,
        num_spec_tokens: NonZeroUsize,
        mut config: Qwen4xExecutorConfig,
    ) -> Result<Self, ModelExecutorError> {
        let depth = num_spec_tokens.get();
        config.validate();
        model_config.normalize()?;
        validate_metal_model_config(&model_config)?;
        validate_metal_model_config(&model_config.mtp_config()?)?;
        // Resolve the requested head before allocating Main or any GPU resources.
        let store = SafeTensorStore::from_model_dir(mtp_dir.as_ref())?;
        resolve_qwen4x_mtp_weight_bindings(&model_config, store.index().tensor_names())?;
        drop(store);
        config.max_tokens = config
            .max_tokens
            .checked_add(
                config
                    .max_requests
                    .min(config.max_tokens)
                    .checked_mul(depth - 1)
                    .expect("MTP token capacity must fit usize"),
            )
            .expect("MTP token capacity must fit usize");
        let mut model = Qwen4xExecutor::new_with_spec_capacity(model_dir, model_config, config, depth)?;
        model.mtp = Some(Box::new(Qwen4xMTPSpeculator::new(&model, mtp_dir.as_ref(), depth)?));
        Ok(model)
    }
}

impl Qwen4xExecutor {
    fn commit_main(&mut self, seq: u64, decisions: &[Qwen35DecodeDecision]) {
        let batch = self.transactions.pending_microbatch(seq).clone();
        let versions = self.transactions.commit(seq, decisions);
        self.state.commit(
            self.runtime.device(),
            self.pages.buffer(),
            |index| &self.main.component().ple_scratch(index).normalized_gated,
            &batch,
            &versions,
        );
        self.gdn.commit(self.pages.buffer(), &versions);
    }
    fn record_rejection(&mut self, recorder: &mut Qwen4xModelOpsRecorder, batch: &Qwen35Microbatch) {
        let mtp = self.mtp.as_mut().unwrap();
        let mut draft_indices = Vec::new();
        for i in 0..batch.num_reqs() {
            if !batch.is_decode_req(i) {
                continue;
            }
            let count = batch.num_spec_tokens(i) as usize;
            let end = batch.cu_tokens()[i + 1] as usize;
            for (j, &token) in batch.flat_token_ids()[end - count..end].iter().enumerate() {
                mtp.probabilities
                    .assert_expected_draft_token(batch.req_slots()[i], j, token as u32);
                draft_indices.push(mtp.probabilities.draft_distribution_index(batch.req_slots()[i], j));
            }
        }
        let rejector = Rc::clone(mtp.rejection.component().rejector());
        let prepared = rejector.prepare_inputs(batch, &draft_indices);
        let configs = sample_sampler_configs(batch);
        let active = self.sampler.active_shape(&configs);
        let shape = rejector.prepare_replay_shape(&prepared, active.top_k);
        let target_shape = active.with_num_total_sampling_inputs(shape.num_total_target_distributions);
        let rejection = RejectionSamplerInput {
            shape,
            target_token_ids: mtp.probabilities.target_token_ids(),
            target_probs: mtp.probabilities.target_probs(),
            draft_token_ids: mtp.probabilities.draft_token_ids(),
            draft_probs: mtp.probabilities.draft_probs(),
        };
        let (key, _) = mtp.rejection.record(
            self.runtime.stream(),
            &RejectionSamplingInput {
                target_shape,
                logits: &self.logits,
                target_sparse: TopKSamplingWriteDistributionOutput {
                    token_ids: mtp.probabilities.target_token_ids(),
                    probs: mtp.probabilities.target_probs(),
                    output_distribution_indices: &mtp.target_indices,
                    max_k: mtp.probabilities.max_k() as u32,
                    num_output_distributions: mtp.probabilities.num_target_distributions(),
                },
                rejection,
            },
        );
        let slots = sample_req_slots(batch);
        let positions = sample_token_positions(batch);
        self.sampler.prepare(&slots, &positions);
        let mut offset = 0;
        let params = prepared
            .decode_req_indices
            .iter()
            .map(|&i| {
                let position = positions[offset];
                offset += batch.num_spec_tokens(i) as usize + 1;
                SparseRejectionSamplingReqParams {
                    seed: batch.sampler_configs()[i].seed(),
                    sample_position: position,
                    top_k: self
                        .sampler_bounds
                        .active_top_k(&batch.sampler_configs()[i])
                        .expect("validated sampler configuration"),
                }
            })
            .collect::<Vec<_>>();
        rejector.set_runtime_params(&params);
        let mut args = ReplayArguments::new();
        self.sampler.add_replay_arguments(target_shape, &mut args);
        rejector.add_replay_arguments(rejection, &mut args);
        recorder.rejection = Some((key, args, prepared));
    }
    fn read_rejection(&self, prepared: &PreparedRejection, batch: &Qwen35Microbatch) -> Vec<Qwen35DecodeDecision> {
        let mtp = self.mtp.as_ref().unwrap();
        let results = mtp.rejection.component().rejector().read_results(
            prepared.num_active_decode_reqs(),
            prepared.num_active_draft_distributions,
        );
        let mut offset = 0;
        prepared
            .decode_req_indices
            .iter()
            .enumerate()
            .map(|(row, &i)| {
                let accepted = results.num_accepted_tokens(row);
                debug_assert!(accepted <= batch.num_spec_tokens(i) as usize);
                let decision = Qwen35DecodeDecision {
                    sampled_token: results.sampled_token_id(row) as u32,
                    sampled_prob: results.sampled_prob(row),
                    validated_tokens: results
                        .accepted_token_ids(offset, accepted)
                        .iter()
                        .map(|&t| t as u32)
                        .collect(),
                    validated_probs: results.accepted_probs(offset, accepted).to_vec(),
                    ..Default::default()
                };
                offset += batch.num_spec_tokens(i) as usize;
                decision
            })
            .collect()
    }
}

impl Qwen4xExecutor {
    fn record_mtp_embed(&mut self, module_index: usize) -> Qwen4xMTPStep {
        let mtp = self.mtp.as_mut().unwrap();
        let batch = plan::batch(
            &mtp.requests,
            module_index,
            mtp.num_spec_tokens,
            self.config.max_requests,
            &mtp.drafts,
        );
        debug_assert!(batch.microbatch.total_tokens() <= self.config.max_tokens);
        self.token_ids.write_typed(0, batch.microbatch.flat_token_ids());
        let count = batch.microbatch.total_tokens() as u32;
        let total = mtp.body.component().capacity(count);
        mtp.states[module_index].prepare_lane(
            self.runtime.device(),
            self.runtime.stream(),
            self.pages.buffer(),
            mtp.sync_blocks.iter(),
            &batch.microbatch,
            total,
            0,
            module_index + 1,
        );
        mtp.hidden_routes.write_typed(0, &batch.hidden_routes);
        mtp.write_routes.write_typed(0, &batch.write_routes);
        mtp.draft_indices.write_typed(0, &batch.distribution_indices);
        let (_, args) = self.embed.component().prepare_replay(count);
        let (key, _) = self.embed.record(
            self.runtime.stream(),
            &Qwen35MainTextEmbedArgs {
                num_tokens: count,
                token_ids: &self.token_ids,
                hidden_output: &self.hidden_input,
            },
        );
        Qwen4xMTPStep {
            module_index,
            batch,
            embed: (key, args),
            body: None,
            output: None,
            sampling: None,
        }
    }
    fn record_mtp_body(&mut self, step: &mut Qwen4xMTPStep) {
        let mtp = self.mtp.as_mut().unwrap();
        let main_channels = self.main.component().channels();
        let input = Qwen4xMTPArgs {
            module_index: step.module_index,
            previous_hidden: main_channels,
            hidden_cache: mtp.hidden_cache.as_ref().unwrap_or(main_channels),
            hidden_routes: &mtp.hidden_routes,
            write_routes: &mtp.write_routes,
            num_write_rows: (step.batch.write_routes.len() / 2) as u32,
            token_embedding: &self.hidden_input,
            hidden_output: &self.hidden_output,
            pages: self.pages.buffer(),
            state: &mtp.states[step.module_index],
        };
        let args = mtp.body.component().replay_arguments(&input);
        let (key, _) = mtp.body.record(self.runtime.stream(), &input);
        step.body = Some((key, args));
    }
    fn record_mtp_output(&mut self, step: &mut Qwen4xMTPStep) {
        if step.batch.sample_slots.is_empty() {
            return;
        }
        let indices = gather_flat_indices(&step.batch.microbatch);
        self.gather_indices.write_typed(0, &indices);
        let count = indices.len() as u32;
        let (_, args) = self.output.component().prepare_replay(count);
        let (key, _) = self.output.record(
            self.runtime.stream(),
            &Qwen35GatherUnembedArgs {
                num_rows: count,
                hidden_input: &self.hidden_output,
                row_indices: &self.gather_indices,
                hidden_output: &self.gathered_hidden,
                logits: &self.logits,
            },
        );
        step.output = Some((key, args));
    }
    fn record_mtp_sampling(&mut self, step: &mut Qwen4xMTPStep) {
        if step.batch.sample_slots.is_empty() {
            return;
        }
        let mtp = self.mtp.as_mut().unwrap();
        let shape = mtp.sampling.component().prepare_shape(&step.batch.sampler_configs);
        let (key, _) = mtp.sampling.record(
            self.runtime.stream(),
            &DraftSamplingInput {
                shape,
                logits: &self.logits,
                output: self.sampler_output.as_output(),
                sparse: TopKSamplingWriteDistributionOutput {
                    token_ids: mtp.probabilities.draft_token_ids(),
                    probs: mtp.probabilities.draft_probs(),
                    output_distribution_indices: &mtp.draft_indices,
                    max_k: mtp.probabilities.max_k() as u32,
                    num_output_distributions: mtp.probabilities.num_draft_distributions(),
                },
            },
        );
        self.sampler
            .prepare(&step.batch.sample_slots, &step.batch.sample_positions);
        let mut args = ReplayArguments::new();
        self.sampler.add_replay_arguments(shape, &mut args);
        step.sampling = Some((key, args));
    }
    fn submit_mtp_step(&self, step: &Qwen4xMTPStep) -> MetalReplaySubmission {
        let mtp = self.mtp.as_ref().unwrap();
        let (body_key, body_args) = step.body.as_ref().unwrap();
        let mut sequence = vec![
            ReplayExecution::new(self.embed.replay(&step.embed.0), &step.embed.1),
            ReplayExecution::new(mtp.body.replay(body_key), body_args),
        ];
        if let Some((key, args)) = &step.output {
            sequence.push(ReplayExecution::new(self.output.replay(key), args));
        }
        if let Some((key, args)) = &step.sampling {
            sequence.push(ReplayExecution::new(mtp.sampling.replay(key), args));
        }
        let stream = self.runtime.stream();
        if stream.gpu_timestamps_enabled() {
            stream
                .submit_replay_sequence_with_gpu_timestamps(&sequence, &[sequence.len()])
                .into()
        } else {
            stream.submit_replay_sequence(&sequence).into()
        }
    }
    fn read_mtp_step(&mut self, batch: &Qwen35Microbatch, gpu_elapsed: Option<Duration>) {
        let start = Instant::now();
        let count = (0..batch.num_reqs()).filter(|&i| batch.is_decode_req(i)).count();
        let tokens = self.sampler_output.token_ids.read_typed(0, count);
        let probs = self.sampler_output.token_probs.read_typed(0, count);
        let mtp = self.mtp.as_mut().unwrap();
        mtp.drafts.push(tokens);
        mtp.draft_probs.push(probs);
        mtp.timing.spec_read_elapsed += start.elapsed();
        mtp.timing.spec_replay_gpu_elapsed = mtp
            .timing
            .spec_replay_gpu_elapsed
            .zip(gpu_elapsed)
            .map(|(total, elapsed)| total + elapsed);
        let module_index = mtp.drafts.len() - 1;
        let versions: Vec<_> = batch
            .token_indices()
            .iter()
            .enumerate()
            .map(|(i, &start)| start + batch.cu_tokens()[i + 1] - batch.cu_tokens()[i])
            .collect();
        mtp.states[module_index].commit(
            self.runtime.device(),
            self.pages.buffer(),
            |_| unreachable!("native MTP has no PLE"),
            batch,
            &versions,
        );
    }
    fn submit_mtp(&mut self, first: &Qwen4xMTPStep) -> MetalReplaySubmission {
        let depth = self.mtp.as_ref().unwrap().num_spec_tokens;
        if depth == 1 {
            self.mtp.as_mut().unwrap().final_batch = Some(first.batch.microbatch.clone());
            return self.submit_mtp_step(first);
        }
        let submission = self.submit_mtp_step(first);
        submission.wait();
        self.read_mtp_step(
            &first.batch.microbatch,
            submission.gpu_timestamp_durations().map(|times| times.iter().sum()),
        );
        for module_index in 1..depth {
            let start = Instant::now();
            let mut step = self.record_mtp_embed(module_index);
            self.record_mtp_body(&mut step);
            self.record_mtp_output(&mut step);
            self.record_mtp_sampling(&mut step);
            self.mtp.as_mut().unwrap().timing.spec_build_elapsed += start.elapsed();
            if module_index + 1 == depth {
                self.mtp.as_mut().unwrap().final_batch = Some(step.batch.microbatch.clone());
                return self.submit_mtp_step(&step);
            }
            let submission = self.submit_mtp_step(&step);
            submission.wait();
            self.read_mtp_step(
                &step.batch.microbatch,
                submission.gpu_timestamp_durations().map(|times| times.iter().sum()),
            );
        }
        unreachable!("MTP depth is positive and the final module returns its submission")
    }
    fn finish_mtp(
        &mut self,
        decisions: &mut [Qwen35DecodeDecision],
        elapsed: Duration,
        gpu_durations: Option<&[Duration]>,
    ) -> ModelOutputTiming {
        let batch = self.mtp.as_mut().unwrap().final_batch.take().unwrap();
        self.read_mtp_step(&batch, gpu_durations.map(|times| times.iter().sum()));
        let mtp = self.mtp.as_mut().unwrap();
        debug_assert_eq!(mtp.drafts.len(), mtp.num_spec_tokens);
        for request in &mtp.requests {
            let (tokens, has_hidden) = match &request.kind {
                plan::Kind::Prefill(lanes) => (lanes.iter().map(|tokens| *tokens.last().unwrap()).collect(), false),
                plan::Kind::Decode {
                    sampled_token_id: sampled,
                    decision_index: decision,
                    ..
                } => {
                    for module_index in 0..mtp.num_spec_tokens {
                        let token = mtp.drafts[module_index][*decision] as u32;
                        mtp.probabilities
                            .set_expected_draft_token(request.req_slot, module_index, token);
                        decisions[*decision].spec_tokens.push(token);
                        decisions[*decision]
                            .spec_probs
                            .push(mtp.draft_probs[module_index][*decision]);
                        decisions[*decision].spec_confidences.push(1.0);
                    }
                    (
                        std::iter::once(*sampled)
                            .chain(
                                mtp.drafts
                                    .iter()
                                    .take(mtp.num_spec_tokens - 1)
                                    .map(|step| step[*decision]),
                            )
                            .collect(),
                        true,
                    )
                },
            };
            mtp.history[request.req_slot as usize] = plan::History {
                pending_token_index: request.next_pending_token_index,
                token_ids: tokens,
                has_hidden,
            };
        }
        mtp.timing.spec_passes = mtp.num_spec_tokens;
        mtp.timing.spec_replay_elapsed = elapsed;
        mtp.timing
    }
}

impl Qwen4xMTPSpeculator {
    fn finish_commits(&mut self) {
        for state in &mut self.states {
            state.finish_commit();
        }
    }
    fn clear_replay_cache(&mut self) {
        self.finish_commits();
        self.body.clear();
        self.sampling.clear();
        self.rejection.clear();
    }
    fn reset_req_slots(&mut self, slots: &[u32]) {
        self.finish_commits();
        self.probabilities.reset_req_slots(slots);
        for &slot in slots {
            self.history[slot as usize] = plan::History::default();
        }
        for state in &mut self.states {
            state.reset_req_slots(slots);
        }
    }
    fn load_weights(&mut self, device: &Device) -> Result<(), ModelExecutorError> {
        let mut store = SafeTensorStore::from_model_dir(&self.model_dir)?;
        self.body
            .component_mut()
            .load_weights(device, &mut store, &self.config, &self.bindings)
    }
    fn snapshot_files(&self, files: &mut Vec<StateSnapshotFile>) {
        for module in 0..self.num_spec_tokens as u32 {
            files.extend([
                StateSnapshotFile::Qwen4xMTPPageTable(module),
                StateSnapshotFile::Qwen4xMTPAuxState(module),
                StateSnapshotFile::Qwen4xMTPRequestState(module),
                StateSnapshotFile::Qwen4xMTPCarryHistory(module),
            ]);
        }
    }
    fn write_state(
        &mut self,
        device: &Device,
        stream: &inference_backend_metal::metal::Stream,
        writer: &mut StateSnapshotWriter,
        slots: &[std::ops::Range<u32>],
    ) -> Result<(), ModelExecutorError> {
        self.finish_commits();
        for (module, state) in self.states.iter().enumerate() {
            let module = module as u32;
            state.pack_snapshot(device, stream);
            state.page_table().write_selected_state(
                writer,
                GQAStateSnapshotFiles::new(StateSnapshotFile::Qwen4xMTPPageTable(module)),
                slots,
            )?;
            writer.write_selected_buffer(
                StateSnapshotFile::Qwen4xMTPAuxState(module),
                state.packed(),
                slots,
                state.state_bytes(),
            )?;
            writer.write_metadata(
                StateSnapshotFile::Qwen4xMTPRequestState(module),
                &state.snapshot_metadata(slots),
            )?;
            writer.write_selected_buffer(
                StateSnapshotFile::Qwen4xMTPCarryHistory(module),
                state.index_states(0),
                &state.carry_history_ranges(slots),
                state.carry_bytes(),
            )?;
        }
        Ok(())
    }
    fn read_state(
        &mut self,
        device: &Device,
        stream: &inference_backend_metal::metal::Stream,
        reader: &mut StateSnapshotReader,
        slots: &[std::ops::Range<u32>],
    ) -> Result<(), ModelExecutorError> {
        for (module, state) in self.states.iter_mut().enumerate() {
            let module = module as u32;
            state.page_table_mut().read_selected_state(
                reader,
                GQAStateSnapshotFiles::new(StateSnapshotFile::Qwen4xMTPPageTable(module)),
                slots,
            )?;
            reader.read_selected_buffer(
                StateSnapshotFile::Qwen4xMTPAuxState(module),
                state.packed(),
                slots,
                state.state_bytes(),
            )?;
            let metadata: Qwen4xRequestState =
                reader.read_metadata(StateSnapshotFile::Qwen4xMTPRequestState(module))?;
            state.load_metadata(metadata, &self.config)?;
            reader.read_selected_buffer(
                StateSnapshotFile::Qwen4xMTPCarryHistory(module),
                state.index_states(0),
                &state.carry_history_ranges(slots),
                state.carry_bytes(),
            )?;
            state.restore_snapshot(device, stream);
        }
        Ok(())
    }
}
