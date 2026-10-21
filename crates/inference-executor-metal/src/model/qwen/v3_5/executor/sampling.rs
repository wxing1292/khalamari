impl Qwen35Executor {
    fn prepare_sample_replay(
        &mut self,
        sampler_configs: &[SamplerConfig],
        sample_req_slots: &[u32],
        sample_positions: &[u32],
    ) -> (TopKSamplingReplayKey, ReplayArguments) {
        assert_eq!(
            sampler_configs.len(),
            sample_positions.len(),
            "qwen3.5 sample runtime configs and positions must have equal lengths"
        );
        let sample_shape = self.sampling.component().prepare_shape(sampler_configs);
        let input = SamplingInput {
            shape: sample_shape,
            logits: &self.unembed_logits,
            output: self.sampler_output.as_output(),
        };
        let runtime = self.runtime.stream();
        let (sample_key, _) = self.sampling.record(runtime, &input);
        self.sampler.prepare(sample_req_slots, sample_positions);
        let mut replay_arguments = ReplayArguments::new();
        self.sampler.add_replay_arguments(sample_shape, &mut replay_arguments);
        (sample_key, replay_arguments)
    }

    fn prepare_mtp_sampling_replay(
        &mut self,
        sampler_configs: &[SamplerConfig],
        sample_req_slots: &[u32],
        sample_positions: &[u32],
    ) -> (TopKSamplingReplayKey, ReplayArguments) {
        assert_eq!(
            sampler_configs.len(),
            sample_positions.len(),
            "qwen3.5 MTP sample runtime configs and positions must have equal lengths"
        );
        let sample_shape = self
            .speculator
            .mtp()
            .sampling
            .component()
            .prepare_shape(sampler_configs);
        let mtp = self.speculator.mtp_mut();
        let input = DraftSamplingInput {
            shape: sample_shape,
            logits: &self.unembed_logits,
            output: self.sampler_output.as_output(),
            sparse: TopKSamplingWriteDistributionOutput {
                token_ids: mtp.common.spec_probs.draft_token_ids(),
                probs: mtp.common.spec_probs.draft_probs(),
                output_distribution_indices: &mtp.draft_distribution_indices,
                max_k: mtp.common.spec_probs.max_k() as u32,
                num_output_distributions: mtp.common.spec_probs.num_draft_distributions(),
            },
        };
        let runtime = self.runtime.stream();
        let (sample_key, _) = mtp.sampling.record(runtime, &input);
        self.sampler.prepare(sample_req_slots, sample_positions);
        let mut replay_arguments = ReplayArguments::new();
        self.sampler.add_replay_arguments(sample_shape, &mut replay_arguments);
        (sample_key, replay_arguments)
    }

    fn record_sampling(&mut self, microbatch: &Qwen35Microbatch) -> (TopKSamplingReplayKey, ReplayArguments) {
        let sampler_configs = sample_sampler_configs(microbatch);
        let sample_req_slots = sample_req_slots(microbatch);
        let sample_positions = sample_token_positions(microbatch);
        self.prepare_sample_replay(&sampler_configs, &sample_req_slots, &sample_positions)
    }

    fn assert_expected_draft_tokens_fit(&self, num_tokens: usize) {
        assert!(num_tokens > 0, "qwen3.5 replay model requires at least one token");
        assert!(
            num_tokens <= self.layout.max_tokens as usize,
            "qwen3.5 replay model tokens={} exceed max_tokens={}",
            num_tokens,
            self.layout.max_tokens
        );
    }

    fn read_sampled_token_ids(&self, num_decode_reqs: usize) -> Qwen35SampledTokens {
        let output = &self.sampler_output;
        Qwen35SampledTokens::new(
            output.token_ids.read_typed::<i32>(0, num_decode_reqs),
            output.token_probs.read_typed::<f32>(0, num_decode_reqs),
        )
    }

    fn read_sample_decisions(&self, num_decode_reqs: usize) -> Vec<Qwen35DecodeDecision> {
        sample_decisions_from_sampled_tokens(&self.read_sampled_token_ids(num_decode_reqs))
    }

    fn read_sampling(
        &mut self,
        num_sample_rows: usize,
        replay_elapsed: Duration,
    ) -> (Vec<Qwen35DecodeDecision>, ModelOutputTiming) {
        let mut timing = ModelOutputTiming {
            main_sample_replay_elapsed: replay_elapsed,
            ..ModelOutputTiming::default()
        };
        let sample_read_start = Instant::now();
        let decisions = self.read_sample_decisions(num_sample_rows);
        trace_decisions("sample_read", &decisions);
        timing.sample_read_elapsed = sample_read_start.elapsed();
        (decisions, timing)
    }

    fn record_rejection_sampling(&mut self, recorder: &mut Qwen35ModelOpsRecorder, microbatch: &Qwen35Microbatch) {
        let sample_positions = sample_token_positions(microbatch);
        let num_main_output_rows = num_main_output_rows(microbatch);
        let sampler_configs = sample_sampler_configs(microbatch);
        let mut flat_draft_distribution_indices = Vec::new();
        let max_spec_tokens = self.num_spec_tokens();
        assert!(
            max_spec_tokens > 0,
            "qwen3.5 target rejection sampling requires a speculator"
        );
        for req_index in 0..microbatch.num_reqs() {
            if !microbatch.is_decode_req(req_index) {
                continue;
            }
            let req_slot = microbatch.req_slots()[req_index];
            let num_spec_tokens = microbatch.num_spec_tokens(req_index) as usize;
            assert!(
                num_spec_tokens <= max_spec_tokens,
                "qwen3.5 replay rejection num_spec_tokens exceeds speculator capacity"
            );
            let q_end = microbatch.cu_tokens()[req_index + 1] as usize;
            for (spec_token_index, &draft_token) in microbatch.flat_token_ids()[q_end - num_spec_tokens..q_end]
                .iter()
                .enumerate()
            {
                self.speculator.common().spec_probs.assert_expected_draft_token(
                    req_slot,
                    spec_token_index,
                    draft_token
                        .try_into()
                        .expect("qwen3.5 request contained a negative draft token ID"),
                );
                flat_draft_distribution_indices.push(
                    self.speculator
                        .common()
                        .spec_probs
                        .draft_distribution_index(req_slot, spec_token_index),
                );
            }
        }
        let rejector = Rc::clone(
            self.speculator
                .common()
                .rejection_sampling
                .component()
                .rejector(),
        );
        let prepared = rejector.prepare_inputs(microbatch, &flat_draft_distribution_indices);
        let num_active_decode_reqs = prepared.num_active_decode_reqs();
        let num_active_target_distributions = prepared.num_active_target_distributions();
        debug_assert_eq!(
            num_main_output_rows, num_active_target_distributions,
            "qwen3.5 Main output rows must match target distributions"
        );
        let active_target_shape = self.sampler.active_shape(&sampler_configs);
        let rejection_shape = rejector.prepare_replay_shape(&prepared, active_target_shape.top_k);
        let target_distribution_shape = active_target_shape
            .with_num_total_sampling_inputs(rejection_shape.num_total_target_distributions);
        let Qwen35SpeculativeResources {
            rejection_sampling,
            spec_probs,
            target_distribution_indices,
        } = self.speculator.common_mut();
        let rejection_input = RejectionSamplerInput {
            shape: rejection_shape,
            target_token_ids: spec_probs.target_token_ids(),
            target_probs: spec_probs.target_probs(),
            draft_token_ids: spec_probs.draft_token_ids(),
            draft_probs: spec_probs.draft_probs(),
        };
        let component_input = RejectionSamplingInput {
            target_shape: target_distribution_shape,
            logits: &self.unembed_logits,
            target_sparse: TopKSamplingWriteDistributionOutput {
                token_ids: spec_probs.target_token_ids(),
                probs: spec_probs.target_probs(),
                output_distribution_indices: target_distribution_indices,
                max_k: spec_probs
                    .max_k()
                    .try_into()
                    .expect("qwen3.5 draft distribution width must fit u32"),
                num_output_distributions: spec_probs.num_target_distributions(),
            },
            rejection: rejection_input,
        };
        {
            let rejection_build_start = Instant::now();
            let runtime = self.runtime.stream();
            let (rejection_key, rejection_cache_hit) = rejection_sampling.record(runtime, &component_input);
            if !rejection_cache_hit {
                recorder.rejection_build_elapsed += rejection_build_start.elapsed();
            }
            recorder.rejection_key = Some(rejection_key);
        }
        self.sampler.prepare(&sample_req_slots(microbatch), &sample_positions);
        let mut rejection_runtime_params = Vec::with_capacity(num_active_decode_reqs);
        let mut target_offset = 0usize;
        for &req_index in &prepared.decode_req_indices {
            let sampler_config = &microbatch.sampler_configs()[req_index];
            let sample_position = sample_positions[target_offset];
            let num_spec_tokens = microbatch.num_spec_tokens(req_index);
            assert!(
                sample_position <= u32::MAX - num_spec_tokens,
                "qwen3.5 rejection sampling positions must fit u32"
            );
            rejection_runtime_params.push(SparseRejectionSamplingReqParams {
                seed: microbatch.sampler_configs()[req_index].seed(),
                sample_position,
                top_k: self
                    .sampler_bounds
                    .active_top_k(sampler_config)
                    .expect("qwen3.5 rejection sampler config should fit bounds"),
            });
            target_offset += num_spec_tokens as usize + 1;
        }
        assert_eq!(
            target_offset, num_active_target_distributions,
            "qwen3.5 rejection target distributions must cover sampled requests"
        );
        rejector.set_runtime_params(&rejection_runtime_params);
        let mut replay_arguments = ReplayArguments::new();
        self.sampler
            .add_replay_arguments(target_distribution_shape, &mut replay_arguments);
        rejector.add_replay_arguments(rejection_input, &mut replay_arguments);
        recorder.rejection_arguments = replay_arguments;
        recorder.rejection_prepared = Some(prepared);
    }

    fn read_rejection_sampling(
        &mut self,
        recorder: &Qwen35ModelOpsRecorder,
        microbatch: &Qwen35Microbatch,
        replay_elapsed: Duration,
    ) -> (Vec<Qwen35DecodeDecision>, ModelOutputTiming) {
        let prepared = recorder
            .rejection_prepared
            .as_ref()
            .expect("qwen3.5 rejection read requires recorded inputs");
        let num_active_decode_reqs = prepared.num_active_decode_reqs();
        let num_active_draft_distributions = prepared.num_active_draft_distributions;
        let mut timing = ModelOutputTiming {
            main_sample_replay_elapsed: replay_elapsed,
            rejection_build_elapsed: recorder.rejection_build_elapsed,
            ..ModelOutputTiming::default()
        };
        let rejection_read_start = Instant::now();
        let results = self
            .speculator
            .common()
            .rejection_sampling
            .component()
            .rejector()
            .read_results(num_active_decode_reqs, num_active_draft_distributions);
        let mut decisions = Vec::with_capacity(num_active_decode_reqs);
        let mut flat_draft_index = 0usize;
        for (decode_req_index, &req_index) in prepared.decode_req_indices.iter().enumerate() {
            let num_accepted_tokens = results.num_accepted_tokens(decode_req_index);
            assert!(
                num_accepted_tokens <= microbatch.num_spec_tokens(req_index) as usize,
                "qwen3.5 replay rejection accepted more tokens than drafts"
            );
            let decision = Qwen35DecodeDecision {
                sampled_token: results
                    .sampled_token_id(decode_req_index)
                    .try_into()
                    .expect("qwen3.5 rejection sampler returned a negative token ID"),
                sampled_prob: results.sampled_prob(decode_req_index),
                validated_tokens: results
                    .accepted_token_ids(flat_draft_index, num_accepted_tokens)
                    .iter()
                    .map(|&token| {
                        token
                            .try_into()
                            .expect("qwen3.5 rejection sampler returned a negative accepted token ID")
                    })
                    .collect(),
                validated_probs: results.accepted_probs(flat_draft_index, num_accepted_tokens).to_vec(),
                ..Qwen35DecodeDecision::default()
            };
            decisions.push(decision);
            flat_draft_index += microbatch.num_spec_tokens(req_index) as usize;
        }
        timing.rejection_read_elapsed = rejection_read_start.elapsed();
        (decisions, timing)
    }

    pub fn num_spec_tokens(&self) -> usize {
        self.speculator.num_spec_tokens()
    }
}
