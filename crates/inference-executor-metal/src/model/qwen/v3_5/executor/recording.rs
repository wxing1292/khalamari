impl Qwen35Executor {
    fn replay_runtime(&self) -> MetalReplayRuntime<'_> {
        MetalReplayRuntime::new(self.runtime.stream())
    }

    fn create_recorder(&self) -> ReplayRecorder {
        let runtime = self.replay_runtime();
        Runtime::create_recorder(&runtime)
    }

    fn submit_replay(&self, replay: &ReplayProgram) {
        self.runtime.submit_replay(replay).wait();
    }

    fn submit_replay_with_arguments(&self, replay: &ReplayProgram, arguments: &ReplayArguments) {
        self.runtime.submit_replay_with_arguments(replay, arguments).wait();
    }

    fn submit_main_recording(&self, recorder: &Qwen35ModelOpsRecorder) -> MetalReplaySubmission {
        let main_text_embed_replay = self.main_text_embed.replay(&recorder.main_text_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let empty_arguments = ReplayArguments::new();
        let mut sequence = vec![ReplayExecution::new(
            main_text_embed_replay,
            &recorder.main_text_embed_arguments,
        )];
        if let Some(resource_embed_key) = &recorder.resource_embed_key {
            sequence.push(ReplayExecution::new(
                self.main_resource_embed
                    .as_ref()
                    .expect("Qwen3.8 ResourceEmbed replay requires MainResourceEmbed")
                    .replay(resource_embed_key),
                &recorder.resource_embed_arguments,
            ));
        }
        sequence.push(ReplayExecution::new(main_replay, &recorder.main_arguments));
        let runtime = self.replay_runtime();
        let mut timestamp_stage_ends = runtime.gpu_timestamps_enabled().then(|| Vec::with_capacity(5));
        if let Some(gather_unembed_key) = &recorder.main_gather_unembed_key {
            assert!(
                recorder.num_main_sample_rows > 0,
                "qwen3.5 GatherUnembed replay requires Main sampling rows"
            );
            sequence.push(ReplayExecution::new(
                self.gather_unembed.replay(gather_unembed_key),
                &recorder.main_gather_unembed_arguments,
            ));
            if let Some(rejection_key) = &recorder.rejection_key {
                assert!(
                    recorder.sampling_key.is_none(),
                    "qwen3.5 Main recording must select one sampling path"
                );
                if let Some(stage_ends) = &mut timestamp_stage_ends {
                    stage_ends.push(sequence.len());
                }
                sequence.push(ReplayExecution::new(
                    self.speculator.common().rejection_sampling.replay(rejection_key),
                    &recorder.rejection_arguments,
                ));
                if let Some(stage_ends) = &mut timestamp_stage_ends {
                    stage_ends.push(sequence.len());
                }
            } else {
                let sampling_key = recorder
                    .sampling_key
                    .as_ref()
                    .expect("qwen3.5 sampled output requires Sampling replay");
                sequence.push(ReplayExecution::new(
                    self.sampling.replay(sampling_key),
                    &recorder.sampling_arguments,
                ));
                if let Some(stage_ends) = &mut timestamp_stage_ends {
                    stage_ends.push(sequence.len());
                }
            }
        } else {
            assert_eq!(
                recorder.num_main_sample_rows, 0,
                "qwen3.5 Main sampling rows require GatherUnembed replay"
            );
            assert!(
                recorder.sampling_key.is_none() && recorder.rejection_key.is_none(),
                "qwen3.5 Main recording without output rows must not contain sampling"
            );
            if let Some(stage_ends) = &mut timestamp_stage_ends {
                stage_ends.push(sequence.len());
            }
        }

        if self.speculator.is_dflash2() {
            let spec_stage_ends = self.speculator.dflash2().execution.append_spec_replays(
                &mut sequence,
                recorder
                    .dflash2_spec_prefill
                    .as_ref()
                    .expect("qwen3.5 combined DFlash2 sequence requires Spec Prefill"),
                recorder.dflash2_spec_decode.as_ref(),
            );
            if let Some(stage_ends) = &mut timestamp_stage_ends {
                if let Some(decode_prepare) = spec_stage_ends.decode_prepare {
                    stage_ends.push(decode_prepare);
                }
                stage_ends.push(spec_stage_ends.prefill);
                if let Some(decode) = spec_stage_ends.decode {
                    stage_ends.push(decode);
                }
            }
        }

        trace::qwen35_state(|| format!("event=submit_main_sequence_start main_key={:?}", recorder.main_key));
        let submission = if let Some(stage_ends) = timestamp_stage_ends {
            runtime.submit_replay_sequence_with_gpu_timestamps(&sequence, &stage_ends)
        } else {
            runtime.submit_replay_sequence(&sequence)
        };
        trace::qwen35_state(|| format!("event=submit_main_sequence_submitted main_key={:?}", recorder.main_key));
        submission
    }
    fn submit_gdn_state_restore(&mut self) -> Duration {
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let start = Instant::now();
        self.main_gdn_state.restore(&runtime, self.pages.buffer());
        let elapsed = start.elapsed();
        trace::qwen35_state(|| format!("event=gdn_restore elapsed_us={}", elapsed.as_micros()));
        elapsed
    }
}
