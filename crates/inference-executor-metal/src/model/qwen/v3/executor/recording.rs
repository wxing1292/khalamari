impl Qwen3Executor {
    fn submit_main_recording(&self, recorder: &Qwen3ModelOpsRecorder) -> MetalReplaySubmission {
        let main_text_embed_replay = self.main_text_embed.replay(&recorder.main_text_embed_key);
        let main_replay = self.main.replay(&recorder.main_key);
        let mut sequence = vec![ReplayExecution::new(
            main_text_embed_replay,
            &recorder.main_text_embed_arguments,
        )];
        if let Some(resource_embed_key) = &recorder.resource_embed_key {
            sequence.push(ReplayExecution::new(
                self.main_resource_embed
                    .as_ref()
                    .expect("Qwen3 ResourceEmbed replay requires MainResourceEmbed")
                    .replay(resource_embed_key),
                &recorder.resource_embed_arguments,
            ));
        }
        sequence.push(ReplayExecution::new(main_replay, &recorder.main_arguments));
        let runtime = self.runtime.stream();
        let mut timestamp_stage_ends = runtime.gpu_timestamps_enabled().then(|| Vec::with_capacity(1));
        if let Some(gather_unembed_key) = &recorder.gather_unembed_key {
            assert!(
                recorder.num_main_sample_rows > 0,
                "qwen3 GatherUnembed replay requires Main sampling rows"
            );
            sequence.push(ReplayExecution::new(
                self.gather_unembed.replay(gather_unembed_key),
                &recorder.gather_unembed_arguments,
            ));

            let sampling_key = recorder
                .sampling_key
                .as_ref()
                .expect("qwen3 sampled output requires Sampling replay");
            sequence.push(ReplayExecution::new(
                self.sampling.replay(sampling_key),
                &recorder.sampling_arguments,
            ));
            if let Some(stage_ends) = &mut timestamp_stage_ends {
                stage_ends.push(sequence.len());
            }
        } else {
            assert_eq!(
                recorder.num_main_sample_rows, 0,
                "qwen3 Main sampling rows require GatherUnembed replay"
            );
            assert!(
                recorder.sampling_key.is_none(),
                "qwen3 Main recording without output rows must not contain sampling"
            );
            if let Some(stage_ends) = &mut timestamp_stage_ends {
                stage_ends.push(sequence.len());
            }
        }
        if let Some(stage_ends) = timestamp_stage_ends {
            runtime.submit_replay_sequence_with_gpu_timestamps(&sequence, &stage_ends).into()
        } else {
            runtime.submit_replay_sequence(&sequence).into()
        }
    }
}
