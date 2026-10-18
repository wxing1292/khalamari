impl Qwen35Executor {
    fn write_gather_flat_indices(&self, microbatch: &Qwen35Microbatch) -> Vec<u32> {
        // The mask selects source hidden states. Its compact indices are a
        // dynamic gather input, not batch state: [F, F, F, T, T, T] -> [3, 4, 5].
        let flat_indices = gather_flat_indices(microbatch);
        debug_assert!(
            !flat_indices.is_empty(),
            "qwen3.5 replay unembed requires Main output rows"
        );
        self.gather_flat_indices.write_typed(0, &flat_indices);
        flat_indices
    }

    fn prepare_gather_unembed_replay(
        &mut self,
        microbatch: &Qwen35Microbatch,
        hidden_input: &Buffer,
    ) -> (Qwen35GatherUnembedReplayKey, ReplayArguments) {
        let num_active_rows = self
            .write_gather_flat_indices(microbatch)
            .len()
            .try_into()
            .expect("qwen3.5 GatherUnembed output row count must fit u32");
        let (gather_unembed_key, gather_unembed_arguments) =
            self.gather_unembed.component().prepare_replay(num_active_rows);
        let input = Qwen35GatherUnembedArgs {
            num_rows: num_active_rows,
            hidden_input,
            row_indices: &self.gather_flat_indices,
            hidden_output: &self.unembed_hidden,
            logits: &self.unembed_logits,
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_key, _) = self.gather_unembed.record(&runtime, &input);
        debug_assert_eq!(
            recorded_key, gather_unembed_key,
            "qwen3.5 GatherUnembed replay input must match the prepared replay key"
        );
        (recorded_key, gather_unembed_arguments)
    }
}
