impl Qwen35Executor {
    fn validate_input(&self, core_batch_req: &BatchDeviceRequest) {
        assert!(
            core_batch_req.dev_reqs.len() <= self.config.max_requests,
            "qwen3.5 replay executor supports at most {} requests per batch, got {}",
            self.config.max_requests,
            core_batch_req.dev_reqs.len()
        );
        assert!(
            core_batch_req.token_cost() <= self.config.max_tokens,
            "qwen3.5 replay executor supports at most {} tokens per batch, got {}",
            self.config.max_tokens,
            core_batch_req.token_cost()
        );
        let max_spec_tokens = self.num_spec_tokens();
        for request in &core_batch_req.dev_reqs {
            assert!(
                request.decoder_query_tokens.token_consumption() <= self.config.max_tokens_per_request,
                "qwen3.5 replay executor request tokens={} exceed scheduler max_tokens_per_request={}",
                request.decoder_query_tokens.token_consumption(),
                self.config.max_tokens_per_request
            );
            let num_spec_tokens = request.decoder_query_tokens.num_spec_tokens();
            assert!(
                num_spec_tokens <= max_spec_tokens,
                "qwen3.5 speculative-token count exceeds speculator capacity"
            );
            if num_spec_tokens > 0 {
                let num_fixed_tokens = request.decoder_query_tokens.token_consumption() - num_spec_tokens;
                assert!(!self.main_gdn_state.backend().uses_recurrent(num_fixed_tokens, 0),
                    "qwen3.5 speculative input exceeds the GDN fixed-token replay window");
            }
        }
        let max_context_tokens = core_batch_req
            .dev_reqs
            .iter()
            .map(|request| {
                request
                    .decoder_query_tokens
                    .token_index()
                    .checked_add(request.decoder_query_tokens.token_consumption())
                    .expect("qwen3.5 GQA request context length overflow")
            })
            .max()
            .expect("qwen3.5 replay executor requires at least one request");
        let num_physical_pages =
            max_context_tokens.div_ceil(self.main_gqa_state.backend().num_tokens_per_page() as usize);
        let page_capacity = self.gqa_page_table_layout.num_physical_pages_per_request();
        assert!(
            num_physical_pages <= page_capacity,
            "qwen3.5 GQA request context needs {} physical pages but capacity is {}",
            num_physical_pages,
            page_capacity
        );
        if self.speculator.is_dspark() {
            let dspark = &self.speculator.dspark().execution;
            let num_dspark_pages = max_context_tokens.div_ceil(dspark.num_tokens_per_page());
            let dspark_page_capacity = dspark.num_physical_pages_per_request();
            assert!(
                num_dspark_pages <= dspark_page_capacity,
                "qwen3.5 DSpark request context needs {} physical pages but capacity is {}",
                num_dspark_pages,
                dspark_page_capacity
            );
        } else if self.speculator.is_dflash2() {
            let dflash2 = &self.speculator.dflash2().execution;
            let num_dflash2_pages = max_context_tokens.div_ceil(dflash2.num_tokens_per_page());
            let dflash2_page_capacity = dflash2.num_physical_pages_per_request();
            assert!(
                num_dflash2_pages <= dflash2_page_capacity,
                "qwen3.5 DFlash2 request context needs {} physical pages but capacity is {}",
                num_dflash2_pages,
                dflash2_page_capacity
            );
        }
    }

    pub fn num_main_gqa_page_ids_per_block(&self) -> usize {
        let layout = self.gqa_page_table_layout;
        let num_page_ids = (layout.num_gqa_layers as u64)
            .checked_mul(layout.num_page_ids_per_block as u64)
            .expect("qwen3.5 main GQA page IDs per block overflow");
        usize::try_from(num_page_ids).expect("qwen3.5 main GQA page IDs per block must fit usize")
    }

    pub fn num_main_lane_gqa_page_ids_per_block(&self) -> usize {
        self.num_gqa_page_ids_per_main_lane_block
    }

    fn prepare_gqa_page_ids(&self, batch: &BatchDeviceRequest) {
        let num_main_gqa_page_ids = self.num_main_gqa_page_ids_per_block();
        let num_speculator_gqa_page_ids = self.speculator.num_gqa_page_ids_per_main_lane_block();
        debug_assert_eq!(
            num_main_gqa_page_ids
                .checked_add(num_speculator_gqa_page_ids)
                .expect("qwen3.5 Main cache-lane page-ID count must fit usize"),
            self.num_gqa_page_ids_per_main_lane_block
        );
        for request in &batch.dev_reqs {
            let page_ids_by_block = request
                .decoder_sync_blocks
                .kv_page_ids()
                .first()
                .expect("qwen3.5 Main GQA request requires runtime cache lane 0");
            for (block_offset, page_ids) in page_ids_by_block.iter().enumerate() {
                let block_index = request
                    .decoder_sync_blocks
                    .block_index()
                    .checked_add(block_offset)
                    .expect("qwen3.5 Main cache-block index must fit usize");
                let (main_page_ids, speculator_page_ids) =
                    split_main_lane_page_ids(page_ids, num_main_gqa_page_ids, num_speculator_gqa_page_ids);
                self.main_gqa_state
                    .write_page_ids(request.req_slot, block_index, main_page_ids);
                self.speculator
                    .write_page_ids(request.req_slot, block_index, speculator_page_ids);
            }
        }
    }

    pub fn num_mtp_gqa_page_ids_per_block(&self) -> Vec<usize> {
        match &self.speculator {
            Qwen35Speculator::Vanilla | Qwen35Speculator::DSpark(_) | Qwen35Speculator::DFlash2(_) => Vec::new(),
            Qwen35Speculator::MTP(mtp) => {
                let layout = mtp.gqa_state.request_page_table().layout();
                vec![layout.num_page_ids_per_block as usize; mtp.num_spec_tokens]
            },
        }
    }

    pub fn num_gdn_state_page_ids_per_block(&self) -> usize {
        self.main_gdn_state.num_pages_per_state_slot()
    }

    fn commit(&mut self, compute_seq: RawComputeSlotSeq, decisions: &[Qwen35DecodeDecision]) {
        let dst_state_versions = self.pending_transactions.commit(compute_seq, decisions);
        trace_decisions("model_commit_decisions", decisions);
        trace::qwen35_state(|| {
            format!("event=model_commit dst_state_versions={dst_state_versions:?}")
        });
        self.main_gdn_state
            .commit(self.pages.buffer(), &dst_state_versions);
        // State reconstruction and publish share one asynchronous submission.
        // The next prepare/reset waits before reusing logs, state slots, or page staging.
    }

    fn finish_gdn_commit(&mut self) {
        let start = Instant::now();
        self.main_gdn_state.finish_commit();
        trace::qwen35_state(|| format!("event=gdn_commit_wait elapsed_us={}", start.elapsed().as_micros()));
    }

    fn write_token_ids(&self, token_ids: &[i32]) {
        self.assert_expected_draft_tokens_fit(token_ids.len());
        self.token_ids.write_typed(0, token_ids);
    }
}
