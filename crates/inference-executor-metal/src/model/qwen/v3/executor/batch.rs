impl Qwen3Executor {
    fn validate_input(&self, core_batch_req: &BatchDeviceRequest) {
        assert!(
            core_batch_req.dev_reqs.len() <= self.config.max_requests,
            "qwen3 replay executor supports at most {} requests per batch, got {}",
            self.config.max_requests,
            core_batch_req.dev_reqs.len()
        );
        assert!(
            core_batch_req.token_cost() <= self.config.max_tokens,
            "qwen3 replay executor supports at most {} tokens per batch, got {}",
            self.config.max_tokens,
            core_batch_req.token_cost()
        );
        for request in &core_batch_req.dev_reqs {
            assert!(
                request.decoder_query_tokens.token_consumption() <= self.config.max_tokens_per_request,
                "qwen3 replay executor request tokens={} exceed scheduler max_tokens_per_request={}",
                request.decoder_query_tokens.token_consumption(),
                self.config.max_tokens_per_request
            );
            assert_eq!(
                request.decoder_query_tokens.num_spec_tokens(),
                0,
                "qwen3 does not accept speculative input tokens"
            );
        }
        let max_context_tokens = core_batch_req
            .dev_reqs
            .iter()
            .map(|request| {
                request
                    .decoder_query_tokens
                    .token_index()
                    .checked_add(request.decoder_query_tokens.token_consumption())
                    .expect("qwen3 GQA request context length overflow")
            })
            .max()
            .expect("qwen3 replay executor requires at least one request");
        let num_physical_pages = max_context_tokens.div_ceil(self.main_gqa_state.num_tokens_per_page());
        let page_capacity = self.gqa_page_table_layout.num_physical_pages_per_request();
        assert!(
            num_physical_pages <= page_capacity,
            "qwen3 GQA request context needs {} physical pages but capacity is {}",
            num_physical_pages,
            page_capacity
        );
    }

    pub fn model_config(&self) -> &Qwen3ModelConfig {
        match &self.checkpoint {
            Qwen3Checkpoint::Text(config) => config,
            Qwen3Checkpoint::Asr(_) => panic!("Qwen3-ASR does not contain a plain Qwen3 model config"),
        }
    }

    pub fn asr_model_config(&self) -> &Qwen3ASRModelConfig {
        match &self.checkpoint {
            Qwen3Checkpoint::Text(_) => panic!("plain Qwen3 does not contain a Qwen3-ASR model config"),
            Qwen3Checkpoint::Asr(config) => config,
        }
    }

    pub fn num_spec_tokens(&self) -> usize {
        0
    }

    pub fn num_main_gqa_page_ids_per_block(&self) -> usize {
        let layout = self.gqa_page_table_layout;
        layout.num_gqa_layers as usize * layout.num_page_ids_per_block as usize
    }

    pub fn num_kv_page_ids_per_block(&self) -> usize {
        self.num_main_gqa_page_ids_per_block()
    }

    fn prepare_gqa_page_ids(&self, batch: &BatchDeviceRequest) {
        let num_main_gqa_page_ids = self.num_main_gqa_page_ids_per_block();
        for request in &batch.dev_reqs {
            let page_ids_by_block = request
                .decoder_sync_blocks
                .kv_page_ids()
                .first()
                .expect("qwen3 Main GQA request requires runtime cache lane 0");
            for (block_offset, page_ids) in page_ids_by_block.iter().enumerate() {
                let block_index = request
                    .decoder_sync_blocks
                    .block_index()
                    .checked_add(block_offset)
                    .expect("qwen3 Main cache-block index must fit usize");
                assert_eq!(
                    page_ids.len(),
                    num_main_gqa_page_ids,
                    "Qwen3 Main cache block must contain exactly its Main page IDs"
                );
                self.main_gqa_state
                    .write_page_ids(request.req_slot, block_index, page_ids);
            }
        }
    }

    fn write_token_ids(&self, token_ids: &[i32]) {
        assert!(!token_ids.is_empty(), "qwen3 replay model requires at least one token");
        assert!(
            token_ids.len() <= self.layout.max_tokens as usize,
            "qwen3 replay model tokens={} exceed max_tokens={}",
            token_ids.len(),
            self.layout.max_tokens
        );
        self.token_ids.write_typed(0, token_ids);
    }
}
