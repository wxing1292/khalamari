// MTP Decode lifecycle, K = 3. Every lane uses cache-local indices from 0.
// C = cached KV (including replaceable drafts); P = pending, not yet executed.
// Main verifies [w, x1, x2, x3] from index 3. It does not replay cached Main tokens.
//
// Before verification:
// +-------+----+----+----+----+
// | Index | 0  | 1  | 2  | 3  |
// | Cache | C  | C  | C  | P  |
// | Main  | t0 | t1 | t2 | w  |
// | MTP0  | t1 | t2 | w  | x1 |
// | MTP1  | t2 | w  | x1 | x2 |
// | MTP2  | w  | x1 | x2 | x3 |
// +-------+----+----+----+----+
// Old hidden cache: MTP0 H0(w) @2; MTP1 H1(w), H1(x1) @1,2; MTP2 none.
//
// Reject all; sample y and propose z1/z2/z3. Next pending index = 4.
// +-------+----+----+----+----+----+
// | Index | 0  | 1  | 2  | 3  | 4  |
// | Cache | C  | C  | C  | C  | P  |
// | Main  | t0 | t1 | t2 | w  | y  |
// | MTP0  | t1 | t2 | w  | y  | z1 |
// | MTP1  | t2 | w  | y  | z1 | z2 |
// | MTP2  | w  | y  | z1 | z2 | z3 |
// +-------+----+----+----+----+----+
// Fresh MTP0 @3: [y]
// Fresh MTP1 @2: [y, z1]
// Fresh MTP2 @1: [y, z1, z2]
//
// Accept x1; reject x2/x3; sample y and propose z1/z2/z3. Next pending index = 5.
// +-------+----+----+----+----+----+----+
// | Index | 0  | 1  | 2  | 3  | 4  | 5  |
// | Cache | C  | C  | C  | C  | C  | P  |
// | Main  | t0 | t1 | t2 | w  | x1 | y  |
// | MTP0  | t1 | t2 | w  | x1 | y  | z1 |
// | MTP1  | t2 | w  | x1 | y  | z1 | z2 |
// | MTP2  | w  | x1 | y  | z1 | z2 | z3 |
// +-------+----+----+----+----+----+----+
// Fresh MTP0 @3: [x1, y]
// Fresh MTP1 @3: [y, z1]
// Fresh MTP2 @2: [y, z1, z2]
//
// Accept x1/x2/x3; sample y and propose z1/z2/z3. Next pending index = 7.
// +-------+----+----+----+----+----+----+----+----+
// | Index | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
// | Cache | C  | C  | C  | C  | C  | C  | C  | P  |
// | Main  | t0 | t1 | t2 | w  | x1 | x2 | x3 | y  |
// | MTP0  | t1 | t2 | w  | x1 | x2 | x3 | y  | z1 |
// | MTP1  | t2 | w  | x1 | x2 | x3 | y  | z1 | z2 |
// | MTP2  | w  | x1 | x2 | x3 | y  | z1 | z2 | z3 |
// +-------+----+----+----+----+----+----+----+----+
// Fresh MTP0 @3: [x1, x2, x3, y]
// Fresh MTP1 @3: [x2, x3, y, z1]
// Fresh MTP2 @3: [x3, y, z1, z2]
//
// For V continuation tokens after the anchor, module m starts at
// P - max(m - V, 0), not always P - m. Its fresh token count is max(V, m) + 1.
// See mtp/decode_plan.rs for the executable plan and replacement-tail behavior.
// Every result retains H0(y) @P'-1 and H1(y), H1(z1) @P'-2,P'-1.
// MTP2 retains no hidden rows. Gather old hidden inputs before writing this new tail.

impl Qwen35Executor {
    fn mtp_requests(&self, microbatch: &Qwen35Microbatch, decisions: &[Qwen35DecodeDecision]) -> Vec<Qwen35MTPRequest> {
        let mut requests = Vec::with_capacity(microbatch.num_reqs());
        let mut decision_index = 0usize;
        let num_spec_tokens = self.speculator.mtp().num_spec_tokens;
        for req_index in 0..microbatch.num_reqs() {
            let flat_start = microbatch.cu_tokens()[req_index] as usize;
            let flat_end = microbatch.cu_tokens()[req_index + 1] as usize;
            let req_slot = microbatch.req_slots()[req_index];
            if microbatch.is_decode_req(req_index) {
                let decision = &decisions[decision_index];
                let num_input_spec_tokens = microbatch.num_spec_tokens(req_index) as usize;
                debug_assert!(num_input_spec_tokens <= num_spec_tokens);
                let num_committed_input_tokens = flat_end - flat_start - num_input_spec_tokens;
                debug_assert!(
                    num_committed_input_tokens > 0,
                    "qwen3.5 MTP proposal requires a non-spec anchor token"
                );
                debug_assert!(
                    decision.validated_tokens.len() <= num_input_spec_tokens,
                    "qwen3.5 MTP proposal validated tokens exceed speculative suffix"
                );
                let spec_start = flat_start + num_committed_input_tokens;
                let validated_end = spec_start + decision.validated_tokens.len();
                debug_assert!(
                    microbatch.flat_token_ids()[spec_start..validated_end]
                        .iter()
                        .copied()
                        .eq(decision.validated_tokens.iter().map(|&token| {
                            i32::try_from(token).expect("qwen3.5 validated token must fit the model i32 token domain")
                        })),
                    "qwen3.5 MTP validated tokens must be the speculative input prefix"
                );
                let pending_token_index = microbatch.token_indices()[req_index];
                let cache_state = self.speculator.mtp().hidden_state_cache.request_state(req_slot);
                let repair_tail = cache_state
                    .requires_tail_repair(pending_token_index, &microbatch.flat_token_ids()[flat_start..flat_end]);
                if !matches!(cache_state, Qwen35MTPCacheState::Decode { .. }) {
                    // Admission and prefix hits leave K known tokens for first Decode.
                    // Every hidden input then comes from this forward, not the cache.
                    debug_assert!(num_committed_input_tokens >= num_spec_tokens);
                }
                let continuation_token_ids = mtp_decode_continuation_token_ids(
                    &microbatch.flat_token_ids()[flat_start..flat_start + num_committed_input_tokens],
                    &microbatch.flat_token_ids()[spec_start..validated_end],
                );
                let num_continuation_tokens = continuation_token_ids.len();
                requests.push(Qwen35MTPRequest {
                    req_slot,
                    block_index: microbatch.block_indices()[req_index],
                    main_hidden_flat_start: flat_start,
                    pending_token_index,
                    repair_tail,
                    next_pending_token_index: pending_token_index
                        .checked_add(num_continuation_tokens as u32)
                        .and_then(|index| index.checked_add(1))
                        .expect("qwen3.5 MTP pending cache-local index must fit u32"),
                    sampler_config: microbatch.sampler_configs()[req_index],
                    kind: Qwen35MTPRequestKind::Decode {
                        continuation_token_ids,
                        sampled_token_id: decision
                            .sampled_token
                            .try_into()
                            .expect("qwen3.5 sampled token must fit the model i32 token domain"),
                        proposal_sample_position: mtp_proposal_sample_position(
                            microbatch.token_indices()[req_index],
                            (num_committed_input_tokens + decision.validated_tokens.len()) as u32,
                            0,
                        ),
                        decision_index,
                    },
                });
                decision_index += 1;
            } else {
                let mut token_ids_by_module = (1..=num_spec_tokens)
                    .map(|lane| microbatch.token_ids_for_lane(req_index, lane).to_vec())
                    .collect::<Vec<_>>();
                let num_input_rows = token_ids_by_module
                    .first()
                    .expect("qwen3.5 MTP Prefill requires logical module tokens")
                    .len();
                debug_assert!(
                    token_ids_by_module
                        .iter()
                        .all(|token_ids| token_ids.len() == num_input_rows),
                    "qwen3.5 MTP Prefill requires rectangular logical module inputs"
                );
                let pending_token_index = microbatch.token_indices()[req_index];
                let source_token_ids = microbatch.flat_token_ids()[flat_start..flat_end]
                    .iter()
                    .copied()
                    .chain(token_ids_by_module.iter().map(|tokens| *tokens.last().unwrap()))
                    .take(num_spec_tokens)
                    .collect::<Vec<_>>();
                let repair_tail = self
                    .speculator
                    .mtp()
                    .hidden_state_cache
                    .request_state(req_slot)
                    .requires_tail_repair(pending_token_index, &source_token_ids);
                if repair_tail {
                    for (module_index, token_ids) in token_ids_by_module.iter_mut().enumerate() {
                        token_ids.splice(0..0, source_token_ids[1..1 + module_index].iter().copied());
                    }
                }
                requests.push(Qwen35MTPRequest {
                    req_slot,
                    block_index: microbatch.block_indices()[req_index],
                    main_hidden_flat_start: flat_start,
                    pending_token_index,
                    repair_tail,
                    next_pending_token_index: pending_token_index
                        .checked_add(num_input_rows as u32)
                        .expect("qwen3.5 MTP Prefill pending cache-local index must fit u32"),
                    sampler_config: microbatch.sampler_configs()[req_index],
                    kind: Qwen35MTPRequestKind::Prefill { token_ids_by_module },
                });
            }
        }
        debug_assert_eq!(
            decision_index,
            decisions.len(),
            "qwen3.5 MTP decisions must match sampled requests"
        );
        requests
    }

    fn mtp_batch(&self, module_index: usize) -> Qwen35MTPBatch {
        let mtp = self.speculator.mtp();
        let requests = &mtp.execution.requests;
        let num_spec_tokens = mtp.num_spec_tokens;
        let mut flat_token_ids = Vec::new();
        let mut flat_sample_mask = Vec::new();
        let mut cu_tokens = Vec::with_capacity(requests.len() + 1);
        let mut token_indices = Vec::with_capacity(requests.len());
        let mut previous_hidden_routes = Vec::new();
        let mut hidden_state_cache_write_routes = Vec::new();
        let mut draft_distribution_indices = Vec::new();
        let mut sampler_configs = Vec::new();
        let mut sample_positions = Vec::new();
        let num_sample_rows = mtp_num_sample_rows(requests);
        let mut previous_module_flat_start = 0usize;
        let mut sample_index = 0usize;
        cu_tokens.push(0);
        for request in requests {
            let flat_start = flat_token_ids.len();
            let (num_input_rows, token_index, num_reused_tokens) = match &request.kind {
                Qwen35MTPRequestKind::Prefill { token_ids_by_module } => {
                    let token_ids = &token_ids_by_module[module_index];
                    flat_token_ids.extend_from_slice(token_ids);
                    let num_reused_tokens = if request.repair_tail { 0 } else { module_index };
                    let token_index = request
                        .pending_token_index
                        .checked_sub((module_index - num_reused_tokens) as u32)
                        .expect("qwen3.5 MTP repair requires a cached tail");
                    (token_ids.len(), token_index, num_reused_tokens)
                },
                Qwen35MTPRequestKind::Decode {
                    continuation_token_ids,
                    sampled_token_id,
                    proposal_sample_position,
                    ..
                } => {
                    let plan =
                        Qwen35MTPDecodePlan::new(num_spec_tokens, continuation_token_ids.len(), request.repair_tail)
                            .module(module_index);
                    for row_offset in 0..plan.num_input_rows() {
                        let token_id = match plan.token_source(row_offset) {
                            Qwen35MTPDecodeTokenSource::Continuation { token_offset } => {
                                continuation_token_ids[token_offset]
                            },
                            Qwen35MTPDecodeTokenSource::Sampled => *sampled_token_id,
                            Qwen35MTPDecodeTokenSource::Draft { step_index } => {
                                mtp.execution.draft_token_ids[step_index * num_sample_rows + sample_index]
                            },
                        };
                        flat_token_ids.push(token_id);
                    }
                    draft_distribution_indices.push(
                        mtp.common
                            .spec_probs
                            .draft_distribution_index(request.req_slot, module_index),
                    );
                    sampler_configs.push(request.sampler_config);
                    sample_positions.push(
                        proposal_sample_position
                            .checked_add(module_index as u32)
                            .expect("qwen3.5 MTP proposal sample position must fit u32"),
                    );
                    sample_index += 1;
                    (
                        plan.num_input_rows(),
                        plan.token_index(request.pending_token_index),
                        plan.num_reused_tokens(),
                    )
                },
            };
            token_indices.push(token_index);
            let hidden_state_cache_rows = if module_index == 0 {
                0..0
            } else {
                mtp.hidden_state_cache.row_range(request.req_slot, module_index - 1)
            };
            Qwen35MTPHiddenStateTransferPlan::new(module_index, num_input_rows, num_reused_tokens).append_routes(
                &mut previous_hidden_routes,
                request.main_hidden_flat_start,
                previous_module_flat_start,
                hidden_state_cache_rows,
            );
            flat_sample_mask.extend(std::iter::repeat_n(false, num_input_rows));
            if matches!(&request.kind, Qwen35MTPRequestKind::Decode { .. }) {
                *flat_sample_mask
                    .last_mut()
                    .expect("qwen3.5 MTP sampled request requires token") = true;
            }
            if module_index + 1 < num_spec_tokens && matches!(&request.kind, Qwen35MTPRequestKind::Decode { .. }) {
                let cache_rows = mtp.hidden_state_cache.row_range(request.req_slot, module_index);
                append_mtp_hidden_state_cache_write_routes(
                    &mut hidden_state_cache_write_routes,
                    cache_rows,
                    flat_start,
                    num_input_rows,
                );
            }
            cu_tokens.push(flat_token_ids.len() as u32);
            if module_index > 0 {
                previous_module_flat_start += mtp_num_input_rows(request, module_index - 1, num_spec_tokens);
            }
        }
        debug_assert_eq!(sample_index, num_sample_rows);
        debug_assert!(flat_token_ids.len() <= self.config.max_tokens);
        Qwen35MTPBatch {
            microbatch: Qwen35Microbatch::new(
                requests.iter().map(|request| request.req_slot).collect(),
                requests.iter().map(|request| request.block_index).collect(),
                token_indices,
                flat_token_ids,
                cu_tokens,
                vec![Vec::new(); requests.len()],
                requests.iter().map(|request| request.sampler_config).collect(),
                flat_sample_mask,
            ),
            previous_hidden_routes,
            hidden_state_cache_write_routes,
            draft_distribution_indices,
            sampler_configs,
            sample_positions,
        }
    }

    fn record_mtp_embed(
        &mut self,
        recorder: &mut Qwen35ModelOpsRecorder,
        microbatch: &Qwen35Microbatch,
        main_hidden: Rc<Buffer>,
        decisions: &[Qwen35DecodeDecision],
    ) -> Rc<Buffer> {
        let num_decode_reqs = (0..microbatch.num_reqs())
            .filter(|&req_index| microbatch.is_decode_req(req_index))
            .count();
        debug_assert_eq!(
            decisions.len(),
            num_decode_reqs,
            "qwen3.5 MTP proposal requires one decision per decode request"
        );
        let requests = self.mtp_requests(microbatch, decisions);
        self.speculator.mtp_mut().execution.begin(requests);
        let module_batch = self.mtp_batch(0);
        let num_tokens = module_batch.microbatch.total_tokens();
        debug_assert!(num_tokens <= self.config.max_tokens);
        let num_active_tokens = num_tokens as u32;
        let num_total_tokens = self
            .speculator
            .mtp()
            .body
            .component()
            .capacity(num_active_tokens);
        let num_mtp_sample_rows = module_batch.sampler_configs.len();
        self.write_token_ids(module_batch.microbatch.flat_token_ids());
        let mtp_gqa_shape = self.speculator.mtp().gqa_state.prepare_metadata(
            module_batch.microbatch.req_slots(),
            module_batch.microbatch.token_indices(),
            module_batch.microbatch.cu_tokens(),
            num_total_tokens,
        );
        self.speculator
            .mtp()
            .previous_hidden_routes
            .write_typed(0, &module_batch.previous_hidden_routes);
        if num_mtp_sample_rows > 0 {
            self.speculator
                .mtp()
                .draft_distribution_indices
                .write_typed(0, &module_batch.draft_distribution_indices);
        }
        let mtp_gqa_topology = self.speculator.mtp().gqa_state.replay_topology();
        let mtp_key =
            self.speculator
                .mtp()
                .body
                .component()
                .prepare_replay(num_active_tokens, mtp_gqa_shape, mtp_gqa_topology);
        let (mtp_embed_key, mtp_embed_arguments) = self
            .speculator
            .mtp()
            .embed
            .component()
            .prepare_replay(num_active_tokens);
        let (mtp_hidden_state_transfer_key, mtp_hidden_state_transfer_arguments) =
            self.speculator.mtp().hidden_state_transfers[0]
                .component()
                .prepare_replay(num_active_tokens, 0);
        let mtp_hidden_input = Rc::clone(&self.speculator.mtp().hidden_input);
        let mtp_embed_build_start = Instant::now();
        let mtp = self.speculator.mtp_mut();
        let hidden_state_cache = mtp.hidden_state_cache.hidden_states().unwrap_or(&main_hidden);
        let hidden_state_transfer_input = Qwen35MTPHiddenStateTransferArgs {
            num_rows: num_active_tokens,
            main_hidden_input: &main_hidden,
            previous_module_hidden_input: &main_hidden,
            hidden_state_cache_input: hidden_state_cache,
            routes: &mtp.previous_hidden_routes,
            previous_hidden_output: &mtp.previous_hidden,
            num_write_rows: 0,
            write_input: &main_hidden,
            write_routes: &mtp.hidden_state_cache_write_routes,
            hidden_state_cache_output: hidden_state_cache,
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_transfer_key, mtp_hidden_state_transfer_cache_hit) =
            mtp.hidden_state_transfers[0].record(&runtime, &hidden_state_transfer_input);
        debug_assert_eq!(
            recorded_transfer_key, mtp_hidden_state_transfer_key,
            "qwen3.5 MTP hidden-state transfer replay input must match its key"
        );
        let input = Qwen35MTPEmbedArgs {
            num_tokens: num_active_tokens,
            prev_hidden_input: &mtp.previous_hidden,
            token_ids: &self.token_ids,
            token_hidden_input: &self.token_hidden_input,
            hidden_output: &mtp_hidden_input,
        };
        let (recorded_key, mtp_embed_cache_hit) = mtp.embed.record(&runtime, &input);
        debug_assert_eq!(
            recorded_key, mtp_embed_key,
            "qwen3.5 MTPEmbed replay input must match its key"
        );
        if !mtp_hidden_state_transfer_cache_hit || !mtp_embed_cache_hit {
            recorder.mtp_build_elapsed += mtp_embed_build_start.elapsed();
        }
        mtp.execution.pending_hidden_state_cache_write_routes = module_batch.hidden_state_cache_write_routes;
        recorder.mtp_sample_req_slots = self
            .speculator
            .mtp()
            .execution
            .requests
            .iter()
            .filter_map(|request| {
                matches!(&request.kind, Qwen35MTPRequestKind::Decode { .. }).then_some(request.req_slot)
            })
            .collect();
        recorder.mtp_sample_decision_indices = self
            .speculator
            .mtp()
            .execution
            .requests
            .iter()
            .filter_map(|request| {
                if let Qwen35MTPRequestKind::Decode { decision_index, .. } = &request.kind {
                    Some(*decision_index)
                } else {
                    None
                }
            })
            .collect();
        trace::qwen35_state(|| {
            format!(
                "event=mtp_embed_record num_tokens={} num_reqs={} num_sample_rows={} transfer_cache_hit={} \
                 embed_cache_hit={} key={:?}",
                num_tokens,
                microbatch.num_reqs(),
                num_mtp_sample_rows,
                mtp_hidden_state_transfer_cache_hit,
                mtp_embed_cache_hit,
                mtp_embed_key,
            )
        });
        recorder.mtp_hidden_state_transfer_key = Some(mtp_hidden_state_transfer_key);
        recorder.mtp_hidden_state_transfer_arguments = mtp_hidden_state_transfer_arguments;
        recorder.mtp_embed_key = Some(mtp_embed_key);
        recorder.mtp_embed_arguments = mtp_embed_arguments;
        recorder.mtp_key = Some(mtp_key);
        recorder.mtp_gqa_shape = Some(mtp_gqa_shape);
        recorder.mtp_gqa_topology = Some(mtp_gqa_topology);
        recorder.mtp_microbatch = Some(module_batch.microbatch);
        recorder.mtp_sampler_configs = module_batch.sampler_configs;
        recorder.mtp_sample_positions = module_batch.sample_positions;
        recorder.mtp_embed_cache_hit = mtp_embed_cache_hit;
        mtp_hidden_input
    }

    fn record_mtp(&mut self, recorder: &mut Qwen35ModelOpsRecorder, mtp_hidden_input: Rc<Buffer>) -> Rc<Buffer> {
        debug_assert!(
            Rc::ptr_eq(&mtp_hidden_input, &self.speculator.mtp().hidden_input),
            "qwen3.5 MTP must consume the MTPEmbed hidden workspace"
        );
        let microbatch = recorder
            .mtp_microbatch
            .as_ref()
            .expect("qwen3.5 MTP recording requires the MTP microbatch");
        self.speculator.mtp().body.component().validate_batch(microbatch);
        let mtp_key = recorder
            .mtp_key
            .as_ref()
            .expect("qwen3.5 MTP recording requires its replay key");
        let num_tokens = microbatch.total_tokens();
        debug_assert!(num_tokens <= self.config.max_tokens);
        let mtp_build_start = Instant::now();
        let mtp = self.speculator.mtp_mut();
        let input = Qwen35MTPArgs {
            num_tokens: num_tokens as u32,
            hidden_input: &mtp_hidden_input,
            hidden_output: &self.hidden_output,
            gqa: mtp.gqa_state.metadata(),
            gqa_replay_topology: mtp.gqa_state.replay_topology(),
            pages: self.pages.buffer(),
        };
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let (recorded_key, mtp_cache_hit) = mtp.body.record(&runtime, &input);
        debug_assert_eq!(&recorded_key, mtp_key, "qwen3.5 MTP replay input must match its key");
        if !mtp_cache_hit {
            recorder.mtp_build_elapsed += mtp_build_start.elapsed();
        }
        trace::qwen35_state(|| {
            format!(
                "event=mtp_record num_tokens={} num_reqs={} num_sample_rows={} mtp_embed_cache_hit={} cache_hit={} \
                 key={:?}",
                num_tokens,
                microbatch.num_reqs(),
                recorder.num_mtp_sample_rows(),
                recorder.mtp_embed_cache_hit,
                mtp_cache_hit,
                mtp_key,
            )
        });
        Rc::clone(&self.hidden_output)
    }

    fn record_mtp_gather_unembed(&mut self, recorder: &mut Qwen35ModelOpsRecorder, mtp_hidden_output: &Rc<Buffer>) {
        debug_assert!(
            Rc::ptr_eq(mtp_hidden_output, &self.hidden_output),
            "qwen3.5 MTP GatherUnembed must consume the MTP hidden workspace"
        );
        if recorder.num_mtp_sample_rows() == 0 {
            return;
        }
        let (gather_unembed_key, gather_unembed_arguments) = self.prepare_gather_unembed_replay(
            recorder
                .mtp_microbatch
                .as_ref()
                .expect("qwen3.5 MTP GatherUnembed requires the MTP microbatch"),
            mtp_hidden_output,
        );
        recorder.mtp_gather_unembed_key = Some(gather_unembed_key);
        recorder.mtp_gather_unembed_arguments = gather_unembed_arguments;
    }

    fn record_mtp_sampling(&mut self, recorder: &mut Qwen35ModelOpsRecorder) {
        if recorder.num_mtp_sample_rows() == 0 {
            return;
        }
        let (mtp_sampling_key, mtp_sampling_arguments) = self.prepare_mtp_sampling_replay(
            &recorder.mtp_sampler_configs,
            &recorder.mtp_sample_req_slots,
            &recorder.mtp_sample_positions,
        );
        recorder.mtp_sampling_key = Some(mtp_sampling_key);
        recorder.mtp_sampling_arguments = mtp_sampling_arguments;
    }

    fn submit_mtp_sequence(&self, executions: &[ReplayExecution<'_>]) -> MetalReplaySubmission {
        let runtime = self.replay_runtime();
        if runtime.gpu_timestamps_enabled() {
            runtime.submit_replay_sequence_with_gpu_timestamps(executions, &[executions.len()])
        } else {
            runtime.submit_replay_sequence(executions)
        }
    }

    fn submit_mtp_step(&self, recorder: &Qwen35ModelOpsRecorder, step_index: usize) -> MetalReplaySubmission {
        let mtp_hidden_state_transfer_key = recorder
            .mtp_hidden_state_transfer_key
            .as_ref()
            .expect("qwen3.5 MTP submission requires hidden-state transfer replay");
        let mtp_embed_key = recorder
            .mtp_embed_key
            .as_ref()
            .expect("qwen3.5 MTP submission requires MTPEmbed replay");
        let mtp_key = recorder
            .mtp_key
            .as_ref()
            .expect("qwen3.5 MTP submission requires MTP replay");
        let mtp = self.speculator.mtp();
        let mtp_hidden_state_transfer_replay = mtp.hidden_state_transfers[0].replay(mtp_hidden_state_transfer_key);
        let mtp_embed_replay = mtp.embed.replay(mtp_embed_key);
        let mtp_replay = mtp.body.replay(mtp_key);
        let gqa_layer_index = step_index as u32;
        let mtp_arguments = mtp.body.component().replay_arguments(
            recorder
                .mtp_gqa_shape
                .expect("qwen3.5 MTP submission requires GQA replay arguments"),
            recorder
                .mtp_gqa_topology
                .expect("qwen3.5 MTP submission requires GQA replay topology"),
            gqa_layer_index,
        );
        if recorder.num_mtp_sample_rows() == 0 {
            return self.submit_mtp_sequence(&[
                ReplayExecution::new(
                    mtp_hidden_state_transfer_replay,
                    &recorder.mtp_hidden_state_transfer_arguments,
                ),
                ReplayExecution::new(mtp_embed_replay, &recorder.mtp_embed_arguments),
                ReplayExecution::new(mtp_replay, &mtp_arguments),
            ]);
        }
        let gather_unembed_key = recorder
            .mtp_gather_unembed_key
            .as_ref()
            .expect("qwen3.5 MTP sampled output requires GatherUnembed replay");
        let mtp_sampling_key = recorder
            .mtp_sampling_key
            .as_ref()
            .expect("qwen3.5 MTP sampled output requires Sampling replay");
        self.submit_mtp_sequence(&[
            ReplayExecution::new(
                mtp_hidden_state_transfer_replay,
                &recorder.mtp_hidden_state_transfer_arguments,
            ),
            ReplayExecution::new(mtp_embed_replay, &recorder.mtp_embed_arguments),
            ReplayExecution::new(mtp_replay, &mtp_arguments),
            ReplayExecution::new(
                self.gather_unembed.replay(gather_unembed_key),
                &recorder.mtp_gather_unembed_arguments,
            ),
            ReplayExecution::new(mtp.sampling.replay(mtp_sampling_key), &recorder.mtp_sampling_arguments),
        ])
    }

    fn record_next_mtp_step(&mut self, module_index: usize) -> Qwen35MTPRecordedStep {
        assert!(
            module_index < self.speculator.mtp().num_spec_tokens,
            "qwen3.5 MTP module index exceeds configured steps"
        );
        let module_batch = self.mtp_batch(module_index);
        let num_active_tokens = module_batch.microbatch.total_tokens() as u32;
        let num_total_tokens = self
            .speculator
            .mtp()
            .body
            .component()
            .capacity(num_active_tokens);
        self.write_token_ids(module_batch.microbatch.flat_token_ids());
        self.speculator
            .mtp()
            .previous_hidden_routes
            .write_typed(0, &module_batch.previous_hidden_routes);
        if !module_batch.draft_distribution_indices.is_empty() {
            self.speculator
                .mtp()
                .draft_distribution_indices
                .write_typed(0, &module_batch.draft_distribution_indices);
        }
        let gqa_shape = self.speculator.mtp().gqa_state.prepare_metadata(
            module_batch.microbatch.req_slots(),
            module_batch.microbatch.token_indices(),
            module_batch.microbatch.cu_tokens(),
            num_total_tokens,
        );
        let gqa_topology = self.speculator.mtp().gqa_state.replay_topology();
        let body_key =
            self.speculator
                .mtp()
                .body
                .component()
                .prepare_replay(num_active_tokens, gqa_shape, gqa_topology);
        let (embed_key, embed_arguments) = self
            .speculator
            .mtp()
            .embed
            .component()
            .prepare_replay(num_active_tokens);
        let hidden_input = Rc::clone(&self.speculator.mtp().hidden_input);
        let build_start = Instant::now();
        let runtime = MetalReplayRuntime::new(self.runtime.stream());
        let mtp = self.speculator.mtp_mut();
        debug_assert_eq!(
            mtp.execution.pending_hidden_state_cache_write_routes.len() % 2,
            0,
            "qwen3.5 MTP hidden-state cache write routes must contain row pairs"
        );
        let num_hidden_state_cache_write_rows =
            (mtp.execution.pending_hidden_state_cache_write_routes.len() / 2) as u32;
        if num_hidden_state_cache_write_rows > 0 {
            mtp.hidden_state_cache_write_routes
                .write_typed(0, &mtp.execution.pending_hidden_state_cache_write_routes);
        }
        let (hidden_state_transfer_key, hidden_state_transfer_arguments) = mtp.hidden_state_transfers[module_index]
            .component()
            .prepare_replay(num_active_tokens, num_hidden_state_cache_write_rows);
        let hidden_state_cache = mtp.hidden_state_cache.hidden_states().unwrap_or(&self.hidden_output);
        let hidden_state_transfer_input = Qwen35MTPHiddenStateTransferArgs {
            num_rows: num_active_tokens,
            main_hidden_input: &self.hidden_output,
            previous_module_hidden_input: &self.hidden_output,
            hidden_state_cache_input: hidden_state_cache,
            routes: &mtp.previous_hidden_routes,
            previous_hidden_output: &mtp.previous_hidden,
            num_write_rows: num_hidden_state_cache_write_rows,
            write_input: &self.hidden_output,
            write_routes: &mtp.hidden_state_cache_write_routes,
            hidden_state_cache_output: hidden_state_cache,
        };
        let (recorded_hidden_state_transfer_key, hidden_state_transfer_cache_hit) =
            mtp.hidden_state_transfers[module_index].record(&runtime, &hidden_state_transfer_input);
        debug_assert_eq!(
            recorded_hidden_state_transfer_key, hidden_state_transfer_key,
            "qwen3.5 MTP hidden-state transfer replay input must match its key"
        );
        let embed_input = Qwen35MTPEmbedArgs {
            num_tokens: num_active_tokens,
            prev_hidden_input: &mtp.previous_hidden,
            token_ids: &self.token_ids,
            token_hidden_input: &self.token_hidden_input,
            hidden_output: &hidden_input,
        };
        let (recorded_embed_key, embed_cache_hit) = mtp.embed.record(&runtime, &embed_input);
        debug_assert_eq!(
            recorded_embed_key, embed_key,
            "qwen3.5 MTPEmbed replay input must match its key"
        );
        let body_input = Qwen35MTPArgs {
            num_tokens: num_active_tokens,
            hidden_input: &hidden_input,
            hidden_output: &self.hidden_output,
            gqa: mtp.gqa_state.metadata(),
            gqa_replay_topology: gqa_topology,
            pages: self.pages.buffer(),
        };
        let (recorded_body_key, body_cache_hit) = mtp.body.record(&runtime, &body_input);
        debug_assert_eq!(
            recorded_body_key, body_key,
            "qwen3.5 MTP replay input must match its key"
        );
        if !hidden_state_transfer_cache_hit || !embed_cache_hit || !body_cache_hit {
            mtp.execution.build_elapsed += build_start.elapsed();
        }
        mtp.execution.pending_hidden_state_cache_write_routes = module_batch.hidden_state_cache_write_routes;
        let gather_unembed = if module_batch.sampler_configs.is_empty() {
            None
        } else {
            let hidden_output = Rc::clone(&self.hidden_output);
            Some(self.prepare_gather_unembed_replay(&module_batch.microbatch, &hidden_output))
        };
        let sample_req_slots = module_batch
            .microbatch
            .req_slots()
            .iter()
            .enumerate()
            .filter_map(|(req_index, &req_slot)| module_batch.microbatch.is_decode_req(req_index).then_some(req_slot))
            .collect::<Vec<_>>();
        let sampling = if module_batch.sampler_configs.is_empty() {
            None
        } else {
            Some(self.prepare_mtp_sampling_replay(
                &module_batch.sampler_configs,
                &sample_req_slots,
                &module_batch.sample_positions,
            ))
        };
        Qwen35MTPRecordedStep {
            hidden_state_transfer_key,
            hidden_state_transfer_arguments,
            embed_key,
            embed_arguments,
            body_key,
            gqa_shape,
            gqa_topology,
            gather_unembed,
            sampling,
            num_sample_rows: module_batch.sampler_configs.len(),
        }
    }

    fn submit_recorded_mtp_step(&self, module_index: usize, step: &Qwen35MTPRecordedStep) -> MetalReplaySubmission {
        let mtp = self.speculator.mtp();
        let body_arguments =
            mtp.body
                .component()
                .replay_arguments(step.gqa_shape, step.gqa_topology, module_index as u32);
        let mut executions = vec![
            ReplayExecution::new(
                mtp.hidden_state_transfers[module_index].replay(&step.hidden_state_transfer_key),
                &step.hidden_state_transfer_arguments,
            ),
            ReplayExecution::new(mtp.embed.replay(&step.embed_key), &step.embed_arguments),
        ];
        executions.push(ReplayExecution::new(mtp.body.replay(&step.body_key), &body_arguments));
        if let Some((gather_unembed_key, gather_unembed_arguments)) = &step.gather_unembed {
            executions.push(ReplayExecution::new(
                self.gather_unembed.replay(gather_unembed_key),
                gather_unembed_arguments,
            ));
        }
        if let Some((sampling_key, sampling_arguments)) = &step.sampling {
            executions.push(ReplayExecution::new(
                mtp.sampling.replay(sampling_key),
                sampling_arguments,
            ));
        }
        self.submit_mtp_sequence(&executions)
    }

    fn read_mtp_step(&self, num_sample_rows: usize) -> (Vec<i32>, Vec<f32>, Duration) {
        let read_start = Instant::now();
        let draft_token_ids = self.sampler_output.token_ids.read_typed::<i32>(0, num_sample_rows);
        let draft_probs = self.sampler_output.token_probs.read_typed::<f32>(0, num_sample_rows);
        (draft_token_ids, draft_probs, read_start.elapsed())
    }

    fn submit_mtp_recording(&mut self, recorder: &Qwen35ModelOpsRecorder) -> MetalReplaySubmission {
        let num_spec_tokens = self.speculator.mtp().num_spec_tokens;
        let mut submission = self.submit_mtp_step(recorder, 0);
        for module_index in 1..num_spec_tokens {
            submission.wait();
            let gpu_elapsed = submission.gpu_timestamp_durations().map(|durations| durations[0]);
            let (draft_token_ids, draft_probs, read_elapsed) = self.read_mtp_step(recorder.num_mtp_sample_rows());
            self.speculator
                .mtp_mut()
                .execution
                .push_step(&draft_token_ids, &draft_probs, read_elapsed, gpu_elapsed);
            let next_step = self.record_next_mtp_step(module_index);
            submission = self.submit_recorded_mtp_step(module_index, &next_step);
        }
        submission
    }

    fn read_mtp_proposal(
        &mut self,
        recorder: &Qwen35ModelOpsRecorder,
        decisions: &mut [Qwen35DecodeDecision],
        replay_elapsed: Duration,
        gpu_timestamp_durations: Option<&[Duration]>,
    ) -> ModelOutputTiming {
        let mut timing = ModelOutputTiming {
            spec_build_elapsed: recorder.mtp_build_elapsed,
            spec_replay_elapsed: replay_elapsed,
            spec_passes: self.speculator.mtp().num_spec_tokens,
            ..ModelOutputTiming::default()
        };
        let num_mtp_sample_rows = recorder.num_mtp_sample_rows();
        assert_eq!(
            recorder.mtp_sample_decision_indices.len(),
            num_mtp_sample_rows,
            "qwen3.5 MTP decisions must match draft sampling rows"
        );
        let (draft_token_ids, draft_probs, read_elapsed) = self.read_mtp_step(num_mtp_sample_rows);
        let mtp = self.speculator.mtp_mut();
        mtp.execution.push_step(
            &draft_token_ids,
            &draft_probs,
            read_elapsed,
            gpu_timestamp_durations.map(|durations| durations[0]),
        );
        timing.spec_replay_gpu_elapsed = mtp.execution.gpu_elapsed;
        assert_eq!(mtp.execution.completed_steps, mtp.num_spec_tokens);
        timing.spec_build_elapsed += mtp.execution.build_elapsed;
        timing.spec_read_elapsed += mtp.execution.read_elapsed;
        if num_mtp_sample_rows > 0 {
            for step_index in 0..mtp.num_spec_tokens {
                for sample_index in 0..num_mtp_sample_rows {
                    let flat_index = step_index * num_mtp_sample_rows + sample_index;
                    let draft_token = mtp.execution.draft_token_ids[flat_index]
                        .try_into()
                        .expect("qwen3.5 sampler returned a negative draft token ID");
                    mtp.common.spec_probs.set_expected_draft_token(
                        recorder.mtp_sample_req_slots[sample_index],
                        step_index,
                        draft_token,
                    );
                    let decision = &mut decisions[recorder.mtp_sample_decision_indices[sample_index]];
                    decision.spec_tokens.push(draft_token);
                    decision.spec_probs.push(mtp.execution.draft_probs[flat_index]);
                    decision.spec_confidences.push(1.0);
                }
            }
        }
        // Keep the previous wave's metadata until every hidden read and write completes.
        for request in &mtp.execution.requests {
            let token_index = request.next_pending_token_index;
            let state = match &request.kind {
                Qwen35MTPRequestKind::Prefill { token_ids_by_module } => {
                    Qwen35MTPCacheState::Prefill {
                        token_index,
                        token_ids: token_ids_by_module
                            .iter()
                            .map(|tokens| *tokens.last().unwrap())
                            .collect(),
                    }
                },
                Qwen35MTPRequestKind::Decode {
                    sampled_token_id,
                    decision_index,
                    ..
                } => {
                    // The last draft is sampled but has no cached KV slot yet.
                    let token_ids = std::iter::once(*sampled_token_id)
                        .chain(
                            decisions[*decision_index]
                                .spec_tokens
                                .iter()
                                .take(mtp.num_spec_tokens - 1)
                                .map(|&token| token as i32),
                        )
                        .collect();
                    Qwen35MTPCacheState::Decode { token_index, token_ids }
                },
            };
            mtp.hidden_state_cache.set_request_state(request.req_slot, state);
        }
        trace_decisions("mtp_propose_done", decisions);
        timing
    }
}

fn mtp_num_input_rows(request: &Qwen35MTPRequest, module_index: usize, num_spec_tokens: usize) -> usize {
    match &request.kind {
        Qwen35MTPRequestKind::Prefill { token_ids_by_module } => token_ids_by_module[module_index].len(),
        Qwen35MTPRequestKind::Decode {
            continuation_token_ids, ..
        } => {
            Qwen35MTPDecodePlan::new(num_spec_tokens, continuation_token_ids.len(), request.repair_tail)
                .module(module_index)
                .num_input_rows()
        },
    }
}

fn mtp_num_sample_rows(requests: &[Qwen35MTPRequest]) -> usize {
    requests
        .iter()
        .filter(|request| matches!(&request.kind, Qwen35MTPRequestKind::Decode { .. }))
        .count()
}

fn mtp_decode_continuation_token_ids(main_input_token_ids: &[i32], validated_token_ids: &[i32]) -> Vec<i32> {
    debug_assert!(!main_input_token_ids.is_empty());
    let mut token_ids = Vec::with_capacity(main_input_token_ids.len() - 1 + validated_token_ids.len());
    token_ids.extend_from_slice(&main_input_token_ids[1..]);
    token_ids.extend_from_slice(validated_token_ids);
    token_ids
}

#[cfg(test)]
mod mtp_tests {
    use super::mtp_decode_continuation_token_ids;

    #[test]
    fn test_decode_continuation_drops_the_anchor_and_appends_validated_tokens() {
        assert_eq!(mtp_decode_continuation_token_ids(&[10], &[20, 21]), vec![20, 21]);
        assert_eq!(
            mtp_decode_continuation_token_ids(&[10, 11, 12], &[20, 21]),
            vec![11, 12, 20, 21]
        );
    }
}
