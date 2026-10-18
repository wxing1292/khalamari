use crate::compute::DeviceRequest;
use crate::compute::DeviceResponse;
use crate::compute::QueryTokens;
use crate::runtime::RawRequestID;
use crate::runtime::decoder::trie_cache::DecoderBlocks;
use crate::runtime::decoder::trie_cache::InitBlockOnceResult;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::request::CompletionReason;
use crate::runtime::request::InternalRequest;
use crate::runtime::request::internal_request::commit_output::prepare_commit_output;
use crate::runtime::scheduler::CancelResult;
use crate::runtime::scheduler::CommitResult;
use crate::runtime::scheduler::ComputePhase;
use crate::runtime::scheduler::PrepareResult;
use crate::runtime::scheduler::ReqTokenInventory;
use crate::runtime::scheduler::UserRequest;
use crate::runtime::tasks::AsyncTaskResp;
use crate::runtime::tasks::ResourceMaterializationReq;
use crate::runtime::tasks::ResourceMaterializationResp;

impl<const N: usize, const P: usize, const L: usize, DBC> UserRequest<DeviceRequest, DeviceResponse>
    for InternalRequest<N, P, L, DBC>
where
    DBC: MultiLaneBlockCache<P, L>,
{
    fn id(&self) -> RawRequestID {
        self.req_id
    }

    fn store_swapped(&self) -> bool {
        InternalRequest::store_swapped(self)
    }

    fn is_terminal(&self) -> bool {
        self.status().is_terminal()
    }

    fn num_in_flight_computes(&self) -> usize {
        self.in_flight_computes.len()
    }

    fn num_in_flight_blocking_async_tasks(&self) -> usize {
        self.num_in_flight_blocking_async_tasks
    }

    fn num_in_flight_nonblocking_async_tasks(&self) -> usize {
        self.num_in_flight_nonblocking_async_tasks
    }

    fn token_estimate(&self) -> ReqTokenInventory<'_> {
        ReqTokenInventory::new::<L>(
            self.req_id,
            self.decoder_blocks.num_ready_tokens(),
            self.decoder_blocks.num_queued_tokens(),
            self.decoder_blocks.num_spec_tokens(),
            self.decoder_blocks.spec_confidences(),
        )
    }

    fn prepare(&mut self, token_budget: usize) -> PrepareResult<DeviceRequest> {
        if self.status().is_terminal() {
            return if self.in_flight_computes.is_empty()
                && self.num_in_flight_blocking_async_tasks == 0
                && self.num_in_flight_nonblocking_async_tasks == 0
            {
                PrepareResult::Terminal
            } else {
                PrepareResult::Skip
            };
        }
        if self.num_in_flight_blocking_async_tasks != 0 {
            return PrepareResult::Skip;
        }
        if matches!(self.in_flight_computes.back(), Some(ComputePhase::Decode { .. })) {
            return PrepareResult::Skip;
        }
        debug_assert_eq!(
            self.token_estimate().token_consumption(token_budget),
            token_budget,
            "request prepare requires a valid token consumption"
        );

        let mut ready_token_slots = self.decoder_blocks.ready_token_slots();
        while ready_token_slots < token_budget {
            ready_token_slots = match self.decoder_blocks.init_block_once() {
                InitBlockOnceResult::ResourceLimitExceeded => {
                    return if self.in_flight_computes.is_empty() {
                        PrepareResult::ResourceLimitExceeded
                    } else {
                        PrepareResult::Skip
                    };
                },
                InitBlockOnceResult::Await { wait } => {
                    return if self.in_flight_computes.is_empty() {
                        PrepareResult::Await { wait }
                    } else {
                        drop(wait);
                        PrepareResult::Skip
                    };
                },
                InitBlockOnceResult::ResourceNotFound { resource_ids, .. } => {
                    self.num_in_flight_blocking_async_tasks += 1;
                    let req = Box::new(ResourceMaterializationReq::new(
                        self.req_id,
                        resource_ids,
                        self.resource_processors.clone(),
                    ));
                    return PrepareResult::BlockingAsyncTask { req };
                },
                InitBlockOnceResult::Success { ready_token_slots } => ready_token_slots,
            }
        }

        match self.decoder_blocks.prepare(token_budget) {
            Some(decoder_query_tokens) => {
                debug_assert!(
                    decoder_query_tokens.token_index() + decoder_query_tokens.token_consumption()
                        <= self.context_window,
                    "request model input exceeds its context window"
                );
                let compute_phase = compute_phase(&decoder_query_tokens);
                self.in_flight_computes.push_back(compute_phase);
                let decoder_sync_blocks = self.decoder_blocks.prepare_blocks();
                let resource_placements = self.decoder_blocks.device_resource_placements(&decoder_query_tokens);
                let token_positions = self
                    .token_positions
                    .as_ref()
                    .map(|positions| positions.query(&decoder_query_tokens));
                let dev_req = DeviceRequest::new(
                    self.req_id(),
                    self.req_slot(),
                    decoder_query_tokens,
                    decoder_sync_blocks,
                    token_positions,
                    resource_placements,
                    self.sampling_config().clone(),
                );
                PrepareResult::Continue { dev_req, compute_phase }
            },
            None => PrepareResult::Skip,
        }
    }

    fn handle_async_task_resp(&mut self, resp: Box<dyn AsyncTaskResp>) {
        // TODO: Replace this single-response-type downcast before the runtime adds another async task response type.
        // Resource materialization is currently the only async task response.
        let resp = resp
            .into_any()
            .downcast::<ResourceMaterializationResp>()
            .unwrap_or_else(|_| panic!("internal request received an unsupported async task response"));
        resp.update(self);
    }

    fn cancel(&mut self, dev_req: DeviceRequest) -> CancelResult {
        let DeviceRequest {
            req_id,
            req_slot,
            decoder_query_tokens,
            decoder_sync_blocks,
            ..
        } = dev_req;
        assert_eq!(self.req_id(), req_id, "cancel response request ID mismatch");
        assert_eq!(self.req_slot(), req_slot, "cancel response request slot mismatch");
        let cancelled_compute = self.in_flight_computes.pop_back();
        debug_assert_eq!(
            cancelled_compute,
            Some(compute_phase(&decoder_query_tokens)),
            "request cancellation must retire the latest compute"
        );

        self.decoder_blocks.cancel_blocks(decoder_sync_blocks);
        self.decoder_blocks.cancel(decoder_query_tokens);

        if self.status().is_terminal() {
            if self.in_flight_computes.is_empty()
                && self.num_in_flight_blocking_async_tasks == 0
                && self.num_in_flight_nonblocking_async_tasks == 0
            {
                CancelResult::Terminal
            } else {
                CancelResult::Pending
            }
        } else if self.num_in_flight_blocking_async_tasks != 0 {
            CancelResult::Pending
        } else {
            debug_assert!(
                !matches!(self.in_flight_computes.back(), Some(ComputePhase::Decode { .. })),
                "cancelling the latest compute must not leave an in-flight Decode"
            );
            CancelResult::Continue
        }
    }

    fn commit(&mut self, dev_resp: DeviceResponse) -> CommitResult {
        let DeviceResponse {
            req_id,
            query_tokens,
            sampled_tokens,
        } = dev_resp;
        assert_eq!(self.req_id, req_id, "device response request ID mismatch");
        let committed_compute = self.in_flight_computes.pop_front();
        debug_assert_eq!(
            committed_compute,
            Some(compute_phase(&query_tokens)),
            "request commit must retire the earliest compute"
        );
        debug_assert!(self.decoder_blocks.num_sampled_tokens() < self.sampling_config().max_sampled_tokens);
        debug_assert!(self.decoder_blocks.num_total_tokens() < self.context_window);
        let remaining_sampled_tokens =
            self.sampling_config().max_sampled_tokens - self.decoder_blocks.num_sampled_tokens();
        let remaining_context_tokens = self.context_window - self.decoder_blocks.num_total_tokens();
        let (token_probs, completion, num_spec_tokens) = prepare_commit_output(
            self.sampling_config.stop_sequences.as_slice(),
            remaining_sampled_tokens,
            remaining_context_tokens,
            self.decoder_blocks.sampled_tokens_rev(),
            &sampled_tokens,
        );
        // Cache metadata needs the complete physical proposal, even when stop or
        // length limits permit fewer speculative tokens in the next query.
        self.decoder_blocks.commit(query_tokens, sampled_tokens);
        self.decoder_blocks.truncate_spec_tokens(num_spec_tokens);
        if !token_probs.is_empty() {
            self.send_token_probs(token_probs);
        }
        match completion {
            Some(CompletionReason::ContextLimit) => {
                self.store_completed(CompletionReason::ContextLimit);
            },
            Some(reason @ (CompletionReason::StopSequence | CompletionReason::LengthLimit)) => {
                self.finish_turn();
                return CommitResult::TurnCompleted(reason);
            },
            None => {},
        }

        if self.status().is_terminal() {
            if self.in_flight_computes.is_empty()
                && self.num_in_flight_blocking_async_tasks == 0
                && self.num_in_flight_nonblocking_async_tasks == 0
            {
                CommitResult::Terminal
            } else {
                CommitResult::Pending
            }
        } else if self.num_in_flight_blocking_async_tasks != 0
            || matches!(self.in_flight_computes.back(), Some(ComputePhase::Decode { .. }))
        {
            CommitResult::Pending
        } else {
            CommitResult::Continue
        }
    }
}

impl ResourceMaterializationResp {
    fn update<const N: usize, const P: usize, const L: usize, DBC>(self, request: &mut InternalRequest<N, P, L, DBC>)
    where
        DBC: MultiLaneBlockCache<P, L>,
    {
        assert!(
            request.num_in_flight_blocking_async_tasks != 0,
            "resource materialization response requires an in-flight async task"
        );
        request.num_in_flight_blocking_async_tasks -= 1;
        let concrete_resources = match self.into_result() {
            Ok(resources) => resources,
            Err(error) => {
                tracing::error!(error = %error, request_id = request.req_id, "resource materialization failed");
                request.store_aborted();
                return;
            },
        };
        for resource in concrete_resources {
            request.decoder_blocks.resource_symbolic_to_concrete(resource);
        }
    }
}

fn compute_phase(query_tokens: &QueryTokens) -> ComputePhase {
    match query_tokens {
        QueryTokens::Prefill { epoch, token_index, .. } => {
            ComputePhase::Prefill {
                epoch: *epoch,
                token_index: *token_index,
            }
        },
        QueryTokens::Decode { epoch, token_index, .. } => {
            ComputePhase::Decode {
                epoch: *epoch,
                token_index: *token_index,
            }
        },
    }
}
