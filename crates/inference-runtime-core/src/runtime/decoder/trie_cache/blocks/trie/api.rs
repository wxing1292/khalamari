use std::cmp::min;
use std::sync::Arc;

use inference_runtime_macro::sanity_check;
use smallvec::SmallVec;

use crate::compute::QueryTokens;
use crate::compute::SampledTokens;
use crate::runtime::Token;
use crate::runtime::decoder::BlockAnnotation;
use crate::runtime::decoder::ResourceSegment;
use crate::runtime::decoder::trie_cache::AllocateMultiLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::BlockMetadata;
use crate::runtime::decoder::trie_cache::CommitMultiLaneMutableBlockResult;
use crate::runtime::decoder::trie_cache::CommitMultiLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::DecoderBlock;
use crate::runtime::decoder::trie_cache::DecoderBlocks;
use crate::runtime::decoder::trie_cache::InitBlockOnceResult;
use crate::runtime::decoder::trie_cache::MultiLaneBlockCache;
use crate::runtime::decoder::trie_cache::ReserveMultiLaneSemiImmutableBlockResult;
use crate::runtime::decoder::trie_cache::TrieDecoderBlocks;
use crate::runtime::decoder::trie_cache::TrieNodeKey;
use crate::runtime::decoder::trie_cache::UninitBlockOnceResult;
use crate::runtime::decoder::trie_cache::blocks::cache_tokens;
use crate::runtime::decoder::trie_cache::blocks::pop_front_queued_tokens;
use crate::runtime::decoder::trie_cache::blocks::push_front_queued_tokens;
use crate::runtime::decoder::trie_cache::blocks::push_tokens;
use crate::runtime::decoder::trie_cache::blocks::schedule_tokens;
use crate::runtime::decoder::trie_cache::blocks::unschedule_tokens;
use crate::runtime::resource::ResourceID;

impl<const N: usize, const P: usize, const L: usize, BC> DecoderBlocks for TrieDecoderBlocks<N, P, L, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    fn ready_token_slots(&self) -> usize {
        self.semi_immutable_blocks
            .iter()
            .map(|block_vec| block_vec[0].ready_token_slots())
            .sum::<usize>()
            + self
                .mutable_blocks
                .iter()
                .map(|block_vec| block_vec[0].ready_token_slots())
                .sum::<usize>()
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn init_block_once(&mut self) -> InitBlockOnceResult {
        let num_cachable_tokens = self.num_queued_tokens().saturating_sub(L - 1);
        debug_assert!(
            num_cachable_tokens == 0
                || self
                    .mutable_blocks
                    .iter()
                    .all(|block_vec| N == block_vec[0].total_tokens().len())
        );

        let block_index = self.immutable_blocks.len() + self.semi_immutable_blocks.len() + self.mutable_blocks.len();
        let annotation_vec = self.block_annotation_vec(block_index);
        let resource_ids = annotation_resource_ids(&annotation_vec);
        let missing_resource_ids = resource_ids
            .iter()
            .copied()
            .filter(|resource_id| {
                self.resources
                    .get(resource_id)
                    .expect("cache block annotation must reference a resource owned by its decoder blocks")
                    .is_symbolic()
            })
            .collect::<Vec<_>>();

        let num_tokens = min(num_cachable_tokens, N);
        let tokens = if num_tokens == 0 {
            vec![]
        } else {
            let tokens = pop_front_queued_tokens::<L>(&mut self.queued_tokens, num_tokens);
            debug_assert_eq!(num_tokens + L - 1, tokens.len());
            tokens
        };
        let use_cache = num_tokens == N
            && !self.queued_tokens.is_empty()
            && self.mutable_blocks.is_empty()
            && self.semi_immutable_blocks.is_empty();
        if !use_cache {
            if !missing_resource_ids.is_empty() {
                push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                return InitBlockOnceResult::ResourceNotFound {
                    ready_token_slots: self.ready_token_slots(),
                    resource_ids: missing_resource_ids,
                };
            }
            match self.block_cache.alloc_mutable_block::<N>() {
                AllocateMultiLaneMutableBlockResult::Mutable { mut block_vec } => {
                    for (block, annotations) in block_vec.iter_mut().zip(annotation_vec) {
                        block.insert_annotations(annotations);
                    }
                    push_tokens::<N, L>(&mut block_vec, &tokens);
                    self.mutable_blocks.push_back(block_vec);
                    InitBlockOnceResult::Success {
                        ready_token_slots: self.ready_token_slots(),
                    }
                },
                AllocateMultiLaneMutableBlockResult::ResourceLimitExceeded => {
                    push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                    InitBlockOnceResult::ResourceLimitExceeded
                },
            }
        } else {
            debug_assert_eq!(N, num_tokens);
            debug_assert!(self.mutable_blocks.is_empty());
            debug_assert!(self.semi_immutable_blocks.is_empty());

            let parent_trie_node_key_vec = self.parent_trie_node_key_vec(block_index);
            let block_metadata_vec: [BlockMetadata<N>; L] = parent_trie_node_key_vec
                .into_iter()
                .zip(annotation_vec)
                .zip(tokens.windows(num_tokens))
                .map(|((parent_trie_node_key, annotations), tokens)| {
                    BlockMetadata::new(parent_trie_node_key, annotations, tokens.to_vec().into())
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            match self.block_cache.reserve_semi_immutable_block(block_metadata_vec) {
                ReserveMultiLaneSemiImmutableBlockResult::Mutable { mut block_vec } => {
                    if !missing_resource_ids.is_empty() {
                        push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                        self.block_cache.free_mutable_block(block_vec);
                        return InitBlockOnceResult::ResourceNotFound {
                            ready_token_slots: self.ready_token_slots(),
                            resource_ids: missing_resource_ids,
                        };
                    }
                    push_tokens::<N, L>(&mut block_vec, &tokens);
                    self.mutable_blocks.push_back(block_vec);
                    InitBlockOnceResult::Success {
                        ready_token_slots: self.ready_token_slots(),
                    }
                },
                ReserveMultiLaneSemiImmutableBlockResult::SemiImmutable { block_vec } => {
                    if !missing_resource_ids.is_empty() {
                        push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                        self.block_cache.free_semi_immutable_block(block_vec);
                        return InitBlockOnceResult::ResourceNotFound {
                            ready_token_slots: self.ready_token_slots(),
                            resource_ids: missing_resource_ids,
                        };
                    }
                    self.semi_immutable_blocks.push_back(block_vec);
                    InitBlockOnceResult::Success {
                        ready_token_slots: self.ready_token_slots(),
                    }
                },
                ReserveMultiLaneSemiImmutableBlockResult::Immutable { block_vec } => {
                    self.immutable_blocks.push(block_vec);
                    InitBlockOnceResult::Success {
                        ready_token_slots: self.ready_token_slots(),
                    }
                },
                ReserveMultiLaneSemiImmutableBlockResult::Wait { wait, .. } => {
                    push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                    InitBlockOnceResult::Await { wait }
                },
                ReserveMultiLaneSemiImmutableBlockResult::ResourceLimitExceeded { .. } => {
                    push_front_queued_tokens::<L>(&mut self.queued_tokens, tokens.into_iter());
                    InitBlockOnceResult::ResourceLimitExceeded
                },
            }
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn uninit_block_once(&mut self) -> UninitBlockOnceResult {
        if let Some(block_vec) = self.mutable_blocks.pop_back() {
            push_front_queued_tokens::<1>(&mut self.queued_tokens, block_vec[0].total_tokens().iter().copied());
            self.block_cache.free_mutable_block(block_vec);
        } else if let Some(block_vec) = self.semi_immutable_blocks.pop_back() {
            push_front_queued_tokens::<1>(&mut self.queued_tokens, block_vec[0].total_tokens().iter().copied());
            self.block_cache.free_semi_immutable_block(block_vec);
        } else if let Some(block_vec) = self.immutable_blocks.pop() {
            push_front_queued_tokens::<1>(&mut self.queued_tokens, block_vec[0].total_tokens().iter().copied());
        }

        UninitBlockOnceResult::Success {
            cached_token_slots: self.num_cached_tokens(),
        }
    }

    // Lane 0 is Main; lane m + 1 is MTP module m. Each lane starts at local index 0.
    // Prefill forwards `window` rows per lane with L - 1 known lookahead tokens.
    // Decode forwards only T unprocessed Main tokens and D submitted drafts.
    // Accepting A drafts advances the cached rectangle from P to P + T + A.
    // The executor owns the non-rectangular MTP work after rejection sampling.
    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn prepare(&mut self, token_budget: usize) -> Option<QueryTokens> {
        debug_assert!(0 < token_budget);
        // caller is expected to init enough ready token slots via init_block_once
        debug_assert!(token_budget <= self.ready_token_slots());

        let num_ready_tokens = self.num_ready_tokens();
        let num_queued_tokens = self.num_queued_tokens();
        let num_validated_tokens = num_ready_tokens + num_queued_tokens;
        debug_assert!(num_validated_tokens != 0);

        let token_index = self.num_cached_tokens() + self.num_scheduled_tokens();
        let token_window = min(token_budget, num_validated_tokens);
        debug_assert!(token_window == num_validated_tokens || token_window + L - 1 <= num_validated_tokens);
        let mut tokens = self
            .ready_tokens()
            .chain(self.queued_tokens())
            .take(token_window + L - 1)
            .collect::<Vec<_>>();
        // Each lane gets the same Main window. Unknown lookahead stays mutable
        // until commit supplies the token identities from the executor result.
        tokens.resize(token_window + L - 1, Token::default());
        self.write_tokens(token_index, token_index + token_window, &tokens);
        self.queued_tokens
            .drain(..token_window.saturating_sub(num_ready_tokens));

        // Query-local token offsets. Lookahead does not advance scheduled progress.
        let mut index_start = 0;
        let mut index_end = 0;
        for block_vec in &mut self.semi_immutable_blocks {
            index_end = index_start + min(token_window - index_start, block_vec[0].ready_tokens().len());
            schedule_tokens(block_vec, index_end - index_start);
            index_start = index_end;
        }
        for block_vec in &mut self.mutable_blocks {
            index_end = index_start + min(token_window - index_start, block_vec[0].ready_tokens().len());
            schedule_tokens(block_vec, index_end - index_start);
            index_start = index_end;
        }
        debug_assert_eq!(token_window, index_start);
        debug_assert_eq!(token_window, index_end);

        if token_budget < num_validated_tokens {
            Some(QueryTokens::Prefill {
                epoch: self.epoch,
                token_index,
                tokens,
                window: token_window,
            })
        } else {
            debug_assert!(self.queued_tokens.is_empty());
            tokens.truncate(token_window);
            let num_selected_spec_tokens = token_budget - token_window;
            debug_assert!(num_selected_spec_tokens <= self.spec_tokens.len());
            self.truncate_spec_tokens(num_selected_spec_tokens);
            let spec_tokens = self.spec_tokens.clone();

            Some(QueryTokens::Decode {
                epoch: self.epoch,
                token_index,
                tokens,
                spec_tokens,
            })
        }
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn cancel(&mut self, query_tokens: QueryTokens) {
        // Cancel the Main rows scheduled by prepare, not lookahead or submitted drafts.
        let num_scheduled_tokens = match query_tokens {
            QueryTokens::Prefill {
                epoch,
                token_index,
                tokens,
                window,
            } => {
                debug_assert_eq!(self.epoch, epoch);
                debug_assert!(token_index < self.num_total_tokens());
                debug_assert!(!tokens.is_empty());
                debug_assert!(1 <= window);
                debug_assert_eq!(window, tokens.len() - (L - 1));

                window
            },
            QueryTokens::Decode {
                epoch,
                token_index,
                tokens,
                spec_tokens,
            } => {
                debug_assert_eq!(self.epoch, epoch);
                debug_assert!(token_index < self.num_total_tokens());
                debug_assert!(!tokens.is_empty());
                debug_assert_eq!(self.spec_tokens, spec_tokens);

                tokens.len()
            },
        };

        // Query-local token offsets, traversed from the last scheduled row.
        let mut index_end = num_scheduled_tokens;
        let mut index_start = index_end;
        'unschedule_loop: {
            for block_vec in self.mutable_blocks.iter_mut().rev() {
                index_start = index_end - min(index_end, block_vec[0].scheduled_tokens().len());
                unschedule_tokens(block_vec, index_end - index_start);
                index_end = index_start;
                if index_end == 0 {
                    break 'unschedule_loop;
                }
            }

            for block_vec in self.semi_immutable_blocks.iter_mut().rev() {
                index_start = index_end - min(index_end, block_vec[0].scheduled_tokens().len());
                unschedule_tokens(block_vec, index_end - index_start);
                index_end = index_start;
                if index_end == 0 {
                    break 'unschedule_loop;
                }
            }
        }

        debug_assert_eq!(0, index_start);
        debug_assert_eq!(0, index_end);
    }

    #[sanity_check(sanity_check_fn = "self.sanity_check()")]
    fn commit(&mut self, query_tokens: QueryTokens, sampled_tokens: SampledTokens) {
        debug_assert_eq!(query_tokens.token_index(), self.num_cached_tokens());

        // Commit rewrites CPU token metadata only; the executor already produced KV/state.
        // Allocation and publication operate on a complete block column, never one lane.
        // C = cached KV (including replaceable drafts); P = pending, not yet executed.
        // The illustrative block size is N = 2. K = 3 can cross more than one block.
        // Before verification, P = 3:
        // +-------+----+----+----+----+
        // | Index | 0  | 1  | 2  | 3  |
        // | Block | 0  | 0  | 1  | 1  |
        // | Cache | C  | C  | C  | P  |
        // | Main  | t0 | t1 | t2 | w  |
        // | MTP0  | t1 | t2 | w  | x1 |
        // | MTP1  | t2 | w  | x1 | x2 |
        // | MTP2  | w  | x1 | x2 | x3 |
        // +-------+----+----+----+----+
        //
        // Reject all; sample y and propose z1/z2/z3. P' = 4.
        // +-------+----+----+----+----+----+
        // | Index | 0  | 1  | 2  | 3  | 4  |
        // | Block | 0  | 0  | 1  | 1  | 2  |
        // | Cache | C  | C  | C  | C  | P  |
        // | Main  | t0 | t1 | t2 | w  | y  |
        // | MTP0  | t1 | t2 | w  | y  | z1 |
        // | MTP1  | t2 | w  | y  | z1 | z2 |
        // | MTP2  | w  | y  | z1 | z2 | z3 |
        // +-------+----+----+----+----+----+
        //
        // Accept x1; reject x2/x3; sample y and propose z1/z2/z3. P' = 5.
        // +-------+----+----+----+----+----+----+
        // | Index | 0  | 1  | 2  | 3  | 4  | 5  |
        // | Block | 0  | 0  | 1  | 1  | 2  | 2  |
        // | Cache | C  | C  | C  | C  | C  | P  |
        // | Main  | t0 | t1 | t2 | w  | x1 | y  |
        // | MTP0  | t1 | t2 | w  | x1 | y  | z1 |
        // | MTP1  | t2 | w  | x1 | y  | z1 | z2 |
        // | MTP2  | w  | x1 | y  | z1 | z2 | z3 |
        // +-------+----+----+----+----+----+----+
        //
        // Accept x1/x2/x3; sample y and propose z1/z2/z3. P' = 7.
        // +-------+----+----+----+----+----+----+----+----+
        // | Index | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
        // | Block | 0  | 0  | 1  | 1  | 2  | 2  | 3  | 3  |
        // | Cache | C  | C  | C  | C  | C  | C  | C  | P  |
        // | Main  | t0 | t1 | t2 | w  | x1 | x2 | x3 | y  |
        // | MTP0  | t1 | t2 | w  | x1 | x2 | x3 | y  | z1 |
        // | MTP1  | t2 | w  | x1 | x2 | x3 | y  | z1 | z2 |
        // | MTP2  | w  | x1 | x2 | x3 | y  | z1 | z2 | z3 |
        // +-------+----+----+----+----+----+----+----+----+
        //
        // Apply each lane's shifted source slice to [P.saturating_sub(K), P').
        // Fixed blocks must match; mutable blocks copy the supplied token IDs.
        // Split at every block boundary. The pending column is not written here.
        // Then advance cached progress. Publish a full column ending at E only when
        // E <= P' and E + K <= canonical token count (including y).
        // For N=2 above: reject all publishes [0,2); accept x1 also [0,2);
        // accept all publishes [0,2) and [2,4). Later columns remain mutable.
        // Immutable and semi-immutable identities do not change.
        // The source includes shifted lookahead; token_window counts Main rows only.
        let (input_token_index, cache_source_tokens, token_window) = match (query_tokens, sampled_tokens) {
            (
                QueryTokens::Prefill {
                    epoch: input_epoch,
                    token_index: input_token_index,
                    tokens: input_tokens,
                    window: input_token_window,
                },
                SampledTokens::Prefill { epoch: output_epoch },
            ) => {
                debug_assert_eq!(self.epoch, input_epoch);
                debug_assert_eq!(self.epoch, output_epoch);
                debug_assert!(input_token_index < self.num_total_tokens());
                debug_assert!(!input_tokens.is_empty());

                (input_token_index, input_tokens, input_token_window)
            },
            (
                QueryTokens::Decode {
                    epoch: input_epoch,
                    token_index: input_token_index,
                    tokens: input_tokens,
                    spec_tokens: input_spec_tokens,
                },
                SampledTokens::Decode {
                    epoch: output_epoch,
                    validated_tokens: output_validated_tokens,
                    validated_probs: output_validated_probs,
                    sampled_token: output_sampled_token,
                    spec_tokens: output_spec_tokens,
                    spec_probs: output_spec_probs,
                    spec_confidences: output_spec_confidences,
                    ..
                },
            ) => {
                debug_assert_eq!(self.epoch, input_epoch);
                debug_assert_eq!(self.epoch, output_epoch);
                debug_assert!(input_token_index < self.num_total_tokens());
                debug_assert!(!input_tokens.is_empty());
                debug_assert!(
                    input_spec_tokens.starts_with(&output_validated_tokens),
                    "validated tokens must equal a prefix of the speculative input suffix"
                );
                debug_assert_eq!(output_validated_tokens.len(), output_validated_probs.len());
                debug_assert_eq!(output_spec_tokens.len(), output_spec_probs.len());
                debug_assert_eq!(output_spec_tokens.len(), output_spec_confidences.len());
                debug_assert!(self.queued_tokens.is_empty());

                let token_window = input_tokens.len() + output_validated_tokens.len();
                let mut cache_source_tokens = input_tokens;
                cache_source_tokens.extend(output_validated_tokens);
                cache_source_tokens.push(output_sampled_token);
                cache_source_tokens.extend_from_slice(&output_spec_tokens);
                self.queued_tokens.push_back(output_sampled_token);
                self.spec_tokens = output_spec_tokens;
                self.spec_probs = output_spec_probs;
                self.spec_confidences = output_spec_confidences;
                (input_token_index, cache_source_tokens, token_window)
            },
            _ => unreachable!(),
        };

        let token_index_start = input_token_index.saturating_sub(L - 1);
        let mut tokens = self
            .cached_tokens()
            .rev()
            .take(input_token_index - token_index_start)
            .collect::<Vec<_>>();
        tokens.reverse();
        tokens.extend_from_slice(&cache_source_tokens);
        // Reconcile token identities before advancing cached progress.
        self.write_tokens(token_index_start, input_token_index + token_window, &tokens);

        // Query-local token offsets. The lookback is already cached and does not advance progress.
        let mut index_start = 0;
        let mut index_end = 0;
        for block_vec in &mut self.semi_immutable_blocks {
            index_end = index_start + min(token_window - index_start, block_vec[0].scheduled_tokens().len());
            cache_tokens(block_vec, &cache_source_tokens, index_start, index_end);
            index_start = index_end;
        }
        for block_vec in &mut self.mutable_blocks {
            let num_tokens = min(token_window - index_start, N - block_vec[0].cached_tokens().len());
            index_end = index_start + num_tokens;
            // Accepted drafts were not part of the scheduled Main token metadata.
            let num_scheduled_tokens = block_vec[0].scheduled_tokens().len();
            schedule_tokens(block_vec, num_tokens.saturating_sub(num_scheduled_tokens));
            cache_tokens(block_vec, &cache_source_tokens, index_start, index_end);
            index_start = index_end;
        }
        debug_assert_eq!(token_window, index_start);
        debug_assert_eq!(token_window, index_end);

        let publish_token_index_end = self.num_total_tokens().saturating_sub(L - 1);
        'publish_loop: {
            while let Some(block_vec) = self.semi_immutable_blocks.front() {
                if block_vec[0].cached_tokens().len() != N
                    || (self.immutable_blocks.len() + 1) * N > publish_token_index_end
                {
                    break 'publish_loop;
                }
                let block_vec = self.semi_immutable_blocks.pop_front().unwrap();
                match self.block_cache.commit_semi_immutable_block(block_vec) {
                    CommitMultiLaneSemiImmutableBlockResult::Immutable { block_vec } => {
                        self.immutable_blocks.push(block_vec);
                    },
                    CommitMultiLaneSemiImmutableBlockResult::ImmutableCollision { block_vec } => {
                        self.immutable_blocks.push(block_vec);
                        self.num_in_sync_blocks = 0;
                    },
                }
            }
            while let Some(block_vec) = self.mutable_blocks.front() {
                if block_vec[0].cached_tokens().len() != N
                    || (self.immutable_blocks.len() + 1) * N > publish_token_index_end
                {
                    break 'publish_loop;
                }
                let parent_trie_node_key_vec = self.parent_trie_node_key_vec(self.immutable_blocks.len());
                let block_vec = self.mutable_blocks.pop_front().unwrap();
                match self
                    .block_cache
                    .commit_mutable_block(parent_trie_node_key_vec, block_vec)
                {
                    CommitMultiLaneMutableBlockResult::Immutable { block_vec } => {
                        self.immutable_blocks.push(block_vec);
                    },
                    CommitMultiLaneMutableBlockResult::ImmutableCollision { block_vec } => {
                        self.immutable_blocks.push(block_vec);
                        self.num_in_sync_blocks = 0;
                    },
                }
            }
        }
        self.try_mark_ready();
        self.unload_cached_resources();
    }
}

fn annotation_resource_ids<const L: usize>(annotation_vec: &[SmallVec<[BlockAnnotation; 1]>; L]) -> Vec<ResourceID> {
    let mut resource_ids = Vec::new();
    for annotation in annotation_vec.iter().flatten() {
        let BlockAnnotation::Resource(segment) = annotation else {
            continue;
        };
        if !resource_ids.contains(&segment.resource_id()) {
            resource_ids.push(segment.resource_id());
        }
    }
    resource_ids
}

impl<const N: usize, const P: usize, const L: usize, BC> TrieDecoderBlocks<N, P, L, BC>
where
    BC: MultiLaneBlockCache<P, L>,
{
    /// Sets token identities in `token_index_start..token_index_end` across all cache lanes.
    /// `tokens[0]` has Main token index `token_index_start`.
    /// Cache lane `l` at cache token index `i` receives `tokens[i - token_index_start + l]`.
    /// Fixed blocks must already contain these IDs. Mutable blocks overwrite or append them.
    /// The interval must be non-empty and fit the allocated blocks. This function does not clip it.
    ///
    /// The source includes L - 1 lookahead slots. Decode preparation may use
    /// placeholders for unknown lookahead in noncached mutable MTP slots.
    /// Commit must supply real token identities before advancing cached progress.
    fn write_tokens(&mut self, token_index_start: usize, token_index_end: usize, tokens: &[Token]) {
        let num_fixed_blocks = self.immutable_blocks.len() + self.semi_immutable_blocks.len();
        debug_assert!(
            token_index_start < token_index_end,
            "token write interval must be non-empty and ordered"
        );
        debug_assert!(
            token_index_end <= (num_fixed_blocks + self.mutable_blocks.len()) * N,
            "token write interval exceeds allocated blocks"
        );
        debug_assert!(token_index_end - token_index_start + L - 1 <= tokens.len());
        for block_index in token_index_start / N..token_index_end.div_ceil(N) {
            let block_token_index_start = block_index * N;
            let write_token_index_start = token_index_start.max(block_token_index_start);
            let write_token_index_end = token_index_end.min(block_token_index_start + N);
            let block_token_offset_start = write_token_index_start - block_token_index_start;
            let block_token_offset_end = write_token_index_end - block_token_index_start;
            for cache_lane_index in 0..L {
                let lane_tokens = &tokens[write_token_index_start - token_index_start + cache_lane_index
                    ..write_token_index_end - token_index_start + cache_lane_index];
                if block_index < self.immutable_blocks.len() {
                    debug_assert_eq!(
                        &self.immutable_blocks[block_index][cache_lane_index].total_tokens()
                            [block_token_offset_start..block_token_offset_end],
                        lane_tokens,
                        "immutable token identities changed"
                    );
                } else if block_index < num_fixed_blocks {
                    debug_assert_eq!(
                        &self.semi_immutable_blocks[block_index - self.immutable_blocks.len()][cache_lane_index]
                            .total_tokens()[block_token_offset_start..block_token_offset_end],
                        lane_tokens,
                        "semi-immutable token identities changed"
                    );
                } else {
                    self.mutable_blocks[block_index - num_fixed_blocks][cache_lane_index]
                        .write_tokens(block_token_offset_start, lane_tokens);
                }
            }
        }
    }

    pub fn start_turn(&mut self, prompt_tokens: Vec<Token>) {
        assert!(!prompt_tokens.is_empty(), "a new turn must include prompt tokens");
        assert_eq!(
            self.num_prompt_tokens(),
            0,
            "a new turn requires completed prompt metadata"
        );
        assert_eq!(
            self.num_sampled_tokens(),
            0,
            "a new turn requires completed sampled-token metadata"
        );
        assert!(
            self.spec_tokens.is_empty(),
            "a new turn cannot retain speculative tokens"
        );
        assert!(
            self.spec_probs.is_empty(),
            "a new turn cannot retain speculative probabilities"
        );
        assert!(
            self.spec_confidences.is_empty(),
            "a new turn cannot retain speculative confidences"
        );
        self.num_prompt_tokens = prompt_tokens.len();
        self.queued_tokens.extend(prompt_tokens);
        self.try_mark_ready();
    }

    pub fn finish_turn(&mut self) {
        assert_eq!(
            self.num_scheduled_tokens(),
            0,
            "a turn cannot finish while tokens are scheduled"
        );
        self.truncate_spec_tokens(0);
        self.num_history_tokens = self.num_total_tokens();
        self.num_prompt_tokens = 0;
        debug_assert_eq!(self.num_sampled_tokens(), 0);
    }

    fn try_mark_ready(&mut self) {
        let mut num_cachable_tokens = self.num_queued_tokens().saturating_sub(L - 1);

        for block_vec in self.mutable_blocks.iter_mut() {
            if num_cachable_tokens == 0 {
                break;
            }
            if N == block_vec[0].total_tokens().len() {
                continue;
            }

            let num_tokens = min(num_cachable_tokens, N - block_vec[0].total_tokens().len());
            num_cachable_tokens -= num_tokens;
            let tokens = pop_front_queued_tokens::<L>(&mut self.queued_tokens, num_tokens);
            debug_assert_eq!(num_tokens + L - 1, tokens.len());
            push_tokens::<N, L>(block_vec, &tokens);
            // TODO when mutable block is full, maybe turn it into semi immutable
        }
    }

    fn parent_trie_node_key_vec(&self, block_index: usize) -> [Option<TrieNodeKey>; L] {
        if block_index == 0 {
            std::array::from_fn(|_| None)
        } else {
            debug_assert!(
                block_index <= self.immutable_blocks.len(),
                "parent_trie_node_key_vec: block_index={block_index} requires previous immutable block"
            );
            let parent_block_vec = &self.immutable_blocks[block_index - 1];
            std::array::from_fn(|lane| Some(parent_block_vec[lane].trie_node_key()))
        }
    }

    fn block_annotation_vec(&self, block_index: usize) -> [SmallVec<[BlockAnnotation; 1]>; L] {
        std::array::from_fn(|lane| {
            let mut annotations: SmallVec<[BlockAnnotation; 1]> = SmallVec::new();
            if block_index == 0 && lane > 0 {
                // MTP root lane identity, L = 4:
                //
                // verified tokens  t0  t1  t2  ...
                // lane 0 root      t0  t1  t2  ...
                // lane 1 root          t1  t2  ...  prefix: [t0]
                // lane 2 root              t2  ...  prefix: [t0, t1]
                // lane 3 root                  ...  prefix: [t0, t1, t2]
                //
                // TODO: Allow a short MTP root to defer its lane-specific PrefixTokens annotation until verified
                // history contains that lane's complete prefix.
                let prefix_tokens: Arc<[Token]> = self.total_tokens().take(lane).collect::<Vec<_>>().into();
                debug_assert_eq!(lane, prefix_tokens.len());
                annotations.push(BlockAnnotation::prefix_tokens(prefix_tokens));
            }

            let block_token_start = block_index * N + lane;
            let block_token_end = block_token_start + N;
            for placement in &self.resource_placements {
                for &(placement_token_start, resource_index, len) in placement.placements() {
                    let placement_token_end = placement_token_start + len;
                    let active_token_start = block_token_start.max(placement_token_start);
                    let active_token_end = block_token_end.min(placement_token_end);
                    if active_token_start >= active_token_end {
                        continue;
                    }

                    let local_token_index = (active_token_start - block_token_start) as u16;
                    let active_resource_index =
                        u32::try_from(resource_index + active_token_start - placement_token_start)
                            .expect("resource index must fit the cache annotation u32 domain");
                    let active_len = (active_token_end - active_token_start) as u16;
                    annotations.push(BlockAnnotation::resource(ResourceSegment::new(
                        placement.resource_id(),
                        local_token_index,
                        active_resource_index,
                        active_len,
                    )));
                }
            }
            annotations.sort_unstable();
            annotations
        })
    }
}

#[cfg(test)]
#[path = "./api_test_vanilla.rs"]
mod api_test_vanilla;

#[cfg(test)]
#[path = "./api_test_w_mtp.rs"]
mod api_test_w_mtp;

#[cfg(test)]
#[path = "./api_test_single_lane.rs"]
mod api_test_single_lane;

#[cfg(test)]
#[path = "./api_test_resource.rs"]
mod api_test_resource;

#[cfg(test)]
#[path = "./api_test_support.rs"]
mod api_test_support;
