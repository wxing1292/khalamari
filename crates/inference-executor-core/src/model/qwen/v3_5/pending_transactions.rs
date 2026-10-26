use std::collections::VecDeque;

use inference_runtime_core::runtime::RawComputeSlotSeq;

use crate::model::qwen::v3_5::Qwen35DecodeDecision;
use crate::model::qwen::v3_5::Qwen35Microbatch;
use crate::model::qwen::v3_5::verified_state_versions_for_decisions;

#[derive(Clone, Debug)]
pub struct Qwen35PendingTransactions {
    transactions: VecDeque<Qwen35PendingTransaction>,
}

#[derive(Clone, Debug)]
struct Qwen35PendingTransaction {
    compute_seq: RawComputeSlotSeq,
    microbatch: Qwen35Microbatch,
}

impl Qwen35PendingTransactions {
    pub fn new() -> Self {
        Self {
            transactions: VecDeque::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.transactions.is_empty()
    }

    pub fn pending_microbatch(&self, compute_seq: RawComputeSlotSeq) -> &Qwen35Microbatch {
        let transaction = self
            .transactions
            .front()
            .expect("qwen3.5 replay sampling requires a pending batch");
        assert_eq!(
            transaction.compute_seq, compute_seq,
            "qwen3.5 pending batch sequence must match the oldest transaction"
        );
        &transaction.microbatch
    }

    pub fn push(&mut self, compute_seq: RawComputeSlotSeq, microbatch: Qwen35Microbatch) {
        if let Some(last_transaction) = self.transactions.back() {
            assert!(
                last_transaction.compute_seq < compute_seq,
                "qwen3.5 pending transaction sequences must increase"
            );
        }
        self.transactions.push_back(Qwen35PendingTransaction {
            compute_seq,
            microbatch,
        });
    }

    pub fn commit(&mut self, compute_seq: RawComputeSlotSeq, decisions: &[Qwen35DecodeDecision]) -> Vec<u32> {
        let front_transaction = self
            .transactions
            .front()
            .expect("qwen3.5 commit requires a pending batch");
        assert_eq!(
            front_transaction.compute_seq, compute_seq,
            "qwen3.5 commit sequence must match the oldest pending transaction"
        );
        let transaction = self
            .transactions
            .pop_front()
            .expect("qwen3.5 verified pending transaction must remain available");
        let microbatch = transaction.microbatch;
        verified_state_versions_for_decisions(&microbatch, decisions)
    }
}

#[cfg(test)]
mod tests {
    use inference_runtime_core::compute::DecoderSyncBlocks;
    use inference_runtime_core::compute::DeviceRequest;
    use inference_runtime_core::compute::QueryTokens;
    use inference_runtime_core::config::SamplingConfig;
    use inference_runtime_core::runtime::Token;

    use super::*;
    use crate::sampling::SamplerConfig;

    #[test]
    fn test_commit() {
        let mut transactions = Qwen35PendingTransactions::new();
        let request = request_with_tokens(vec![11], vec![12]);

        transactions.push(1, request);
        assert!(!transactions.transactions.is_empty());
        let decisions = vec![Qwen35DecodeDecision {
            validated_tokens: vec![12],
            sampled_token: 99,
            sampled_prob: 0.4,
            ..Qwen35DecodeDecision::default()
        }];
        assert_eq!(transactions.commit(1, &decisions), vec![6]);
        assert!(transactions.transactions.is_empty());
    }

    #[test]
    fn test_main_only() {
        let mut transactions = Qwen35PendingTransactions::new();
        let request = request_with_tokens(vec![11, 12], vec![]);

        transactions.push(1, request);

        assert_eq!(transactions.commit(1, &[Qwen35DecodeDecision::default()]), vec![6]);
    }

    #[test]
    fn test_prefill_commits_through_pending_transactions() {
        let mut transactions = Qwen35PendingTransactions::new();

        transactions.push(1, prefill_request(vec![11, 12]));

        assert_eq!(transactions.commit(1, &[]), vec![6]);
    }

    #[test]
    fn test_pending_transactions_commit_in_sequence_order() {
        let mut transactions = Qwen35PendingTransactions::new();
        transactions.push(1, request_with_tokens(vec![11], vec![]));
        transactions.push(2, request_with_tokens(vec![12], vec![]));

        assert_eq!(transactions.pending_microbatch(1).flat_token_ids(), &[11]);
        assert_eq!(transactions.commit(1, &[Qwen35DecodeDecision::default()]), vec![5]);
        assert_eq!(transactions.pending_microbatch(2).flat_token_ids(), &[12]);
        assert_eq!(transactions.commit(2, &[Qwen35DecodeDecision::default()]), vec![5]);
        assert!(transactions.transactions.is_empty());
    }

    fn request_with_tokens(tokens: Vec<u32>, spec_tokens: Vec<u32>) -> Qwen35Microbatch {
        let request = DeviceRequest::new(
            10,
            0,
            QueryTokens::Decode {
                epoch: 1,
                token_index: 4,
                tokens: tokens.into_iter().map(Token::new).collect(),
                spec_tokens: spec_tokens.into_iter().map(Token::new).collect(),
            },
            DecoderSyncBlocks::new(0, vec![], vec![]),
            vec![],
            SamplingConfig::default(),
        );
        Qwen35Microbatch::from_requests(&[request], vec![SamplerConfig::default()])
    }

    fn prefill_request(tokens: Vec<u32>) -> Qwen35Microbatch {
        let window = tokens.len();
        let request = DeviceRequest::new(
            10,
            0,
            QueryTokens::Prefill {
                epoch: 1,
                token_index: 4,
                tokens: tokens.into_iter().map(Token::new).collect(),
                window,
            },
            DecoderSyncBlocks::new(0, vec![], vec![]),
            vec![],
            SamplingConfig::default(),
        );
        Qwen35Microbatch::from_requests(&[request], vec![SamplerConfig::default()])
    }
}
