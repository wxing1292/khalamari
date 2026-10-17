use std::collections::HashSet;

use inference_backend_metal::components::ngram_embedding;
use inference_backend_metal::components::ple;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::ngram_embedding::core::NGramEmbeddingCore;

pub struct PLERequest<'a> {
    pub token_ids: &'a [u32],
    /// Exactly `ngram_size - 1` IDs, with EOS padding at the start of a request.
    pub previous_tokens: &'a [u32],
    pub src_state_slot: u32,
    pub dst_state_slot: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct PLEActive {
    pub num_active_tokens: u32,
    pub num_active_requests: u32,
}

/// Upload owner for one batch. Source slots are immutable until the forward and commit complete.
pub struct PLEMetadataBuffers {
    core: NGramEmbeddingCore,
    max_tokens: u32,
    max_requests: u32,
    num_state_slots: u32,
    token_ids: Buffer,
    previous_tokens: Buffer,
    hash_tokens: Buffer,
    conv_tokens: Buffer,
    requests: Buffer,
    accepted_tokens: Buffer,
    request_tokens: Vec<u32>,
}
impl PLEMetadataBuffers {
    pub fn new(
        device: &Device,
        core: NGramEmbeddingCore,
        max_tokens: u32,
        max_requests: u32,
        num_state_slots: u32,
    ) -> Self {
        core.validate();
        assert!(max_tokens > 0 && max_requests > 0 && max_requests <= max_tokens && num_state_slots > 0);
        let buffer = |rows: u32, width: usize| {
            Buffer::new_zeroed(
                device,
                (rows as usize)
                    .checked_mul(width)
                    .expect("PLE metadata bytes must fit usize"),
            )
        };
        Self {
            core,
            max_tokens,
            max_requests,
            num_state_slots,
            token_ids: buffer(max_tokens, 4),
            previous_tokens: buffer(max_requests, (core.ngram_size - 1) as usize * 4),
            hash_tokens: buffer(max_tokens, size_of::<ngram_embedding::Token>()),
            conv_tokens: buffer(max_tokens, size_of::<ple::Token>()),
            requests: buffer(max_requests, size_of::<ple::Request>()),
            accepted_tokens: buffer(max_requests, 4),
            request_tokens: Vec::new(),
        }
    }
    pub fn update(&mut self, requests: &[PLERequest<'_>]) -> PLEActive {
        assert!(!requests.is_empty() && requests.len() <= self.max_requests as usize);
        let sources: HashSet<_> = requests.iter().map(|r| r.src_state_slot).collect();
        let mut destinations = HashSet::new();
        let mut ids = Vec::new();
        let mut history = Vec::new();
        let mut hash_tokens = Vec::new();
        let mut conv_tokens = Vec::new();
        let mut records = Vec::new();
        self.request_tokens.clear();
        for (request_index, r) in requests.iter().enumerate() {
            assert!(!r.token_ids.is_empty());
            assert_eq!(r.previous_tokens.len(), self.core.ngram_size as usize - 1);
            assert!(
                r.token_ids
                    .iter()
                    .chain(r.previous_tokens)
                    .all(|&id| id < self.core.vocab_size)
            );
            assert!(r.src_state_slot < self.num_state_slots && r.dst_state_slot < self.num_state_slots);
            assert!(
                !sources.contains(&r.dst_state_slot) && destinations.insert(r.dst_state_slot),
                "PLE destinations must be distinct and disjoint from all sources"
            );
            let start = u32::try_from(ids.len()).expect("PLE batch start must fit u32");
            let count = u32::try_from(r.token_ids.len()).expect("PLE request token count must fit u32");
            assert!(u64::from(start) + u64::from(count) <= u64::from(self.max_tokens));
            ids.extend_from_slice(r.token_ids);
            history.extend_from_slice(r.previous_tokens);
            for _ in 0..count {
                hash_tokens.extend_from_slice(&[start, request_index as u32]);
                conv_tokens.extend_from_slice(&[start, r.src_state_slot]);
            }
            records.extend_from_slice(&[start, r.src_state_slot, r.dst_state_slot]);
            self.request_tokens.push(count);
        }
        self.token_ids.write_typed(0, &ids);
        self.previous_tokens.write_typed(0, &history);
        self.hash_tokens.write_typed(0, &hash_tokens);
        self.conv_tokens.write_typed(0, &conv_tokens);
        self.requests.write_typed(0, &records);
        self.accepted_tokens.write_typed(0, &self.request_tokens);
        PLEActive {
            num_active_tokens: ids.len() as u32,
            num_active_requests: requests.len() as u32,
        }
    }
    /// Must run after the forward completes. Zero acceptance copies the source state unchanged.
    pub fn set_accepted_tokens(&self, accepted: &[u32]) {
        assert_eq!(accepted.len(), self.request_tokens.len());
        assert!(accepted.iter().zip(&self.request_tokens).all(|(a, n)| a <= n));
        self.accepted_tokens.write_typed(0, accepted);
    }
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }
    pub fn ngram_core(&self) -> NGramEmbeddingCore {
        self.core
    }
    pub fn max_requests(&self) -> u32 {
        self.max_requests
    }
    pub fn num_state_slots(&self) -> u32 {
        self.num_state_slots
    }
    pub fn token_ids(&self) -> &Buffer {
        &self.token_ids
    }
    pub fn previous_tokens(&self) -> &Buffer {
        &self.previous_tokens
    }
    pub fn hash_tokens(&self) -> &Buffer {
        &self.hash_tokens
    }
    pub fn conv_tokens(&self) -> &Buffer {
        &self.conv_tokens
    }
    pub fn requests(&self) -> &Buffer {
        &self.requests
    }
    pub fn accepted_tokens(&self) -> &Buffer {
        &self.accepted_tokens
    }
}
