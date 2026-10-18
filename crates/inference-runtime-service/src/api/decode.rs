use std::sync::atomic::Ordering;

use inference_runtime_core::Error;
use inference_runtime_core::Result;
use inference_runtime_core::config::MAX_SAMPLING_TOP_K;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::CompletionReason;
use inference_runtime_core::runtime::ExternalRequest;
use inference_runtime_core::runtime::RawRequestID;
use inference_runtime_core::runtime::RequestEvent;
use inference_runtime_core::runtime::RequestStatus;
use inference_runtime_core::runtime::RequestTokenPositions;
use inference_runtime_core::runtime::Resource;
use inference_runtime_core::runtime::ResourcePlacement;
use inference_runtime_core::runtime::Token;
use inference_runtime_core::runtime::TokenProbs;

use crate::api::Inference;

impl<const N: usize, const L: usize, const P: usize> Inference<N, L, P> {
    pub fn create_session(&self, request: DecodeRequest) -> Result<DecodeSession<N, L, P>> {
        self.validate_input_tokens(&request.tokens)?;
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        assert_ne!(request_id, 0, "decode request ID allocator wrapped to zero");

        let DecodeRequest {
            tokens,
            token_positions,
            resource_entries,
            mut sampling,
        } = request;
        let num_history_tokens = tokens.len();
        merge_stop_sequences(&mut sampling.stop_sequences, &self.default_stop_sequences);

        let external_request = self.runtime.create_session(
            request_id,
            vec![],
            tokens,
            vec![],
            token_positions,
            resource_entries,
            sampling,
        )?;
        Ok(DecodeSession::new(external_request, num_history_tokens))
    }

    pub async fn resume_session(&self, session: &mut DecodeSession<N, L, P>, request: DecodeRequest) -> Result<()> {
        assert!(
            session.num_turn_output_tokens.is_none(),
            "decode session cannot resume before turn completion",
        );
        self.validate_input_tokens(&request.tokens)?;
        let DecodeRequest {
            tokens,
            token_positions,
            resource_entries,
            mut sampling,
        } = request;
        let num_prompt_tokens = tokens.len();
        if token_positions.is_some() {
            return Err(Error::invalid_argument(
                "a resumed decode turn cannot set explicit token positions",
            ));
        }
        if !resource_entries.is_empty() {
            return Err(Error::invalid_argument(
                "a resumed decode turn cannot add resource entries",
            ));
        }
        merge_stop_sequences(&mut sampling.stop_sequences, &self.default_stop_sequences);
        self.runtime.resume_session(&session.request, tokens, sampling).await?;
        session.start_turn(num_prompt_tokens);
        Ok(())
    }

    fn validate_input_tokens(&self, tokens: &[Token]) -> Result<()> {
        if let Some(token) = tokens.iter().find(|token| token.value() as usize >= self.vocab_size) {
            return Err(Error::invalid_argument(format!(
                "input token ID {} must be below model vocab_size={}",
                token.value(),
                self.vocab_size
            )));
        }
        Ok(())
    }
}

pub struct DecodeRequest {
    tokens: Vec<Token>,
    token_positions: Option<RequestTokenPositions>,
    resource_entries: Vec<(Resource, ResourcePlacement)>,
    sampling: SamplingConfig,
}

impl DecodeRequest {
    pub fn new(
        tokens: Vec<Token>,
        token_positions: Option<RequestTokenPositions>,
        resource_entries: Vec<(Resource, ResourcePlacement)>,
        sampling: SamplingConfig,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(Error::invalid_argument(
                "decode request must include at least one token",
            ));
        }
        if sampling.max_sampled_tokens == 0 {
            return Err(Error::invalid_argument("max_sampled_tokens must be greater than 0"));
        }
        if !sampling.temperature.is_finite() || sampling.temperature < 0.0 {
            return Err(Error::invalid_argument(format!(
                "temperature must be finite and non-negative, got {}",
                sampling.temperature
            )));
        }
        if !(1..=MAX_SAMPLING_TOP_K).contains(&sampling.top_k) {
            return Err(Error::invalid_argument(format!(
                "top_k must be in [1, {MAX_SAMPLING_TOP_K}], got {}",
                sampling.top_k
            )));
        }
        if !sampling.top_p.is_finite() || !(0.0..=1.0).contains(&sampling.top_p) {
            return Err(Error::invalid_argument(format!(
                "top_p must be finite and in [0, 1], got {}",
                sampling.top_p
            )));
        }
        if sampling.stop_sequences.iter().any(Vec::is_empty) {
            return Err(Error::invalid_argument(
                "stop sequences must not include empty sequences",
            ));
        }
        Ok(Self {
            tokens,
            token_positions,
            resource_entries,
            sampling,
        })
    }
}

pub struct DecodeSession<const N: usize, const L: usize, const P: usize> {
    request: ExternalRequest,
    num_history_tokens: usize,
    num_turn_output_tokens: Option<usize>,
}

impl<const N: usize, const L: usize, const P: usize> DecodeSession<N, L, P> {
    fn new(request: ExternalRequest, num_history_tokens: usize) -> Self {
        Self {
            request,
            num_history_tokens,
            num_turn_output_tokens: Some(0),
        }
    }

    pub fn request_id(&self) -> RawRequestID {
        self.request.req_id()
    }

    pub fn num_history_tokens(&self) -> usize {
        self.num_history_tokens
    }

    fn start_turn(&mut self, num_prompt_tokens: usize) {
        assert!(
            self.num_turn_output_tokens.is_none(),
            "decode session cannot start two active turns"
        );
        self.num_history_tokens += num_prompt_tokens;
        self.num_turn_output_tokens = Some(0);
    }

    pub async fn recv_event(&mut self) -> Result<DecodeEvent> {
        match self.request.event_rx().recv().await {
            Ok(RequestEvent::TokenProbs(token_probs)) => {
                debug_assert!(!token_probs.tokens.is_empty());
                debug_assert_eq!(token_probs.tokens.len(), token_probs.probs.len());
                self.num_history_tokens += token_probs.tokens.len();
                *self
                    .num_turn_output_tokens
                    .as_mut()
                    .expect("token output requires an active decode turn") += token_probs.tokens.len();
                Ok(DecodeEvent::TokenProbs(token_probs))
            },
            Ok(RequestEvent::TurnCompleted(reason)) => {
                let num_output_tokens = self
                    .num_turn_output_tokens
                    .take()
                    .expect("turn completion requires an active decode turn");
                Ok(DecodeEvent::Completed {
                    reason,
                    num_output_tokens,
                })
            },
            Err(_) => {
                match self.request.status() {
                    RequestStatus::Completed(reason) => {
                        Ok(DecodeEvent::Completed {
                            reason,
                            num_output_tokens: self
                                .num_turn_output_tokens
                                .take()
                                .expect("terminal completion requires an active decode turn"),
                        })
                    },
                    RequestStatus::Cancelled => Err(Error::cancelled("request was cancelled")),
                    RequestStatus::TimedOut => Err(Error::deadline_exceeded("request deadline exceeded")),
                    RequestStatus::Aborted => Err(Error::aborted("request was aborted")),
                    RequestStatus::Evicted => Err(Error::evicted("request was evicted")),
                    status => panic!("request output closed in non-terminal state: {status:?}"),
                }
            },
        }
    }

    /// Waits for an idle session to lose its internal request and returns the terminal error.
    pub async fn wait_for_session_end(&mut self) -> Error {
        match self.recv_event().await {
            Err(error) => error,
            Ok(_) => panic!("an idle decode session cannot produce a request event"),
        }
    }
}

#[derive(Debug)]
pub enum DecodeEvent {
    TokenProbs(TokenProbs),
    Completed {
        reason: CompletionReason,
        num_output_tokens: usize,
    },
}

fn merge_stop_sequences(input: &mut Vec<Vec<Token>>, defaults: &[Vec<Token>]) {
    input.extend(defaults.iter().cloned());
    input.sort_unstable();
    input.dedup();
}

#[cfg(test)]
#[path = "./decode_test.rs"]
mod decode_test;
