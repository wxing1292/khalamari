use std::future::ready;

use dyn_clone::DynClone;
use futures_lite::future::Boxed;

use crate::Error;
use crate::Result;
use crate::runtime::Token;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateMachineStatus {
    InProgress,
    /// The output is complete. More tokens may still be accepted.
    Accepting,
    Terminated,
}

#[derive(Debug)]
pub enum StateMachineTransition {
    Allow,
    Reject {
        token: Token,
        /// Byte offset within the rejected token, if known.
        byte_offset: Option<usize>,
        reason: String,
    },
    Error(Error),
}

#[derive(Debug)]
pub enum Mask {
    AllowAll,
    /// Packed bits: bit `token_id % 32` in word `token_id / 32` allows that token.
    Bitset(Vec<u32>),
}

pub type MaskFuture = Boxed<Result<Mask>>;

/// Tracks which tokens are valid at the current grammar position.
///
/// Cloning must be cheap. Share immutable data, and copy mutable state or use copy-on-write.
/// Changes to a clone must not affect the original.
pub trait GuidedDecodeSM: DynClone + Send {
    /// Tries to consume `token`.
    ///
    /// Rejection and errors can leave this instance changed. Clone it first if you need
    /// to preserve its state. This method does not generate masks or append other tokens.
    fn try_apply(&mut self, token: Token) -> StateMachineTransition;

    fn status(&self) -> StateMachineStatus;

    /// Generates the next-token mask from the current state.
    /// Allowed tokens must match `try_apply` at this state.
    ///
    /// Consumes this snapshot. Submit any background work before returning.
    fn fill_mask(self: Box<Self>) -> MaskFuture;
}

dyn_clone::clone_trait_object!(GuidedDecodeSM);

/// Generates masks for the current state and each valid speculative prefix.
///
/// The current state must already include the anchor token. Mask `0` predicts the
/// first draft; mask `i + 1` predicts the token after draft `i`.
/// Stops before applying any draft after the first rejection. The returned length
/// is the valid draft count plus one. This function leaves `state_machine` unchanged.
/// The final mask also covers the first rejected draft. Returning fewer masks does
/// not accept the preceding drafts or replace rejection sampling with a bonus draw.
pub fn fill_spec_masks(
    state_machine: &dyn GuidedDecodeSM,
    spec_tokens: impl IntoIterator<Item = Token>,
) -> Result<Vec<MaskFuture>> {
    let mut state_machine = dyn_clone::clone_box(state_machine);
    let mut masks = vec![state_machine.clone().fill_mask()];
    for token in spec_tokens {
        match state_machine.try_apply(token) {
            StateMachineTransition::Allow => masks.push(state_machine.clone().fill_mask()),
            StateMachineTransition::Reject { .. } => break,
            StateMachineTransition::Error(error) => return Err(error),
        }
    }
    Ok(masks)
}

/// The default state machine permits every token and never terminates.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultStateMachine;

impl GuidedDecodeSM for DefaultStateMachine {
    fn try_apply(&mut self, _token: Token) -> StateMachineTransition {
        StateMachineTransition::Allow
    }

    fn status(&self) -> StateMachineStatus {
        StateMachineStatus::Accepting
    }

    fn fill_mask(self: Box<Self>) -> MaskFuture {
        Box::pin(ready(Ok(Mask::AllowAll)))
    }
}

#[cfg(test)]
#[path = "guided_decode_test.rs"]
mod tests;
