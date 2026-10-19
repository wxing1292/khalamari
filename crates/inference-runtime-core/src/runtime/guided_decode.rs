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
    ///
    /// Consumes this snapshot. Submit any background work before returning.
    fn fill_mask(self: Box<Self>) -> MaskFuture;
}

dyn_clone::clone_trait_object!(GuidedDecodeSM);

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
mod tests {
    use futures_lite::future::block_on;

    use super::*;

    #[test]
    fn test_mask_future_outlives_state_machine() {
        let future = {
            let sm: Box<dyn GuidedDecodeSM> = Box::new(DefaultStateMachine);
            let future = sm.clone().fill_mask();
            assert_eq!(sm.status(), StateMachineStatus::Accepting);
            future
        };

        assert!(matches!(block_on(future), Ok(Mask::AllowAll)));
    }
}
