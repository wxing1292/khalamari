use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Sender;
use crossbeam_channel::unbounded;
use futures_lite::future::block_on;

use super::*;
use crate::channel::Shutdown;
use crate::runtime::tasks::SyncTaskPool;

#[derive(Clone)]
struct SequenceStateMachine {
    tokens: Arc<[Token]>,
    next_token: usize,
    pool: SyncTaskPool,
    submitted: Sender<usize>,
}

impl GuidedDecodeSM for SequenceStateMachine {
    fn try_apply(&mut self, token: Token) -> StateMachineTransition {
        let expected = self.tokens[self.next_token];
        self.next_token += 1;
        if token == expected {
            StateMachineTransition::Allow
        } else if token == Token::new(99) {
            StateMachineTransition::Error(Error::internal("dummy transition failed"))
        } else {
            StateMachineTransition::Reject {
                token,
                byte_offset: None,
                reason: format!("expected token {}", expected.value()),
            }
        }
    }

    fn status(&self) -> StateMachineStatus {
        StateMachineStatus::InProgress
    }

    fn fill_mask(self: Box<Self>) -> MaskFuture {
        let pool = self.pool.clone();
        let receiver = pool.spawn(move || {
            let _ = self.submitted.send(self.next_token);
            let token_id = self.tokens[self.next_token].value();
            let mut words = vec![0; 4];
            words[token_id as usize / 32] |= 1 << (token_id % 32);
            Ok(Mask::Bitset(words))
        });
        Box::pin(async move { receiver.await.expect("mask task must return a result") })
    }
}

#[test]
fn test_spec_masks_match_next_token_positions() {
    let pool = SyncTaskPool::new(Shutdown::new(), 2).unwrap();
    let (submitted, started) = unbounded();
    let mut state_machine = SequenceStateMachine {
        tokens: Arc::from([10, 35, 68, 101].map(Token::new)),
        next_token: 0,
        pool,
        submitted,
    };
    assert!(matches!(
        state_machine.try_apply(Token::new(10)),
        StateMachineTransition::Allow
    ));

    let masks = fill_spec_masks(&state_machine, [35, 68].map(Token::new)).unwrap();
    assert_eq!(masks.len(), 3);
    assert_eq!(
        state_machine.next_token, 1,
        "drafts must not advance the original state"
    );

    // Every snapshot starts work before any mask future is polled.
    let mut positions = (0..3)
        .map(|_| started.recv_timeout(Duration::from_secs(5)).unwrap())
        .collect::<Vec<_>>();
    positions.sort_unstable();
    assert_eq!(positions, [1, 2, 3]);
    drop(state_machine);

    // Read futures in reverse order to check that each owns its original snapshot.
    for (future, token_id) in masks.into_iter().rev().zip([101, 68, 35]) {
        assert_single_token_mask(future, token_id);
    }
}

#[test]
fn test_spec_masks_stop_at_rejected_draft() {
    let pool = SyncTaskPool::new(Shutdown::new(), 1).unwrap();
    let (submitted, _started) = unbounded();
    let mut state_machine = SequenceStateMachine {
        tokens: Arc::from([10, 35, 68, 101].map(Token::new)),
        next_token: 0,
        pool,
        submitted,
    };
    assert!(matches!(
        state_machine.try_apply(Token::new(10)),
        StateMachineTransition::Allow
    ));

    for (drafts, expected_masks) in [
        (&[][..], &[35][..]),
        (&[35][..], &[35, 68][..]),
        (&[35, 7, 99][..], &[35, 68][..]),
        (&[7, 99][..], &[35][..]),
    ] {
        let masks = fill_spec_masks(&state_machine, drafts.iter().copied().map(Token::new)).unwrap();
        assert_eq!(masks.len(), expected_masks.len());
        for (future, &token_id) in masks.into_iter().zip(expected_masks) {
            assert_single_token_mask(future, token_id);
        }
        assert_eq!(state_machine.next_token, 1);
    }
    assert!(matches!(
        fill_spec_masks(&state_machine, [Token::new(99)]),
        Err(Error::Internal(message)) if message == "dummy transition failed"
    ));
    assert_eq!(state_machine.next_token, 1);
}

#[test]
fn test_default_spec_masks() {
    for drafts in [&[][..], &[10, 35, 68][..]] {
        let masks = fill_spec_masks(&DefaultStateMachine, drafts.iter().copied().map(Token::new)).unwrap();
        assert_eq!(masks.len(), drafts.len() + 1);
        for future in masks {
            assert!(matches!(block_on(future), Ok(Mask::AllowAll)));
        }
    }
}

fn assert_single_token_mask(future: MaskFuture, token_id: u32) {
    let Mask::Bitset(words) = block_on(future).unwrap() else {
        panic!("dummy must return a bitset")
    };
    for (word_index, word) in words.into_iter().enumerate() {
        let expected = if word_index == token_id as usize / 32 {
            1 << (token_id % 32)
        } else {
            0
        };
        assert_eq!(word, expected);
    }
}
