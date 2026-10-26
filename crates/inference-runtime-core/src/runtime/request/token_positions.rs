use crate::compute::QueryTokens;

/// Model-owned token positions for one initial request and its sequential continuation.
///
/// Runtime core stores and slices this data. It does not interpret an axis or
/// calculate model-specific token positions.
// TODO(token-positions): Restore request transport after the executor consumes model-owned positions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestTokenPositions {
    initial: Vec<[u32; 3]>,
    continuation_start: [u32; 3],
}

impl RequestTokenPositions {
    pub fn new(initial: Vec<[u32; 3]>, continuation_start: [u32; 3]) -> Self {
        assert!(
            !initial.is_empty(),
            "explicit request token positions require initial tokens"
        );
        Self {
            initial,
            continuation_start,
        }
    }

    pub fn initial(&self) -> &[[u32; 3]] {
        &self.initial
    }

    pub const fn continuation_start(&self) -> [u32; 3] {
        self.continuation_start
    }

    pub fn query(&self, query_tokens: &QueryTokens) -> Vec<[u32; 3]> {
        let token_start = query_tokens.token_index();
        let token_end = token_start + query_tokens.token_consumption();
        (token_start..token_end)
            .map(|token_index| {
                self.initial.get(token_index).copied().unwrap_or_else(|| {
                    let continuation_index = token_index - self.initial.len();
                    self.continuation_start
                        .map(|position| position + continuation_index as u32)
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Token;

    #[test]
    fn test_query_initial_and_continuation() {
        let positions = RequestTokenPositions::new(vec![[0, 0, 0], [1, 2, 3]], [4, 5, 6]);
        let query = QueryTokens::Prefill {
            epoch: 0,
            token_index: 1,
            tokens: vec![Token::new(7), Token::new(8), Token::new(9)],
            window: 3,
        };

        assert_eq!(positions.query(&query), [[1, 2, 3], [4, 5, 6], [5, 6, 7]]);
    }
}
