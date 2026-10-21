use inference_executor_core::def::ModelExecutorError;
use inference_executor_core::sampling::MAX_TOP_K;
use inference_executor_core::sampling::SamplerConfig;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TopKSamplingShape {
    pub num_active_sampling_inputs: u32,
    pub num_total_sampling_inputs: u32,
    pub vocab_size: u32,
    pub top_k: u32,
}

impl TopKSamplingShape {
    pub fn top_k_count(self) -> usize {
        (self.num_total_sampling_inputs as usize)
            .checked_mul(self.top_k as usize)
            .expect("top-k sampling output element count must fit usize")
    }

    pub fn with_num_total_sampling_inputs(mut self, num_total_sampling_inputs: u32) -> Self {
        assert!(
            self.num_active_sampling_inputs <= num_total_sampling_inputs,
            "top-k sampling active inputs exceed replay total"
        );
        self.num_total_sampling_inputs = num_total_sampling_inputs;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopKSamplingBounds {
    pub max_sampling_inputs: u32,
    pub vocab_size: u32,
    pub top_k: u32,
}

impl TopKSamplingBounds {
    pub fn from_config(
        config: &SamplerConfig,
        max_sampling_inputs: u32,
        vocab_size: u32,
    ) -> Result<Self, ModelExecutorError> {
        config.validate()?;
        let top_k = top_k_sampling_len(config, vocab_size as usize)? as u32;
        Ok(Self {
            max_sampling_inputs,
            vocab_size,
            top_k,
        })
    }

    pub fn validate(self) {
        assert!(self.max_sampling_inputs > 0);
        assert!(self.vocab_size > 0);
        assert!(self.top_k > 0);
        assert!(self.top_k <= self.vocab_size);
        assert!(
            i32::try_from(self.vocab_size).is_ok(),
            "top-k sampling token ID must fit the signed token-buffer ABI"
        );
        (self.max_sampling_inputs as usize)
            .checked_mul(self.vocab_size as usize)
            .expect("top-k sampling logits element count must fit usize");
    }

    pub fn max_shape(self) -> TopKSamplingShape {
        TopKSamplingShape {
            num_active_sampling_inputs: self.max_sampling_inputs,
            num_total_sampling_inputs: self.max_sampling_inputs,
            vocab_size: self.vocab_size,
            top_k: self.top_k,
        }
    }

    pub fn active_top_k(self, config: &SamplerConfig) -> Result<u32, ModelExecutorError> {
        let top_k = top_k_sampling_len(config, self.vocab_size as usize)? as u32;
        assert!(
            top_k <= self.top_k,
            "top-k sampling active top_k={} exceed capacity={}",
            top_k,
            self.top_k
        );
        Ok(top_k)
    }

    pub fn active_shape(self, configs: &[SamplerConfig]) -> Result<TopKSamplingShape, ModelExecutorError> {
        assert!(!configs.is_empty(), "top-k sampling requires inputs");
        assert!(
            configs.len() <= self.max_sampling_inputs as usize,
            "top-k sampling active inputs={} exceed max={}",
            configs.len(),
            self.max_sampling_inputs
        );
        let num_active_sampling_inputs = configs.len() as u32;
        let top_k = configs.iter().try_fold(0, |top_k, config| {
            self.active_top_k(config).map(|value| top_k.max(value))
        })?;
        Ok(TopKSamplingShape {
            num_active_sampling_inputs,
            num_total_sampling_inputs: num_active_sampling_inputs,
            vocab_size: self.vocab_size,
            top_k,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopKSamplingLogitsDtype {
    Float32,
    Bfloat16,
}

fn top_k_sampling_len(config: &SamplerConfig, vocab_size: usize) -> Result<usize, ModelExecutorError> {
    if vocab_size == 0 {
        return Err(ModelExecutorError::custom(
            "top-k sampling requires non-empty vocab_size",
        ));
    }
    if config.is_greedy() {
        return Ok(1);
    }
    if config.top_k == 0 || config.top_k > vocab_size {
        return Err(ModelExecutorError::custom(format!(
            "top-k sampling requires bounded top_k with 0 < top_k <= vocab_size, got top_k={} vocab_size={vocab_size}",
            config.top_k
        )));
    }
    if config.top_k > MAX_TOP_K {
        return Err(ModelExecutorError::custom(format!(
            "top-k sampling supports top_k <= {}, got top_k={}",
            MAX_TOP_K, config.top_k
        )));
    }
    Ok(config.top_k)
}

#[cfg(test)]
mod tests {
    use inference_executor_core::sampling::SamplerConfig;

    use super::TopKSamplingBounds;

    #[test]
    fn test_top_k_sampling_bounds_uses_full_topk_contract() {
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 32,
            top_p: 0.8,
            seed: 7,
        };

        let bounds = TopKSamplingBounds::from_config(&config, 2, 1000).unwrap();
        let shape = bounds.max_shape();

        assert_eq!(shape.top_k, 32);
        assert_eq!(shape.top_k_count(), 64);
    }

    #[test]
    fn test_top_k_sampling_bounds_keeps_greedy_as_specialization() {
        let config = SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
        };

        let bounds = TopKSamplingBounds::from_config(&config, 4, 1000).unwrap();

        assert_eq!(bounds.top_k, 1);
    }

    #[test]
    #[should_panic(expected = "top-k sampling token ID must fit the signed token-buffer ABI")]
    fn test_bounds_rejects_vocab_outside_i32_token_domain() {
        TopKSamplingBounds {
            max_sampling_inputs: 1,
            vocab_size: i32::MAX as u32 + 1,
            top_k: 1,
        }
        .validate();
    }

    #[test]
    fn test_active_shape_uses_batch_max_top_k() {
        let capacity = SamplerConfig {
            temperature: 1.0,
            top_k: 64,
            top_p: 1.0,
            seed: 7,
        };
        let small = SamplerConfig { top_k: 8, ..capacity };
        let greedy = SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            ..capacity
        };
        let bounds = TopKSamplingBounds::from_config(&capacity, 4, 1000).unwrap();

        let mixed = bounds.active_shape(&[small, capacity, greedy]).unwrap();
        let uniform = bounds.active_shape(&[small, small]).unwrap();

        assert_eq!(mixed.num_active_sampling_inputs, 3);
        assert_eq!(mixed.top_k, 64);
        assert_eq!(uniform.top_k, 8);
    }

    #[test]
    fn test_top_k_sampling_rejects_unbounded_nucleus_without_topk() {
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.8,
            seed: 42,
        };

        let err = TopKSamplingBounds::from_config(&config, 1, 1000).unwrap_err();

        assert!(err.to_string().contains("bounded top_k"));
    }
}
