use crate::metal::Dtype;

pub mod sliced_kv;
pub mod sliced_q;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MapThreadBlockConstants {
    pub max_q_tokens: u32,
    pub max_q_heads: u32,
    /// KV interval used to partition Map tasks. Sliced-Q also uses it as its staging width.
    /// Sliced-KV processes 16 KV tokens per SIMDgroup per step independently of this interval.
    pub kv_tokens_per_iteration: u32,
    pub required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KVCacheConstants {
    pub tokens_per_page: u32,
}

/// Algorithm and Q-token/head assignment within one Map threadblock.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MapKernelKind {
    SlicedKv,
    QTokensPerHead,
    QHeadsPerToken,
    QTokensAndHeads,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MapKernelConstants {
    pub kind: MapKernelKind,
    pub thread_block: MapThreadBlockConstants,
    pub kv_cache: KVCacheConstants,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReduceThreadBlockConstants {
    pub max_q_tokens: u32,
    pub max_q_heads: u32,
    pub required_threads: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReduceKernelConstants {
    pub thread_block: ReduceThreadBlockConstants,
}

/// One compatible GQA SDPA Map and Reduce execution variant.
///
/// `map.kind` identifies the Q-token/head assignment. Token capacity does not identify the
/// algorithm: sliced-Q variants can also process one Q token.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionVariant {
    pub map: MapKernelConstants,
    pub reduce: ReduceKernelConstants,
}

impl ExecutionVariant {
    pub fn sliced_kv(config: Config, kv_tokens_per_iteration: u32, required_threads: u32, max_q_heads: u32) -> Self {
        config.validate();
        let reduce_output_coordinates = 256 / config.head_dim.min(256);
        let reduce_max_q_heads = config.num_q_heads.min(reduce_output_coordinates).max(1);
        let reduce_max_q_tokens = (reduce_output_coordinates / reduce_max_q_heads).max(1);
        let execution = Self {
            map: MapKernelConstants {
                kind: MapKernelKind::SlicedKv,
                thread_block: MapThreadBlockConstants {
                    max_q_tokens: 1,
                    max_q_heads,
                    kv_tokens_per_iteration,
                    required_threads,
                },
                kv_cache: KVCacheConstants {
                    tokens_per_page: config.tokens_per_page,
                },
            },
            reduce: ReduceKernelConstants {
                thread_block: ReduceThreadBlockConstants {
                    max_q_tokens: reduce_max_q_tokens,
                    max_q_heads: reduce_max_q_heads,
                    required_threads: 256,
                },
            },
        };
        assert!(execution.supports(config));
        execution
    }

    pub fn sliced_q_per_head(
        config: Config,
        max_q_tokens: u32,
        kv_tokens_per_iteration: u32,
        max_q_heads: u32,
    ) -> Self {
        Self::sliced_q(
            config,
            MapKernelKind::QTokensPerHead,
            max_q_tokens,
            kv_tokens_per_iteration,
            max_q_heads,
        )
    }

    pub fn sliced_q(
        config: Config,
        kind: MapKernelKind,
        max_q_tokens: u32,
        kv_tokens_per_iteration: u32,
        max_q_heads: u32,
    ) -> Self {
        config.validate();
        assert!(matches!(max_q_tokens, 1 | 2 | 4 | 8 | 16 | 32));
        assert!(max_q_heads > 0 && max_q_heads <= config.q_heads_per_kv_head());
        let num_simdgroups = match kind {
            MapKernelKind::SlicedKv => panic!("SlicedKv uses its own constructor"),
            MapKernelKind::QTokensPerHead => {
                max_q_tokens
                    .div_ceil(8)
                    .checked_mul(max_q_heads)
                    .expect("GQA tile geometry must fit u32")
            },
            MapKernelKind::QHeadsPerToken => {
                max_q_heads
                    .div_ceil(8)
                    .checked_mul(max_q_tokens)
                    .expect("GQA tile geometry must fit u32")
            },
            MapKernelKind::QTokensAndHeads => {
                max_q_tokens
                    .checked_mul(max_q_heads)
                    .expect("GQA tile geometry must fit u32")
                    .div_ceil(8)
            },
        };
        let required_threads = num_simdgroups.checked_mul(32).expect("GQA thread count must fit u32");
        let execution = Self {
            map: MapKernelConstants {
                kind,
                thread_block: MapThreadBlockConstants {
                    max_q_tokens,
                    max_q_heads,
                    kv_tokens_per_iteration,
                    required_threads,
                },
                kv_cache: KVCacheConstants {
                    tokens_per_page: config.tokens_per_page,
                },
            },
            reduce: ReduceKernelConstants {
                thread_block: ReduceThreadBlockConstants {
                    max_q_tokens,
                    max_q_heads: 1,
                    required_threads,
                },
            },
        };
        assert!(execution.supports(config));
        execution
    }

    pub fn supports(self, config: Config) -> bool {
        let map = self.map.thread_block;
        let reduce = self.reduce.thread_block;
        if self.map.kv_cache.tokens_per_page != config.tokens_per_page
            || map.max_q_tokens == 0
            || map.max_q_heads == 0
            || map.max_q_heads > config.q_heads_per_kv_head()
            || map.kv_tokens_per_iteration == 0
            || map.required_threads == 0
            || reduce.max_q_tokens == 0
            || reduce.max_q_heads == 0
            || reduce.required_threads == 0
        {
            return false;
        }

        if self.map.kind == MapKernelKind::SlicedKv {
            let reduce_output_coordinates = 256 / config.head_dim.min(256);
            let expected_reduce_max_q_heads = config.num_q_heads.min(reduce_output_coordinates).max(1);
            let expected_reduce_max_q_tokens = (reduce_output_coordinates / expected_reduce_max_q_heads).max(1);
            return map.max_q_tokens == 1
                && map.max_q_heads <= 8
                && map.kv_tokens_per_iteration <= 1024
                && map.required_threads.is_power_of_two()
                && (32..=256).contains(&map.required_threads)
                && reduce.max_q_tokens == expected_reduce_max_q_tokens
                && reduce.max_q_heads == expected_reduce_max_q_heads
                && reduce.required_threads == 256;
        }

        let expected_threads = match self.map.kind {
            MapKernelKind::SlicedKv => unreachable!(),
            MapKernelKind::QTokensPerHead => map.max_q_tokens.div_ceil(8).checked_mul(map.max_q_heads),
            MapKernelKind::QHeadsPerToken => map.max_q_heads.div_ceil(8).checked_mul(map.max_q_tokens),
            MapKernelKind::QTokensAndHeads => {
                map.max_q_tokens
                    .checked_mul(map.max_q_heads)
                    .map(|q_token_heads| q_token_heads.div_ceil(8))
            },
        };
        let Some(expected_threads) = expected_threads.and_then(|groups| groups.checked_mul(32)) else {
            return false;
        };
        let profile = (config.head_dim, config.tokens_per_page);
        config.io_dtype == Dtype::Bfloat16
            && matches!(profile, (128, 8 | 16) | (256, 8 | 16 | 32))
            && matches!(map.max_q_tokens, 1 | 2 | 4 | 8 | 16 | 32)
            && matches!(map.kv_tokens_per_iteration, 8 | 16 | 32 | 64 | 128)
            && map.required_threads == expected_threads
            && map.required_threads <= 1024
            && reduce.max_q_tokens == map.max_q_tokens
            && reduce.max_q_heads == 1
            && reduce.required_threads == map.required_threads
    }
}

/// Static GQA SDPA workload facts used for capability filtering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub io_dtype: Dtype,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub tokens_per_page: u32,
}

impl Config {
    pub fn validate(self) {
        assert!(matches!(self.io_dtype, Dtype::Float32 | Dtype::Bfloat16));
        assert!(self.num_q_heads > 0);
        assert!(self.num_kv_heads > 0);
        assert_eq!(self.num_q_heads % self.num_kv_heads, 0);
        assert!(self.head_dim > 0);
        assert!(self.tokens_per_page > 0);
    }

    pub fn q_heads_per_kv_head(self) -> u32 {
        self.num_q_heads / self.num_kv_heads
    }
}

/// Backend-owned registry of legal GQA SDPA execution variants.
///
/// The registry performs only static capability filtering. The executor
/// selector owns dynamic workload selection and complete-candidate comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Registry {
    config: Config,
    variants: Vec<ExecutionVariant>,
}

impl Registry {
    pub fn new(config: Config) -> Self {
        config.validate();
        let mut variants = vec![sliced_kv_variant(config)];
        if supports_sliced_q(config) {
            let full_q_heads = config.q_heads_per_kv_head().min(max_sliced_q_heads(8));
            if !matches!((config.head_dim, config.tokens_per_page), (128, 8 | 16)) {
                let half_q_heads = config.q_heads_per_kv_head().div_ceil(2).min(max_sliced_q_heads(8));
                if half_q_heads != full_q_heads {
                    variants.push(sliced_q_variant(config, half_q_heads));
                }
            }
            variants.push(sliced_q_variant(config, full_q_heads));
            if config.head_dim == 256 {
                variants.push(ExecutionVariant::sliced_q_per_head(config, 8, 32, full_q_heads));
            }
        }
        Self::from_variants(config, variants)
    }

    /// Creates a registry from explicit backend variants.
    ///
    /// The executor lookup uses this constructor for its reachable variants.
    /// Backend benchmarks can use it to force one legal variant.
    pub fn from_variants(config: Config, variants: Vec<ExecutionVariant>) -> Self {
        config.validate();
        assert!(!variants.is_empty(), "GQA SDPA registry requires an execution variant");
        assert!(
            variants.iter().all(|variant| variant.supports(config)),
            "GQA SDPA registry contains an unsupported execution variant"
        );
        Self { config, variants }
    }

    pub fn config(&self) -> Config {
        self.config
    }

    pub fn variants(&self) -> &[ExecutionVariant] {
        &self.variants
    }

    pub fn max_q_tokens_per_map_task(&self) -> u32 {
        self.variants
            .iter()
            .map(|variant| variant.map.thread_block.max_q_tokens)
            .max()
            .expect("GQA SDPA registry requires an execution variant")
    }
}

fn supports_sliced_q(config: Config) -> bool {
    let profile = (config.head_dim, config.tokens_per_page);
    config.io_dtype == Dtype::Bfloat16
        && matches!(profile, (128, 8 | 16) | (256, 8 | 16 | 32))
        && config.q_heads_per_kv_head() <= 8
}

fn sliced_kv_variant(config: Config) -> ExecutionVariant {
    let kv_tokens_per_iteration = config.head_dim.clamp(32, 256).next_power_of_two();
    // SlicedKv partitions KV tiles across SIMDgroups. Four groups retain enough
    // parallelism without the shared-memory cost of the former eight-group map.
    let required_threads = kv_tokens_per_iteration.min(128);
    let max_q_heads = config.q_heads_per_kv_head().min(8);
    ExecutionVariant::sliced_kv(config, kv_tokens_per_iteration, required_threads, max_q_heads)
}

fn sliced_q_variant(config: Config, max_q_heads: u32) -> ExecutionVariant {
    let max_q_tokens = 8;
    ExecutionVariant::sliced_q_per_head(config, max_q_tokens, 16, max_q_heads)
}

fn max_sliced_q_heads(max_q_tokens: u32) -> u32 {
    256 / (max_q_tokens / 8 * 32)
}

#[cfg(test)]
mod tests {
    use super::Config;
    use super::Registry;
    use crate::metal::Dtype;

    fn config(head_dim: u32, q_heads_per_kv_head: u32, tokens_per_page: u32) -> Config {
        let num_kv_heads = 2;
        Config {
            io_dtype: Dtype::Bfloat16,
            num_q_heads: num_kv_heads * q_heads_per_kv_head,
            num_kv_heads,
            head_dim,
            tokens_per_page,
        }
    }

    #[test]
    fn test_registry_filters_static_execution_capabilities() {
        let d256_page32 = Registry::new(config(256, 6, 32));
        let d256_page16 = Registry::new(config(256, 6, 16));
        let d256_page8 = Registry::new(config(256, 6, 8));
        let d128_page16 = Registry::new(config(128, 8, 16));
        let d128_page8 = Registry::new(config(128, 8, 8));
        let unsupported = Registry::new(config(256, 8, 4));

        assert_eq!(d256_page32.variants().len(), 4);
        assert_eq!(d256_page16.variants().len(), 4);
        assert_eq!(d256_page8.variants().len(), 4);
        assert_eq!(d128_page16.variants().len(), 2);
        assert_eq!(d128_page8.variants().len(), 2);
        assert_eq!(unsupported.variants().len(), 1);
        for registry in [
            d256_page32,
            d256_page16,
            d256_page8,
            d128_page16,
            d128_page8,
            unsupported,
        ] {
            assert!(
                registry
                    .variants()
                    .iter()
                    .all(|execution| execution.supports(registry.config()))
            );
        }
    }
}
