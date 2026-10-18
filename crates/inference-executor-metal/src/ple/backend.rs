use inference_backend_metal::components::ngram_embedding;
use inference_backend_metal::components::ple;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::metal::ReplayU32;
use inference_backend_metal::operators::affine_quantized;
use inference_executor_core::backend::recorder::Recorder;
use inference_executor_core::ngram_embedding::core::HashParameters;
use inference_executor_core::ngram_embedding::core::NGramEmbeddingCore;
use inference_executor_core::ple::core::PLECore;

use super::batch_metadata::PLEMetadataBuffers;
use super::scratch::PLEScratch;
use crate::def::quantized_affine::QuantizedAffineLayout;
use crate::def::quantized_affine::QuantizedAffineWeights;
use crate::def::replay_op::ReplayOp;

#[derive(Clone, Copy, Debug)]
pub struct PLEMetalConfig {
    pub key: QuantizedAffineLayout,
    pub value: QuantizedAffineLayout,
    pub io_dtype: Dtype,
    pub table_scale: f32,
}
#[derive(Clone, Copy)]
pub struct PLEWeights<'a> {
    pub key: QuantizedAffineWeights<'a>,
    pub value: QuantizedAffineWeights<'a>,
    /// Raw BF16 checkpoint offsets. Normalization applies `1 + weight` in FP32.
    pub key_norm: &'a Buffer,
    pub query_norm: &'a Buffer,
    pub conv_norm: &'a Buffer,
    pub conv: &'a Buffer,
}

pub struct PLEPrepareInput<'a, 's> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub metadata: &'a PLEMetadataBuffers,
    pub scratch: &'a PLEScratch,
    pub shards: &'s [ngram_embedding::Shard<'a>],
    pub weights: PLEWeights<'a>,
}
pub struct PLEProjectInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub embedding: &'a Buffer,
    pub scratch: &'a PLEScratch,
    pub weights: PLEWeights<'a>,
}
pub struct PLEApplyInput<'a> {
    pub num_total_tokens: u32,
    pub num_active_tokens: ReplayU32,
    pub metadata: &'a PLEMetadataBuffers,
    pub scratch: &'a PLEScratch,
    pub residual: &'a Buffer,
    pub output: &'a Buffer,
    pub states: &'a Buffer,
    pub weights: PLEWeights<'a>,
}
pub struct PLECommitInput<'a> {
    pub num_active_requests: ReplayU32,
    pub metadata: &'a PLEMetadataBuffers,
    pub scratch: &'a PLEScratch,
    pub states: &'a Buffer,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PLEReplayTopology {
    pub key: affine_quantized::KernelKind,
    pub value: affine_quantized::KernelKind,
}

/// PLE computation. The caller owns table storage, requests, state versions, and scheduling.
///
/// ```text
/// PREPARE, independent of residual:
///   token IDs [T] + per-request history [R,N-1] -> hash IDs [T,A]
///   resident table shards -> gather/dequant -> E [T,E]
///   SSD caller: host hash -> file rows -> staging/dequant -> E [T,E]
///   Both paths use PROJECT:
///   E -> key affine -> per-channel norm -> K [T,C,H]
///   E -> value affine -----------------> V [T,H]
///
/// APPLY, after the layer residual is ready:
///   X [T,C,H] -> per-channel norm -> Q [T,C,H]
///   Q,K -> dot/signed-sqrt/sigmoid -> gate [T,C,1]
///   gate * V -> Z [T,C,H] -> norm -> dilated depthwise conv -> SiLU
///   X + (Z + conv) -> X' [T,C,H]
///
/// COMMIT, after acceptance:
///   immutable source state + normalized Z prefix -> destination state
/// ```
/// Prepare/apply can use different streams only when the model owner establishes their dependency.
pub struct PLE {
    ngram_core: NGramEmbeddingCore,
    ngram: ngram_embedding::Compute,
    hash_parameters: [Buffer; 3],
    num_table_rows: u32,
    key: affine_quantized::Matmul,
    value: affine_quantized::Matmul,
    compute: ple::Compute,
    table_scale: f32,
}
impl PLE {
    pub fn new(
        device: &Device,
        core: PLECore,
        ngram_core: NGramEmbeddingCore,
        parameters: HashParameters,
        config: PLEMetalConfig,
    ) -> Self {
        core.validate();
        ngram_core.validate();
        assert_eq!(core.embedding_dim, ngram_core.embedding_dim);
        assert!(config.table_scale.is_finite());
        match config.io_dtype {
            Dtype::Bfloat16 => {},
            Dtype::Float32 => todo!("F32 PLE model boundary is not supported"),
            dtype => panic!("unsupported PLE model boundary dtype {dtype:?}"),
        }
        assert_eq!(parameters.multipliers.len(), ngram_core.ngram_size as usize);
        assert_eq!(parameters.head_vocab_sizes.len(), ngram_core.num_heads() as usize);
        assert_eq!(parameters.head_offsets.len(), ngram_core.num_heads() as usize);
        let mut end = 0_u32;
        for (&size, &offset) in parameters.head_vocab_sizes.iter().zip(&parameters.head_offsets) {
            assert!(size > 0);
            assert_eq!(offset, end);
            end = end.checked_add(size).expect("n-gram table rows must fit u32");
        }
        let affine = |layout: QuantizedAffineLayout, output_dim: u32| {
            affine_quantized::Matmul::new(
                device,
                layout.config(output_dim as usize, core.embedding_dim as usize, config.io_dtype),
            )
        };
        Self {
            ngram_core,
            ngram: ngram_embedding::Compute::new(
                device,
                ngram_embedding::Config {
                    ngram_size: ngram_core.ngram_size,
                    heads_per_ngram: ngram_core.heads_per_ngram,
                    embedding_dim: ngram_core.embedding_dim,
                    eos_token_id: ngram_core.eos_token_id,
                },
            ),
            hash_parameters: [
                Buffer::from_slice(device, &parameters.multipliers),
                Buffer::from_slice(device, &parameters.head_vocab_sizes),
                Buffer::from_slice(device, &parameters.head_offsets),
            ],
            num_table_rows: end,
            key: affine(config.key, core.residual_dim()),
            value: affine(config.value, core.hidden_dim),
            table_scale: config.table_scale,
            compute: ple::Compute::new(
                device,
                ple::Config {
                    num_channels: core.num_channels,
                    hidden_dim: core.hidden_dim,
                    conv_kernel_size: core.conv_kernel_size,
                    conv_dilation: core.conv_dilation,
                    norm_eps: core.norm_eps,
                },
            ),
        }
    }
    pub fn replay_topology(&self, num_total_tokens: u32) -> PLEReplayTopology {
        PLEReplayTopology {
            key: self.key.topology(num_total_tokens),
            value: self.value.topology(num_total_tokens),
        }
    }
    pub fn lookup_topology(
        &self,
        num_total_tokens: u32,
        shards: &[ngram_embedding::Shard<'_>],
    ) -> ngram_embedding::KernelKind {
        self.ngram.lookup_topology(num_total_tokens, shards)
    }
    pub fn replay_token_topology_boundaries(&self) -> Box<[u32]> {
        let mut boundaries = self.key.topology_boundaries().into_vec();
        boundaries.extend_from_slice(&self.value.topology_boundaries());
        boundaries.extend(self.ngram.lookup_token_topology_boundaries());
        boundaries.sort_unstable();
        boundaries.dedup();
        boundaries.into_boxed_slice()
    }
    pub fn prepare<'a, R>(&'a self, recorder: &mut R, input: PLEPrepareInput<'a, '_>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        assert_eq!(input.metadata.ngram_core(), self.ngram_core);
        let num_total_tokens = input.num_total_tokens;
        let num_active_tokens = input.num_active_tokens;
        let scratch = input.scratch;
        let last = input.shards.last().expect("PLE resident table requires shards");
        assert!(u64::from(last.row_start) + u64::from(last.num_rows) >= u64::from(self.num_table_rows));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.ngram.hash(
            num_total_tokens,
            input.metadata.max_requests(),
            num_active_tokens,
            ngram_embedding::HashBuffers {
                token_ids: input.metadata.token_ids(),
                previous_tokens: input.metadata.previous_tokens(),
                tokens: input.metadata.hash_tokens(),
                multipliers: &self.hash_parameters[0],
                head_vocab_sizes: &self.hash_parameters[1],
                head_offsets: &self.hash_parameters[2],
                row_ids: &scratch.row_ids,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.ngram.lookup(
            num_total_tokens,
            num_active_tokens,
            &scratch.row_ids,
            &scratch.embedding,
            self.table_scale,
            input.shards,
        )));
        self.project(
            recorder,
            PLEProjectInput {
                num_total_tokens,
                num_active_tokens,
                embedding: &scratch.embedding,
                scratch,
                weights: input.weights,
            },
        );
    }
    /// Project a decoded embedding from either resident lookup or an external row loader.
    pub fn project<'a, R>(&'a self, recorder: &mut R, input: PLEProjectInput<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_total_tokens = input.num_total_tokens;
        let num_active_tokens = input.num_active_tokens;
        let scratch = input.scratch;
        let weights = input.weights.key;
        recorder.record_with_barrier_before(ReplayOp::opaque(self.key.invoke(
            num_total_tokens,
            num_active_tokens,
            &scratch.key,
            0,
            input.embedding,
            0,
            weights.weight,
            weights.weight_offset_bytes,
            weights.scales,
            weights.scales_offset_bytes,
            weights.biases,
            weights.biases_offset_bytes,
        )));
        let weights = input.weights.value;
        recorder.record(ReplayOp::opaque(self.value.invoke(
            num_total_tokens,
            num_active_tokens,
            &scratch.value,
            0,
            input.embedding,
            0,
            weights.weight,
            weights.weight_offset_bytes,
            weights.scales,
            weights.scales_offset_bytes,
            weights.biases,
            weights.biases_offset_bytes,
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.norm(
            num_total_tokens,
            num_active_tokens,
            ple::NormBuffers {
                input: &scratch.key,
                weight: input.weights.key_norm,
                output: &scratch.normalized_key,
            },
        )));
    }
    pub fn apply<'a, R>(&'a self, recorder: &mut R, input: PLEApplyInput<'a>) -> &'a Buffer
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        let num_total_tokens = input.num_total_tokens;
        let num_active_tokens = input.num_active_tokens;
        let scratch = input.scratch;
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.norm(
            num_total_tokens,
            num_active_tokens,
            ple::NormBuffers {
                input: input.residual,
                weight: input.weights.query_norm,
                output: &scratch.normalized_query,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.gate(
            num_total_tokens,
            num_active_tokens,
            ple::GateBuffers {
                key: &scratch.normalized_key,
                query: &scratch.normalized_query,
                value: &scratch.value,
                output: &scratch.gated,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.norm(
            num_total_tokens,
            num_active_tokens,
            ple::NormBuffers {
                input: &scratch.gated,
                weight: input.weights.conv_norm,
                output: &scratch.normalized_gated,
            },
        )));
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.conv(
            num_total_tokens,
            num_active_tokens,
            input.metadata.num_state_slots(),
            ple::ConvBuffers {
                normalized: &scratch.normalized_gated,
                gated: &scratch.gated,
                residual: input.residual,
                weight: input.weights.conv,
                tokens: input.metadata.conv_tokens(),
                states: input.states,
                output: input.output,
            },
        )));
        input.output
    }
    pub fn commit<'a, R>(&'a self, recorder: &mut R, input: PLECommitInput<'a>)
    where
        R: Recorder<'a, Operator = ReplayOp<'a>>,
    {
        recorder.record_with_barrier_before(ReplayOp::opaque(self.compute.commit(
            input.metadata.max_tokens(),
            input.metadata.max_requests(),
            input.num_active_requests,
            input.metadata.num_state_slots(),
            ple::CommitBuffers {
                normalized: &input.scratch.normalized_gated,
                requests: input.metadata.requests(),
                accepted_tokens: input.metadata.accepted_tokens(),
                states: input.states,
            },
        )));
    }
}

#[cfg(test)]
#[path = "backend_test.rs"]
mod tests;
