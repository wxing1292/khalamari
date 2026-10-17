use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_executor_core::ngram_embedding::core::NGramEmbeddingCore;
use inference_executor_core::ple::core::PLECore;

/// Preparation outputs remain live through apply. Normalized Z remains live through accepted-prefix commit.
pub struct PLEScratch {
    pub row_ids: Buffer,
    pub embedding: Buffer,
    pub key: Buffer,
    pub normalized_key: Buffer,
    pub value: Buffer,
    pub normalized_query: Buffer,
    pub gated: Buffer,
    pub normalized_gated: Buffer,
}
impl PLEScratch {
    pub fn new(device: &Device, core: PLECore, ngram: NGramEmbeddingCore, max_tokens: u32) -> Self {
        core.validate();
        ngram.validate();
        assert!(max_tokens > 0);
        assert_eq!(core.embedding_dim, ngram.embedding_dim);
        let buffer = |width: u32, bytes: usize| {
            Buffer::new_zeroed(
                device,
                (max_tokens as usize)
                    .checked_mul(width as usize)
                    .and_then(|n| n.checked_mul(bytes))
                    .expect("PLE scratch bytes must fit usize"),
            )
        };
        let wide = core.residual_dim();
        Self {
            row_ids: buffer(ngram.num_heads(), 4),
            embedding: buffer(core.embedding_dim, 2),
            key: buffer(wide, 2),
            normalized_key: buffer(wide, 2),
            value: buffer(core.hidden_dim, 2),
            normalized_query: buffer(wide, 2),
            gated: buffer(wide, 2),
            normalized_gated: buffer(wide, 2),
        }
    }
}
