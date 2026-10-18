pub fn ple_core(config: &Qwen4xModelConfig) -> PLECore {
    let t = &config.text_config;
    PLECore {
        num_channels: t.hc_count as u32,
        hidden_dim: t.hidden_size as u32,
        embedding_dim: t.ple_embed_dim as u32,
        conv_kernel_size: t.ple_conv_kernel_size as u32,
        conv_dilation: t.ngram_size as u32,
        norm_eps: t.rms_norm_eps,
    }
}
pub fn ngram_core(config: &Qwen4xModelConfig) -> Result<NGramEmbeddingCore, ModelExecutorError> {
    let t = &config.text_config;
    Ok(NGramEmbeddingCore {
        vocab_size: t.vocab_size as u32,
        eos_token_id: config.eos_token_id()?,
        ngram_size: t.ngram_size as u32,
        heads_per_ngram: t.heads_per_ngram as u32,
        embedding_dim: t.ple_embed_dim as u32,
    })
}

struct ResidentGroup {
    row_start: u32,
    num_rows: u32,
    layout: QuantizedAffineLayout,
    weights: AffineWeights,
}
impl ResidentGroup {
    fn shard(&self) -> ngram_embedding::Shard<'_> {
        ngram_embedding::Shard {
            row_start: self.row_start,
            num_rows: self.num_rows,
            weights: ngram_embedding::TableWeights::Affine {
                bits: self.layout.bits,
                group_size: self.layout.group_size,
                weight: &self.weights.weight,
                scales: &self.weights.scales,
                biases: &self.weights.biases,
            },
        }
    }
}
struct PLEWeightsOwned {
    key: AffineWeights,
    value: AffineWeights,
    key_norm: Buffer,
    query_norm: Buffer,
    conv_norm: Buffer,
    conv: Buffer,
    table: Vec<ResidentGroup>,
}
impl PLEWeightsOwned {
    fn bindings(&self) -> PLEWeights<'_> {
        PLEWeights {
            key: self.key.bindings(),
            value: self.value.bindings(),
            key_norm: &self.key_norm,
            query_norm: &self.query_norm,
            conv_norm: &self.conv_norm,
            conv: &self.conv,
        }
    }
    fn load(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen4xModelConfig,
        ple_index: usize,
        bindings: &Qwen4xPLEWeightBindings,
    ) -> Result<Self, ModelExecutorError> {
        let core = ple_core(config);
        let ngram = ngram_core(config)?;
        let parameters = HashParameters::new(
            ngram,
            config.text_config.ngram_vocab_size_base,
            ple_index as u32,
            config.text_config.seed,
        );
        let rows = (parameters.num_rows() as usize).div_ceil(config.text_config.make_ngram_vocab_size_divisible_by)
            * config.text_config.make_ngram_vocab_size_divisible_by;
        let parts = bindings.table.len();
        if parts > rows {
            return Err(ModelExecutorError::custom("Qwen4x PLE has more shards than table rows"));
        }
        let sizes: Vec<_> = (0..parts)
            .map(|i| rows / parts + usize::from(i < rows % parts))
            .collect();
        let mut table = Vec::new();
        let mut row_start = 0usize;
        // Stream each checkpoint shard into its final resident allocation. No full-table join exists.
        for start in (0..parts).step_by(parts.div_ceil(8)) {
            let end = (start + parts.div_ceil(8)).min(parts);
            let names: Vec<_> = bindings.table[start..end].iter().map(|b| b.weight.as_str()).collect();
            let layout = uniform_layout(config, &names, "Qwen4x PLE resident group")?;
            let head_dim = ngram.head_dim() as usize;
            if !head_dim.is_multiple_of(layout.group_size as usize) {
                return Err(ModelExecutorError::custom(
                    "Qwen4x PLE head dimension must align to its quantization group",
                ));
            }
            let num_rows = sizes[start..end].iter().sum::<usize>();
            let row_weight_bytes = head_dim * layout.bits as usize / 8;
            let row_affine_bytes = head_dim / layout.group_size as usize * 2;
            let allocate = |width: usize| {
                Buffer::new_uninit(
                    device,
                    num_rows
                        .checked_mul(width)
                        .expect("PLE resident group byte count must fit usize"),
                )
            };
            let weights = AffineWeights {
                weight: allocate(row_weight_bytes),
                scales: allocate(row_affine_bytes),
                biases: allocate(row_affine_bytes),
            };
            let mut dst_row = 0;
            for shard in start..end {
                for (name, dtype, width, destination) in [
                    (
                        &bindings.table[shard].weight,
                        safetensors::Dtype::U32,
                        row_weight_bytes,
                        &weights.weight,
                    ),
                    (
                        &bindings.table[shard].scales,
                        safetensors::Dtype::BF16,
                        row_affine_bytes,
                        &weights.scales,
                    ),
                    (
                        &bindings.table[shard].biases,
                        safetensors::Dtype::BF16,
                        row_affine_bytes,
                        &weights.biases,
                    ),
                ] {
                    let tensor = store.tensor_bytes(name, dtype)?;
                    validate_shape(name, tensor.shape(), &[sizes[shard], width / (dtype.bitsize() / 8)])?;
                    destination.write_bytes(dst_row * width, tensor.data());
                    store.unload_all();
                }
                dst_row += sizes[shard];
            }
            table.push(ResidentGroup {
                row_start: row_start.try_into().expect("PLE row starts must fit u32"),
                num_rows: num_rows.try_into().expect("PLE group rows must fit u32"),
                layout,
                weights,
            });
            row_start += num_rows;
        }
        let norm = |store: &mut SafeTensorStore, name: &str| {
            load_qwen3x_norm_weight(device, store, name, &[core.residual_dim() as usize])
        };
        let raw_conv = store.tensor_bytes(&bindings.conv, safetensors::Dtype::BF16)?;
        validate_shape(
            &bindings.conv,
            raw_conv.shape(),
            &[core.residual_dim() as usize, core.conv_kernel_size as usize, 1],
        )?;
        Ok(Self {
            key: AffineWeights::load(
                device,
                store,
                &bindings.key,
                uniform_layout(config, &[&bindings.key.weight], "Qwen4x PLE key")?,
                core.residual_dim() as usize,
                core.embedding_dim as usize,
            )?,
            value: AffineWeights::load(
                device,
                store,
                &bindings.value,
                uniform_layout(config, &[&bindings.value.weight], "Qwen4x PLE value")?,
                core.hidden_dim as usize,
                core.embedding_dim as usize,
            )?,
            key_norm: norm(store, &bindings.key_norm)?,
            query_norm: norm(store, &bindings.query_norm)?,
            conv_norm: norm(store, &bindings.conv_norm)?,
            conv: Buffer::from_slice(device, raw_conv.data()),
            table,
        })
    }
}

struct Qwen4xPLE {
    compute: PLE,
    weights: Option<PLEWeightsOwned>,
    scratch: PLEScratch,
    ple_index: usize,
}
impl Qwen4xPLE {
    fn new(
        device: &Device,
        config: &Qwen4xModelConfig,
        bindings: &Qwen4xPLEWeightBindings,
        ple_index: usize,
        max_tokens: u32,
    ) -> Result<Self, ModelExecutorError> {
        let core = ple_core(config);
        let ngram = ngram_core(config)?;
        let parameters = HashParameters::new(
            ngram,
            config.text_config.ngram_vocab_size_base,
            ple_index as u32,
            config.text_config.seed,
        );
        Ok(Self {
            compute: PLE::new(
                device,
                core,
                ngram,
                parameters,
                PLEMetalConfig {
                    key: uniform_layout(config, &[&bindings.key.weight], "Qwen4x PLE key")?,
                    value: uniform_layout(config, &[&bindings.value.weight], "Qwen4x PLE value")?,
                    io_dtype: Dtype::Bfloat16,
                    table_scale: 1.0,
                },
            ),
            weights: None,
            scratch: PLEScratch::new(device, core, ngram, max_tokens),
            ple_index,
        })
    }
    fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen4xModelConfig,
        bindings: &Qwen4xPLEWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        self.weights = Some(PLEWeightsOwned::load(device, store, config, self.ple_index, bindings)?);
        Ok(())
    }
    fn prepare<'a, R: Recorder<'a, Operator = ReplayOp<'a>>>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        state: &'a Qwen4xState,
    ) {
        let weights = self.weights.as_ref().expect("Qwen4x PLE weights must be loaded");
        let shards: Vec<_> = weights.table.iter().map(ResidentGroup::shard).collect();
        self.compute.prepare(
            recorder,
            PLEPrepareInput {
                num_total_tokens,
                num_active_tokens,
                metadata: state.ple_metadata(),
                scratch: &self.scratch,
                shards: &shards,
                weights: weights.bindings(),
            },
        );
    }
    fn apply<'a, R: Recorder<'a, Operator = ReplayOp<'a>>>(
        &'a self,
        recorder: &mut R,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        residual: &'a Buffer,
        state: &'a Qwen4xState,
    ) {
        self.compute.apply(
            recorder,
            PLEApplyInput {
                num_total_tokens,
                num_active_tokens,
                metadata: state.ple_metadata(),
                scratch: &self.scratch,
                residual,
                output: residual,
                states: state.ple_states(self.ple_index),
                weights: self
                    .weights
                    .as_ref()
                    .expect("Qwen4x PLE weights must be loaded")
                    .bindings(),
            },
        );
    }
}
