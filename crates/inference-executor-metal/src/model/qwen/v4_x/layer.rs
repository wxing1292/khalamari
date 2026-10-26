struct Qwen4xLayer {
    attention: Qwen4xAttention,
    attention_residual: Option<GRWeights>,
    mlp: Qwen4xMLP,
    mlp_residual: Option<GRWeights>,
    ple: Option<Qwen4xPLE>,
}
#[allow(clippy::upper_case_acronyms)]
enum Qwen4xAttention {
    GDN(Qwen3xGDN),
    QSA {
        backend: Box<QSA>,
        weights: Option<QSAWeights>,
        core: GQACore,
        indexer_core: QSAIndexerCore,
        cache_index: usize,
    },
}

const QWEN4X_MAIN_NUM_ACTIVE_TOKENS: ReplayParameterKey = ReplayParameterKey::new("qwen4x.main.num_active_tokens");

pub struct Qwen4xMain {
    layers: Vec<Qwen4xLayer>,
    residual: GatedResidual,
    mixer: Option<GRWeights>,
    gr_scratch: GatedResidualScratch,
    mixed: Buffer,
    branch: Buffer,
    gate: Buffer,
    channels: Buffer,
    expand: inference_backend_metal::operators::repeat_rows::Compute,
    qsa_scratch: crate::attn::qsa::scratch::QSAScratch,
    replay_bucket_policy: ReplayBucketPolicy,
}

pub struct Qwen4xMainArgs<'a> {
    pub hidden_input: &'a Buffer,
    pub hidden_output: &'a Buffer,
    pub pages: &'a Buffer,
    pub state: &'a Qwen4xState,
    pub gdn: &'a Qwen3xGDNState,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Qwen4xMainReplayKey {
    num_total_tokens: u32,
    num_total_gdn_requests: u32,
    gdn: crate::attn::gdn::backend::GDNReplayTopology,
    sdpa: crate::attn::qsa::sdpa::QSASDPAShape,
    indexer: qsa_indexer::Shape,
    attention: Box<[crate::attn::qsa::backend::QSAReplayTopology]>,
    mlp: Box<[Qwen4xMLPReplayTopology]>,
    ple: Box<[crate::ple::backend::PLEReplayTopology]>,
    lookup: Box<[ngram_embedding::KernelKind]>,
}

impl Qwen4xMain {
    pub fn new(
        device: &Device,
        config: &Qwen4xModelConfig,
        bindings: &Qwen4xModelWeightBindings,
        max_tokens: u32,
        gdn: &Qwen3xGDNState,
        state: &Qwen4xState,
    ) -> Result<Self, ModelExecutorError> {
        let core = gr_core(config);
        core.validate();
        let mut layers = Vec::new();
        let mut gdn_index = 0;
        let mut qsa_index = 0;
        let mut ple_index = 0;
        let mlp_scratch = Qwen4xMLPScratch::new(device, config, &bindings.layers[0].mlp, max_tokens as usize)?;
        let mut qsa_scratch = None;
        let mut boundaries = gdn.replay_token_topology_boundaries().into_vec();
        for (index, bindings) in bindings.layers.iter().enumerate() {
            let attention = match &bindings.attention {
                Qwen4xAttentionWeightBindings::GDN(bindings) => {
                    let (core, metal) = gdn_configs(config, index, bindings)?;
                    let attention = Qwen3xGDN::new(
                        gdn_index,
                        core,
                        metal,
                        Rc::clone(gdn.backend()),
                        Rc::clone(gdn.scratch()),
                        Rc::clone(gdn.request_state_resources()),
                    );
                    gdn_index += 1;
                    Qwen4xAttention::GDN(attention)
                },
                Qwen4xAttentionWeightBindings::QSA { main, indexer } => {
                    let (core, metal, indexer_core, indexer_metal) = qsa_configs(config, index, main, indexer)?;
                    let backend = Box::new(QSA::new(
                        device,
                        core.clone(),
                        metal,
                        indexer_core,
                        indexer_metal,
                        state.page_table_layout(),
                    ));
                    boundaries.extend(backend.replay_token_topology_boundaries());
                    if qsa_scratch.is_none() {
                        qsa_scratch = Some(backend.new_scratch(state.max_sdpa_shape(), state.max_indexer_shape()));
                    }
                    let attention = Qwen4xAttention::QSA {
                        backend,
                        weights: None,
                        core,
                        indexer_core,
                        cache_index: qsa_index,
                    };
                    qsa_index += 1;
                    attention
                },
            };
            let mlp = Qwen4xMLP::new(device, config, index, &bindings.mlp, &mlp_scratch)?;
            boundaries.extend(mlp.replay_topology_boundaries());
            let ple = bindings
                .ple
                .as_ref()
                .map(|bindings| {
                    let ple = Qwen4xPLE::new(device, config, bindings, ple_index, max_tokens)?;
                    ple_index += 1;
                    boundaries.extend(ple.compute.replay_token_topology_boundaries());
                    Ok::<_, ModelExecutorError>(ple)
                })
                .transpose()?;
            layers.push(Qwen4xLayer {
                attention,
                attention_residual: None,
                mlp,
                mlp_residual: None,
                ple,
            });
        }
        let hidden = config.text_config.hidden_size;
        let bf16 = |width: usize| {
            Buffer::new_zeroed_elements(
                device,
                (max_tokens as usize)
                    .checked_mul(width)
                    .expect("Qwen4x scratch elements must fit usize"),
                Dtype::Bfloat16,
            )
        };
        Ok(Self {
            layers,
            residual: GatedResidual::new(device, core),
            mixer: None,
            gr_scratch: GatedResidualScratch::new(device, core, max_tokens as usize),
            mixed: bf16(hidden),
            branch: bf16(hidden),
            gate: bf16(config.text_config.hc_count),
            channels: bf16(core.residual_dim()),
            expand: inference_backend_metal::operators::repeat_rows::Compute::new(
                device,
                hidden as u32,
                config.text_config.hc_count as u32,
            ),
            qsa_scratch: qsa_scratch.expect("Qwen4x requires QSA layers"),
            replay_bucket_policy: ReplayBucketPolicy::with_topology_boundaries(max_tokens, &boundaries),
        })
    }
    pub fn load_weights(
        &mut self,
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen4xModelConfig,
        bindings: &Qwen4xModelWeightBindings,
    ) -> Result<(), ModelExecutorError> {
        for (layer, bindings) in self.layers.iter_mut().zip(&bindings.layers) {
            layer.attention_residual = Some(GRWeights::load(device, store, config, &bindings.attention_residual)?);
            layer.mlp_residual = Some(GRWeights::load(device, store, config, &bindings.mlp_residual)?);
            layer.mlp.load_weights(device, store, bindings.mlp.clone())?;
            match (&mut layer.attention, &bindings.attention) {
                (Qwen4xAttention::GDN(gdn), Qwen4xAttentionWeightBindings::GDN(bindings)) => {
                    gdn.load_weights(device, store, bindings.clone())?
                },
                (
                    Qwen4xAttention::QSA {
                        weights,
                        core,
                        indexer_core,
                        ..
                    },
                    Qwen4xAttentionWeightBindings::QSA { main, indexer },
                ) => {
                    *weights = Some(QSAWeights::load(
                        device,
                        store,
                        config,
                        core,
                        *indexer_core,
                        main,
                        indexer,
                    )?)
                },
                _ => unreachable!("Qwen4x layer bindings match the initialized model"),
            }
            if let Some(ple) = &mut layer.ple {
                ple.load_weights(device, store, config, bindings.ple.as_ref().unwrap())?;
            }
            store.unload_all();
        }
        self.mixer = Some(GRWeights::load(device, store, config, &bindings.mixer)?);
        Ok(())
    }
    pub fn unload_weights(&mut self) {
        self.mixer.take();
        for layer in self.layers.iter_mut().rev() {
            layer.attention_residual.take();
            layer.mlp_residual.take();
            layer.mlp.unload_weights();
            match &mut layer.attention {
                Qwen4xAttention::GDN(gdn) => gdn.unload_weights(),
                Qwen4xAttention::QSA { weights, .. } => {
                    weights.take();
                },
            }
            if let Some(ple) = &mut layer.ple {
                ple.weights.take();
            }
        }
    }
    pub fn unload_state(&mut self) {
        for layer in &mut self.layers {
            if let Qwen4xAttention::GDN(gdn) = &mut layer.attention {
                gdn.unload_state();
            }
        }
    }
    pub fn load_state(&mut self, state: &Qwen3xGDNState) {
        for layer in &mut self.layers {
            if let Qwen4xAttention::GDN(gdn) = &mut layer.attention {
                gdn.load_state(state);
            }
        }
    }
    pub fn channels(&self) -> &Buffer {
        &self.channels
    }
    pub fn capacity(&self, num_active_tokens: u32) -> u32 {
        self.replay_bucket_policy.capacity(num_active_tokens)
    }
    pub fn ple_scratch(&self, index: usize) -> &PLEScratch {
        &self
            .layers
            .iter()
            .filter_map(|l| l.ple.as_ref())
            .nth(index)
            .expect("PLE ordinal must exist")
            .scratch
    }
    pub fn replay_arguments(&self, input: &Qwen4xMainArgs<'_>) -> ReplayArguments {
        let batch = input.state.batch();
        let mut args =
            ReplayArguments::new().with_u32(QWEN4X_MAIN_NUM_ACTIVE_TOKENS, batch.active.sdpa.num_active_tokens);
        batch
            .active
            .add_replay_arguments(batch.indexer_shape, batch.sdpa.config().top_k, &mut args);
        input.gdn.add_private_replay_arguments(&mut args);
        args
    }
    fn read<'a>(
        &'a self,
        recorder: &mut ReplayRecorder,
        weights: &'a GRWeights,
        num_total_tokens: u32,
        num_active_tokens: ReplayU32,
        output: &'a Buffer,
    ) {
        self.residual.read(
            recorder,
            ReadInput {
                num_total_tokens,
                num_active_tokens,
                residual: &self.channels,
                hidden_state: output,
                scratch: &self.gr_scratch,
                weights: weights.read(),
                write_gate: weights.write.as_ref().map(|weight| {
                    gated_residual::WriteGate {
                        weight,
                        output: &self.gate,
                    }
                }),
            },
        );
    }
    fn write(&self, recorder: &mut ReplayRecorder, num_total_tokens: u32, num_active_tokens: ReplayU32) {
        self.residual.write(
            recorder,
            WriteInput {
                num_total_tokens,
                num_active_tokens,
                residual: &self.channels,
                gate: &self.gate,
                hidden_state: &self.branch,
                output: &self.channels,
            },
        );
    }
}
impl ReplayComponent for Qwen4xMain {
    type Key = Qwen4xMainReplayKey;
    type Input<'a> = Qwen4xMainArgs<'a>;
    fn replay_key(&self, input: &Self::Input<'_>) -> Self::Key {
        let batch = input.state.batch();
        let num_total_tokens = batch.sdpa.shape().num_total_tokens;
        Qwen4xMainReplayKey {
            num_total_tokens,
            num_total_gdn_requests: input.gdn.metadata().replay_shape().num_total_reqs,
            gdn: input.gdn.replay_topology(),
            sdpa: batch.sdpa.shape(),
            indexer: batch.indexer_shape,
            attention: self
                .layers
                .iter()
                .filter_map(|l| {
                    match &l.attention {
                        Qwen4xAttention::QSA { backend, .. } => Some(backend.replay_topology(num_total_tokens)),
                        _ => None,
                    }
                })
                .collect(),
            mlp: self
                .layers
                .iter()
                .map(|l| l.mlp.replay_topology(num_total_tokens))
                .collect(),
            ple: self
                .layers
                .iter()
                .filter_map(|l| l.ple.as_ref().map(|p| p.compute.replay_topology(num_total_tokens)))
                .collect(),
            lookup: self
                .layers
                .iter()
                .filter_map(|l| l.ple.as_ref())
                .map(|p| {
                    let shards: Vec<_> = p
                        .weights
                        .as_ref()
                        .expect("Qwen4x PLE weights must be loaded")
                        .table
                        .iter()
                        .map(ResidentGroup::shard)
                        .collect();
                    p.compute.lookup_topology(num_total_tokens, &shards)
                })
                .collect(),
        }
    }
    fn record<'a>(&'a self, recorder: &mut ReplayRecorder, input: &Self::Input<'a>) {
        let batch = input.state.batch();
        let num_total_tokens = batch.sdpa.shape().num_total_tokens;
        let num_active_tokens = ReplayU32::Parameter(QWEN4X_MAIN_NUM_ACTIVE_TOKENS);
        // Token-only preparation precedes layer 0. This initial schedule executes it serially.
        for layer in &self.layers {
            if let Some(ple) = &layer.ple {
                ple.prepare(recorder, num_total_tokens, num_active_tokens, input.state);
            }
        }
        recorder.record_with_barrier_before(ReplayOp::opaque(self.expand.invoke(
            num_total_tokens,
            num_active_tokens,
            input.hidden_input,
            &self.channels,
        )));
        for layer in &self.layers {
            if let Some(ple) = &layer.ple {
                ple.apply(
                    recorder,
                    num_total_tokens,
                    num_active_tokens,
                    &self.channels,
                    input.state,
                );
            }
            self.read(
                recorder,
                layer
                    .attention_residual
                    .as_ref()
                    .expect("Qwen4x GR weights must be loaded"),
                num_total_tokens,
                num_active_tokens,
                &self.mixed,
            );
            match &layer.attention {
                Qwen4xAttention::GDN(gdn) => {
                    gdn.record(
                        recorder,
                        &self.mixed,
                        &self.branch,
                        input.gdn.metadata(),
                        num_active_tokens,
                    )
                },
                Qwen4xAttention::QSA {
                    backend,
                    weights,
                    cache_index,
                    ..
                } => {
                    let mut scratch = self.qsa_scratch.bindings();
                    scratch.sdpa_shape = batch.sdpa.shape();
                    scratch.indexer_shape = batch.indexer_shape;
                    backend.record(
                        recorder,
                        QSAInput {
                            batch_metadata: &batch.sdpa,
                            indexer_metadata: &batch.indexer,
                            page_table_index: ReplayU32::Fixed(*cache_index as u32),
                            hidden_state: &self.mixed,
                            next_hidden_state: &self.branch,
                            kv_cache: QSAKVCacheBindings {
                                pages: input.pages,
                                page_ids: input.state.page_ids(),
                            },
                            indexer_cache: QSAIndexerCacheBindings {
                                pages: input.pages,
                                page_ids: input.state.index_page_ids(*cache_index),
                                raw_states: input.state.index_states(*cache_index),
                            },
                            weights: weights.as_ref().expect("Qwen4x QSA weights must be loaded").bindings(),
                            scratch,
                            active: QSAReplay::parameters(),
                        },
                    );
                },
            }
            self.write(recorder, num_total_tokens, num_active_tokens);
            self.read(
                recorder,
                layer.mlp_residual.as_ref().unwrap(),
                num_total_tokens,
                num_active_tokens,
                &self.mixed,
            );
            layer
                .mlp
                .record(recorder, &self.mixed, &self.branch, num_total_tokens, num_active_tokens);
            self.write(recorder, num_total_tokens, num_active_tokens);
        }
        self.read(
            recorder,
            self.mixer.as_ref().expect("Qwen4x final mixer weights must be loaded"),
            num_total_tokens,
            num_active_tokens,
            input.hidden_output,
        );
    }
}
