use std::collections::BTreeMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use half::bf16;
use inference_backend_metal::components::ReplayRecorder;
use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::ReplayArguments;
use inference_backend_metal::metal::ReplayProgram;
use inference_backend_metal::metal::Stream;
use inference_executor_core::checkpoint::SafeTensorStore;
use inference_executor_core::model::qwen::v3_5::LayerType;
use inference_executor_core::model::qwen::v3_5::Qwen35ModelConfig;
use inference_executor_core::model::qwen::v3_5::init_qwen35_model_config;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35LayerWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35MainWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::Qwen35ModelWeightBindings;
use inference_executor_core::model::qwen::v3_5::weight_layout::resolve_qwen35_model_weight_bindings;
use inference_executor_metal::attn::gqa::request_page_table::GQAPageTableLayout;
use inference_executor_metal::def::layer::ReplayLayer;
use inference_executor_metal::mlp::dense::scratch::DenseMLPScratch;
use inference_executor_metal::mlp::moe::scratch::MoEScratch;
use inference_executor_metal::model::page_arena::PageArena;
use inference_executor_metal::model::qwen::v3_5::QWEN35_PAGE_SIZE_BYTES;
use inference_executor_metal::model::qwen::v3_5::component_config::Qwen35MetalDefaults;
use inference_executor_metal::model::qwen::v3_5::component_config::derive_qwen35_dense_mlp_configs;
use inference_executor_metal::model::qwen::v3_5::component_config::derive_qwen35_gdn_configs;
use inference_executor_metal::model::qwen::v3_5::component_config::derive_qwen35_gqa_configs;
use inference_executor_metal::model::qwen::v3_5::component_config::derive_qwen35_moe_configs;
use inference_executor_metal::model::qwen::v3_5::component_config::qwen35_layer_counts;
use inference_executor_metal::model::qwen::v3_5::main::Qwen35Main;
use inference_executor_metal::model::qwen::v3_5::main::Qwen35MainArgs;
use inference_executor_metal::model::qwen::v3_5::main::Qwen35MainReplayKey;
use inference_executor_metal::model::qwen::v3_5::main::layer::Qwen35MainLayer;
use inference_executor_metal::model::qwen::v3_5::main::layer::Qwen35MainLayerInput;
use inference_executor_metal::model::qwen::v3_5::main::layer::Qwen35MainLayerScratch;
use inference_executor_metal::model::qwen::v3_x::state::Qwen3xGDNState;
use inference_executor_metal::model::qwen::v3_x::state::Qwen3xGQAState;
use inference_executor_metal::replay::Replay;

const DEFAULT_TOKENS: u32 = 1;
const DEFAULT_CONTEXT: u32 = 32;
const DEFAULT_MAX_TOKENS: usize = 128;
const CACHE_BLOCK_TOKENS: usize = 2048;

#[derive(Clone, Copy, Debug)]
struct BenchShape {
    num_tokens: u32,
    context: u32,
}

impl BenchShape {
    fn context_end(self, max_position_embeddings: usize) -> usize {
        let context_end = (self.context as usize)
            .checked_add(self.num_tokens as usize)
            .expect("qwen3.5 Main benchmark context length must fit usize");
        assert!(
            context_end <= max_position_embeddings,
            "qwen3.5 Main benchmark context end {context_end} exceeds model position capacity \
             {max_position_embeddings}"
        );
        u32::try_from(context_end).expect("qwen3.5 Main benchmark context end must fit u32");
        context_end
    }
}

#[derive(Clone, Copy, Debug)]
enum Case {
    Layer(usize),
    FirstLayers(usize),
    Layers { start: usize, end: usize },
    AllLayers,
}

impl Case {
    fn layer_indices(self, num_main_layers: usize) -> Vec<usize> {
        match self {
            Self::Layer(model_layer_index) => {
                assert!(
                    model_layer_index < num_main_layers,
                    "selected layer exceeds the model depth"
                );
                vec![model_layer_index]
            },
            Self::FirstLayers(count) => {
                assert!(
                    count <= num_main_layers,
                    "selected first-layer count exceeds the model depth"
                );
                (0..count).collect()
            },
            Self::Layers { start, end } => {
                assert!(start < end, "explicit layer range must not be empty");
                assert!(end <= num_main_layers, "explicit layer range exceeds the model depth");
                (start..end).collect()
            },
            Self::AllLayers => (0..num_main_layers).collect(),
        }
    }

    fn key(self) -> String {
        match self {
            Self::Layer(model_layer_index) => format!("layer{model_layer_index}"),
            Self::FirstLayers(count) => format!("first{count}"),
            Self::Layers { start, end } => format!("layers{start}-{end}"),
            Self::AllLayers => "all_layers".to_string(),
        }
    }
}

#[derive(Clone, Copy)]
enum Target {
    Main,
    MainLayers,
}

struct BenchArgs {
    model_dir: PathBuf,
    cases: Vec<Case>,
    shapes: Vec<BenchShape>,
    max_tokens: usize,
    iters: usize,
    warmup_iters: usize,
    runs: usize,
}

impl BenchArgs {
    fn parse(target: Target) -> Self {
        let mut args = Self {
            model_dir: PathBuf::new(),
            cases: match target {
                Target::Main => Vec::new(),
                Target::MainLayers => selected_cases(),
            },
            shapes: vec![BenchShape {
                num_tokens: DEFAULT_TOKENS,
                context: DEFAULT_CONTEXT,
            }],
            max_tokens: DEFAULT_MAX_TOKENS,
            iters: 200,
            warmup_iters: 50,
            runs: 3,
        };
        let mut num_tokens = None;
        let mut contexts = None;
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => print_help_and_exit(target),
                "--model-dir" => args.model_dir = PathBuf::from(next_arg(&mut iter, &arg)),
                "--cases" => {
                    assert!(
                        matches!(target, Target::MainLayers),
                        "qwen35_main has one implicit main case and does not accept --cases"
                    );
                    args.cases = parse_cases(&next_arg(&mut iter, &arg));
                },
                "--tokens" => num_tokens = Some(parse_u32_list(&next_arg(&mut iter, &arg), &arg)),
                "--contexts" => contexts = Some(parse_u32_list(&next_arg(&mut iter, &arg), &arg)),
                "--max-tokens" => args.max_tokens = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--iters" => args.iters = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--warmup-iters" => args.warmup_iters = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--runs" => args.runs = parse_usize_arg(&next_arg(&mut iter, &arg), &arg),
                "--bench" => {},
                other => panic!("unknown argument {other:?}; pass --help for usage"),
            }
        }
        let mut num_tokens = num_tokens.unwrap_or_else(|| vec![DEFAULT_TOKENS]);
        let mut contexts = contexts.unwrap_or_else(|| vec![DEFAULT_CONTEXT]);
        assert!(!num_tokens.is_empty(), "--tokens must include at least one value");
        assert!(!contexts.is_empty(), "--contexts must include at least one value");
        assert!(!args.model_dir.as_os_str().is_empty(), "--model-dir is required");
        assert!(args.max_tokens > 0, "--max-tokens must be positive");
        assert!(u32::try_from(args.max_tokens).is_ok(), "--max-tokens must fit u32");
        assert!(args.iters > 0, "--iters must be positive");
        assert!(args.runs > 0, "--runs must be positive");
        sort_unique(&mut num_tokens, "--tokens");
        sort_unique(&mut contexts, "--contexts");
        args.shapes = num_tokens
            .iter()
            .flat_map(|&num_tokens| {
                assert!(num_tokens > 0, "--tokens entries must be positive");
                assert!(
                    num_tokens as usize <= args.max_tokens,
                    "--tokens entries must not exceed --max-tokens"
                );
                contexts.iter().map(move |&context| BenchShape { num_tokens, context })
            })
            .collect();
        args
    }
}

struct FullMainFixture {
    stream: Stream,
    input: Buffer,
    output: Buffer,
    main: Replay<Qwen35Main>,
    gqa_state: Qwen3xGQAState,
    gdn_state: Qwen3xGDNState,
    pages: PageArena,
    max_tokens: usize,
    shape: BenchShape,
}

struct PreparedMainReplay {
    key: Qwen35MainReplayKey,
    arguments: ReplayArguments,
}

impl FullMainFixture {
    fn new(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        bindings: Qwen35MainWeightBindings,
        shape: BenchShape,
        max_tokens: usize,
    ) -> Self {
        let defaults = Qwen35MetalDefaults::from_quantization(config.quantization.as_ref())
            .expect("qwen3.5 Main benchmark requires supported quantization");
        let counts = qwen35_layer_counts(config).expect("qwen3.5 Main benchmark requires a valid layer schedule");
        assert!(counts.gqa > 0, "qwen3.5 Main benchmark requires GQA layers");
        assert!(counts.gdn > 0, "qwen3.5 Main benchmark requires GDN layers");
        let first_gqa_layer = first_layer_of_type(config, LayerType::FullAttention);
        let (gqa_core, gqa_metal) = derive_qwen35_gqa_configs(first_gqa_layer, &config.text_config, defaults)
            .expect("qwen3.5 Main benchmark requires valid GQA geometry");
        let num_page_ids_per_block = CACHE_BLOCK_TOKENS.div_ceil(gqa_metal.num_tokens_per_page(&gqa_core) as usize);
        let context_end = shape.context_end(config.text_config.max_position_embeddings);
        let num_blocks = context_end.div_ceil(CACHE_BLOCK_TOKENS);
        let num_cache_pages = counts
            .gqa
            .checked_mul(num_blocks)
            .and_then(|value| value.checked_mul(num_page_ids_per_block))
            .expect("qwen3.5 Main benchmark cache page count must fit usize");
        let gqa_state = Qwen3xGQAState::new(
            device,
            gqa_core,
            gqa_metal,
            GQAPageTableLayout {
                num_req_slots: 1,
                num_blocks: num_blocks
                    .try_into()
                    .expect("qwen3.5 Main benchmark block count must fit u32"),
                num_gqa_layers: counts
                    .gqa
                    .try_into()
                    .expect("qwen3.5 Main benchmark GQA layer count must fit u32"),
                num_page_ids_per_block: num_page_ids_per_block
                    .try_into()
                    .expect("qwen3.5 Main benchmark page count must fit u32"),
            },
            max_tokens,
            num_cache_pages,
        );
        write_gqa_page_ids(
            &gqa_state,
            counts.gqa,
            num_blocks,
            num_page_ids_per_block,
            num_cache_pages,
        );

        let gdn_layers = layer_indices_of_type(config, LayerType::GDN);
        let gdn_cores = gdn_layers
            .iter()
            .map(|&index| {
                derive_qwen35_gdn_configs(index, &config.text_config, defaults)
                    .expect("qwen3.5 Main benchmark requires valid GDN geometry")
                    .0
            })
            .collect::<Vec<_>>();
        let gdn_metal = derive_qwen35_gdn_configs(gdn_layers[0], &config.text_config, defaults)
            .expect("qwen3.5 Main benchmark requires valid GDN geometry")
            .1;
        let gdn_state = Qwen3xGDNState::new(
            device,
            Stream::new(device),
            &gdn_cores,
            gdn_metal,
            1,
            inference_executor_metal::attn::gdn::state_table::GDNStateCapacity::new(2, 1, 1, max_tokens),
            max_tokens,
            CACHE_BLOCK_TOKENS,
            num_cache_pages,
            QWEN35_PAGE_SIZE_BYTES,
        );
        let layer_scratch = Rc::new(Qwen35MainLayerScratch::new(
            device,
            max_tokens,
            config.text_config.hidden_size,
        ));
        let dense_scratch = dense_scratch(device, config, defaults, counts.has_dense_mlp, max_tokens);
        let moe_scratch = moe_scratch(device, config, defaults, counts.has_moe, max_tokens);
        let mut main = Qwen35Main::new(
            device,
            config,
            max_tokens,
            defaults,
            &gqa_state,
            &gdn_state,
            None,
            layer_scratch,
            dense_scratch.as_ref(),
            moe_scratch.as_ref(),
        )
        .unwrap_or_else(|error| panic!("unable to construct the Qwen3.5 Main owner: {error}"));
        main.load_weights(device, store, config, bindings)
            .unwrap_or_else(|error| panic!("unable to load the Qwen3.5 Main owner: {error}"));
        let num_total_tokens = main.capacity(shape.num_tokens);
        gqa_state.prepare_metadata(&[0], &[shape.context], &[0, shape.num_tokens], num_total_tokens);
        prepare_gdn_metadata(&gdn_state, shape.num_tokens, num_total_tokens);

        Self {
            stream: Stream::new(device),
            input: Buffer::from_slice(device, &hidden_fixture(max_tokens, config.text_config.hidden_size)),
            output: Buffer::new_zeroed_elements(
                device,
                max_tokens * config.text_config.hidden_size,
                inference_backend_metal::metal::Dtype::Bfloat16,
            ),
            main: Replay::new("qwen3.5 Main benchmark", main),
            gqa_state,
            gdn_state,
            pages: PageArena::new(device, num_cache_pages, QWEN35_PAGE_SIZE_BYTES),
            max_tokens,
            shape,
        }
    }

    fn prepare(&mut self) -> PreparedMainReplay {
        let input = Qwen35MainArgs {
            num_tokens: self.shape.num_tokens,
            hidden_input: &self.input,
            hidden_output: &self.output,
            gqa: self.gqa_state.metadata(),
            gqa_replay_topology: self.gqa_state.replay_topology(),
            gdn: self.gdn_state.metadata(),
            gdn_replay_topology: self.gdn_state.replay_topology(),
            pages: self.pages.buffer(),
        };
        let (expected_key, mut arguments) = self.main.component().prepare_replay(
            self.shape.num_tokens,
            self.gqa_state.metadata().replay_shape(),
            self.gqa_state.replay_topology(),
            self.gdn_state.metadata().replay_shape(),
            self.gdn_state.replay_topology(),
        );
        self.gqa_state.add_private_replay_arguments(&mut arguments);
        self.gdn_state.add_private_replay_arguments(&mut arguments);
        let runtime = &self.stream;
        let (key, _) = self.main.record(runtime, &input);
        assert_eq!(key, expected_key, "Main prepared and recorded replay keys must match");
        PreparedMainReplay { key, arguments }
    }

    fn run(&self, prepared: &PreparedMainReplay) {
        self.stream
            .submit_replay_with_arguments(self.main.replay(&prepared.key), &prepared.arguments)
            .wait();
    }

    fn print_gqa_split_plan(&self) {
        print_gqa_split_plan(&self.gqa_state, self.max_tokens, "qwen35-main");
    }
}

struct LayerRangeFixture {
    stream: Stream,
    input: Buffer,
    layers: Vec<Qwen35MainLayer>,
    gqa_state: Qwen3xGQAState,
    gdn_state: Qwen3xGDNState,
    pages: PageArena,
    has_gqa: bool,
    max_tokens: usize,
    shape: BenchShape,
}

impl LayerRangeFixture {
    fn new(
        device: &Device,
        store: &mut SafeTensorStore,
        config: &Qwen35ModelConfig,
        weight_bindings: &Qwen35ModelWeightBindings,
        case: Case,
        shape: BenchShape,
        max_tokens: usize,
    ) -> Self {
        let defaults = Qwen35MetalDefaults::from_quantization(config.quantization.as_ref())
            .expect("qwen3.5 layer bench requires supported quantization");
        let counts = qwen35_layer_counts(config).expect("qwen3.5 layer bench requires a valid layer schedule");
        assert!(counts.gqa > 0, "qwen3.5 layer bench requires GQA layers");
        assert!(counts.gdn > 0, "qwen3.5 layer bench requires GDN layers");
        let first_gqa_layer = first_layer_of_type(config, LayerType::FullAttention);
        let (gqa_core, gqa_metal) = derive_qwen35_gqa_configs(first_gqa_layer, &config.text_config, defaults)
            .expect("qwen3.5 layer bench requires valid GQA geometry");
        let num_page_ids_per_block = CACHE_BLOCK_TOKENS.div_ceil(gqa_metal.num_tokens_per_page(&gqa_core) as usize);
        let context_end = shape.context_end(config.text_config.max_position_embeddings);
        let num_blocks = context_end.div_ceil(CACHE_BLOCK_TOKENS);
        let num_cache_pages = counts
            .gqa
            .checked_mul(num_blocks)
            .and_then(|value| value.checked_mul(num_page_ids_per_block))
            .expect("qwen3.5 layer bench cache page count must fit usize");
        let gqa_state = Qwen3xGQAState::new(
            device,
            gqa_core,
            gqa_metal,
            GQAPageTableLayout {
                num_req_slots: 1,
                num_blocks: num_blocks
                    .try_into()
                    .expect("qwen3.5 layer bench block count must fit u32"),
                num_gqa_layers: counts
                    .gqa
                    .try_into()
                    .expect("qwen3.5 layer bench GQA layer count must fit u32"),
                num_page_ids_per_block: num_page_ids_per_block
                    .try_into()
                    .expect("qwen3.5 layer bench page count must fit u32"),
            },
            max_tokens,
            num_cache_pages,
        );
        write_gqa_page_ids(
            &gqa_state,
            counts.gqa,
            num_blocks,
            num_page_ids_per_block,
            num_cache_pages,
        );
        gqa_state.prepare_metadata(&[0], &[shape.context], &[0, shape.num_tokens], shape.num_tokens);

        let gdn_layers = layer_indices_of_type(config, LayerType::GDN);
        let gdn_cores = gdn_layers
            .iter()
            .map(|&index| {
                derive_qwen35_gdn_configs(index, &config.text_config, defaults)
                    .expect("qwen3.5 layer bench requires valid GDN geometry")
                    .0
            })
            .collect::<Vec<_>>();
        let gdn_metal = derive_qwen35_gdn_configs(gdn_layers[0], &config.text_config, defaults)
            .expect("qwen3.5 layer bench requires valid GDN geometry")
            .1;
        let gdn_state = Qwen3xGDNState::new(
            device,
            Stream::new(device),
            &gdn_cores,
            gdn_metal,
            1,
            inference_executor_metal::attn::gdn::state_table::GDNStateCapacity::new(2, 1, 1, max_tokens),
            max_tokens,
            CACHE_BLOCK_TOKENS,
            num_cache_pages,
            QWEN35_PAGE_SIZE_BYTES,
        );
        prepare_gdn_metadata(&gdn_state, shape.num_tokens, shape.num_tokens);

        let layer_scratch = Rc::new(Qwen35MainLayerScratch::new(
            device,
            max_tokens,
            config.text_config.hidden_size,
        ));
        let dense_scratch = dense_scratch(device, config, defaults, counts.has_dense_mlp, max_tokens);
        let moe_scratch = moe_scratch(device, config, defaults, counts.has_moe, max_tokens);

        let layer_indices = case.layer_indices(config.text_config.num_hidden_layers);
        assert!(!layer_indices.is_empty(), "qwen3.5 layer bench requires layers");
        let has_gqa = layer_indices.iter().any(|&index| {
            config
                .layer_type_at(index)
                .is_ok_and(|kind| kind == LayerType::FullAttention)
        });
        let mut layers = Vec::with_capacity(layer_indices.len());
        for model_layer_index in layer_indices {
            let bindings = weight_bindings
                .main
                .layers
                .get(model_layer_index)
                .unwrap_or_else(|| panic!("qwen3.5 layer bench missing bindings for layer {model_layer_index}"))
                .clone();
            layers.push(load_layer(
                device,
                store,
                config,
                defaults,
                model_layer_index,
                bindings,
                &gqa_state,
                &gdn_state,
                Rc::clone(&layer_scratch),
                dense_scratch.as_ref(),
                moe_scratch.as_ref(),
            ));
            store.unload_all();
        }

        Self {
            stream: Stream::new(device),
            input: Buffer::from_slice(device, &hidden_fixture(max_tokens, config.text_config.hidden_size)),
            layers,
            gqa_state,
            gdn_state,
            pages: PageArena::new(device, num_cache_pages, QWEN35_PAGE_SIZE_BYTES),
            has_gqa,
            max_tokens,
            shape,
        }
    }

    fn build_replay(&self) -> ReplayProgram {
        let mut recorder = ReplayRecorder::new(self.stream.create_replay_program());
        let mut hidden = &self.input;
        for layer in &self.layers {
            hidden = <Qwen35MainLayer as ReplayLayer>::record(
                layer,
                &mut recorder,
                Qwen35MainLayerInput {
                    gdn: self.gdn_state.metadata(),
                    gqa: self.gqa_state.metadata(),
                    num_tokens: self.shape.num_tokens,
                    pages: self.pages.buffer(),
                    residual_input: hidden,
                    residual_output: layer.residual_output(),
                    residual_capture_dest: None,
                },
            );
        }
        recorder.build()
    }

    fn run(&self, replay: &ReplayProgram) {
        self.stream.submit_replay(replay).wait();
    }

    fn print_gqa_split_plan(&self) {
        if !self.has_gqa {
            return;
        }
        print_gqa_split_plan(&self.gqa_state, self.max_tokens, "qwen35-main-layers");
    }
}

fn first_layer_of_type(config: &Qwen35ModelConfig, layer_type: LayerType) -> usize {
    (0..config.text_config.num_hidden_layers)
        .find(|&index| config.layer_type_at(index).is_ok_and(|kind| kind == layer_type))
        .unwrap_or_else(|| panic!("Qwen3.5 model must contain a {layer_type:?} layer"))
}

fn layer_indices_of_type(config: &Qwen35ModelConfig, layer_type: LayerType) -> Vec<usize> {
    (0..config.text_config.num_hidden_layers)
        .filter(|&index| config.layer_type_at(index).is_ok_and(|kind| kind == layer_type))
        .collect()
}

fn write_gqa_page_ids(
    gqa_state: &Qwen3xGQAState,
    num_gqa_layers: usize,
    num_blocks: usize,
    num_page_ids_per_block: usize,
    num_cache_pages: usize,
) {
    let mut next_page_id = 0u32;
    for layer_index in 0..num_gqa_layers {
        for block_index in 0..num_blocks {
            let page_ids = (0..num_page_ids_per_block)
                .map(|_| {
                    let page_id = next_page_id;
                    next_page_id = next_page_id
                        .checked_add(1)
                        .expect("qwen3.5 Main benchmark page ID must fit u32");
                    page_id
                })
                .collect::<Vec<_>>();
            gqa_state
                .request_page_table()
                .write_page_ids(0, layer_index, block_index, &page_ids);
        }
    }
    assert_eq!(
        next_page_id as usize, num_cache_pages,
        "qwen3.5 Main benchmark must initialize every cache page ID"
    );
}

fn prepare_gdn_metadata(gdn_state: &Qwen3xGDNState, num_active_tokens: u32, num_total_tokens: u32) {
    let mut flat_materialized_state_slots = vec![u32::MAX; num_active_tokens as usize];
    flat_materialized_state_slots[num_active_tokens as usize - 1] = 1;
    gdn_state.metadata().update(
        &[0, num_active_tokens],
        0,
        &[0],
        &[0],
        &flat_materialized_state_slots,
        &flat_materialized_state_slots,
        1,
        num_total_tokens,
    );
}

fn dense_scratch(
    device: &Device,
    config: &Qwen35ModelConfig,
    defaults: Qwen35MetalDefaults,
    required: bool,
    max_tokens: usize,
) -> Option<Rc<DenseMLPScratch>> {
    required.then(|| {
        let index = (0..config.text_config.num_hidden_layers)
            .find(|&index| !config.layer_uses_moe(index))
            .expect("qwen3.5 Main dense schedule must contain a dense layer");
        let (core, metal) = derive_qwen35_dense_mlp_configs(index, &config.text_config, defaults)
            .expect("qwen3.5 Main benchmark requires valid dense MLP geometry");
        Rc::new(DenseMLPScratch::new(device, &core, metal.io_dtype, max_tokens))
    })
}

fn moe_scratch(
    device: &Device,
    config: &Qwen35ModelConfig,
    defaults: Qwen35MetalDefaults,
    required: bool,
    max_tokens: usize,
) -> Option<Rc<MoEScratch>> {
    required.then(|| {
        let index = (0..config.text_config.num_hidden_layers)
            .find(|&index| config.layer_uses_moe(index))
            .expect("qwen3.5 Main MoE schedule must contain an MoE layer");
        let (core, metal) = derive_qwen35_moe_configs(&format!("layers.{index}"), index, config, defaults)
            .expect("qwen3.5 Main benchmark requires valid MoE geometry");
        Rc::new(MoEScratch::new(device, &core, metal, max_tokens))
    })
}

fn print_gqa_split_plan(gqa_state: &Qwen3xGQAState, max_tokens: usize, component: &str) {
    let metadata = gqa_state.metadata();
    let shape = metadata.replay_shape();
    let map = metadata.variant().map.thread_block;
    let variant_name =
        if metadata.variant().map.kind == inference_backend_metal::components::gqa::sdpa::MapKernelKind::SlicedKv {
            "sliced_kv"
        } else {
            "sliced_q"
        };
    let max_q_tokens = map.max_q_tokens;
    let kv_tokens_per_iteration = map.kv_tokens_per_iteration;
    let cu_kv_splits = metadata
        .cu_sdpa_partial_outputs()
        .read_typed::<u32>(0, shape.num_q_token_tiles as usize + 1);
    let num_map_task_templates_per_q_token_range = cu_kv_splits.windows(2).map(|cu| cu[1] - cu[0]).collect::<Vec<_>>();
    let num_active_q_tokens_per_q_token_range = if max_q_tokens == 1 {
        vec![1; shape.num_q_token_tiles as usize]
    } else {
        metadata
            .q_token_ranges()
            .read_typed::<u32>(0, shape.num_q_token_tiles as usize * 2)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|range| range[1] - range[0])
            .collect::<Vec<_>>()
    };
    let num_active_partial_state_groups = num_active_q_tokens_per_q_token_range
        .iter()
        .zip(&num_map_task_templates_per_q_token_range)
        .map(|(&num_q_tokens, &num_templates)| u64::from(num_q_tokens) * u64::from(num_templates))
        .sum::<u64>();
    let map_task_template_values = metadata
        .sdpa_map_task_templates()
        .read_typed::<u32>(0, shape.num_sdpa_map_task_templates as usize * 3);
    let mut kv_iterations_per_map_task_template_histogram = BTreeMap::new();
    for num_kv_iterations in map_task_template_values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|task| (task[2] - task[1]).div_ceil(kv_tokens_per_iteration))
    {
        *kv_iterations_per_map_task_template_histogram
            .entry(num_kv_iterations)
            .or_insert(0usize) += 1;
    }
    let reserved_partial_state_groups = shape
        .num_sdpa_map_task_templates
        .checked_mul(max_q_tokens)
        .expect("GQA reserved partial-state-group count must fit u32");
    let replay_reserved_partial_state_groups = shape
        .num_total_sdpa_map_task_templates
        .checked_mul(max_q_tokens)
        .expect("GQA replay reserved partial-state-group count must fit u32");
    println!(
        "split_plan component={component} variant={variant_name} max_tokens={max_tokens} num_active_q_tokens={} \
         num_q_token_ranges={} num_active_q_tokens_per_q_token_range={num_active_q_tokens_per_q_token_range:?} \
         num_map_task_templates={} max_map_task_templates={max_tokens} \
         num_active_partial_state_groups={num_active_partial_state_groups} \
         num_reserved_partial_state_groups={reserved_partial_state_groups} num_replay_map_task_template_slots={} \
         num_replay_reserved_partial_state_groups={replay_reserved_partial_state_groups} \
         num_map_task_templates_per_q_token_range={num_map_task_templates_per_q_token_range:?} \
         num_kv_iterations_per_map_task_template_histogram={kv_iterations_per_map_task_template_histogram:?}",
        shape.num_tokens,
        shape.num_q_token_tiles,
        shape.num_sdpa_map_task_templates,
        shape.num_total_sdpa_map_task_templates,
    );
}

#[allow(clippy::too_many_arguments)]
fn load_layer(
    device: &Device,
    store: &mut SafeTensorStore,
    config: &Qwen35ModelConfig,
    defaults: Qwen35MetalDefaults,
    model_layer_index: usize,
    bindings: Qwen35LayerWeightBindings,
    gqa_state: &Qwen3xGQAState,
    gdn_state: &Qwen3xGDNState,
    layer_scratch: Rc<Qwen35MainLayerScratch>,
    dense_scratch: Option<&Rc<DenseMLPScratch>>,
    moe_scratch: Option<&Rc<MoEScratch>>,
) -> Qwen35MainLayer {
    let compact_gqa_layer_index = (0..model_layer_index)
        .filter(|&index| {
            config
                .layer_type_at(index)
                .is_ok_and(|kind| kind == LayerType::FullAttention)
        })
        .count();
    let compact_gdn_layer_index = model_layer_index - compact_gqa_layer_index;
    let attn_layer_index = match config
        .layer_type_at(model_layer_index)
        .expect("benchmark layer type must be valid")
    {
        LayerType::FullAttention => compact_gqa_layer_index,
        LayerType::GDN => compact_gdn_layer_index,
    };
    let mut layer = Qwen35MainLayer::new(
        device,
        config,
        defaults,
        model_layer_index,
        attn_layer_index,
        gqa_state,
        gdn_state,
        layer_scratch,
        dense_scratch,
        moe_scratch,
    )
    .unwrap_or_else(|err| panic!("unable to construct layer {model_layer_index}: {err}"));
    layer
        .load_weights(device, store, config, bindings)
        .unwrap_or_else(|err| panic!("unable to load layer {model_layer_index}: {err}"));
    layer
}

pub fn run_main() {
    let args = BenchArgs::parse(Target::Main);
    let device = Device::system_default();
    let model_config = init_qwen35_model_config(&args.model_dir)
        .unwrap_or_else(|err| panic!("unable to init Qwen3.5 config from {}: {err}", args.model_dir.display()));
    let mut store = SafeTensorStore::from_model_dir(&args.model_dir).unwrap_or_else(|err| {
        panic!(
            "unable to load safetensors store from {}: {err}",
            args.model_dir.display()
        )
    });
    let weight_bindings = resolve_qwen35_model_weight_bindings(&model_config, store.index().tensor_names())
        .unwrap_or_else(|err| {
            panic!(
                "unable to resolve Qwen3.5 weight layout from {}: {err}",
                args.model_dir.display()
            )
        });
    for shape in args.shapes {
        let mut fixture = FullMainFixture::new(
            &device,
            &mut store,
            &model_config,
            weight_bindings.main.clone(),
            shape,
            args.max_tokens,
        );
        store.unload_all();
        let prepared = fixture.prepare();
        fixture.print_gqa_split_plan();
        fixture.run(&prepared);
        let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || fixture.run(&prepared));
        print_main_perf(
            &args.model_dir,
            shape,
            args.max_tokens,
            args.iters,
            fixture.main.replay(&prepared.key),
            &samples,
        );
    }
}

pub fn run_main_layers() {
    let args = BenchArgs::parse(Target::MainLayers);
    let device = Device::system_default();
    let model_config = init_qwen35_model_config(&args.model_dir)
        .unwrap_or_else(|err| panic!("unable to init Qwen3.5 config from {}: {err}", args.model_dir.display()));
    let mut store = SafeTensorStore::from_model_dir(&args.model_dir).unwrap_or_else(|err| {
        panic!(
            "unable to load safetensors store from {}: {err}",
            args.model_dir.display()
        )
    });
    let weight_bindings = resolve_qwen35_model_weight_bindings(&model_config, store.index().tensor_names())
        .unwrap_or_else(|err| {
            panic!(
                "unable to resolve Qwen3.5 weight layout from {}: {err}",
                args.model_dir.display()
            )
        });
    for shape in args.shapes {
        for &case in &args.cases {
            let fixture = LayerRangeFixture::new(
                &device,
                &mut store,
                &model_config,
                &weight_bindings,
                case,
                shape,
                args.max_tokens,
            );
            store.unload_all();
            let replay = fixture.build_replay();
            fixture.print_gqa_split_plan();
            fixture.run(&replay);
            let samples = measure_runs(args.runs, args.warmup_iters, args.iters, || fixture.run(&replay));
            print_main_layers_perf(
                &args.model_dir,
                case,
                shape,
                args.max_tokens,
                args.iters,
                &replay,
                &samples,
            );
        }
    }
}

fn selected_cases() -> Vec<Case> {
    vec![Case::Layer(0), Case::Layer(4), Case::FirstLayers(4)]
}

fn parse_cases(value: &str) -> Vec<Case> {
    let cases = value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            match part {
                "layer0" => Case::Layer(0),
                "layer3" => Case::Layer(3),
                "layer4" => Case::Layer(4),
                "first4" => Case::FirstLayers(4),
                "all_layers" => Case::AllLayers,
                _ => {
                    parse_explicit_layer_range(part).unwrap_or_else(|| {
                        panic!(
                            "invalid case {part:?}; expected layer0, layer3, layer4, first4, all_layers, or \
                             layersSTART-END"
                        )
                    })
                },
            }
        })
        .collect::<Vec<_>>();
    assert!(!cases.is_empty(), "--cases must include at least one case");
    let keys = cases.iter().map(|case| case.key()).collect::<Vec<_>>();
    assert!(
        keys.iter().enumerate().all(|(index, key)| !keys[..index].contains(key)),
        "--cases must not contain duplicates"
    );
    cases
}

fn parse_explicit_layer_range(value: &str) -> Option<Case> {
    let range = value.strip_prefix("layers")?;
    let (start, end) = range.split_once('-')?;
    let start = start.parse::<usize>().ok()?;
    let end = end.parse::<usize>().ok()?;
    assert!(start < end, "explicit layer ranges must use START < END");
    Some(Case::Layers { start, end })
}

fn hidden_fixture(num_tokens: usize, hidden_dim: usize) -> Vec<u16> {
    (0..num_tokens * hidden_dim)
        .map(|index| bf16::from_f32(((index % 23) as f32 - 11.0) * 0.03125).to_bits())
        .collect()
}

fn next_arg(iter: &mut impl Iterator<Item = String>, name: &str) -> String {
    iter.next()
        .unwrap_or_else(|| panic!("{name} requires a value; pass --help for usage"))
}

fn parse_u32_list(value: &str, name: &str) -> Vec<u32> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse()
                .unwrap_or_else(|err| panic!("invalid {name} value {part:?}: {err}"))
        })
        .collect::<Vec<_>>();
    assert!(!values.is_empty(), "{name} must include at least one value");
    values
}

fn parse_usize_arg(value: &str, name: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|err| panic!("invalid {name} value {value:?}: {err}"))
}

fn sort_unique(values: &mut [u32], name: &str) {
    values.sort_unstable();
    assert!(
        values.windows(2).all(|pair| pair[0] != pair[1]),
        "{name} must not contain duplicates"
    );
}

fn print_help_and_exit(target: Target) -> ! {
    let bench_name = match target {
        Target::Main => "qwen35_main",
        Target::MainLayers => "qwen35_main_layers",
    };
    println!("{bench_name} bench");
    println!();
    println!("Usage: cargo bench --bench {bench_name} -- [options]");
    println!();
    println!("Options:");
    println!("--model-dir PATH");
    if matches!(target, Target::MainLayers) {
        println!("--cases layer0,layer3,layer4,first4,all_layers,layersSTART-END");
        println!("  layersSTART-END selects the half-open range [START, END)");
    }
    println!("--tokens 1,2,4");
    println!("--contexts 0,32,128");
    println!("--max-tokens N");
    println!("--iters N");
    println!("--warmup-iters N");
    println!("--runs N");
    std::process::exit(0);
}

fn measure_runs(runs: usize, warmup_iters: usize, iters: usize, mut run: impl FnMut()) -> Vec<f64> {
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        for _ in 0..warmup_iters {
            run();
        }
        let mut duration = Duration::ZERO;
        for _ in 0..iters {
            let start = Instant::now();
            run();
            duration += start.elapsed();
        }
        samples.push(duration.as_secs_f64() * 1_000_000.0 / iters as f64);
    }
    samples
}

fn print_main_layers_perf(
    model_dir: &std::path::Path,
    case: Case,
    shape: BenchShape,
    max_tokens: usize,
    iters: usize,
    replay: &ReplayProgram,
    samples: &[f64],
) {
    let median_us = median(samples);
    let stats = replay.stats();
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component=qwen35-main-layers impl=selected-layer-replay model_dir={} case={} num_tokens={} context={} \
         max_tokens={max_tokens} commands={} retained_buffers={} retained_pipelines={} constant_bytes={} \
         iters={iters} runs={} median_us={median_us:.3} samples_us=[{sample_text}]",
        model_dir.display(),
        case.key(),
        shape.num_tokens,
        shape.context,
        stats.command_count,
        stats.retained_buffer_count,
        stats.retained_pipeline_count,
        stats.parameter_buffer_bytes,
        samples.len()
    );
}

fn print_main_perf(
    model_dir: &std::path::Path,
    shape: BenchShape,
    max_tokens: usize,
    iters: usize,
    replay: &ReplayProgram,
    samples: &[f64],
) {
    let median_us = median(samples);
    let stats = replay.stats();
    let sample_text = samples
        .iter()
        .map(|sample| format!("{sample:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "perf component=qwen35-main impl=production-main-replay model_dir={} case=main num_tokens={} context={} \
         max_tokens={max_tokens} commands={} retained_buffers={} retained_pipelines={} constant_bytes={} \
         iters={iters} runs={} median_us={median_us:.3} samples_us=[{sample_text}]",
        model_dir.display(),
        shape.num_tokens,
        shape.context,
        stats.command_count,
        stats.retained_buffer_count,
        stats.retained_pipeline_count,
        stats.parameter_buffer_bytes,
        samples.len(),
    );
}

fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_explicit_layer_range_uses_half_open_coordinates() {
        let case = super::parse_explicit_layer_range("layers3-7").expect("range must parse");
        assert_eq!(case.key(), "layers3-7");
        assert_eq!(case.layer_indices(12), vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_all_layers_names_only_the_transformer_layer_range() {
        let case = super::Case::AllLayers;
        assert_eq!(case.key(), "all_layers");
        assert_eq!(case.layer_indices(4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_legacy_main_all_case_is_rejected() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| super::parse_cases("main_all")));
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_case_duplicates_are_rejected() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| super::parse_cases("layer0,layer0")));
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_axes_are_sorted_and_reject_duplicates() {
        let mut values = vec![7, 1, 4];
        super::sort_unique(&mut values, "--values");
        assert_eq!(values, vec![1, 4, 7]);

        let mut duplicates = [1, 7, 1];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::sort_unique(&mut duplicates, "--values");
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_context_end_respects_model_capacity() {
        let shape = super::BenchShape {
            num_tokens: 2,
            context: 6,
        };
        assert_eq!(shape.context_end(8), 8);
        let result = std::panic::catch_unwind(|| shape.context_end(7));
        assert!(result.is_err());
    }
}
