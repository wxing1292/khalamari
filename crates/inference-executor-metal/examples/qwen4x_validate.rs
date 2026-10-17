//! Real-checkpoint Main capture. The default layer limit is eight.
use std::path::PathBuf;

use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_core::model::qwen::v4_x::init_qwen4x_model_config;
use inference_executor_metal::model::qwen::v4_x::Qwen4xExecutor;
use inference_executor_metal::model::qwen::v4_x::Qwen4xExecutorConfig;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::DecoderSyncBlocks;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::Token;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(
        args.next()
            .ok_or("usage: qwen4x_validate MODEL_DIR [LAYERS|all] [TOKEN_IDS]")?,
    );
    let limit = args.next().unwrap_or_else(|| "8".to_owned());
    let tokens = args
        .next()
        .unwrap_or_else(|| "1,2,3,4,5,6,7,8".to_owned())
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.is_empty() {
        return Err("token IDs must not be empty".into());
    }
    let mut config = init_qwen4x_model_config(&model_dir)?;
    let layers = if limit == "all" {
        config.text_config.num_hidden_layers
    } else {
        limit.parse::<usize>()?
    };
    if layers == 0 || layers > config.text_config.num_hidden_layers {
        return Err("layer limit must select a nonempty model prefix".into());
    }
    config.text_config.num_hidden_layers = layers;
    config.text_config.layer_types.truncate(layers);
    config.text_config.ple_layer_ids.retain(|&id| id <= layers);
    let hidden_dim = config.text_config.hidden_size;
    let mut model = Qwen4xExecutor::new(
        &model_dir,
        config,
        Qwen4xExecutorConfig {
            max_requests: 1,
            max_tokens: tokens.len(),
            max_tokens_per_request: tokens.len(),
            num_cache_pages: 4096,
            num_tokens_per_block: 2048,
        },
    )?;
    let blocks = tokens.len().div_ceil(2048);
    let kv_count = model.num_kv_page_ids_per_block();
    let state_count = model.num_state_page_ids_per_block();
    let needed = blocks
        .checked_mul(kv_count + state_count)
        .ok_or("cache page count overflow")?;
    if needed > 4096 {
        return Err("capture exceeds the validation page arena".into());
    }
    let mut next_page = 0u32;
    let mut allocate = |n: usize| {
        let result = (next_page..next_page + n as u32).collect::<Vec<_>>();
        next_page += n as u32;
        result
    };
    let kv = (0..blocks).map(|_| allocate(kv_count)).collect();
    let states = (0..blocks).map(|_| allocate(state_count)).collect();
    let request = DeviceRequest::new(
        1,
        0,
        QueryTokens::Prefill {
            epoch: 1,
            token_index: 0,
            tokens: tokens.iter().map(|&id| Token::new(id)).collect(),
            window: tokens.len(),
        },
        DecoderSyncBlocks::new(0, vec![kv], vec![states]),
        None,
        vec![],
        SamplingConfig::default(),
    );
    let mut request = BatchDeviceRequest {
        seq: 1,
        dev_reqs: vec![request],
    };
    let batch = model.prepare_batch(&mut request);
    let mut recorder = model.begin_ops_recording(&batch);
    let hidden = model.embed_main(&mut recorder, &batch);
    let hidden = model.forward_main(&mut recorder, &batch, hidden);
    model.submit_main(&recorder).wait();
    let values: Vec<f32> = hidden
        .read_typed::<u16>(0, tokens.len() * hidden_dim)
        .into_iter()
        .map(|bits| half::bf16::from_bits(bits).to_f32())
        .collect();
    if !values.iter().all(|v| v.is_finite()) {
        return Err("Main capture contains non-finite values".into());
    }
    model.commit_batch(request, model.empty_sampled_output());
    model.clear_replay_cache();
    println!(
        "{{\"layers\":{layers},\"tokens\":{tokens:?},\"shape\":[{},{}],\"hidden\":{values:?}}}",
        tokens.len(),
        hidden_dim
    );
    Ok(())
}
