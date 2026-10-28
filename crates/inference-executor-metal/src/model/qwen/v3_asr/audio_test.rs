use std::sync::Arc;

use half::bf16;
use inference_backend_metal::metal::Device;
use inference_executor_core::model::EncoderExecutorLifecycle;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRAudioSource;
use inference_executor_core::model::qwen::v3_asr::init_qwen3_asr_config;
use inference_executor_core::model::qwen::v3_asr::weight_layout::resolve_qwen3_asr_weight_bindings;

use super::AudioEncoderExecutor;
use super::BF16_BYTES;
use crate::model::resource_arena::new_resource_arena;

#[test]
#[ignore = "requires KHALAMARI_QWEN3_ASR_MODEL_DIR and the pinned local checkpoint"]
fn test_audio_encoder_fixed() {
    let model_dir = std::env::var("KHALAMARI_QWEN3_ASR_MODEL_DIR").unwrap();
    let config = init_qwen3_asr_config(&model_dir).unwrap();
    let store = inference_executor_core::checkpoint::SafeTensorStore::from_model_dir(&model_dir).unwrap();
    let names = store.index().tensor_names().collect::<Vec<_>>();
    let bindings = resolve_qwen3_asr_weight_bindings(&config, names).unwrap();
    let device = Device::system_default();
    let arena = Arc::new(new_resource_arena(&device, 16 * 1024));
    let executor = AudioEncoderExecutor::load(&model_dir, config.audio, bindings.audio, Arc::clone(&arena)).unwrap();
    let features = (0..128 * 8)
        .map(|index| (index as f32 * 0.017).sin() + (index as f32 * 0.003).cos())
        .collect();
    let source = Qwen3ASRAudioSource::new(features, 128, 8).unwrap();
    executor.stop();
    executor.start().unwrap();
    let allocation = futures_lite::future::block_on(executor.encode(Arc::new(source))).unwrap();
    let actual = arena
        .storage()
        .buffer()
        .read_typed::<u16>(allocation.offset_bytes() as usize / BF16_BYTES, 2048)
        .into_iter()
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = [
        0.02923481,
        -0.00056352373,
        0.013302455,
        -0.012556037,
        0.0009782473,
        -0.02472848,
        0.010675542,
        0.0057406314,
    ];
    for (index, expected) in expected.into_iter().enumerate() {
        assert!(
            (actual[index] - expected).abs() < 0.003,
            "audio tower mismatch at {index}: actual={} expected={expected}",
            actual[index]
        );
    }
    let mean = actual.iter().sum::<f32>() / actual.len() as f32;
    let l2 = actual.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!((mean - -0.0006288766).abs() < 0.0002, "mean={mean}");
    assert!((l2 - 0.7997648).abs() < 0.08, "l2={l2}");
}
