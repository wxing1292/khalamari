use std::sync::Arc;

use inference_backend_metal::metal::Device;
use inference_executor_core::model::qwen::v3_asr::QWEN3_ASR_AUDIO_RESOURCE_TYPE;
use inference_executor_core::model::qwen::v3_asr::Qwen3ASRAudioSource;
use inference_executor_core::model::qwen::v3_asr::init_qwen3_asr_config;
use inference_executor_core::model::qwen::v3_asr::weight_layout::resolve_qwen3_asr_weight_bindings;
use inference_runtime_core::runtime::ResourceID;
use inference_runtime_core::runtime::ResourceURI;
use inference_runtime_core::runtime::resource::processor::ResourceProcessor;

use super::AudioSourceStore;
use super::Qwen3ASRAudioProcessor;
use crate::model::resource_arena::new_resource_arena;

#[test]
fn test_audio_source_registration_owns_store_entry() {
    let sources = Arc::new(AudioSourceStore::default());
    let resource_id = ResourceID::new(QWEN3_ASR_AUDIO_RESOURCE_TYPE);
    let uri = ResourceURI::new("qwen3-asr://prepared/test".to_string());
    let source = Qwen3ASRAudioSource::new(vec![0.0; 12], 3, 4).unwrap();
    let registration = sources.register(resource_id, uri.clone(), source);

    let (resolved_uri, resolved_source) = sources.resolve(resource_id).unwrap();
    assert_eq!(uri, resolved_uri);
    drop(registration);
    assert!(sources.resolve(resource_id).is_none());
    assert_eq!(4, resolved_source.num_frames());
}

#[test]
#[ignore = "requires KHALAMARI_QWEN3_ASR_MODEL_DIR and the pinned local checkpoint"]
fn test_audio_processor_materializes_registered_source() {
    let model_dir = std::env::var("KHALAMARI_QWEN3_ASR_MODEL_DIR").unwrap();
    let config = init_qwen3_asr_config(&model_dir).unwrap();
    let store = inference_executor_core::checkpoint::SafeTensorStore::from_model_dir(&model_dir).unwrap();
    let names = store.index().tensor_names().collect::<Vec<_>>();
    let bindings = resolve_qwen3_asr_weight_bindings(&config, names).unwrap();
    let arena = Arc::new(new_resource_arena(&Device::system_default(), 16 * 1024));
    let processor = Qwen3ASRAudioProcessor::load(&model_dir, &config, &bindings, Arc::clone(&arena)).unwrap();
    let resource_id = ResourceID::new(QWEN3_ASR_AUDIO_RESOURCE_TYPE);
    let features = (0..128 * 8)
        .map(|index| (index as f32 * 0.017).sin() + (index as f32 * 0.003).cos())
        .collect();
    let source = Qwen3ASRAudioSource::new(features, 128, 8).unwrap();
    let (uri, _registration) = processor.register_source(resource_id, source);

    let concrete = futures_lite::future::block_on(processor.process(resource_id)).unwrap();

    assert_eq!(concrete.id(), resource_id);
    assert_eq!(concrete.uri(), &uri);
    assert_eq!(concrete.num_resource_tokens(), 1);
    assert_eq!(concrete.source().len_bytes(), 2048 * size_of::<half::bf16>() as u64);
    let output = arena
        .storage()
        .buffer()
        .read_typed::<u16>(concrete.source().offset_bytes() as usize / size_of::<u16>(), 2048)
        .into_iter()
        .map(|bits| half::bf16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let l2 = output.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!((l2 - 0.7997648).abs() < 0.08, "l2={l2}");
}
