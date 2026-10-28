use inference_runtime_core::tokenizer::Tokenizer;

use super::*;

#[test]
#[ignore = "requires KHALAMARI_QWEN3_ASR_MODEL_DIR"]
fn test_target_tokenizer_contract() {
    let model_dir = std::env::var("KHALAMARI_QWEN3_ASR_MODEL_DIR").unwrap();
    let tokenizer = load(Path::new(&model_dir)).unwrap();

    assert_eq!(tokenizer.token("<|audio_start|>").unwrap().value(), 151_669);
    assert_eq!(tokenizer.token("<|audio_end|>").unwrap().value(), 151_670);
    assert_eq!(tokenizer.token("<|audio_pad|>").unwrap().value(), 151_676);
    assert_eq!(tokenizer.token("<asr_text>").unwrap().value(), 151_704);

    let prompt = concat!(
        "<|im_start|>system\nIt's 2026.\n<|im_end|>\n",
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n",
        "<|im_start|>assistant\nlanguage English<asr_text>"
    );
    let tokens = tokenizer.encode(prompt).unwrap();
    assert_eq!(
        tokens.iter().map(|token| token.value()).collect::<Vec<_>>(),
        [
            151_644, 8_948, 198, 2_132, 594, 220, 17, 15, 17, 21, 624, 151_645, 198, 151_644, 872, 198, 151_669,
            151_676, 151_670, 151_645, 198, 151_644, 77_091, 198, 11_528, 6_364, 151_704,
        ]
    );
    assert_eq!(tokens.iter().filter(|token| token.value() == 151_676).count(), 1);
    assert_eq!(tokenizer.decode(&tokens).unwrap(), prompt);
}
