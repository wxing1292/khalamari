use std::path::Path;

use inference_executor_core::model::qwen::v3_asr::init_qwen3_asr_config;
use inference_runtime_core::tokenizer::Tokenizer;

use super::*;

#[test]
fn test_parse_transcription_metadata() {
    for (output, requested, text, language) in [
        ("  hello world  ", Some("English"), "hello world", "English"),
        ("header\nLANGUAGE cHINese\n<asr_text>  你好  ", None, "你好", "Chinese"),
        ("  hello world  ", None, "hello world", ""),
        ("language None<asr_text>", None, "", ""),
        ("language None<asr_text>unexpected", None, "unexpected", ""),
    ] {
        assert_eq!(
            parse_transcription(output.into(), requested.map(str::to_owned)),
            Transcription {
                text: text.into(),
                language: language.into()
            }
        );
    }
}

#[test]
#[ignore = "requires KHALAMARI_QWEN3_ASR_MODEL_DIR"]
fn test_target_prompt_contract() {
    let model_dir = std::env::var("KHALAMARI_QWEN3_ASR_MODEL_DIR").unwrap();
    let model_dir = Path::new(&model_dir);
    let config = init_qwen3_asr_config(model_dir).unwrap();
    let tokenizer = tokenizer::load(model_dir).unwrap();

    let (tokens, audio_token_index) = tokenize_prompt(&tokenizer, &config, "It's 2026.", Some("English"), 2).unwrap();
    assert_eq!(audio_token_index, 17);
    assert_eq!(
        tokens.iter().map(|token| token.value()).collect::<Vec<_>>(),
        [
            151_644, 8_948, 198, 2_132, 594, 220, 17, 15, 17, 21, 13, 151_645, 198, 151_644, 872, 198, 151_669,
            151_676, 151_676, 151_670, 151_645, 198, 151_644, 77_091, 198, 11_528, 6_364, 151_704,
        ]
    );
    assert_eq!(
        tokenizer.decode(&tokens).unwrap(),
        concat!(
            "<|im_start|>system\nIt's 2026.<|im_end|>\n",
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|>",
            "<|audio_end|><|im_end|>\n",
            "<|im_start|>assistant\nlanguage English<asr_text>"
        )
    );
}
