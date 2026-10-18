use std::num::NonZeroUsize;
use std::time::Duration;

use clap::Parser;
use inference_runtime_core::Error;
use inference_runtime_core::config::ExecutorHibernationMode;
use inference_runtime_launcher::args::Qwen3ASRArgs;
use inference_runtime_launcher::args::Qwen3Args;
use inference_runtime_launcher::args::Qwen35Args;

use super::Qwen3ASRConfig;
use super::Qwen3Config;
use super::Qwen35Config;
use super::Qwen35ModelMode;

fn parse_qwen35(extra: &[&str]) -> Qwen35Args {
    let mut args = vec!["qwen3.5", "--hf-model-dir", "model"];
    args.extend_from_slice(extra);
    Qwen35Args::try_parse_from(args).unwrap()
}

fn parse_qwen3(extra: &[&str]) -> Qwen3Args {
    let mut args = vec!["qwen3", "--hf-model-dir", "model"];
    args.extend_from_slice(extra);
    Qwen3Args::try_parse_from(args).unwrap()
}

fn parse_qwen3_asr(extra: &[&str]) -> Qwen3ASRArgs {
    let mut args = vec!["qwen3_asr", "--hf-model-dir", "model"];
    args.extend_from_slice(extra);
    Qwen3ASRArgs::try_parse_from(args).unwrap()
}

#[test]
fn test_executor_hibernation_timeout() {
    let qwen3_default = Qwen3Config::from_args(parse_qwen3(&[])).unwrap();
    let qwen3_asr_default = Qwen3ASRConfig::from_args(parse_qwen3_asr(&[])).unwrap();
    let qwen35_default = Qwen35Config::from_args(parse_qwen35(&[])).unwrap();
    let qwen3_override = Qwen3Config::from_args(parse_qwen3(&["--executor-hibernation-timeout-secs", "17"])).unwrap();
    let qwen3_asr_override =
        Qwen3ASRConfig::from_args(parse_qwen3_asr(&["--executor-hibernation-timeout-secs", "17"])).unwrap();
    let qwen35_override =
        Qwen35Config::from_args(parse_qwen35(&["--executor-hibernation-timeout-secs", "17"])).unwrap();

    assert_eq!(qwen3_default.executor_hibernation_timeout(), Duration::from_secs(300));
    assert_eq!(
        qwen3_asr_default.executor_hibernation_timeout(),
        Duration::from_secs(300)
    );
    assert_eq!(qwen35_default.executor_hibernation_timeout(), Duration::from_secs(300));
    assert_eq!(qwen3_override.executor_hibernation_timeout(), Duration::from_secs(17));
    assert_eq!(
        qwen3_asr_override.executor_hibernation_timeout(),
        Duration::from_secs(17)
    );
    assert_eq!(qwen35_override.executor_hibernation_timeout(), Duration::from_secs(17));
}

#[test]
fn test_executor_hibernation_mode() {
    let qwen3_default = Qwen3Config::from_args(parse_qwen3(&[])).unwrap();
    let qwen3_asr_default = Qwen3ASRConfig::from_args(parse_qwen3_asr(&[])).unwrap();
    let qwen35_default = Qwen35Config::from_args(parse_qwen35(&[])).unwrap();
    let qwen3_all = Qwen3Config::from_args(parse_qwen3(&["--executor-hibernation-mode", "all"])).unwrap();
    let qwen3_asr_all = Qwen3ASRConfig::from_args(parse_qwen3_asr(&["--executor-hibernation-mode", "all"])).unwrap();
    let qwen35_all = Qwen35Config::from_args(parse_qwen35(&["--executor-hibernation-mode", "all"])).unwrap();

    assert_eq!(
        qwen3_default.executor_hibernation_mode(),
        ExecutorHibernationMode::Selected
    );
    assert_eq!(
        qwen3_asr_default.executor_hibernation_mode(),
        ExecutorHibernationMode::Selected
    );
    assert_eq!(
        qwen35_default.executor_hibernation_mode(),
        ExecutorHibernationMode::Selected
    );
    assert_eq!(qwen3_all.executor_hibernation_mode(), ExecutorHibernationMode::All);
    assert_eq!(qwen3_asr_all.executor_hibernation_mode(), ExecutorHibernationMode::All);
    assert_eq!(qwen35_all.executor_hibernation_mode(), ExecutorHibernationMode::All);
}

#[test]
fn test_scheduler_rejects_per_request_token_capacity_above_batch_capacity() {
    assert!(matches!(
        Qwen3Config::from_args(parse_qwen3(&["--max-tokens", "3", "--max-tokens-per-request", "4"])),
        Err(Error::InvalidArgument(message))
            if message.contains("--max-tokens-per-request=4 must not exceed --max-tokens=3")
    ));
    assert!(matches!(
        Qwen3ASRConfig::from_args(parse_qwen3_asr(&[
            "--max-tokens",
            "3",
            "--max-tokens-per-request",
            "4",
        ])),
        Err(Error::InvalidArgument(message))
            if message.contains("--max-tokens-per-request=4 must not exceed --max-tokens=3")
    ));
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&["--max-tokens", "3", "--max-tokens-per-request", "4"])),
        Err(Error::InvalidArgument(message))
            if message.contains("--max-tokens-per-request=4 must not exceed --max-tokens=3")
    ));
}

#[test]
fn test_request_capacity_follows_max_requests() {
    let qwen3 = Qwen3Config::from_args(parse_qwen3(&["--max-requests", "9"])).unwrap();
    assert_eq!(qwen3.scheduler_config().max_requests, 9);
    assert_eq!(qwen3.max_running_requests(), 9);

    let qwen3_asr = Qwen3ASRConfig::from_args(parse_qwen3_asr(&["--max-requests", "9"])).unwrap();
    assert_eq!(qwen3_asr.scheduler_config().max_requests, 9);
    assert_eq!(qwen3_asr.max_running_requests(), 9);

    let qwen35_configs = [
        Qwen35Config::from_args(parse_qwen35(&["--max-requests", "9"])).unwrap(),
        Qwen35Config::from_args(parse_qwen35(&[
            "--hf-spec-model-dir",
            "mtp-model",
            "--spec-type",
            "mtp",
            "--max-requests",
            "9",
        ]))
        .unwrap(),
        Qwen35Config::from_args(parse_qwen35(&[
            "--hf-spec-model-dir",
            "dflash2-model",
            "--spec-type",
            "dflash2",
            "--max-requests",
            "9",
        ]))
        .unwrap(),
    ];
    for config in qwen35_configs {
        assert_eq!(config.scheduler_config().max_requests, 9);
        assert_eq!(config.max_running_requests(), 9);
    }
}

#[test]
fn test_mtp_inputs_normalize_model_mode_and_cache_lanes() {
    let main = Qwen35Config::from_args(parse_qwen35(&[])).unwrap();
    assert_eq!(main.model_mode(), &Qwen35ModelMode::Vanilla);
    assert_eq!(main.num_cache_lanes(), 1);

    let default_mtp = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-spec-model-dir",
        "mtp-model",
        "--spec-type",
        "mtp",
    ]))
    .unwrap();
    assert_eq!(
        default_mtp.model_mode(),
        &Qwen35ModelMode::MTP {
            model_dir: "mtp-model".into(),
            num_spec_tokens: NonZeroUsize::MIN,
        }
    );
    assert_eq!(default_mtp.num_cache_lanes(), 2);

    let four_steps = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-spec-model-dir",
        "mtp-model",
        "--spec-type",
        "mtp",
        "--num-spec-tokens",
        "4",
    ]))
    .unwrap();
    assert_eq!(
        four_steps.model_mode(),
        &Qwen35ModelMode::MTP {
            model_dir: "mtp-model".into(),
            num_spec_tokens: NonZeroUsize::new(4).unwrap(),
        }
    );
    assert_eq!(four_steps.num_cache_lanes(), 5);
}

#[test]
fn test_dflash2_inputs_normalize_model_mode_and_cache_lanes() {
    let qwen35 = Qwen35Config::from_args(parse_qwen35(&[
        "--hf-spec-model-dir",
        "qwen35-dflash2",
        "--spec-type",
        "dflash2",
    ]))
    .unwrap();
    assert_eq!(
        qwen35.model_mode(),
        &Qwen35ModelMode::DFlash2 {
            model_dir: "qwen35-dflash2".into(),
            num_spec_tokens: None,
        }
    );
    assert_eq!(qwen35.num_cache_lanes(), 1);
}

#[test]
fn test_spec_tokens_require_a_speculator() {
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&["--num-spec-tokens", "1"])),
        Err(Error::InvalidArgument(message)) if message.contains("requires --spec-type")
    ));
}

#[test]
fn test_mtp_spec_tokens_fit_per_request_budget() {
    assert!(matches!(
        Qwen35Config::from_args(parse_qwen35(&[
            "--hf-spec-model-dir",
            "mtp-model",
            "--spec-type",
            "mtp",
            "--num-spec-tokens",
            "4",
            "--max-tokens-per-request",
            "3",
        ])),
        Err(Error::InvalidArgument(message)) if message.contains("--max-tokens-per-request=3")
    ));
    Qwen35Config::from_args(parse_qwen35(&[
        "--hf-spec-model-dir",
        "mtp-model",
        "--spec-type",
        "mtp",
        "--num-spec-tokens",
        "4",
        "--max-tokens-per-request",
        "4",
    ]))
    .unwrap();
}

#[test]
fn test_num_spec_tokens_overrides_block_spec_modes() {
    for count in ["1", "2", "3", "4", "7"] {
        let num_spec_tokens = Some(NonZeroUsize::new(count.parse().unwrap()).unwrap());
        let config = Qwen35Config::from_args(parse_qwen35(&[
            "--hf-spec-model-dir",
            "spec-model",
            "--spec-type",
            "dflash2",
            "--num-spec-tokens",
            count,
        ]))
        .unwrap();
        assert_eq!(
            config.model_mode(),
            &Qwen35ModelMode::DFlash2 {
                model_dir: "spec-model".into(),
                num_spec_tokens,
            }
        );
        assert_eq!(config.num_cache_lanes(), 1);
    }
}

#[test]
fn test_spec_checkpoint_arguments_must_be_specified_together() {
    for incomplete_args in [["--hf-spec-model-dir", "spec-model"], ["--spec-type", "mtp"]] {
        assert!(matches!(
            Qwen35Config::from_args(parse_qwen35(&incomplete_args)),
            Err(Error::InvalidArgument(message)) if message.contains("must be specified together")
        ));
    }
}

#[test]
fn test_qwen3_rejects_speculative_options() {
    assert!(Qwen3Args::try_parse_from(["qwen3", "--hf-model-dir", "model", "--spec-type", "dflash2"]).is_err());
}
