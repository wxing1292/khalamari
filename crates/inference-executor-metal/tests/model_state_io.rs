use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use inference_executor_core::model::ExecutionSubmission;
use inference_executor_core::model::ReplayableDecoderModel;
use inference_executor_metal::model::qwen::v3_5::executor::Qwen35ExecutorConfig;
use inference_executor_metal::model::qwen::v3_5::executor::init_qwen_3_5_model_with_mtp;
use inference_runtime_core::compute::BatchDeviceRequest;
use inference_runtime_core::compute::DecoderSyncBlocks;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::ExecutorHibernationPlan;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::runtime::Token;

const MODEL_27B_DIR_ENV: &str = "PSI_DEC_MODEL_STATE_IO_TEST_27B_MODEL_DIR";
const MTP_27B_DIR_ENV: &str = "PSI_DEC_MODEL_STATE_IO_TEST_27B_MTP_MODEL_DIR";
const MODEL_35B_DIR_ENV: &str = "PSI_DEC_MODEL_STATE_IO_TEST_35B_MODEL_DIR";
const MTP_35B_DIR_ENV: &str = "PSI_DEC_MODEL_STATE_IO_TEST_35B_MTP_MODEL_DIR";
const NUM_CACHE_PAGES: usize = 1024;
const SNAPSHOT_COMPARE_CHUNK_BYTES: usize = 16 * 1024 * 1024;

#[test]
#[ignore = "requires Qwen3.6 27B Main and MTP checkpoints and substantial unified memory"]
fn model_state_io_27b_mtp_full_state_unload_load() {
    let model = init_qwen_3_5_model_with_mtp(
        model_dir(MODEL_27B_DIR_ENV),
        model_dir(MTP_27B_DIR_ENV),
        num_spec_tokens(),
        executor_config(),
    )
    .expect("Qwen3.6 27B with MTP must initialize");
    run_full_state_unload_load(model);
}

#[test]
#[ignore = "requires Qwen3.6 27B Main and MTP checkpoints and substantial unified memory"]
fn model_state_io_27b_mtp_selected_state_unload_load() {
    let model = init_qwen_3_5_model_with_mtp(
        model_dir(MODEL_27B_DIR_ENV),
        model_dir(MTP_27B_DIR_ENV),
        num_spec_tokens(),
        executor_config(),
    )
    .expect("Qwen3.6 27B with MTP must initialize");
    run_selected_state_unload_load(model);
}

#[test]
#[ignore = "requires Qwen3.6 35B Main and MTP checkpoints and substantial unified memory"]
fn model_state_io_35b_mtp_full_state_unload_load() {
    let model = init_qwen_3_5_model_with_mtp(
        model_dir(MODEL_35B_DIR_ENV),
        model_dir(MTP_35B_DIR_ENV),
        num_spec_tokens(),
        executor_config(),
    )
    .expect("Qwen3.6 35B with MTP must initialize");
    run_full_state_unload_load(model);
}

#[test]
#[ignore = "requires Qwen3.6 35B Main and MTP checkpoints and substantial unified memory"]
fn model_state_io_35b_mtp_selected_state_unload_load() {
    let model = init_qwen_3_5_model_with_mtp(
        model_dir(MODEL_35B_DIR_ENV),
        model_dir(MTP_35B_DIR_ENV),
        num_spec_tokens(),
        executor_config(),
    )
    .expect("Qwen3.6 35B with MTP must initialize");
    run_selected_state_unload_load(model);
}

fn model_dir(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{variable} must name a Qwen3.6 model directory"))
}

fn num_spec_tokens() -> NonZeroUsize {
    NonZeroUsize::new(1).unwrap()
}

fn executor_config() -> Qwen35ExecutorConfig {
    Qwen35ExecutorConfig {
        max_requests: 1,
        max_tokens: 1,
        max_tokens_per_request: 128,
        num_cache_pages: NUM_CACHE_PAGES,
        num_tokens_per_block: 128,
    }
}

fn run_full_state_unload_load(mut model: inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor) {
    run_hibernation_plan_unload_load(&mut model, "all", &ExecutorHibernationPlan::All);
}

fn run_selected_state_unload_load(mut model: inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor) {
    let num_page_ids = std::iter::once(model.num_main_lane_gqa_page_ids_per_block())
        .chain(model.num_mtp_gqa_page_ids_per_block())
        .sum::<usize>();
    let num_page_ids = u32::try_from(num_page_ids).expect("test page ID count must fit u32");
    let plan = ExecutorHibernationPlan::selected(
        std::iter::once(0..1).collect(),
        std::iter::once(0..num_page_ids).collect(),
    );
    run_hibernation_plan_unload_load(&mut model, "selected", &plan);
}

fn run_hibernation_plan_unload_load(
    model: &mut inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor,
    plan_name: &str,
    plan: &ExecutorHibernationPlan,
) {
    run_one_decode(model);
    let temp_dir = std::env::temp_dir().join(format!(
        "psi-dec-model-state-io-{plan_name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must not precede the Unix epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_dir).expect("test snapshot directory must be created");
    let first_snapshot = temp_dir.join("first.state");
    let second_snapshot = temp_dir.join("second.state");

    model.clear_replay_cache();
    model
        .unload_state(&first_snapshot, plan)
        .expect("initial model state must unload");
    model.unload_weights();
    model.load_weights().expect("model weights must reload");
    model
        .load_state(&first_snapshot, plan)
        .expect("initial model state must reload");
    model
        .unload_state(&second_snapshot, plan)
        .expect("restored model state must unload");
    assert_snapshot_directories_equal(&first_snapshot, &second_snapshot);
    model
        .load_state(&second_snapshot, plan)
        .expect("verified model state must reload");
    model.reset_req_slots(&[0]);
    run_one_decode(model);
    model.reset_req_slots(&[0]);

    std::fs::remove_dir_all(&temp_dir).expect("test snapshot directory must be removed");
}

fn assert_snapshot_directories_equal(expected: &Path, actual: &Path) {
    let expected_files = snapshot_file_names(expected);
    let actual_files = snapshot_file_names(actual);
    assert_eq!(expected_files, actual_files, "model state snapshot file sets differ");

    let mut expected_bytes = vec![0; SNAPSHOT_COMPARE_CHUNK_BYTES];
    let mut actual_bytes = vec![0; SNAPSHOT_COMPARE_CHUNK_BYTES];
    for file_name in expected_files {
        let expected_path = expected.join(&file_name);
        let actual_path = actual.join(&file_name);
        let mut expected_file = File::open(&expected_path).expect("expected snapshot file must open");
        let mut actual_file = File::open(&actual_path).expect("actual snapshot file must open");
        let expected_len = expected_file
            .metadata()
            .expect("expected snapshot file metadata must load")
            .len();
        let actual_len = actual_file
            .metadata()
            .expect("actual snapshot file metadata must load")
            .len();
        assert_eq!(
            expected_len, actual_len,
            "model state snapshot file lengths differ: file={file_name:?}"
        );

        let mut remaining = expected_len;
        while remaining > 0 {
            let chunk_bytes = usize::try_from(remaining.min(SNAPSHOT_COMPARE_CHUNK_BYTES as u64)).unwrap();
            expected_file
                .read_exact(&mut expected_bytes[..chunk_bytes])
                .expect("expected snapshot file must read");
            actual_file
                .read_exact(&mut actual_bytes[..chunk_bytes])
                .expect("actual snapshot file must read");
            assert_eq!(
                expected_bytes[..chunk_bytes],
                actual_bytes[..chunk_bytes],
                "model state snapshot file contents differ: file={file_name:?} offset={}",
                expected_len - remaining
            );
            remaining -= chunk_bytes as u64;
        }
    }
}

fn snapshot_file_names(path: &Path) -> Vec<OsString> {
    let mut file_names = std::fs::read_dir(path)
        .expect("model state snapshot directory must open")
        .map(|entry| {
            let entry = entry.expect("model state snapshot directory entry must load");
            assert!(
                entry
                    .file_type()
                    .expect("model state snapshot entry type must load")
                    .is_file(),
                "model state snapshot entry must be a regular file: {:?}",
                entry.path()
            );
            entry.file_name()
        })
        .collect::<Vec<_>>();
    file_names.sort_unstable();
    file_names
}

fn run_one_decode(model: &mut inference_executor_metal::model::qwen::v3_5::executor::Qwen35Executor) {
    let page_ids_per_lane = std::iter::once(model.num_main_lane_gqa_page_ids_per_block())
        .chain(model.num_mtp_gqa_page_ids_per_block())
        .collect::<Vec<_>>();
    let num_page_ids = page_ids_per_lane.iter().sum::<usize>();
    assert!(
        num_page_ids <= NUM_CACHE_PAGES,
        "test cache must fit one complete Qwen3.6 runtime block"
    );
    let mut next_page_id = 0_u32;
    let kv_page_ids = page_ids_per_lane
        .into_iter()
        .map(|num_lane_page_ids| {
            let begin = next_page_id;
            next_page_id = next_page_id
                .checked_add(u32::try_from(num_lane_page_ids).unwrap())
                .expect("test page ID must fit u32");
            vec![(begin..next_page_id).collect()]
        })
        .collect();
    let mut core_batch = BatchDeviceRequest::new(
        0,
        [DeviceRequest::new(
            0,
            0,
            QueryTokens::Decode {
                epoch: 0,
                token_index: 0,
                tokens: vec![Token::new(11)],
                spec_tokens: Vec::new(),
            },
            DecoderSyncBlocks::new(0, kv_page_ids, Vec::new()),
            vec![],
            Default::default(),
        )],
    );
    let model_batch = model.prepare_batch(&mut core_batch);
    let mut recorder = model.begin_ops_recording(&model_batch);
    let hidden = model.embed_main(&mut recorder, &model_batch);
    let hidden = model.forward_main(&mut recorder, &model_batch, hidden);
    let output = model.unembed_main(&mut recorder, &model_batch, &hidden);
    model.sample_main(&mut recorder, &model_batch, &output);
    let submission = model.submit_main(&recorder);
    submission.wait();
    let gpu_timestamp_durations = submission.gpu_timestamp_durations();
    drop(submission);
    let mut sampled = model.read_main(
        &recorder,
        &model_batch,
        Duration::ZERO,
        gpu_timestamp_durations.as_deref(),
    );
    let run_spec = model.run_spec(&model_batch, &sampled);
    let run_spec_prefill = model.run_spec_prefill(&model_batch);
    let run_spec_decode = model.run_spec_decode(&model_batch, &sampled);
    debug_assert!(!run_spec || (!run_spec_prefill && !run_spec_decode));
    if run_spec || run_spec_prefill || run_spec_decode {
        if run_spec_prefill {
            model.prefill_spec(&mut recorder, &model_batch, &sampled);
        }
        if run_spec_decode {
            model.decode_spec(&mut recorder, &model_batch, &sampled);
        }
        if run_spec {
            let spec_hidden = model.embed_spec(&mut recorder, &model_batch, &hidden, &sampled);
            let spec_hidden = model.forward_spec(&mut recorder, &model_batch, spec_hidden);
            let spec_output = model.unembed_spec(&mut recorder, &model_batch, &spec_hidden);
            model.sample_spec(&mut recorder, &model_batch, &spec_output);
        }
        let submission = model.submit_spec(&recorder);
        submission.wait();
        let gpu_timestamp_durations = submission.gpu_timestamp_durations();
        if run_spec || run_spec_decode {
            sampled = model.read_spec(
                &recorder,
                &model_batch,
                sampled,
                Duration::ZERO,
                gpu_timestamp_durations.as_deref(),
            );
        }
    }
    drop(recorder);
    model.commit_batch(core_batch, sampled);
}
