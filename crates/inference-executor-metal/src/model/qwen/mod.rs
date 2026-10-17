use std::time::Duration;

use inference_executor_core::model::ModelOutputTiming;

pub mod v3;
pub mod v3_asr;
pub mod v3_x;
pub mod v3_5;

fn apply_main_gpu_timing(
    timing: &mut ModelOutputTiming,
    gpu_timestamp_durations: Option<&[Duration]>,
    integrated_spec: bool,
    has_rejection: bool,
    has_spec_decode: bool,
) {
    let Some(durations) = gpu_timestamp_durations else {
        return;
    };

    let mut index = 0usize;
    timing.main_gpu_elapsed = Some(durations[index]);
    index += 1;
    if has_rejection {
        timing.rejection_gpu_elapsed = Some(durations[index]);
        index += 1;
    } else if integrated_spec {
        timing.rejection_gpu_elapsed = Some(Duration::ZERO);
    }
    if integrated_spec {
        if has_spec_decode {
            timing.spec_prepare_gpu_elapsed = Some(durations[index]);
            index += 1;
        } else {
            timing.spec_prepare_gpu_elapsed = Some(Duration::ZERO);
        }
        timing.spec_prefill_gpu_elapsed = Some(durations[index]);
        index += 1;
        if has_spec_decode {
            timing.spec_decode_gpu_elapsed = Some(durations[index]);
        } else {
            timing.spec_decode_gpu_elapsed = Some(Duration::ZERO);
        }
    }
}

fn split_main_lane_page_ids(
    page_ids: &[u32],
    num_main_gqa_page_ids: usize,
    num_speculator_gqa_page_ids: usize,
) -> (&[u32], &[u32]) {
    let expected_gqa_page_ids = num_main_gqa_page_ids
        .checked_add(num_speculator_gqa_page_ids)
        .expect("Qwen Main cache-lane page-ID count must fit usize");
    assert_eq!(
        page_ids.len(),
        expected_gqa_page_ids,
        "Qwen Main cache block must contain the exact Main and speculator page IDs"
    );
    page_ids.split_at(num_main_gqa_page_ids)
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Device;
    use inference_executor_core::attn::GQAPageTableLayout;

    use super::split_main_lane_page_ids;
    use crate::attn::gqa::request_page_table::GQARequestPageTable;

    #[test]
    fn test_split_main_lane_page_ids_accepts_main_only_block() {
        let page_ids = [10, 11, 20, 21];

        let (main_page_ids, speculator_page_ids) = split_main_lane_page_ids(&page_ids, 4, 0);

        assert_eq!(main_page_ids, page_ids);
        assert!(speculator_page_ids.is_empty());
    }

    #[test]
    fn test_split_main_lane_page_ids_writes_independent_main_and_dspark_tables() {
        let device = Device::system_default();
        let main = GQARequestPageTable::new(
            &device,
            GQAPageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 2,
                num_blocks: 3,
                num_page_ids_per_block: 2,
            },
        );
        let dspark = GQARequestPageTable::new(
            &device,
            GQAPageTableLayout {
                num_req_slots: 2,
                num_gqa_layers: 1,
                num_blocks: 3,
                num_page_ids_per_block: 3,
            },
        );
        let page_ids = [10, 11, 20, 21, 30, 31, 32];
        let (main_page_ids, dspark_page_ids) = split_main_lane_page_ids(&page_ids, 4, 3);

        let (main_page_ids_by_layer, remainder) = main_page_ids.as_chunks::<2>();
        assert!(remainder.is_empty());
        for (layer_index, layer_page_ids) in main_page_ids_by_layer.iter().enumerate() {
            main.write_page_ids(1, layer_index, 2, layer_page_ids);
        }
        dspark.write_page_ids(1, 0, 2, dspark_page_ids);

        assert_eq!(main.read_page_ids(1, 0, 2), vec![10, 11]);
        assert_eq!(main.read_page_ids(1, 1, 2), vec![20, 21]);
        assert_eq!(dspark.read_page_ids(1, 0, 2), vec![30, 31, 32]);
    }

    #[test]
    #[should_panic(expected = "Qwen Main cache block must contain the exact Main and speculator page IDs")]
    fn test_split_main_lane_page_ids_rejects_wrong_combined_length() {
        split_main_lane_page_ids(&[10, 11, 20, 21, 30, 31], 4, 3);
    }
}
pub mod v4_x;
