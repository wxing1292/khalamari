use inference_backend_metal::components::resource_embed::Config;
use inference_backend_metal::metal::Dtype;
use inference_runtime_core::compute::DecoderSyncBlocks;
use inference_runtime_core::compute::DeviceRequest;
use inference_runtime_core::compute::DeviceResourcePlacement;
use inference_runtime_core::compute::QueryTokens;
use inference_runtime_core::config::SamplingConfig;
use inference_runtime_core::runtime::Token;

use super::build_mapping_table;

#[test]
fn test_query_intersection_and_batch_rows() {
    let config = Config {
        hidden_dim: 4,
        io_dtype: Dtype::Bfloat16,
    };
    let requests = [
        new_request(
            prefill(10, 3),
            vec![DeviceResourcePlacement::new(
                64,
                64,
                vec![(1, 4, 2), (9, 0, 4), (30, 6, 2)],
            )],
        ),
        new_request(
            prefill(20, 2),
            vec![DeviceResourcePlacement::new(128, 64, vec![(20, 5, 1)])],
        ),
    ];

    let table = build_mapping_table(config, &requests).unwrap();

    assert_eq!(4, table.shape().num_total_mappings);
    assert_eq!(&[0, 72, 0, 1, 80, 0, 2, 88, 0, 3, 168, 0], table.encoded_u32s());
}

#[test]
fn test_text_only_batch_builds_no_mapping_table() {
    let requests = [new_request(prefill(0, 3), vec![])];

    assert!(
        build_mapping_table(
            Config {
                hidden_dim: 4,
                io_dtype: Dtype::Bfloat16,
            },
            &requests,
        )
        .is_none()
    );
}

fn new_request(query_tokens: QueryTokens, resource_placements: Vec<DeviceResourcePlacement>) -> DeviceRequest {
    DeviceRequest::new(
        0,
        0,
        query_tokens,
        DecoderSyncBlocks::new(0, vec![], vec![]),
        resource_placements,
        SamplingConfig::default(),
    )
}

fn prefill(token_index: usize, len: usize) -> QueryTokens {
    QueryTokens::Prefill {
        epoch: 0,
        token_index,
        tokens: (0..len).map(|index| Token::new(index as u32)).collect(),
        window: len,
    }
}
