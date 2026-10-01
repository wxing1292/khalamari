use super::*;

#[test]
fn test_projected_indexer_success() {
    let core = QSAIndexerCore {
        hidden_dim: 2,
        num_q_heads: 1,
        head_dim: 2,
        compress_ratio: 2,
        top_k: 1,
        rope_dim: 2,
        rope_theta: 10000.0,
        norm_eps: 1.0,
    };
    // Query [sqrt(2),0], pooled K [0,sqrt(2)] after norm at position zero.
    let root = 2.0_f32.sqrt();
    let result = projected_indexer(Input {
        core,
        precision: Precision::F32,
        qk: &[root, 0.0, 0.0, root, root, 0.0, 0.0, root],
        raw_history: &[],
        group_start: 0,
        q_norm: &[0.0; 2],
        k_norm: &[0.0; 2],
    });
    assert_eq!(result.group_ids, vec![vec![], vec![0]]);
    assert!((result.keys[1] - 1.0).abs() < 1e-6);
    assert!((result.q[2] - 1.0_f32.cos()).abs() < 1e-6);
    assert!((result.scores[1][0] - 1.0_f32.sin() / root).abs() < 1e-6);
}
