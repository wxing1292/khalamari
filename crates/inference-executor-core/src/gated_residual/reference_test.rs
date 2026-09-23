use super::*;

#[test]
fn test_read_success() {
    let core = GatedResidualCore {
        num_branches: 2,
        hidden_dim: 2,
        low_rank_dim: 1,
        eps: 3.0,
    };
    let result = read(ReadInput {
        core,
        precision: Precision::F32,
        num_tokens: 1,
        residual: &[1.0, -1.0, 2.0, -2.0],
        weights: ReadWeights {
            norm: &[1.0, 0.0, 0.0, -0.5],
            down: &[0.5, 0.0, -1.0, 0.0],
            up: &[1.0, -1.0, 2.0, 0.0],
        },
    });
    let n = [1.0, -0.5, 2.0 / 7.0_f32.sqrt(), -1.0 / 7.0_f32.sqrt()];
    let down = (0.5 * n[0] - n[2]) / 2.0;
    let activated = down / (1.0 + (-down).exp());
    let expected = [
        (n[0] / (1.0 + (-activated).exp()) + n[2] / (1.0 + (-2.0 * activated).exp())) / 2.0,
        (n[1] / (1.0 + activated.exp()) + n[3] * 0.5) / 2.0,
    ];
    assert_close(&result.normalized_residual, &n);
    assert_close(&result.hidden_state, &expected);
}

#[test]
fn test_write_success() {
    let core = GatedResidualCore {
        num_branches: 2,
        hidden_dim: 2,
        low_rank_dim: 1,
        eps: 1.0e-6,
    };
    let logit = 2.0 * 3.0_f32.ln();
    let result = write(WriteInput {
        core,
        precision: Precision::F32,
        num_tokens: 1,
        residual: &[10.0, 20.0, -10.0, -20.0],
        normalized_residual: &[1.0, 0.0, 0.0, 0.0],
        hidden_state: &[2.0, -4.0],
        weight: &[-logit, 0.0, 0.0, 0.0, logit, 0.0, 0.0, 0.0],
    });
    assert_close(&result, &[11.0, 18.0, -7.0, -26.0]);
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
    }
}
