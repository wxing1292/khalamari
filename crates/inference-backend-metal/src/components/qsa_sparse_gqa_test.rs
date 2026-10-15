use inference_executor_core::replay::ReplayBucketPolicy;

use super::*;
use crate::components::gqa::fp8::fp8_e4m3_to_bf16;
use crate::metal::Stream;

#[test]
fn test_replay_bucketing() {
    let policy = ReplayBucketPolicy::with_topology_boundaries(1024, &Compute::topology_boundaries());
    for active in 1..=1024 {
        assert_eq!(
            Selector::key(active),
            Selector::key(policy.capacity(active)),
            "active={active}"
        );
    }
    assert!(matches!(Selector::key(2), KernelKind::Simd { num_splits: 32, .. }));
    assert_eq!(Selector::key(4), KernelKind::Mma { num_splits: 32 });
    assert_eq!(Selector::key(5), KernelKind::Mma { num_splits: 16 });
    assert_eq!(Selector::key(513), KernelKind::Mma { num_splits: 1 });
}

#[test]
fn test_fp8x8_to_half_success() {
    let device = Device::system_default();
    let stream = Stream::new(&device);
    let codes = (u8::MIN..=u8::MAX).collect::<Vec<_>>();
    let input = Buffer::from_slice(&device, &codes);
    let output = Buffer::new_zeroed_elements(&device, codes.len(), Dtype::Float32);
    // Compile the production decoder without the attention entry point.
    let helpers = SIMD_SOURCE.split("kernel void qsa_sparse_gqa_map").next().unwrap();
    let source = format!(
        r"{helpers}
kernel void decode(
    device const uint2* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{{
    const uint4 bits = qsa_fp8x8_to_half(input[tid]);
    output[tid * 2] = float4(as_type<half4>(bits.xy)) * 256.0f;
    output[tid * 2 + 1] = float4(as_type<half4>(bits.zw)) * 256.0f;
}}"
    );
    let kernel = CompiledKernel::new(&device, &source, "decode");
    let mut recorder = stream.create_replay_program();
    recorder.record(Decode {
        kernel: &kernel,
        input: &input,
        output: &output,
    });
    stream.submit_replay(&recorder.build()).wait();
    let actual = output.read_typed::<f32>(0, codes.len());
    for (&code, value) in codes.iter().zip(actual) {
        let expected = fp8_e4m3_to_bf16(code).to_f32();
        if expected.is_nan() {
            assert!(value.is_nan(), "code={code:#04x}");
        } else {
            assert_eq!(value.to_bits(), expected.to_bits(), "code={code:#04x}");
        }
    }
}

struct Decode<'a> {
    kernel: &'a CompiledKernel,
    input: &'a Buffer,
    output: &'a Buffer,
}

impl Operator for Decode<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_read(0, self.input, 0);
        recorder.set_buffer_write(1, self.output, 0);
        recorder.dispatch_threadblocks((1, 1, 1), (32, 1, 1));
    }
}
