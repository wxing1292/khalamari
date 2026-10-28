//! Dense BF16 matrix multiplication.

use std::collections::HashSet;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::mlx_headers::find_mlx_metal_header_root;
use crate::mlx_headers::read_mlx_metal_header;

const GEMM_BM: u32 = 64;
const GEMM_BN: u32 = 64;
const GEMM_BK: u32 = 16;
const GEMM_WM: usize = 2;
const GEMM_WN: usize = 2;
const GEMM_KERNEL_NAME: &str = "khalamari_matmul_bf16_nt_bm64_bn64_bk16_wm2_wn2";
const GEMV_BM4_KERNEL_NAME: &str = "gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0";
const GEMV_BM8_KERNEL_NAME: &str = "gemv_bfloat16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0";
const GEMV_TM: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub input_dim: u32,
    pub output_dim: u32,
}

impl Config {
    fn validate(self) {
        assert!(self.input_dim > 0, "BF16 matmul input dimension must be positive");
        assert!(self.output_dim > 0, "BF16 matmul output dimension must be positive");
        assert!(
            self.input_dim.is_multiple_of(GEMM_BK),
            "BF16 matmul input dimension must be a multiple of {GEMM_BK}"
        );
        i32::try_from(self.input_dim).expect("BF16 matmul input dimension must fit i32");
        i32::try_from(self.output_dim).expect("BF16 matmul output dimension must fit i32");
        self.weight_bytes();
    }

    fn input_bytes(self, num_rows: u32) -> usize {
        tensor_bytes(num_rows, self.input_dim)
    }

    fn output_bytes(self, num_rows: u32) -> usize {
        tensor_bytes(num_rows, self.output_dim)
    }

    fn weight_bytes(self) -> usize {
        tensor_bytes(self.output_dim, self.input_dim)
    }
}

pub struct Matmul {
    config: Config,
    selector: Selector,
}

struct Selector {
    registry: Registry,
}

struct Registry {
    entries: Vec<(KernelKind, CompiledKernel)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernelKind {
    GemvBm4,
    GemvBm8,
    GemmBm64Bn64Bk16Wm2Wn2,
}

impl Matmul {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            selector: Selector::new(device, config),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        num_rows: u32,
        output: &'a Buffer,
        output_offset_bytes: usize,
        input: &'a Buffer,
        input_offset_bytes: usize,
        weight: &'a Buffer,
        weight_offset_bytes: usize,
    ) -> Invocation<'a> {
        assert!(num_rows > 0, "BF16 matmul row count must be positive");
        assert!(num_rows <= i32::MAX as u32, "BF16 matmul row count must fit i32");
        if num_rows == 1 {
            // MLX moves a partial four-row SIMDgroup back to output_dim - GEMV_TM.
            assert!(
                self.config.output_dim >= GEMV_TM,
                "BF16 matmul GEMV requires at least {GEMV_TM} output columns"
            );
            // MLX forms weight row bases and local offsets as signed int.
            let max_weight_index = self.config.output_dim as u64 * self.config.input_dim as u64 - 1;
            assert!(
                max_weight_index <= i32::MAX as u64,
                "BF16 matmul GEMV weight element index must fit i32"
            );
        }
        assert_range(
            input,
            input_offset_bytes,
            self.config.input_bytes(num_rows),
            "BF16 matmul input",
        );
        assert_range(
            output,
            output_offset_bytes,
            self.config.output_bytes(num_rows),
            "BF16 matmul output",
        );
        assert_range(
            weight,
            weight_offset_bytes,
            self.config.weight_bytes(),
            "BF16 matmul weight",
        );
        let (kind, kernel) = self.selector.select(self.config, num_rows);
        Invocation {
            config: self.config,
            kind,
            kernel,
            num_rows,
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
        }
    }
}

impl Selector {
    fn new(device: &Device, config: Config) -> Self {
        Self {
            registry: Registry::new(device, config),
        }
    }

    fn select(&self, config: Config, num_rows: u32) -> (KernelKind, &CompiledKernel) {
        let key = Self::key(config, num_rows);
        (key, self.registry.get(key))
    }

    fn key(config: Config, num_rows: u32) -> KernelKind {
        if num_rows > 1 {
            KernelKind::GemmBm64Bn64Bk16Wm2Wn2
        } else if config.output_dim >= 4096 {
            KernelKind::GemvBm8
        } else {
            KernelKind::GemvBm4
        }
    }
}

impl Registry {
    fn new(device: &Device, config: Config) -> Self {
        let gemv_source = gemv_source();
        let gemm_source = gemm_source();
        let gemv_kind = Selector::key(config, 1);
        let gemv_kernel_name = match gemv_kind {
            KernelKind::GemvBm4 => GEMV_BM4_KERNEL_NAME,
            KernelKind::GemvBm8 => GEMV_BM8_KERNEL_NAME,
            KernelKind::GemmBm64Bn64Bk16Wm2Wn2 => unreachable!(),
        };
        Self {
            entries: vec![
                (gemv_kind, CompiledKernel::new(device, gemv_source, gemv_kernel_name)),
                (
                    KernelKind::GemmBm64Bn64Bk16Wm2Wn2,
                    CompiledKernel::new(device, gemm_source, GEMM_KERNEL_NAME),
                ),
            ],
        }
    }

    fn get(&self, key: KernelKind) -> &CompiledKernel {
        self.entries
            .iter()
            .find_map(|(candidate_key, kernel)| (*candidate_key == key).then_some(kernel))
            .unwrap_or_else(|| panic!("missing BF16 matmul execution variant {key:?}"))
    }
}

pub struct Invocation<'a> {
    config: Config,
    kind: KernelKind,
    kernel: &'a CompiledKernel,
    num_rows: u32,
    output: &'a Buffer,
    output_offset_bytes: usize,
    input: &'a Buffer,
    input_offset_bytes: usize,
    weight: &'a Buffer,
    weight_offset_bytes: usize,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.config;
        recorder.set_kernel(self.kernel);
        recorder.set_buffer_write(3, self.output, self.output_offset_bytes);
        match self.kind {
            KernelKind::GemvBm4 | KernelKind::GemvBm8 => {
                recorder.set_buffer_read(0, self.weight, self.weight_offset_bytes);
                recorder.set_buffer_read(1, self.input, self.input_offset_bytes);
                record_gemv(recorder, config, self.kind);
            },
            KernelKind::GemmBm64Bn64Bk16Wm2Wn2 => {
                recorder.set_buffer_read(0, self.input, self.input_offset_bytes);
                recorder.set_buffer_read(1, self.weight, self.weight_offset_bytes);
                record_gemm(recorder, self.num_rows, config);
            },
        }
    }
}

fn record_gemv(recorder: &CommandRecorder<'_>, config: Config, kind: KernelKind) {
    let bm = match kind {
        KernelKind::GemvBm4 => 4,
        KernelKind::GemvBm8 => 8,
        KernelKind::GemmBm64Bn64Bk16Wm2Wn2 => unreachable!(),
    };
    recorder.set_i32(4, config.input_dim as i32);
    recorder.set_i32(5, config.output_dim as i32);
    recorder.set_i32(6, config.input_dim as i32);
    recorder.set_i32(9, 1);
    recorder.set_i32_slice(10, &[1]);
    recorder.set_i64_slice(11, &[0]);
    recorder.set_i64_slice(12, &[0]);
    recorder.dispatch_threadblocks(
        (config.output_dim.div_ceil(bm * GEMV_TM) as usize, 1, 1),
        (32, 1, bm as usize),
    );
}

fn record_gemm(recorder: &CommandRecorder<'_>, num_rows: u32, config: Config) {
    let tiles_m = num_rows.div_ceil(GEMM_BM);
    let tiles_n = config.output_dim.div_ceil(GEMM_BN);
    recorder.set_i32_slice(4, &gemm_params(num_rows, config, tiles_m, tiles_n));
    recorder.dispatch_threadblocks((tiles_n as usize, tiles_m as usize, 1), (32, GEMM_WN, GEMM_WM));
}

fn gemm_params(num_rows: u32, config: Config, tiles_m: u32, tiles_n: u32) -> [i32; 18] {
    let m = num_rows as i32;
    let n = config.output_dim as i32;
    let k = config.input_dim as i32;
    [
        m,
        n,
        k,
        k,
        k,
        n,
        tiles_n as i32,
        tiles_m as i32,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        k / GEMM_BK as i32,
        0,
        0,
    ]
}

fn tensor_bytes(rows: u32, columns: u32) -> usize {
    rows as usize * columns as usize * Dtype::Bfloat16.item_size()
}

fn assert_range(buffer: &Buffer, offset_bytes: usize, len_bytes: usize, name: &str) {
    assert!(
        offset_bytes.is_multiple_of(Dtype::Bfloat16.item_size()),
        "{name} byte offset must be BF16-aligned"
    );
    let buffer_bytes = buffer.len_bytes();
    assert!(
        offset_bytes <= buffer_bytes && len_bytes <= buffer_bytes - offset_bytes,
        "{name} byte range exceeds its buffer"
    );
}

fn gemv_source() -> &'static str {
    static EXPANDED_SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    EXPANDED_SOURCE.get_or_init(|| {
        let root = find_mlx_metal_header_root("gemv.metal", |_| true, "BF16 matmul GEMV");
        read_mlx_metal_header(&root, "mlx/backend/metal/kernels/gemv.metal", &mut HashSet::new())
    })
}

fn gemm_source() -> &'static str {
    static EXPANDED_SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    EXPANDED_SOURCE.get_or_init(|| {
        let root = find_mlx_metal_header_root("steel/gemm/kernels/steel_gemm_fused.h", |_| true, "BF16 matmul GEMM");
        let mut included = HashSet::new();
        let mut source = read_mlx_metal_header(&root, "mlx/backend/metal/kernels/defines.h", &mut included);
        source.push_str(&read_mlx_metal_header(
            &root,
            "mlx/backend/metal/kernels/utils.h",
            &mut included,
        ));
        source.push_str(&read_mlx_metal_header(
            &root,
            "mlx/backend/metal/kernels/steel/gemm/gemm.h",
            &mut included,
        ));
        let mut fused = read_mlx_metal_header(
            &root,
            "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h",
            &mut included,
        );
        for (declaration, value) in [
            (
                "constant bool has_batch [[function_constant(10)]];",
                "constant bool has_batch = false;",
            ),
            (
                "constant bool use_out_source [[function_constant(100)]];",
                "constant bool use_out_source = false;",
            ),
            (
                "constant bool do_axpby [[function_constant(110)]];",
                "constant bool do_axpby = false;",
            ),
            (
                "constant bool align_M [[function_constant(200)]];",
                "constant bool align_M = false;",
            ),
            (
                "constant bool align_N [[function_constant(201)]];",
                "constant bool align_N = false;",
            ),
            (
                "constant bool align_K [[function_constant(202)]];",
                "constant bool align_K = true;",
            ),
        ] {
            let declaration_start = fused
                .find(declaration)
                .unwrap_or_else(|| panic!("BF16 matmul MLX source is missing {declaration:?}"));
            fused.replace_range(declaration_start..declaration_start + declaration.len(), value);
        }
        source.push_str(&fused);
        source.push_str(&format!(
            "\ntemplate [[host_name(\"{GEMM_KERNEL_NAME}\")]] [[kernel]] decltype(gemm<bfloat16_t, 64, 64, 16, 2, 2, \
             false, true, float>) gemm<bfloat16_t, 64, 64, 16, 2, 2, false, true, float>;\n"
        ));
        source
    })
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::metal::ReplayArguments;
    use crate::metal::Stream;

    #[test]
    fn test_invoke_gemv() {
        assert_invoke_case(1);
    }

    #[test]
    fn test_invoke_gemm() {
        assert_invoke_case(3);
    }

    #[test]
    fn test_selector_variant_boundaries() {
        let device = Device::system_default();
        let small = Matmul::new(
            &device,
            Config {
                input_dim: 16,
                output_dim: 2048,
            },
        );
        let large = Matmul::new(
            &device,
            Config {
                input_dim: 16,
                output_dim: 4096,
            },
        );

        assert_eq!(small.selector.select(small.config, 1).0, KernelKind::GemvBm4);
        assert_eq!(
            small.selector.select(small.config, 2).0,
            KernelKind::GemmBm64Bn64Bk16Wm2Wn2
        );
        assert_eq!(large.selector.select(large.config, 1).0, KernelKind::GemvBm8);
    }

    fn assert_invoke_case(num_rows: u32) {
        let device = Device::system_default();
        let stream = Stream::new(&device);
        let config = Config {
            input_dim: 16,
            output_dim: 64,
        };
        let input_offset_values = 16_usize;
        let weight_offset_values = 16_usize;
        let output_offset_values = 64_usize;
        let input_values = (0..num_rows * config.input_dim)
            .map(|index| bf16::from_f32((index % 11) as f32 * 0.0625 - 0.25))
            .collect::<Vec<_>>();
        let weight_values = (0..config.output_dim * config.input_dim)
            .map(|index| bf16::from_f32((index % 13) as f32 * 0.03125 - 0.1875))
            .collect::<Vec<_>>();
        let mut padded_input = vec![0; input_offset_values];
        padded_input.extend(input_values.iter().map(|value| value.to_bits()));
        let mut padded_weight = vec![0; weight_offset_values];
        padded_weight.extend(weight_values.iter().map(|value| value.to_bits()));
        let input = Buffer::from_slice(&device, &padded_input);
        let weight = Buffer::from_slice(&device, &padded_weight);
        let output = Buffer::new_zeroed_elements(
            &device,
            output_offset_values + (num_rows * config.output_dim) as usize,
            Dtype::Bfloat16,
        );
        let matmul = Matmul::new(&device, config);
        let mut recorder = stream.create_replay_program();
        recorder.record(matmul.invoke(
            num_rows,
            &output,
            output_offset_values * size_of::<u16>(),
            &input,
            input_offset_values * size_of::<u16>(),
            &weight,
            weight_offset_values * size_of::<u16>(),
        ));
        let program = recorder.build();
        stream
            .submit_replay_with_arguments(&program, &ReplayArguments::new())
            .wait();

        assert_eq!(output.read_typed::<u16>(0, output_offset_values), vec![0; 64]);
        let actual = output
            .read_typed::<u16>(output_offset_values, (num_rows * config.output_dim) as usize)
            .into_iter()
            .map(bf16::from_bits)
            .collect::<Vec<_>>();
        for row in 0..num_rows as usize {
            for column in 0..config.output_dim as usize {
                let expected = (0..config.input_dim as usize)
                    .map(|inner| {
                        input_values[row * config.input_dim as usize + inner].to_f32()
                            * weight_values[column * config.input_dim as usize + inner].to_f32()
                    })
                    .sum::<f32>();
                assert!((actual[row * config.output_dim as usize + column].to_f32() - expected).abs() < 0.02);
            }
        }
    }
}
