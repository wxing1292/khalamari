use std::collections::HashSet;
use std::mem::size_of;
use std::path::Path;
use std::path::PathBuf;

use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::CompiledKernel;
use crate::metal::Device;
use crate::metal::Dtype;
use crate::metal::Operator;
use crate::metal::ReplayParameterKey;
use crate::metal::ReplayU32;
use crate::mlx_headers::find_mlx_metal_header_root;
use crate::mlx_headers::read_mlx_metal_header;

fn checked_product(name: &str, factors: &[usize]) -> usize {
    factors
        .iter()
        .try_fold(1usize, |product, &factor| product.checked_mul(factor))
        .unwrap_or_else(|| panic!("{name} must fit usize"))
}

fn checked_bytes(name: &str, dimensions: &[usize], dtype: Dtype) -> usize {
    checked_product(name, dimensions)
        .checked_mul(dtype.item_size())
        .unwrap_or_else(|| panic!("{name} byte length must fit usize"))
}

fn checked_range_end(name: &str, offset_bytes: usize, required_bytes: usize) -> usize {
    offset_bytes
        .checked_add(required_bytes)
        .unwrap_or_else(|| panic!("{name} byte range must fit usize"))
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub n: i32,
    pub k: i32,
    pub group_size: i32,
    pub bits: i32,
    pub input_dtype: Dtype,
    pub output_dtype: Dtype,
    pub scale_bias_dtype: Dtype,
}

impl Config {
    pub fn same_dtype(n: i32, k: i32, group_size: i32, bits: i32, dtype: Dtype) -> Self {
        Self {
            n,
            k,
            group_size,
            bits,
            input_dtype: dtype,
            output_dtype: dtype,
            scale_bias_dtype: dtype,
        }
    }

    pub fn validate(self) {
        assert!(self.n > 0);
        assert!(self.k > 0);
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert_eq!(self.k % self.group_size, 0);
        assert!(matches!(
            self.input_dtype,
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ));
        assert!(matches!(
            self.output_dtype,
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ));
        assert!(matches!(
            self.scale_bias_dtype,
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ));
    }

    pub fn output_bytes(self, m: i32) -> usize {
        self.validate();
        assert!(m > 0);
        checked_bytes(
            "affine matmul output",
            &[m as usize, self.n as usize],
            self.output_dtype,
        )
    }

    pub fn input_bytes(self, m: i32) -> usize {
        self.validate();
        assert!(m > 0);
        checked_bytes("affine matmul input", &[m as usize, self.k as usize], self.input_dtype)
    }

    pub fn weight_bytes(self) -> usize {
        self.validate();
        let pack_factor = if self.bits == 3 {
            8
        } else if self.bits == 6 {
            4
        } else {
            8 / self.bits
        };
        let bytes_per_pack = if self.bits == 3 || self.bits == 6 { 3 } else { 1 };
        checked_product(
            "affine matmul packed weight byte length",
            &[self.n as usize, self.k as usize, bytes_per_pack as usize],
        ) / pack_factor as usize
    }

    pub fn scale_or_bias_bytes(self) -> usize {
        self.validate();
        checked_bytes(
            "affine matmul scale or bias",
            &[self.n as usize, (self.k / self.group_size) as usize],
            self.scale_bias_dtype,
        )
    }

    fn uses_same_dtype(self) -> bool {
        self.input_dtype == self.output_dtype && self.input_dtype == self.scale_bias_dtype
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Epilogue {
    Identity,
    SwiGLU,
}

impl Epilogue {
    fn output_divisor(self) -> i32 {
        match self {
            Self::Identity => 1,
            Self::SwiGLU => 2,
        }
    }
}

pub struct Kernel {
    config: Config,
    epilogue: Epilogue,
    kind: KernelKind,
    kernel: CompiledKernel,
}

pub struct Matmul {
    config: Config,
    registry: Registry,
}

struct Registry {
    epilogue: Epilogue,
    entries: Vec<(KernelKind, Kernel)>,
}

struct Selector;

#[derive(Clone, Copy, Debug)]
pub struct ExpertConfig {
    pub num_experts: i32,
    pub matmul: Config,
}

impl ExpertConfig {
    pub fn validate(self) {
        assert!(self.num_experts > 0);
        self.matmul.validate();
        // TODO: Generalize the gather and ragged expert templates to separate
        // input, output, and scale/bias element types.
        assert!(
            self.matmul.uses_same_dtype(),
            "expert affine quantized kernels do not yet support mixed dtypes"
        );
    }

    pub fn output_bytes(self, num_vectors: i32) -> usize {
        self.validate();
        self.matmul.output_bytes(num_vectors)
    }

    pub fn input_bytes(self, num_vectors: i32) -> usize {
        self.validate();
        self.matmul.input_bytes(num_vectors)
    }

    pub fn weight_bytes_per_expert(self) -> usize {
        self.validate();
        self.matmul.weight_bytes()
    }

    pub fn affine_param_bytes_per_expert(self) -> usize {
        self.validate();
        self.matmul.scale_or_bias_bytes()
    }

    fn weight_bytes(self) -> usize {
        checked_product(
            "expert affine weight byte length",
            &[self.num_experts as usize, self.weight_bytes_per_expert()],
        )
    }

    fn affine_param_bytes(self) -> usize {
        checked_product(
            "expert affine parameter byte length",
            &[self.num_experts as usize, self.affine_param_bytes_per_expert()],
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GatherShape {
    pub num_routes: i32,
    pub num_input_vectors: i32,
}

impl GatherShape {
    pub fn validate(self) {
        assert!(self.num_routes > 0);
        assert!(self.num_input_vectors > 0);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RaggedExpertMajorShape {
    pub num_routes: i32,
}

impl RaggedExpertMajorShape {
    pub fn validate(self) {
        assert!(self.num_routes > 0);
    }
}

#[derive(Clone, Copy)]
struct RouteActivity {
    num_total_tokens: u32,
    num_experts_per_token: u32,
    num_active_tokens: ReplayU32,
}

impl RouteActivity {
    fn new(
        config: ExpertConfig,
        num_total_routes: i32,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens: ReplayU32,
    ) -> Self {
        assert!(num_total_tokens > 0);
        assert!(num_experts_per_token > 0);
        assert!(
            num_experts_per_token <= config.num_experts as u32,
            "expert affine routes per token must not exceed expert count"
        );
        let derived_num_total_routes = num_total_tokens
            .checked_mul(num_experts_per_token)
            .expect("expert affine total route count must fit u32");
        assert_eq!(
            u32::try_from(num_total_routes).expect("expert affine total route count must be positive"),
            derived_num_total_routes,
            "expert affine total routes must equal total tokens times experts per token"
        );
        i32::try_from(derived_num_total_routes).expect("expert affine total route count must fit i32");
        if let ReplayU32::Fixed(num_active_tokens) = num_active_tokens {
            assert_eq!(num_active_tokens, num_total_tokens);
        }
        Self {
            num_total_tokens,
            num_experts_per_token,
            num_active_tokens,
        }
    }

    fn bind(self, recorder: &CommandRecorder<'_>, token_binding_index: usize, topk_binding_index: usize) {
        match self.num_active_tokens {
            ReplayU32::Fixed(value) => recorder.set_u32(token_binding_index, value),
            ReplayU32::Parameter(key) => recorder.bind_u32(token_binding_index, key, 1, self.num_total_tokens),
        }
        recorder.set_u32(topk_binding_index, self.num_experts_per_token);
    }

    fn uses_parameter(self) -> bool {
        matches!(self.num_active_tokens, ReplayU32::Parameter(_))
    }
}

pub struct GatherMatmulKernel {
    config: ExpertConfig,
    kernel: CompiledKernel,
    parameterized_kernel: CompiledKernel,
}

pub struct GatherGateUpSwiGLUKernel {
    config: ExpertConfig,
    kernel: CompiledKernel,
    parameterized_kernel: CompiledKernel,
}

pub struct RaggedExpertMajorGateUpSwiGLUKernel {
    config: ExpertConfig,
    kernels: ExpertMajorKernels,
}

pub struct RaggedExpertMajorMatmulKernel {
    config: ExpertConfig,
    kernels: ExpertMajorKernels,
}

impl GatherMatmulKernel {
    pub fn new(device: &Device, config: ExpertConfig) -> Self {
        config.validate();
        let matmul = config.matmul;
        validate_mlx_qmv_output_dim(matmul.n);
        let type_string = metal_type_string(matmul.input_dtype);
        let bn = 8;
        let fast = matmul.n % bn == 0 && matmul.k % 512 == 0;
        let func = if fast { "gather_qmv_fast" } else { "gather_qmv" };
        let kernel_name = format!("{func}_{type_string}_gs_{}_b_{}", matmul.group_size, matmul.bits);
        let exact_template_definition = template_definition(
            &kernel_name,
            func,
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
            ],
        );
        let source = affine_quantized_source(&exact_template_definition);
        let kernel = CompiledKernel::new(device, &source, &kernel_name);
        let parameterized_kernel_name = format!(
            "token_major_down_matmul_bucketed_{type_string}_gs_{}_b_{}",
            matmul.group_size, matmul.bits
        );
        let parameterized_template_definition = template_definition(
            &parameterized_kernel_name,
            "token_major_down_matmul_bucketed",
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
                fast.to_string(),
            ],
        );
        let parameterized_source =
            affine_quantized_source(&format!("{GATE_UP_SWIGLU_SOURCE}\n{parameterized_template_definition}"));
        let parameterized_kernel = CompiledKernel::new(device, &parameterized_source, &parameterized_kernel_name);
        Self {
            config,
            kernel,
            parameterized_kernel,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: GatherShape,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens: ReplayU32,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        lhs_indices: &'a Buffer,
        rhs_indices: &'a Buffer,
    ) -> GatherMatmulInvocation<'a> {
        validate_gather_buffer_ranges(
            self.config,
            shape,
            output,
            input,
            weight,
            scales,
            biases,
            lhs_indices,
            rhs_indices,
        );
        GatherMatmulInvocation {
            kernel: self,
            shape,
            output,
            input,
            weight,
            scales,
            biases,
            lhs_indices,
            rhs_indices,
            route_activity: RouteActivity::new(
                self.config,
                shape.num_routes,
                num_total_tokens,
                num_experts_per_token,
                num_active_tokens,
            ),
        }
    }
}

impl GatherGateUpSwiGLUKernel {
    pub fn new(device: &Device, config: ExpertConfig) -> Self {
        config.validate();
        let matmul = config.matmul;
        let type_string = metal_type_string(matmul.input_dtype);
        let kernel_name = format!(
            "token_major_gate_up_swiglu_{type_string}_gs_{}_b_{}",
            matmul.group_size, matmul.bits
        );
        let exact_template_definition = template_definition(
            &kernel_name,
            "token_major_gate_up_swiglu",
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
            ],
        );
        let source = affine_quantized_source(&format!("{GATE_UP_SWIGLU_SOURCE}\n{exact_template_definition}"));
        let kernel = CompiledKernel::new(device, &source, &kernel_name);
        let parameterized_kernel_name = format!(
            "token_major_gate_up_swiglu_bucketed_{type_string}_gs_{}_b_{}",
            matmul.group_size, matmul.bits
        );
        let parameterized_template_definition = template_definition(
            &parameterized_kernel_name,
            "token_major_gate_up_swiglu_bucketed",
            &[
                type_string.to_string(),
                matmul.group_size.to_string(),
                matmul.bits.to_string(),
            ],
        );
        let parameterized_source =
            affine_quantized_source(&format!("{GATE_UP_SWIGLU_SOURCE}\n{parameterized_template_definition}"));
        let parameterized_kernel = CompiledKernel::new(device, &parameterized_source, &parameterized_kernel_name);
        Self {
            config,
            kernel,
            parameterized_kernel,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: GatherShape,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens: ReplayU32,
        output: &'a Buffer,
        input: &'a Buffer,
        gate_weight: &'a Buffer,
        gate_scales: &'a Buffer,
        gate_biases: &'a Buffer,
        up_weight: &'a Buffer,
        up_scales: &'a Buffer,
        up_biases: &'a Buffer,
        lhs_indices: &'a Buffer,
        rhs_indices: &'a Buffer,
    ) -> GatherGateUpSwiGLUInvocation<'a> {
        validate_gather_gate_up_swiglu_buffer_ranges(
            self.config,
            shape,
            output,
            input,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            lhs_indices,
            rhs_indices,
        );
        GatherGateUpSwiGLUInvocation {
            kernel: self,
            shape,
            output,
            input,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            lhs_indices,
            rhs_indices,
            route_activity: RouteActivity::new(
                self.config,
                shape.num_routes,
                num_total_tokens,
                num_experts_per_token,
                num_active_tokens,
            ),
        }
    }
}

struct ExpertMajorKernels {
    qmv: CompiledKernel,
    qmv_all: CompiledKernel,
    qmm: [CompiledKernel; EXPERT_QMM_KINDS.len()],
}

impl ExpertMajorKernels {
    fn new(device: &Device, config: ExpertConfig, entry: &str) -> Self {
        let args = [
            metal_type_string(config.matmul.input_dtype).to_string(),
            config.matmul.group_size.to_string(),
            config.matmul.bits.to_string(),
            EXPERT_QMM_MIN_ROWS.to_string(),
        ];
        let qmm_function = format!("{entry}_qmm");
        let qmm = EXPERT_QMM_KINDS.map(|kind| {
            let bm = kind.qmm_shape().unwrap().0;
            let qmm_name = format!("{qmm_function}_bm{bm}");
            let mut qmm_args = args.to_vec();
            qmm_args.push(config.num_experts.to_string());
            qmm_args.push(bm.to_string());
            let template = template_definition(&qmm_name, &qmm_function, &qmm_args);
            let source = affine_quantized_source(&format!(
                "{MIXED_AFFINE_SOURCE}\n{STEEL_SOURCE}\n{EXPERT_MAJOR_STEEL_SOURCE}\n{template}"
            ));
            let qmm = CompiledKernel::new(device, &source, &qmm_name);
            validate_qmm_pipeline(
                device,
                config.matmul.input_dtype,
                kind,
                QMM_BK,
                expert_qmm_num_simdgroups(bm),
                &qmm,
            );
            qmm
        });
        let qmv_name = format!("{entry}_qmv");
        let template = template_definition(&qmv_name, &qmv_name, &args);
        let source = affine_quantized_source(&format!(
            "{GATE_UP_SWIGLU_SOURCE}\n{EXPERT_MAJOR_QMV_SOURCE}\n{template}"
        ));
        let qmv = CompiledKernel::new(device, &source, &qmv_name);
        let qmv_all_name = format!("{entry}_qmv_all");
        let template = template_definition(&qmv_all_name, &qmv_all_name, &args[..3]);
        let source = affine_quantized_source(&format!(
            "{GATE_UP_SWIGLU_SOURCE}\n{EXPERT_MAJOR_QMV_SOURCE}\n{template}"
        ));
        let qmv_all = CompiledKernel::new(device, &source, &qmv_all_name);
        Self { qmv, qmv_all, qmm }
    }

    fn record(
        &self,
        recorder: &CommandRecorder<'_>,
        config: ExpertConfig,
        activity: RouteActivity,
        num_total_routes: i32,
        output: &Buffer,
        bind: impl Fn(),
    ) {
        // Small batches use direct QMV to avoid the complementary dispatch overhead.
        // Larger batches dispatch both families. Actual expert lengths assign their outputs.
        let num_kernels = if activity.num_total_tokens < adaptive_qmv_batch_limit(config.matmul) as u32 {
            1
        } else {
            2
        };
        let qmv = if num_kernels == 1 { &self.qmv_all } else { &self.qmv };
        let qmv_row_tasks = if num_kernels == 1 {
            num_total_routes
        } else {
            num_total_routes.min(ceil_div_i32(EXPERT_QMV_THREADBLOCKS, ceil_div_i32(config.matmul.n, 8)))
        };
        let nominal_rows = num_total_routes / config.num_experts;
        let large_bm = EXPERT_QMM_KINDS[2].qmm_shape().unwrap().0;
        let qmm_index = if nominal_rows < EXPERT_QMM_MIN_ROWS {
            0
        } else if config.matmul.input_dtype == Dtype::Bfloat16
            && config.matmul.bits == 4
            && nominal_rows >= 2 * large_bm
        {
            2
        } else {
            1
        };
        let (bm, bn) = EXPERT_QMM_KINDS[qmm_index].qmm_shape().unwrap();
        // Sum of per-expert ceil(rows / BM) is bounded by ceil(routes / BM) + experts - 1.
        let num_total_row_tiles = (num_total_routes as u32)
            .min((num_total_routes as u32).div_ceil(bm as u32) + config.num_experts as u32 - 1)
            as i32;
        let kernels = [
            (qmv, 8, qmv_row_tasks, 2),
            (
                &self.qmm[qmm_index],
                bn,
                num_total_row_tiles,
                expert_qmm_num_simdgroups(bm),
            ),
        ];
        // Both families select disjoint expert segments from the same offsets.
        recorder.record_disjoint_buffers(&[output], || {
            for &(kernel, bn, num_row_tasks, num_simdgroups) in &kernels[..num_kernels] {
                recorder.set_kernel(kernel);
                bind();
                recorder.dispatch_threadblocks(
                    (num_row_tasks as usize, ceil_div_i32(config.matmul.n, bn) as usize, 1),
                    (32, num_simdgroups, 1),
                );
            }
        });
    }
}

impl RaggedExpertMajorGateUpSwiGLUKernel {
    pub fn new(device: &Device, config: ExpertConfig) -> Self {
        config.validate();
        Self {
            config,
            kernels: ExpertMajorKernels::new(device, config, "expert_major_gate_up_swiglu"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: RaggedExpertMajorShape,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens: ReplayU32,
        output: &'a Buffer,
        input: &'a Buffer,
        gate_weight: &'a Buffer,
        gate_scales: &'a Buffer,
        gate_biases: &'a Buffer,
        up_weight: &'a Buffer,
        up_scales: &'a Buffer,
        up_biases: &'a Buffer,
        experts_by_route: &'a Buffer,
        expert_offsets: &'a Buffer,
    ) -> RaggedExpertMajorGateUpSwiGLUInvocation<'a> {
        validate_ragged_expert_major_gate_up_swiglu_buffer_ranges(
            self.config,
            shape,
            output,
            input,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            experts_by_route,
            expert_offsets,
        );
        RaggedExpertMajorGateUpSwiGLUInvocation {
            kernel: self,
            shape,
            output,
            input,
            gate_weight,
            gate_scales,
            gate_biases,
            up_weight,
            up_scales,
            up_biases,
            experts_by_route,
            expert_offsets,
            route_activity: RouteActivity::new(
                self.config,
                shape.num_routes,
                num_total_tokens,
                num_experts_per_token,
                num_active_tokens,
            ),
        }
    }
}

impl RaggedExpertMajorMatmulKernel {
    pub fn new(device: &Device, config: ExpertConfig) -> Self {
        config.validate();
        Self {
            config,
            kernels: ExpertMajorKernels::new(device, config, "expert_major_down"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        shape: RaggedExpertMajorShape,
        num_total_tokens: u32,
        num_experts_per_token: u32,
        num_active_tokens: ReplayU32,
        output: &'a Buffer,
        input: &'a Buffer,
        weight: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        experts_by_route: &'a Buffer,
        expert_offsets: &'a Buffer,
    ) -> RaggedExpertMajorMatmulInvocation<'a> {
        validate_ragged_expert_major_down_matmul_buffer_ranges(
            self.config,
            shape,
            output,
            input,
            weight,
            scales,
            biases,
            experts_by_route,
            expert_offsets,
        );
        RaggedExpertMajorMatmulInvocation {
            kernel: self,
            shape,
            output,
            input,
            weight,
            scales,
            biases,
            experts_by_route,
            expert_offsets,
            route_activity: RouteActivity::new(
                self.config,
                shape.num_routes,
                num_total_tokens,
                num_experts_per_token,
                num_active_tokens,
            ),
        }
    }
}

pub struct GatherMatmulInvocation<'a> {
    kernel: &'a GatherMatmulKernel,
    shape: GatherShape,
    output: &'a Buffer,
    input: &'a Buffer,
    weight: &'a Buffer,
    scales: &'a Buffer,
    biases: &'a Buffer,
    lhs_indices: &'a Buffer,
    rhs_indices: &'a Buffer,
    route_activity: RouteActivity,
}

pub struct GatherGateUpSwiGLUInvocation<'a> {
    kernel: &'a GatherGateUpSwiGLUKernel,
    shape: GatherShape,
    output: &'a Buffer,
    input: &'a Buffer,
    gate_weight: &'a Buffer,
    gate_scales: &'a Buffer,
    gate_biases: &'a Buffer,
    up_weight: &'a Buffer,
    up_scales: &'a Buffer,
    up_biases: &'a Buffer,
    lhs_indices: &'a Buffer,
    rhs_indices: &'a Buffer,
    route_activity: RouteActivity,
}

pub struct RaggedExpertMajorGateUpSwiGLUInvocation<'a> {
    kernel: &'a RaggedExpertMajorGateUpSwiGLUKernel,
    shape: RaggedExpertMajorShape,
    output: &'a Buffer,
    input: &'a Buffer,
    gate_weight: &'a Buffer,
    gate_scales: &'a Buffer,
    gate_biases: &'a Buffer,
    up_weight: &'a Buffer,
    up_scales: &'a Buffer,
    up_biases: &'a Buffer,
    experts_by_route: &'a Buffer,
    expert_offsets: &'a Buffer,
    route_activity: RouteActivity,
}

pub struct RaggedExpertMajorMatmulInvocation<'a> {
    kernel: &'a RaggedExpertMajorMatmulKernel,
    shape: RaggedExpertMajorShape,
    output: &'a Buffer,
    input: &'a Buffer,
    weight: &'a Buffer,
    scales: &'a Buffer,
    biases: &'a Buffer,
    experts_by_route: &'a Buffer,
    expert_offsets: &'a Buffer,
    route_activity: RouteActivity,
}

impl Operator for RaggedExpertMajorMatmulInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        let matmul = config.matmul;
        let shape = self.shape;
        self.kernel.kernels.record(
            recorder,
            config,
            self.route_activity,
            shape.num_routes,
            self.output,
            || {
                recorder.set_buffer_read(0, self.weight, 0);
                recorder.set_buffer_read(1, self.scales, 0);
                recorder.set_buffer_read(2, self.biases, 0);
                recorder.set_buffer_read(3, self.input, 0);
                recorder.set_buffer_read(4, self.experts_by_route, 0);
                recorder.set_buffer_write(5, self.output, 0);
                recorder.set_i32(6, matmul.k);
                recorder.set_i32(7, matmul.n);
                self.route_activity.bind(recorder, 8, 9);
                recorder.set_buffer_read(10, self.expert_offsets, 0);
            },
        );
    }
}

impl Operator for RaggedExpertMajorGateUpSwiGLUInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        let matmul = config.matmul;
        let shape = self.shape;
        self.kernel.kernels.record(
            recorder,
            config,
            self.route_activity,
            shape.num_routes,
            self.output,
            || {
                recorder.set_buffer_read(0, self.gate_weight, 0);
                recorder.set_buffer_read(1, self.gate_scales, 0);
                recorder.set_buffer_read(2, self.gate_biases, 0);
                recorder.set_buffer_read(3, self.up_weight, 0);
                recorder.set_buffer_read(4, self.up_scales, 0);
                recorder.set_buffer_read(5, self.up_biases, 0);
                recorder.set_buffer_read(6, self.input, 0);
                recorder.set_buffer_read(7, self.experts_by_route, 0);
                recorder.set_buffer_write(8, self.output, 0);
                recorder.set_i32(9, matmul.k);
                recorder.set_i32(10, matmul.n);
                self.route_activity.bind(recorder, 11, 12);
                recorder.set_buffer_read(13, self.expert_offsets, 0);
            },
        );
    }
}

impl Operator for GatherGateUpSwiGLUInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        let matmul = config.matmul;
        let shape = self.shape;
        let kernel = if self.route_activity.uses_parameter() {
            &self.kernel.parameterized_kernel
        } else {
            &self.kernel.kernel
        };
        recorder.set_kernel(kernel);
        recorder.set_buffer_read(0, self.gate_weight, 0);
        recorder.set_buffer_read(1, self.gate_scales, 0);
        recorder.set_buffer_read(2, self.gate_biases, 0);
        recorder.set_buffer_read(3, self.up_weight, 0);
        recorder.set_buffer_read(4, self.up_scales, 0);
        recorder.set_buffer_read(5, self.up_biases, 0);
        recorder.set_buffer_read(6, self.input, 0);
        recorder.set_buffer_read(7, self.lhs_indices, 0);
        recorder.set_buffer_read(8, self.rhs_indices, 0);
        recorder.set_buffer_write(9, self.output, 0);
        recorder.set_i32(10, matmul.k);
        recorder.set_i32(11, matmul.n);
        recorder.set_i32(12, config.num_experts);
        if self.route_activity.uses_parameter() {
            self.route_activity.bind(recorder, 13, 14);
        }
        recorder.dispatch_threadblocks(
            (1, ceil_div_i32(matmul.n, 8) as usize, shape.num_routes as usize),
            (32, 2, 1),
        );
    }
}

impl Operator for GatherMatmulInvocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let config = self.kernel.config;
        let matmul = config.matmul;
        let shape = self.shape;
        let packed_k = packed_dim(matmul.k, matmul.bits);
        let groups = matmul.k / matmul.group_size;
        let kernel = if self.route_activity.uses_parameter() {
            &self.kernel.parameterized_kernel
        } else {
            &self.kernel.kernel
        };
        recorder.set_kernel(kernel);
        recorder.set_buffer_read(0, self.weight, 0);
        recorder.set_buffer_read(1, self.scales, 0);
        recorder.set_buffer_read(2, self.biases, 0);
        recorder.set_buffer_read(3, self.input, 0);
        recorder.set_buffer_read(4, self.lhs_indices, 0);
        recorder.set_buffer_read(5, self.rhs_indices, 0);
        recorder.set_buffer_write(6, self.output, 0);
        recorder.set_i32(7, matmul.k);
        recorder.set_i32(8, matmul.n);

        let x_shape = [shape.num_input_vectors, 1, 1, matmul.k];
        let k_stride = i64::from(matmul.k);
        let x_strides = [k_stride, k_stride, k_stride, 1_i64];
        let w_shape = [config.num_experts, matmul.n, packed_k];
        let w_expert_stride = i64::from(matmul.n)
            .checked_mul(i64::from(packed_k))
            .expect("gather affine weight stride must fit i64");
        let affine_expert_stride = i64::from(matmul.n)
            .checked_mul(i64::from(groups))
            .expect("gather affine scale/bias stride must fit i64");
        let w_strides = [w_expert_stride, i64::from(packed_k), 1_i64];
        let affine_strides = [affine_expert_stride, i64::from(groups), 1_i64];
        let batch_shape = [shape.num_routes];
        let route_strides = [1_i64];

        recorder.set_i32(9, 2);
        recorder.set_i32_slice(10, &x_shape);
        recorder.set_i64_slice(11, &x_strides);
        recorder.set_i32(12, 1);
        recorder.set_i32_slice(13, &w_shape);
        recorder.set_i64_slice(14, &w_strides);
        recorder.set_i64_slice(15, &affine_strides);
        recorder.set_i64_slice(16, &affine_strides);
        recorder.set_i32(17, 1);
        recorder.set_i32_slice(18, &batch_shape);
        recorder.set_i64_slice(19, &route_strides);
        recorder.set_i64_slice(20, &route_strides);
        if self.route_activity.uses_parameter() {
            self.route_activity.bind(recorder, 21, 22);
        }

        let bn = 8;
        let bk = 32;
        recorder.dispatch_threadblocks(
            (1, ceil_div_i32(matmul.n, bn) as usize, shape.num_routes as usize),
            (bk as usize, 2, 1),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_gather_buffer_ranges(
    config: ExpertConfig,
    shape: GatherShape,
    output: &Buffer,
    input: &Buffer,
    weight: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    lhs_indices: &Buffer,
    rhs_indices: &Buffer,
) {
    shape.validate();
    let output_bytes = config.output_bytes(shape.num_routes);
    let input_bytes = config.input_bytes(shape.num_input_vectors);
    let weight_bytes = config.weight_bytes();
    let affine_param_bytes = config.affine_param_bytes();
    assert!(
        output_bytes <= output.len_bytes(),
        "gather affine quantized matmul output range out of bounds: shape={shape:?} required_bytes={output_bytes} \
         buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_bytes <= input.len_bytes(),
        "gather affine quantized matmul input range out of bounds: shape={shape:?} required_bytes={input_bytes} \
         buffer_bytes={}",
        input.len_bytes()
    );
    assert!(
        weight.len_bytes() >= weight_bytes,
        "gather affine quantized matmul weight stack too short: config={config:?} shape={shape:?} \
         required_bytes={weight_bytes} buffer_bytes={}",
        weight.len_bytes()
    );
    assert!(
        affine_param_bytes <= scales.len_bytes(),
        "gather affine quantized matmul scales stack too short: config={config:?} shape={shape:?} \
         required_bytes={affine_param_bytes} buffer_bytes={}",
        scales.len_bytes()
    );
    assert!(
        affine_param_bytes <= biases.len_bytes(),
        "gather affine quantized matmul biases stack too short: config={config:?} shape={shape:?} \
         required_bytes={affine_param_bytes} buffer_bytes={}",
        biases.len_bytes()
    );
    let index_bytes = shape.num_routes as usize * size_of::<u32>();
    assert!(
        index_bytes <= lhs_indices.len_bytes(),
        "gather affine quantized matmul lhs index buffer too short: shape={shape:?} required_bytes={index_bytes} \
         buffer_bytes={}",
        lhs_indices.len_bytes()
    );
    assert!(
        index_bytes <= rhs_indices.len_bytes(),
        "gather affine quantized matmul rhs index buffer too short: shape={shape:?} required_bytes={index_bytes} \
         buffer_bytes={}",
        rhs_indices.len_bytes()
    );
}

#[allow(clippy::too_many_arguments)]
fn validate_gather_gate_up_swiglu_buffer_ranges(
    config: ExpertConfig,
    shape: GatherShape,
    output: &Buffer,
    input: &Buffer,
    gate_weight: &Buffer,
    gate_scales: &Buffer,
    gate_biases: &Buffer,
    up_weight: &Buffer,
    up_scales: &Buffer,
    up_biases: &Buffer,
    lhs_indices: &Buffer,
    rhs_indices: &Buffer,
) {
    shape.validate();
    let output_bytes = config.output_bytes(shape.num_routes);
    let input_bytes = config.input_bytes(shape.num_input_vectors);
    let weight_bytes = config.weight_bytes();
    let affine_param_bytes = config.affine_param_bytes();
    assert!(
        output_bytes <= output.len_bytes(),
        "gather affine quantized gate/up/swiglu output range out of bounds: shape={shape:?} \
         required_bytes={output_bytes} buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_bytes <= input.len_bytes(),
        "gather affine quantized gate/up/swiglu input range out of bounds: shape={shape:?} \
         required_bytes={input_bytes} buffer_bytes={}",
        input.len_bytes()
    );
    assert!(
        gate_weight.len_bytes() >= weight_bytes,
        "gather affine quantized gate weight stack too short: config={config:?} shape={shape:?} \
         required_bytes={weight_bytes} buffer_bytes={}",
        gate_weight.len_bytes()
    );
    assert!(weight_bytes <= up_weight.len_bytes());
    assert!(affine_param_bytes <= gate_scales.len_bytes());
    assert!(affine_param_bytes <= gate_biases.len_bytes());
    assert!(affine_param_bytes <= up_scales.len_bytes());
    assert!(affine_param_bytes <= up_biases.len_bytes());
    let index_bytes = shape.num_routes as usize * size_of::<u32>();
    assert!(index_bytes <= lhs_indices.len_bytes());
    assert!(index_bytes <= rhs_indices.len_bytes());
}

#[allow(clippy::too_many_arguments)]
fn validate_ragged_expert_major_gate_up_swiglu_buffer_ranges(
    config: ExpertConfig,
    shape: RaggedExpertMajorShape,
    output: &Buffer,
    input: &Buffer,
    gate_weight: &Buffer,
    gate_scales: &Buffer,
    gate_biases: &Buffer,
    up_weight: &Buffer,
    up_scales: &Buffer,
    up_biases: &Buffer,
    experts_by_route: &Buffer,
    expert_offsets: &Buffer,
) {
    shape.validate();
    let output_bytes = config.output_bytes(shape.num_routes);
    let input_bytes = config.input_bytes(shape.num_routes);
    let weight_bytes = config.weight_bytes();
    let affine_param_bytes = config.affine_param_bytes();
    let route_index_bytes = checked_product(
        "ragged expert-major gate/up route-index byte length",
        &[shape.num_routes as usize, size_of::<u32>()],
    );
    assert!(
        output_bytes <= output.len_bytes(),
        "ragged expert-major gate/up/swiglu output range out of bounds: shape={shape:?} required_bytes={output_bytes} \
         buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_bytes <= input.len_bytes(),
        "ragged expert-major gate/up/swiglu input range out of bounds: shape={shape:?} required_bytes={input_bytes} \
         buffer_bytes={}",
        input.len_bytes()
    );
    assert!(weight_bytes <= gate_weight.len_bytes());
    assert!(affine_param_bytes <= gate_scales.len_bytes());
    assert!(affine_param_bytes <= gate_biases.len_bytes());
    assert!(weight_bytes <= up_weight.len_bytes());
    assert!(affine_param_bytes <= up_scales.len_bytes());
    assert!(affine_param_bytes <= up_biases.len_bytes());
    assert!(route_index_bytes <= experts_by_route.len_bytes());
    assert!((config.num_experts as usize + 1) * size_of::<u32>() <= expert_offsets.len_bytes());
}

#[allow(clippy::too_many_arguments)]
fn validate_ragged_expert_major_down_matmul_buffer_ranges(
    config: ExpertConfig,
    shape: RaggedExpertMajorShape,
    output: &Buffer,
    input: &Buffer,
    weight: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    experts_by_route: &Buffer,
    expert_offsets: &Buffer,
) {
    shape.validate();
    let output_bytes = config.output_bytes(shape.num_routes);
    let input_bytes = config.input_bytes(shape.num_routes);
    let weight_bytes = config.weight_bytes();
    let affine_param_bytes = config.affine_param_bytes();
    let route_index_bytes = checked_product(
        "ragged expert-major down route-index byte length",
        &[shape.num_routes as usize, size_of::<u32>()],
    );
    assert!(
        output_bytes <= output.len_bytes(),
        "ragged expert-major matmul output range out of bounds: shape={shape:?} required_bytes={output_bytes} \
         buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        input_bytes <= input.len_bytes(),
        "ragged expert-major matmul input range out of bounds: shape={shape:?} required_bytes={input_bytes} \
         buffer_bytes={}",
        input.len_bytes()
    );
    assert!(weight_bytes <= weight.len_bytes());
    assert!(affine_param_bytes <= scales.len_bytes());
    assert!(affine_param_bytes <= biases.len_bytes());
    assert!(route_index_bytes <= experts_by_route.len_bytes());
    assert!((config.num_experts as usize + 1) * size_of::<u32>() <= expert_offsets.len_bytes());
}

fn packed_dim(k: i32, bits: i32) -> i32 {
    assert!(k > 0);
    assert!(matches!(bits, 2 | 3 | 4 | 6 | 8));
    let total_bits = k.checked_mul(bits).expect("packed affine dimension must fit i32");
    assert_eq!(total_bits % 32, 0);
    total_bits / 32
}

/// Stable identity for the kernel topology recorded by one affine invocation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelKind {
    QmvBn4Bk32,
    QmvBn8Bk32,
    QmvQuadBn64,
    QmmSlicedK { row_tiles: u32, column_tiles: u32 },
    QmmSlicedKBm3Bn4,
    QmmSimdgroupsBm4Bn32,
    QmmBm8Bn16,
    QmmBm16Bn16,
    QmmBm32Bn16,
    QmmBm8Bn32,
    QmmBm16Bn32,
    QmmBm32Bn32,
    QmmBm64Bn32,
    QmmBm32Bn64,
    QmmBm64Bn64,
}

impl KernelKind {
    fn sliced_k_shape(self) -> Option<(u32, u32)> {
        match self {
            Self::QmmSlicedK {
                row_tiles,
                column_tiles,
            } => Some((row_tiles, column_tiles)),
            _ => None,
        }
    }

    fn small_qmm_shape(self) -> Option<(u32, usize)> {
        match self {
            Self::QmmSlicedKBm3Bn4 => Some((3, 2)),
            Self::QmmSimdgroupsBm4Bn32 => Some((4, 8)),
            _ => None,
        }
    }

    fn qmm_shape(self) -> Option<(i32, i32)> {
        match self {
            Self::QmmBm8Bn16 => Some((8, 16)),
            Self::QmmBm16Bn16 => Some((16, 16)),
            Self::QmmBm32Bn16 => Some((32, 16)),
            Self::QmmBm8Bn32 => Some((8, 32)),
            Self::QmmBm16Bn32 => Some((16, 32)),
            Self::QmmBm32Bn32 => Some((32, 32)),
            Self::QmmBm64Bn32 => Some((64, 32)),
            Self::QmmBm32Bn64 => Some((32, 64)),
            Self::QmmBm64Bn64 => Some((64, 64)),
            Self::QmvBn4Bk32
            | Self::QmvBn8Bk32
            | Self::QmvQuadBn64
            | Self::QmmSlicedK { .. }
            | Self::QmmSlicedKBm3Bn4
            | Self::QmmSimdgroupsBm4Bn32 => None,
        }
    }
}

impl Kernel {
    pub fn new(device: &Device, config: Config, kind: KernelKind) -> Self {
        Self::new_with_epilogue(device, config, kind, Epilogue::Identity)
            .expect("affine kernel exceeds device resources")
    }

    /// Fuses stacked gate/up projection and SwiGLU. `config.n` is twice the output width.
    pub fn new_gate_up_swiglu(device: &Device, config: Config, kind: KernelKind) -> Self {
        Self::new_with_epilogue(device, config, kind, Epilogue::SwiGLU).expect("affine kernel exceeds device resources")
    }

    fn new_with_epilogue(device: &Device, config: Config, kind: KernelKind, epilogue: Epilogue) -> Option<Self> {
        config.validate();
        assert_eq!(config.n % epilogue.output_divisor(), 0);
        validate_kernel_kind(config, kind, epilogue);
        if kind == KernelKind::QmvBn8Bk32 && epilogue == Epilogue::Identity && config.uses_same_dtype() {
            validate_mlx_qmv_output_dim(config.n);
            // MLX computes the packed row numerator and weight row offsets as signed int.
            let bytes_per_pack = if config.bits == 3 || config.bits == 6 { 3 } else { 4 };
            let pack_factor = match config.bits {
                3 => 8,
                6 => 4,
                bits => 32 / bits as u64,
            };
            let row_byte_numerator = config.k as u64 * bytes_per_pack;
            assert!(
                row_byte_numerator <= i32::MAX as u64,
                "MLX QMV packed row byte numerator {row_byte_numerator} must fit i32"
            );
            let weight_row_bytes = row_byte_numerator / pack_factor;
            // This weight byte domain also bounds the smaller scale/bias element offsets.
            let max_weight_byte_index = config.n as u64 * weight_row_bytes - 1;
            assert!(
                max_weight_byte_index <= i32::MAX as u64,
                "MLX QMV packed weight byte index {max_weight_byte_index} must fit i32"
            );
        }
        let sliced_k_threadblock_memory = kind.sliced_k_shape().map(|(row_tiles, column_tiles)| {
            let (num_simdgroups, _) = sliced_k_geometry(config, epilogue);
            epilogue.output_divisor() as usize
                * row_tiles as usize
                * num_simdgroups
                * column_tiles as usize
                * 32
                * size_of::<[f32; 2]>()
        });
        if sliced_k_threadblock_memory.is_some_and(|bytes| bytes > device.max_threadblock_memory_length()) {
            return None;
        }
        let (kernel_name, source) = if matches!(
            kind,
            KernelKind::QmmSlicedK { .. } | KernelKind::QmmSlicedKBm3Bn4 | KernelKind::QmmSimdgroupsBm4Bn32
        ) {
            small_qmm_source(config, kind, epilogue)
        } else {
            match epilogue {
                Epilogue::Identity => affine_kernel_source(config, kind),
                Epilogue::SwiGLU => gate_up_swiglu_source(config, kind),
            }
        };
        let kernel = CompiledKernel::new(device, &source, &kernel_name);
        if let Some(threadblock_memory) = sliced_k_threadblock_memory {
            let (num_simdgroups, _) = sliced_k_geometry(config, epilogue);
            assert_eq!(kernel.thread_execution_width(), 32);
            if 32 * num_simdgroups > kernel.max_total_threads_per_threadblock() {
                return None;
            }
            assert_eq!(kernel.static_threadblock_memory_length(), threadblock_memory);
        } else if let Some((bm, num_simdgroups)) = kind.small_qmm_shape() {
            let sliced_k = kind == KernelKind::QmmSlicedKBm3Bn4;
            assert_eq!(kernel.thread_execution_width(), 32);
            assert!(
                32 * num_simdgroups * epilogue.output_divisor() as usize <= kernel.max_total_threads_per_threadblock()
            );
            let threadblock_memory = if sliced_k || epilogue == Epilogue::SwiGLU {
                epilogue.output_divisor() as usize * num_simdgroups * 4 * bm as usize * size_of::<f32>()
            } else {
                0
            };
            assert!(threadblock_memory <= device.max_threadblock_memory_length());
            assert_eq!(kernel.static_threadblock_memory_length(), threadblock_memory);
        } else if let Some((bm, _)) = kind.qmm_shape() {
            let operand_dtype = if config.uses_same_dtype() {
                config.input_dtype
            } else {
                Dtype::Float32
            };
            validate_qmm_pipeline(
                device,
                operand_dtype,
                kind,
                qmm_bk(config, bm),
                qmm_num_simdgroups(bm, epilogue),
                &kernel,
            );
        }
        Some(Self {
            config,
            epilogue,
            kind,
            kernel,
        })
    }

    pub fn kind(&self) -> KernelKind {
        self.kind
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        num_total_rows: u32,
        num_active_rows: ReplayU32,
        output: &'a Buffer,
        output_offset_bytes: usize,
        input: &'a Buffer,
        input_offset_bytes: usize,
        weight: &'a Buffer,
        weight_offset_bytes: usize,
        scales: &'a Buffer,
        scales_offset_bytes: usize,
        biases: &'a Buffer,
        biases_offset_bytes: usize,
    ) -> Invocation<'a> {
        validate_num_total_rows(num_total_rows);
        if let Some((bm, _)) = self.kind.small_qmm_shape() {
            assert!(num_total_rows <= bm, "small QMM recorded rows must fit its tile");
        }
        if self.kind == KernelKind::QmvBn8Bk32 && self.epilogue == Epilogue::Identity && self.config.uses_same_dtype() {
            let config = self.config;
            let pack_factor = match config.bits {
                3 => 8,
                6 => 4,
                bits => 32 / bits as u64,
            };
            let fast = config.n % 8 == 0 && config.k % 512 == 0;
            let packs_per_lane = if fast && config.bits != 2 { 2 } else { 1 };
            let values_per_lane = pack_factor * packs_per_lane;
            // MLX forms these base offsets as uint before pointer arithmetic.
            let max_input_base = (num_total_rows as u64 - 1) * config.k as u64 + 31 * values_per_lane;
            assert!(
                max_input_base <= u32::MAX as u64,
                "MLX QMV input base element offset {max_input_base} must fit u32"
            );
            let max_output_index = num_total_rows as u64 * config.n as u64 - 1;
            assert!(
                max_output_index <= u32::MAX as u64,
                "MLX QMV output element index {max_output_index} must fit u32"
            );
        }
        validate_buffer_ranges(
            self.config,
            self.epilogue,
            num_total_rows as i32,
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
            scales,
            scales_offset_bytes,
            biases,
            biases_offset_bytes,
        );
        Invocation {
            kernel: self,
            num_total_rows,
            num_active_rows_key: match num_active_rows {
                ReplayU32::Fixed(value) => {
                    assert_eq!(value, num_total_rows);
                    None
                },
                ReplayU32::Parameter(key) => Some(key),
            },
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
            scales,
            scales_offset_bytes,
            biases,
            biases_offset_bytes,
        }
    }
}

impl Matmul {
    pub fn new(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            registry: Registry::new(device, config, Epilogue::Identity),
        }
    }

    /// Fuses stacked gate/up projection and SwiGLU. `config.n` is twice the output width.
    pub fn new_gate_up_swiglu(device: &Device, config: Config) -> Self {
        config.validate();
        Self {
            config,
            registry: Registry::new(device, config, Epilogue::SwiGLU),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn invoke<'a>(
        &'a self,
        num_total_rows: u32,
        num_active_rows: ReplayU32,
        output: &'a Buffer,
        output_offset_bytes: usize,
        input: &'a Buffer,
        input_offset_bytes: usize,
        weight: &'a Buffer,
        weight_offset_bytes: usize,
        scales: &'a Buffer,
        scales_offset_bytes: usize,
        biases: &'a Buffer,
        biases_offset_bytes: usize,
    ) -> Invocation<'a> {
        let (_, kernel) = Selector::select(
            &self.registry,
            self.config,
            i32::try_from(num_total_rows).expect("affine total row count must fit i32"),
        );
        kernel.invoke(
            num_total_rows,
            num_active_rows,
            output,
            output_offset_bytes,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
            scales,
            scales_offset_bytes,
            biases,
            biases_offset_bytes,
        )
    }

    /// Returns the recorded kernel topology for one total row count.
    pub fn topology(&self, num_total_rows: u32) -> KernelKind {
        validate_num_total_rows(num_total_rows);
        Selector::select(
            &self.registry,
            self.config,
            i32::try_from(num_total_rows).expect("affine total row count must fit i32"),
        )
        .0
    }

    /// Returns the first row count for each change in recorded kernel topology.
    pub fn topology_boundaries(&self) -> Box<[u32]> {
        adaptive_topology_boundaries(self.config, |rows| {
            Selector::select(&self.registry, self.config, rows).0
        })
    }

    pub fn selected_kernel(&self, m: i32) -> &Kernel {
        assert!(m > 0);
        Selector::select(&self.registry, self.config, m).1
    }
}

impl Registry {
    fn new(device: &Device, config: Config, epilogue: Epilogue) -> Self {
        let qmv_key = match epilogue {
            Epilogue::Identity => Selector::qmv_key(config),
            Epilogue::SwiGLU => KernelKind::QmvBn4Bk32,
        };
        let qmm = match epilogue {
            Epilogue::Identity => [KernelKind::QmmBm8Bn32, KernelKind::QmmBm16Bn32, KernelKind::QmmBm32Bn32],
            Epilogue::SwiGLU => [KernelKind::QmmBm8Bn16, KernelKind::QmmBm16Bn16, KernelKind::QmmBm32Bn16],
        };
        let kinds = [qmv_key, qmm[0], qmm[1], qmm[2]]
            .into_iter()
            .chain(
                uses_large_qmm(config)
                    .then_some(match epilogue {
                        Epilogue::Identity => [KernelKind::QmmBm32Bn64, KernelKind::QmmBm64Bn64],
                        Epilogue::SwiGLU => [KernelKind::QmmBm32Bn32, KernelKind::QmmBm64Bn32],
                    })
                    .into_iter()
                    .flatten(),
            )
            .chain(uses_small_qmm(config, epilogue).then_some(small_qmm_kind(config, epilogue)))
            .chain(
                (8..=64)
                    .step_by(8)
                    .filter_map(|rows| sliced_k_kind(config, rows, epilogue)),
            );
        let mut entries = Vec::new();
        for kind in kinds {
            if entries.iter().any(|(key, _)| *key == kind) {
                continue;
            }
            if let Some(kernel) = Kernel::new_with_epilogue(device, config, kind, epilogue) {
                entries.push((kind, kernel));
            }
        }
        Self { epilogue, entries }
    }

    fn get(&self, key: KernelKind) -> Option<&Kernel> {
        self.entries
            .iter()
            .find_map(|(candidate_key, kernel)| (*candidate_key == key).then_some(kernel))
    }
}

impl Selector {
    fn select(registry: &Registry, config: Config, num_rows: i32) -> (KernelKind, &Kernel) {
        let key = Self::key(config, num_rows, registry.epilogue);
        if let Some(kernel) = registry.get(key) {
            return (key, kernel);
        }
        if key.sliced_k_shape().is_none() {
            panic!("missing affine quantized matmul execution variant {key:?}");
        }
        let key = Self::base_key(config, num_rows, registry.epilogue);
        (
            key,
            registry
                .get(key)
                .expect("missing base affine quantized matmul execution variant"),
        )
    }

    fn key(config: Config, num_rows: i32, epilogue: Epilogue) -> KernelKind {
        sliced_k_kind(config, num_rows, epilogue).unwrap_or_else(|| Self::base_key(config, num_rows, epilogue))
    }

    fn base_key(config: Config, num_rows: i32, epilogue: Epilogue) -> KernelKind {
        if num_rows == 3 && uses_small_qmm(config, epilogue) {
            return small_qmm_kind(config, epilogue);
        }
        if uses_large_qmm(config) && num_rows > QMM_BM16_MAX_ROWS {
            // A single BM32/BN64 row tile can leave too few independent threadblocks.
            // BM16/BN32 splits both output axes for more parallel work without partial sums.
            if epilogue == Epilogue::Identity
                && num_rows <= 2 * QMM_BM16_MAX_ROWS
                && ceil_div_i32(config.n, 64) < QMM_MIN_THREADBLOCKS
            {
                return KernelKind::QmmBm16Bn32;
            }
            match (epilogue, num_rows >= QMM_LARGE_MIN_ROWS) {
                (Epilogue::Identity, false) => KernelKind::QmmBm32Bn64,
                (Epilogue::Identity, true) => KernelKind::QmmBm64Bn64,
                (Epilogue::SwiGLU, false) => KernelKind::QmmBm32Bn32,
                (Epilogue::SwiGLU, true) => KernelKind::QmmBm64Bn32,
            }
        } else if num_rows < adaptive_qmv_batch_limit(config) {
            match epilogue {
                Epilogue::Identity => Self::qmv_key(config),
                Epilogue::SwiGLU => KernelKind::QmvBn4Bk32,
            }
        } else if num_rows <= QMM_BM8_MAX_ROWS {
            match epilogue {
                Epilogue::Identity => KernelKind::QmmBm8Bn32,
                Epilogue::SwiGLU => KernelKind::QmmBm8Bn16,
            }
        } else if num_rows <= QMM_BM16_MAX_ROWS {
            match epilogue {
                Epilogue::Identity => KernelKind::QmmBm16Bn32,
                Epilogue::SwiGLU => KernelKind::QmmBm16Bn16,
            }
        } else {
            match epilogue {
                Epilogue::Identity => KernelKind::QmmBm32Bn32,
                Epilogue::SwiGLU => KernelKind::QmmBm32Bn16,
            }
        }
    }

    fn qmv_key(config: Config) -> KernelKind {
        if config.uses_same_dtype() && matches!(config.k, 64 | 128) && is_power_of_two(config.bits) {
            KernelKind::QmvQuadBn64
        } else {
            KernelKind::QmvBn8Bk32
        }
    }
}

pub struct Invocation<'a> {
    kernel: &'a Kernel,
    num_total_rows: u32,
    num_active_rows_key: Option<ReplayParameterKey>,
    output: &'a Buffer,
    output_offset_bytes: usize,
    input: &'a Buffer,
    input_offset_bytes: usize,
    weight: &'a Buffer,
    weight_offset_bytes: usize,
    scales: &'a Buffer,
    scales_offset_bytes: usize,
    biases: &'a Buffer,
    biases_offset_bytes: usize,
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let kernel = self.kernel;
        let output = self.output;
        let output_offset_bytes = self.output_offset_bytes;
        let input = self.input;
        let input_offset_bytes = self.input_offset_bytes;
        let weight = self.weight;
        let weight_offset_bytes = self.weight_offset_bytes;
        let scales = self.scales;
        let scales_offset_bytes = self.scales_offset_bytes;
        let biases = self.biases;
        let biases_offset_bytes = self.biases_offset_bytes;
        let config = kernel.config;
        let num_total_rows = self.num_total_rows;
        let num_total_rows_i32 = num_total_rows as i32;
        recorder.set_kernel(&kernel.kernel);
        recorder.set_buffer_read(0, weight, weight_offset_bytes);
        recorder.set_buffer_read(1, scales, scales_offset_bytes);
        recorder.set_buffer_read(2, biases, biases_offset_bytes);
        recorder.set_buffer_read(3, input, input_offset_bytes);
        recorder.set_buffer_write(4, output, output_offset_bytes);
        recorder.set_i32(5, config.k);
        let output_cols = config.n / kernel.epilogue.output_divisor();
        recorder.set_i32(6, output_cols);
        record_num_active_rows(recorder, 7, num_total_rows, self.num_active_rows_key);

        if let Some((row_tiles, column_tiles)) = kernel.kind.sliced_k_shape() {
            let (num_simdgroups, _) = sliced_k_geometry(config, kernel.epilogue);
            recorder.dispatch_threadblocks(
                (
                    ceil_div_i32(num_total_rows_i32, 8 * row_tiles as i32) as usize,
                    output_cols as usize / (8 * column_tiles as usize),
                    1,
                ),
                (32 * num_simdgroups, 1, 1),
            );
        } else if let Some((_, num_simdgroups)) = kernel.kind.small_qmm_shape() {
            let sliced_k = kernel.kind == KernelKind::QmmSlicedKBm3Bn4;
            let columns = if sliced_k { 4 } else { 32 };
            recorder.dispatch_threadblocks(
                (1, output_cols as usize / columns, 1),
                (32 * num_simdgroups * kernel.epilogue.output_divisor() as usize, 1, 1),
            );
        } else if let Some((bm, bn)) = kernel.kind.qmm_shape() {
            recorder.dispatch_threadblocks(
                (
                    ceil_div_i32(output_cols, bn) as usize,
                    ceil_div_i32(num_total_rows_i32, bm) as usize,
                    1,
                ),
                (32, qmm_num_simdgroups(bm, kernel.epilogue), 1),
            );
        } else {
            let (bn, num_simdgroups) = match kernel.kind {
                KernelKind::QmvBn4Bk32 => (4, 2),
                KernelKind::QmvBn8Bk32 => (8, 2),
                KernelKind::QmvQuadBn64 => (64, 1),
                _ => unreachable!(),
            };
            recorder.dispatch_threadblocks(
                (num_total_rows as usize, ceil_div_i32(output_cols, bn) as usize, 1),
                (32, num_simdgroups, 1),
            );
        }
    }
}

fn record_num_active_rows(
    recorder: &CommandRecorder,
    binding_index: usize,
    num_total_rows: u32,
    key: Option<ReplayParameterKey>,
) {
    match key {
        Some(key) => recorder.bind_u32(binding_index, key, 1, num_total_rows),
        None => recorder.set_u32(binding_index, num_total_rows),
    }
}

fn validate_num_total_rows(num_total_rows: u32) {
    assert!(num_total_rows > 0, "affine total row count must be positive");
    assert!(
        i32::try_from(num_total_rows).is_ok(),
        "affine total row count must fit i32"
    );
}

#[allow(clippy::too_many_arguments)]
fn validate_buffer_ranges(
    config: Config,
    epilogue: Epilogue,
    m: i32,
    output: &Buffer,
    output_offset_bytes: usize,
    input: &Buffer,
    input_offset_bytes: usize,
    weight: &Buffer,
    weight_offset_bytes: usize,
    scales: &Buffer,
    scales_offset_bytes: usize,
    biases: &Buffer,
    biases_offset_bytes: usize,
) {
    let output_bytes = checked_bytes(
        "affine matmul output",
        &[m as usize, (config.n / epilogue.output_divisor()) as usize],
        config.output_dtype,
    );
    let input_bytes = config.input_bytes(m);
    let weight_bytes = config.weight_bytes();
    let scale_or_bias_bytes = config.scale_or_bias_bytes();
    for (name, offset_bytes, item_size) in [
        ("output", output_offset_bytes, config.output_dtype.item_size()),
        ("input", input_offset_bytes, config.input_dtype.item_size()),
        ("weight", weight_offset_bytes, size_of::<u32>()),
        ("scales", scales_offset_bytes, config.scale_bias_dtype.item_size()),
        ("biases", biases_offset_bytes, config.scale_bias_dtype.item_size()),
    ] {
        assert!(
            offset_bytes.is_multiple_of(item_size),
            "affine quantized matmul {name} byte offset must align to its dtype"
        );
    }
    assert!(
        checked_range_end("affine quantized matmul output", output_offset_bytes, output_bytes) <= output.len_bytes(),
        "affine quantized matmul output range out of bounds: config={config:?} m={m} \
         offset_bytes={output_offset_bytes} required_bytes={output_bytes} buffer_bytes={}",
        output.len_bytes()
    );
    assert!(
        checked_range_end("affine quantized matmul input", input_offset_bytes, input_bytes) <= input.len_bytes(),
        "affine quantized matmul input range out of bounds: config={config:?} m={m} offset_bytes={input_offset_bytes} \
         required_bytes={input_bytes} buffer_bytes={}",
        input.len_bytes()
    );
    assert!(
        checked_range_end("affine quantized matmul weight", weight_offset_bytes, weight_bytes) <= weight.len_bytes(),
        "affine quantized matmul weight range out of bounds: config={config:?} offset_bytes={weight_offset_bytes} \
         required_bytes={weight_bytes} buffer_bytes={}",
        weight.len_bytes()
    );
    assert!(
        checked_range_end(
            "affine quantized matmul scales",
            scales_offset_bytes,
            scale_or_bias_bytes,
        ) <= scales.len_bytes(),
        "affine quantized matmul scales range out of bounds: config={config:?} offset_bytes={scales_offset_bytes} \
         required_bytes={scale_or_bias_bytes} buffer_bytes={}",
        scales.len_bytes()
    );
    assert!(
        checked_range_end(
            "affine quantized matmul biases",
            biases_offset_bytes,
            scale_or_bias_bytes,
        ) <= biases.len_bytes(),
        "affine quantized matmul biases range out of bounds: config={config:?} offset_bytes={biases_offset_bytes} \
         required_bytes={scale_or_bias_bytes} buffer_bytes={}",
        biases.len_bytes()
    );
}

fn gate_up_swiglu_source(config: Config, kind: KernelKind) -> (String, String) {
    let input_type = metal_type_string(config.input_dtype);
    let output_type = metal_type_string(config.output_dtype);
    let param_type = metal_type_string(config.scale_bias_dtype);
    let shape = kind.qmm_shape();
    let (function, source) = if shape.is_some() {
        ("affine_qmm_gate_up_swiglu", DENSE_GATE_UP_SWIGLU_STEEL_SOURCE)
    } else {
        ("affine_qmv_gate_up_swiglu", DENSE_GATE_UP_SWIGLU_SOURCE)
    };
    let mut args = vec![
        input_type.to_string(),
        param_type.to_string(),
        output_type.to_string(),
        config.group_size.to_string(),
        config.bits.to_string(),
    ];
    if let Some((bm, bn)) = shape {
        args.push(bm.to_string());
        args.push(bn.to_string());
        args.push(qmm_bk(config, bm).to_string());
    } else {
        let bn = if kind == KernelKind::QmvBn4Bk32 { 4 } else { 8 };
        args.push(bn.to_string());
        let pack_factor = match config.bits {
            3 => 8,
            6 => 4,
            bits => 32 / bits,
        };
        let aligned = config.n / 2 % bn == 0 && config.k % (64 * pack_factor) == 0;
        args.push(aligned.to_string());
    }
    let kernel_name = format!("{function}_{}", args.join("_"));
    let definition = template_definition(&kernel_name, function, &args);
    let dependencies = if shape.is_some() {
        MIXED_AFFINE_SOURCE.to_owned()
    } else {
        String::new()
    };
    (
        kernel_name,
        affine_quantized_source(&format!("{dependencies}\n{source}\n{definition}")),
    )
}

fn affine_kernel_source(config: Config, kind: KernelKind) -> (String, String) {
    if let Some((bm, bn)) = kind.qmm_shape() {
        affine_qmm_source(config, bm, bn)
    } else {
        match kind {
            KernelKind::QmvBn8Bk32 => affine_qmv_bn8_bk32_source(config),
            KernelKind::QmvQuadBn64 => affine_qmv_quad_bn64_source(config),
            _ => unreachable!(),
        }
    }
}

fn sliced_k_geometry(config: Config, epilogue: Epilogue) -> (usize, usize) {
    let output_cols = config.n / epilogue.output_divisor();
    if output_cols >= 65_536 {
        (4, 4)
    } else {
        (8, if config.k >= 16_384 { 2 } else { 1 })
    }
}

fn uses_sliced_k(config: Config, epilogue: Epilogue, column_tiles: u32) -> bool {
    let (num_simdgroups, _) = sliced_k_geometry(config, epilogue);
    let output_cols = config.n / epilogue.output_divisor();
    config.uses_same_dtype()
        && matches!(config.input_dtype, Dtype::Bfloat16 | Dtype::Float16)
        && config.bits == 4
        && output_cols >= 1024
        && output_cols % (8 * column_tiles as i32) == 0
        && config.k % (config.group_size * num_simdgroups as i32) == 0
}

// Row tiles reuse packed weights without changing each row's K reduction.
// Keep the unmeasured dtype/group-size domain on the existing Steel policy.
fn sliced_k_kind(config: Config, num_rows: i32, epilogue: Epilogue) -> Option<KernelKind> {
    if !(4..=64).contains(&num_rows) {
        return None;
    }
    let (_, default_column_tiles) = sliced_k_geometry(config, epilogue);
    let output_cols = config.n / epilogue.output_divisor();
    let (row_tiles, mut column_tiles) = if num_rows <= 8 {
        (1, default_column_tiles as u32)
    } else {
        if !uses_large_qmm(config) || output_cols < 4096 {
            return None;
        }
        let tiles = ceil_div_i32(num_rows, 8);
        // Wide projections already expose enough Steel threadblocks at these capacities.
        if (output_cols >= 65_536 && matches!(tiles, 4 | 7 | 8)) || (epilogue == Epilogue::SwiGLU && tiles == 8) {
            return None;
        }
        let row_tiles = match tiles {
            2 => 2,
            3 | 6 => 3,
            4 if epilogue == Epilogue::SwiGLU => 4,
            4 | 8 => 2,
            5 | 7 => 1,
            _ => unreachable!(),
        };
        let column_tiles = if row_tiles == 1 {
            2
        } else if epilogue == Epilogue::SwiGLU {
            1
        } else {
            4
        };
        (row_tiles, column_tiles)
    };
    if num_rows > 8 {
        // Narrow CT until complete column tiles preserve enough independent work.
        while column_tiles > 1
            && (output_cols % (8 * column_tiles as i32) != 0
                || (epilogue == Epilogue::Identity
                    && row_tiles > 1
                    && ceil_div_i32(num_rows, 8 * row_tiles as i32) * (output_cols / (8 * column_tiles as i32))
                        < QMM_MIN_THREADBLOCKS))
        {
            column_tiles /= 2;
        }
    }
    uses_sliced_k(config, epilogue, column_tiles).then_some(KernelKind::QmmSlicedK {
        row_tiles,
        column_tiles,
    })
}

fn uses_small_qmm(config: Config, epilogue: Epilogue) -> bool {
    let output_cols = config.n / epilogue.output_divisor();
    config.uses_same_dtype()
        && matches!(config.input_dtype, Dtype::Bfloat16 | Dtype::Float16)
        && matches!(config.bits, 4 | 8)
        && config.k % 64 == 0
        && output_cols >= 4096
        && output_cols % 4 == 0
}

fn small_qmm_kind(config: Config, epilogue: Epilogue) -> KernelKind {
    let output_cols = config.n / epilogue.output_divisor();
    if output_cols >= 100_000 && output_cols % 32 == 0 {
        KernelKind::QmmSimdgroupsBm4Bn32
    } else {
        KernelKind::QmmSlicedKBm3Bn4
    }
}

fn small_qmm_source(config: Config, kind: KernelKind, epilogue: Epilogue) -> (String, String) {
    let (function, args) = if let Some((row_tiles, column_tiles)) = kind.sliced_k_shape() {
        let (num_simdgroups, _) = sliced_k_geometry(config, epilogue);
        (
            "affine_qmm_sliced_k",
            vec![
                metal_type_string(config.input_dtype).to_string(),
                config.k.to_string(),
                (config.n / epilogue.output_divisor()).to_string(),
                config.group_size.to_string(),
                num_simdgroups.to_string(),
                column_tiles.to_string(),
                (epilogue == Epilogue::SwiGLU).to_string(),
                row_tiles.to_string(),
            ],
        )
    } else {
        let (bm, num_simdgroups) = kind.small_qmm_shape().unwrap();
        let sliced_k = kind == KernelKind::QmmSlicedKBm3Bn4;
        (
            "affine_qmm_small",
            vec![
                metal_type_string(config.input_dtype).to_string(),
                config.group_size.to_string(),
                config.bits.to_string(),
                bm.to_string(),
                num_simdgroups.to_string(),
                sliced_k.to_string(),
                (epilogue == Epilogue::SwiGLU).to_string(),
            ],
        )
    };
    let kernel_name = format!("{function}_{}", args.join("_"));
    let definition = template_definition(&kernel_name, function, &args);
    let source = if let Some((bm, _)) = kind.small_qmm_shape() {
        let (pointers, dot) = small_qmm_dot_source(config.bits, bm, kind == KernelKind::QmmSlicedKBm3Bn4);
        SMALL_QMM_SOURCE
            .replace("/* SMALL_QMM_COLUMN_POINTERS */", &pointers)
            .replace("/* SMALL_QMM_PACKED_DOT */", &dot)
    } else {
        SMALL_QMM_SOURCE.to_string()
    };
    (kernel_name, affine_quantized_source(&format!("{source}\n{definition}")))
}

// Preserve the upstream's literal accumulator indices and column-load order.
// SIMDgroup-column kernels unroll the nibble loop across all four columns.
fn small_qmm_dot_source(bits: i32, bm: u32, sliced_k: bool) -> (String, String) {
    let packed_stride = if bits == 4 { "packs" } else { "K / 4" };
    let mut pointers = vec![
        format!("const device uint* wp0 = w + ulong(n0) * ({packed_stride});"),
        "const device T* sp0 = scales + ulong(n0) * groups;".to_string(),
        "const device T* bp0 = biases + ulong(n0) * groups;".to_string(),
    ];
    for column in 1..4 {
        pointers.push(format!(
            "const device uint* wp{column} = wp0 + {column} * ({packed_stride});"
        ));
        pointers.push(format!("const device T* sp{column} = sp0 + {column} * groups;"));
        pointers.push(format!("const device T* bp{column} = bp0 + {column} * groups;"));
    }
    let mut lines = Vec::new();
    let mut row_loads: Vec<_> = (0..bm).map(|row| format!("Vec8 v{row} = x{row}[pack];")).collect();
    if sliced_k {
        lines.append(&mut row_loads);
    }
    for column in 0..4 {
        if bits == 4 {
            lines.push(format!("uint p{column} = wp{column}[pack];"));
        } else {
            lines.push(format!("uint pa{column} = wp{column}[pack * 2];"));
            lines.push(format!("uint pb{column} = wp{column}[pack * 2 + 1];"));
        }
    }
    if !sliced_k {
        lines.extend(row_loads);
    }
    for column in 0..4 {
        lines.push(format!("float s{column} = float(sp{column}[gi]);"));
        lines.push(format!("float b{column} = float(bp{column}[gi]);"));
    }
    let append_dot = |lines: &mut Vec<String>, column: u32, high: bool| {
        let index = if high { "ki + 4" } else { "ki" };
        let word = if bits == 4 {
            format!("p{column}")
        } else {
            format!("p{}{column}", if high { "b" } else { "a" })
        };
        let shift = bits;
        let mask = if bits == 4 { "0xFu" } else { "0xFFu" };
        lines.push(format!(
            "float w{column} = float(({word} >> (ki * {shift})) & {mask}) * s{column} + b{column};"
        ));
        for row in 0..bm {
            lines.push(format!(
                "acc[{}] += float(v{row}[{index}]) * w{column};",
                column * bm + row
            ));
        }
    };
    let values = 32 / bits;
    if sliced_k {
        for column in 0..4 {
            lines.push(format!("for (int ki = 0; ki < {values}; ++ki) {{"));
            append_dot(&mut lines, column, false);
            if bits == 8 {
                lines.push("{".to_string());
                append_dot(&mut lines, column, true);
                lines.push("}".to_string());
            }
            lines.push("}".to_string());
        }
    } else {
        for high in [false, true].into_iter().take(if bits == 4 { 1 } else { 2 }) {
            lines.push(format!("#pragma unroll\nfor (int ki = 0; ki < {values}; ++ki) {{"));
            for column in 0..4 {
                append_dot(&mut lines, column, high);
            }
            lines.push("}".to_string());
        }
    }
    (pointers.join("\n"), lines.join("\n"))
}

fn affine_qmm_source(config: Config, bm: i32, bn: i32) -> (String, String) {
    let input_type = metal_type_string(config.input_dtype);
    let output_type = metal_type_string(config.output_dtype);
    let scale_bias_type = metal_type_string(config.scale_bias_dtype);
    let bk = qmm_bk(config, bm);
    let kernel_name = format!(
        "affine_qmm_steel_bm{bm}_bn{bn}_bk{bk}_{input_type}_{scale_bias_type}_{output_type}_gs_{}_b_{}",
        config.group_size, config.bits
    );
    let template_definition = template_definition(
        &kernel_name,
        "affine_qmm_steel",
        &[
            input_type.to_string(),
            scale_bias_type.to_string(),
            output_type.to_string(),
            config.group_size.to_string(),
            config.bits.to_string(),
            bm.to_string(),
            bn.to_string(),
            bk.to_string(),
        ],
    );
    (
        kernel_name,
        affine_quantized_source(&format!(
            "{MIXED_AFFINE_SOURCE}\n{QMM_STEEL_SOURCE}\n{template_definition}"
        )),
    )
}

fn affine_qmv_bn8_bk32_source(config: Config) -> (String, String) {
    if !config.uses_same_dtype() {
        let input_type = metal_type_string(config.input_dtype);
        let output_type = metal_type_string(config.output_dtype);
        let scale_bias_type = metal_type_string(config.scale_bias_dtype);
        let fast = config.n % 8 == 0 && config.k % 512 == 0;
        let function_name = if fast { "mixed_qmv_fast" } else { "mixed_qmv" };
        let kernel_name = format!(
            "{function_name}_{input_type}_{scale_bias_type}_{output_type}_gs_{}_b_{}",
            config.group_size, config.bits
        );
        let template_definition = template_definition(
            &kernel_name,
            function_name,
            &[
                input_type.to_string(),
                scale_bias_type.to_string(),
                output_type.to_string(),
                config.group_size.to_string(),
                config.bits.to_string(),
            ],
        );
        return (
            kernel_name,
            affine_quantized_source(&format!("{MIXED_AFFINE_SOURCE}\n{template_definition}")),
        );
    }

    let type_string = metal_type_string(config.input_dtype);
    let fast = config.n % 8 == 0 && config.k % 512 == 0;
    let function_name = if fast { "psi_dec_qmv_fast" } else { "psi_dec_qmv" };
    let kernel_name = format!(
        "{function_name}_{type_string}_gs_{}_b_{}_batch_0",
        config.group_size, config.bits
    );
    let template_definition = template_definition(
        &kernel_name,
        function_name,
        &[
            type_string.to_string(),
            config.group_size.to_string(),
            config.bits.to_string(),
            "false".to_string(),
        ],
    );
    (
        kernel_name,
        affine_quantized_source(&format!("{GUARDED_AFFINE_SOURCE}\n{template_definition}")),
    )
}

fn affine_qmv_quad_bn64_source(config: Config) -> (String, String) {
    let type_string = metal_type_string(config.input_dtype);
    let kernel_name = format!(
        "psi_dec_qmv_quad_{type_string}_gs_{}_b_{}_d_{}_batch_0",
        config.group_size, config.bits, config.k
    );
    let template_definition = template_definition(
        &kernel_name,
        "psi_dec_qmv_quad",
        &[
            type_string.to_string(),
            config.group_size.to_string(),
            config.bits.to_string(),
            config.k.to_string(),
            "false".to_string(),
        ],
    );
    (
        kernel_name,
        affine_quantized_source(&format!("{GUARDED_AFFINE_SOURCE}\n{template_definition}")),
    )
}

fn ceil_div_i32(value: i32, divisor: i32) -> i32 {
    // Callers supply validated positive dimensions and fixed positive tile sizes.
    1 + (value - 1) / divisor
}

fn is_power_of_two(value: i32) -> bool {
    value > 0 && (value & (value - 1)) == 0
}

fn qmv_batch_limit(input_dim: i32, output_dim: i32) -> i32 {
    if input_dim <= 2048 && output_dim <= 2048 {
        18
    } else if input_dim <= 4096 && output_dim <= 4096 {
        12
    } else {
        10
    }
}

fn adaptive_qmv_batch_limit(config: Config) -> i32 {
    if config.input_dtype == Dtype::Bfloat16 && config.n >= 65_536 {
        if config.k <= 2048 { 5 } else { 6 }
    } else if config.n > 4096 || config.k > 4096 {
        6
    } else {
        qmv_batch_limit(config.k, config.n)
    }
}

const EXPERT_QMV_THREADBLOCKS: i32 = 1024;
const EXPERT_QMM_MIN_ROWS: i32 = 4;
const EXPERT_QMM_KINDS: [KernelKind; 3] = [KernelKind::QmmBm8Bn32, KernelKind::QmmBm16Bn32, KernelKind::QmmBm32Bn32];

fn expert_qmm_num_simdgroups(bm: i32) -> usize {
    // The shared expert tile uses WM2 for BM32 and WM1 for smaller rows, with WN2.
    if bm >= 32 { 4 } else { 2 }
}

const QMM_BM8_MAX_ROWS: i32 = 8;
const QMM_BM16_MAX_ROWS: i32 = 16;
const QMM_MIN_THREADBLOCKS: i32 = 256;
const QMM_LARGE_MIN_ROWS: i32 = 2048;
// BK32 is the default for Steel 8x8 fragments and supports every group size.
const QMM_BK: usize = 32;

fn qmm_bk(config: Config, bm: i32) -> usize {
    if bm <= 16
        && config.uses_same_dtype()
        && config.input_dtype == Dtype::Bfloat16
        && config.group_size >= 64
        && config.n < 65_536
    {
        64
    } else {
        QMM_BK
    }
}

fn qmm_num_simdgroups(bm: i32, epilogue: Epilogue) -> usize {
    if bm >= 32 || matches!(epilogue, Epilogue::Identity) {
        4
    } else {
        2
    }
}

fn uses_large_qmm(config: Config) -> bool {
    config.uses_same_dtype()
        && config.input_dtype == Dtype::Bfloat16
        && config.bits == 4
        && config.group_size == 64
        && config.n >= 4096
        && config.k >= 4096
}

fn adaptive_topology_boundaries(config: Config, key: impl Fn(i32) -> KernelKind) -> Box<[u32]> {
    let mut candidates = [
        3,
        4,
        adaptive_qmv_batch_limit(config),
        QMM_BM8_MAX_ROWS + 1,
        QMM_BM16_MAX_ROWS + 1,
        25,
        33,
        41,
        49,
        57,
        65,
        QMM_LARGE_MIN_ROWS,
    ];
    candidates.sort_unstable();
    let mut boundaries = Vec::with_capacity(candidates.len());
    for boundary in candidates {
        if boundary > 1 && key(boundary - 1) != key(boundary) {
            let boundary = boundary as u32;
            if boundaries.last() != Some(&boundary) {
                boundaries.push(boundary);
            }
        }
    }
    boundaries.into_boxed_slice()
}

fn validate_mlx_qmv_output_dim(n: i32) {
    // Below eight columns, MLX's first SIMDgroup loops over N entries in a four-entry result array.
    assert!(
        n <= 4 || n >= 8,
        "MLX QMV output width must be at most four or at least eight"
    );
}

fn validate_kernel_kind(config: Config, kind: KernelKind, epilogue: Epilogue) {
    let supported = match epilogue {
        Epilogue::Identity => {
            matches!(
                kind,
                KernelKind::QmvBn8Bk32
                    | KernelKind::QmvQuadBn64
                    | KernelKind::QmmSlicedK { .. }
                    | KernelKind::QmmSlicedKBm3Bn4
                    | KernelKind::QmmSimdgroupsBm4Bn32
                    | KernelKind::QmmBm8Bn32
                    | KernelKind::QmmBm16Bn32
                    | KernelKind::QmmBm32Bn32
                    | KernelKind::QmmBm64Bn32
                    | KernelKind::QmmBm32Bn64
                    | KernelKind::QmmBm64Bn64
            )
        },
        Epilogue::SwiGLU => {
            matches!(
                kind,
                KernelKind::QmvBn4Bk32
                    | KernelKind::QmvBn8Bk32
                    | KernelKind::QmmSlicedK { .. }
                    | KernelKind::QmmSlicedKBm3Bn4
                    | KernelKind::QmmSimdgroupsBm4Bn32
                    | KernelKind::QmmBm8Bn16
                    | KernelKind::QmmBm16Bn16
                    | KernelKind::QmmBm32Bn16
                    | KernelKind::QmmBm32Bn32
                    | KernelKind::QmmBm64Bn32
            )
        },
    };
    assert!(supported, "affine kernel {kind:?} does not support this epilogue");
    match kind {
        KernelKind::QmvBn4Bk32
        | KernelKind::QmvBn8Bk32
        | KernelKind::QmmBm8Bn16
        | KernelKind::QmmBm16Bn16
        | KernelKind::QmmBm32Bn16
        | KernelKind::QmmBm8Bn32
        | KernelKind::QmmBm16Bn32
        | KernelKind::QmmBm32Bn32
        | KernelKind::QmmBm64Bn32
        | KernelKind::QmmBm32Bn64
        | KernelKind::QmmBm64Bn64 => {},
        KernelKind::QmmSlicedKBm3Bn4 | KernelKind::QmmSimdgroupsBm4Bn32 => {
            assert!(uses_small_qmm(config, epilogue));
            assert_eq!(kind, small_qmm_kind(config, epilogue));
        },
        KernelKind::QmmSlicedK {
            row_tiles,
            column_tiles,
        } => {
            assert!((1..=4).contains(&row_tiles));
            assert!(matches!(column_tiles, 1 | 2 | 4));
            assert!(
                uses_sliced_k(config, epilogue, column_tiles),
                "sliced-K MMA requires aligned Q4 FP16 or BF16 affine operands"
            );
        },
        KernelKind::QmvQuadBn64 => {
            assert!(
                config.uses_same_dtype(),
                "QMV quad BN=64 affine matmul requires one dtype"
            );
            assert!(
                matches!(config.k, 64 | 128) && is_power_of_two(config.bits),
                "QMV quad BN=64 affine matmul requires K=64 or K=128 and power-of-two weight bits"
            );
        },
    }
}

fn validate_qmm_pipeline(
    device: &Device,
    operand_dtype: Dtype,
    kind: KernelKind,
    bk: usize,
    num_simdgroups: usize,
    kernel: &CompiledKernel,
) {
    let (bm, bn) = kind.qmm_shape().unwrap();
    let num_threads = num_simdgroups * kernel.thread_execution_width();
    assert_eq!(
        kernel.thread_execution_width(),
        32,
        "QMM BM={bm} BN={bn} requires a 32-thread SIMDgroup"
    );
    assert!(
        num_threads <= kernel.max_total_threads_per_threadblock(),
        "QMM BM={bm} BN={bn} requires {num_threads} threads, pipeline supports {}",
        kernel.max_total_threads_per_threadblock()
    );
    let item_size = operand_dtype.item_size();
    let bk_padded = bk + 16 / item_size;
    // Dimensions come from the bounded tile family and validated quantization group.
    let expected_threadblock_memory = (bm as usize + bn as usize) * bk_padded * item_size;
    assert!(
        expected_threadblock_memory <= device.max_threadblock_memory_length(),
        "QMM BM={bm} BN={bn} requires {expected_threadblock_memory} bytes of threadblock memory, device supports {}",
        device.max_threadblock_memory_length()
    );
    assert_eq!(
        kernel.static_threadblock_memory_length(),
        expected_threadblock_memory,
        "QMM BM={bm} BN={bn} pipeline threadblock memory does not match its tile"
    );
}

fn metal_type_string(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Float32 => "float",
        Dtype::Float16 => "float16_t",
        Dtype::Bfloat16 => "bfloat16_t",
        _ => panic!("affine quantized matmul dtype must be f32, f16, or bf16"),
    }
}

fn template_definition(kernel_name: &str, function_name: &str, args: &[String]) -> String {
    let instantiation = format!("{function_name}<{}>", args.join(", "));
    format!("\ntemplate [[host_name(\"{kernel_name}\")]] [[kernel]] decltype({instantiation}) {instantiation};\n")
}

fn affine_quantized_source(template_definition: &str) -> String {
    static HEADERS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let headers = HEADERS.get_or_init(|| {
        let root = mlx_metal_header_root();
        let mut included = HashSet::new();
        let mut source = String::new();
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
        source.push_str(&read_mlx_metal_header(
            &root,
            "mlx/backend/metal/kernels/quantized.h",
            &mut included,
        ));
        source
    });
    format!("{headers}{template_definition}")
}

const EXPERT_MAJOR_QMV_SOURCE: &str = include_str!("metal/affine_quantized_expert_major_qmv.metal");

const EXPERT_MAJOR_STEEL_SOURCE: &str = include_str!("metal/affine_quantized_expert_major_steel.metal");

const STEEL_SOURCE: &str = include_str!("metal/affine_quantized_steel.metal");

const QMM_STEEL_SOURCE: &str = include_str!("metal/affine_quantized_qmm_steel.metal");
const SMALL_QMM_SOURCE: &str = include_str!("metal/affine_quantized_small_qmm.metal");

const MIXED_AFFINE_SOURCE: &str = include_str!("metal/affine_quantized_mixed.metal");

const GUARDED_AFFINE_SOURCE: &str = include_str!("metal/affine_quantized_guarded_qmv.metal");

const DENSE_GATE_UP_SWIGLU_SOURCE: &str = include_str!("metal/affine_quantized_gate_up_swiglu.metal");
const DENSE_GATE_UP_SWIGLU_STEEL_SOURCE: &str = include_str!("metal/affine_quantized_gate_up_swiglu_steel.metal");
const GATE_UP_SWIGLU_SOURCE: &str = include_str!("metal/affine_quantized_gate_up_swiglu_qmv.metal");

fn mlx_metal_header_root() -> PathBuf {
    find_mlx_metal_header_root(
        "quantized.h",
        has_compatible_quantized_headers,
        "affine quantized matmul",
    )
}

fn has_compatible_quantized_headers(root: &Path) -> bool {
    let quantized = root.join("mlx/backend/metal/kernels/quantized.h");
    let Ok(content) = std::fs::read_to_string(quantized) else {
        return false;
    };
    content.contains("METAL_FUNC void qmv_quad_impl(")
        && content.contains("METAL_FUNC void qmv_fast_impl(")
        && content.contains("METAL_FUNC void qmv_impl(")
        && content.contains("METAL_FUNC void qmm_t_impl(")
}

#[cfg(test)]
#[path = "affine_quantized_test.rs"]
mod tests;
