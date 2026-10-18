use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Dtype;
use inference_backend_metal::operators::affine_quantized;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantizedAffineLayout {
    pub group_size: u32,
    pub bits: u32,
    pub scale_bias_dtype: Dtype,
}

impl QuantizedAffineLayout {
    pub fn validate(self) {
        assert!(matches!(self.group_size, 32 | 64 | 128));
        assert!(matches!(self.bits, 2 | 3 | 4 | 6 | 8));
        assert!(matches!(self.scale_bias_dtype, Dtype::Float32 | Dtype::Bfloat16));
    }

    pub fn config(self, output_dim: usize, input_dim: usize, io_dtype: Dtype) -> affine_quantized::Config {
        self.validate();
        affine_quantized::Config {
            n: output_dim.try_into().expect("affine output dimension must fit i32"),
            k: input_dim.try_into().expect("affine input dimension must fit i32"),
            group_size: self.group_size.try_into().expect("affine group size must fit i32"),
            bits: self.bits.try_into().expect("affine bit width must fit i32"),
            input_dtype: io_dtype,
            output_dtype: io_dtype,
            scale_bias_dtype: self.scale_bias_dtype,
        }
    }
}

/// Quantized affine operands with byte offsets from the start of each buffer.
#[derive(Clone, Copy)]
pub struct QuantizedAffineWeights<'a> {
    pub weight: &'a Buffer,
    pub weight_offset_bytes: usize,
    pub scales: &'a Buffer,
    pub scales_offset_bytes: usize,
    pub biases: &'a Buffer,
    pub biases_offset_bytes: usize,
}

impl<'a> QuantizedAffineWeights<'a> {
    pub fn new(weight: &'a Buffer, scales: &'a Buffer, biases: &'a Buffer) -> Self {
        Self {
            weight,
            weight_offset_bytes: 0,
            scales,
            scales_offset_bytes: 0,
            biases,
            biases_offset_bytes: 0,
        }
    }
}
