#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Bfloat16,
    Complex64,
}

impl Dtype {
    pub fn item_size(self) -> usize {
        match self {
            Self::Bool => size_of::<bool>(),
            Self::Uint8 => size_of::<u8>(),
            Self::Uint16 => size_of::<u16>(),
            Self::Uint32 => size_of::<u32>(),
            Self::Uint64 => size_of::<u64>(),
            Self::Int8 => size_of::<i8>(),
            Self::Int16 => size_of::<i16>(),
            Self::Int32 => size_of::<i32>(),
            Self::Int64 => size_of::<i64>(),
            Self::Float16 => size_of::<u16>(),
            Self::Float32 => size_of::<f32>(),
            Self::Float64 => size_of::<f64>(),
            Self::Bfloat16 => size_of::<u16>(),
            Self::Complex64 => 2 * size_of::<f32>(),
        }
    }
}

pub trait MetalBufferElement: Copy {
    const DTYPE: Dtype;
}

macro_rules! impl_buffer_element {
    ($ty:ty, $dtype:expr) => {
        impl MetalBufferElement for $ty {
            const DTYPE: Dtype = $dtype;
        }
    };
}

impl_buffer_element!(u8, Dtype::Uint8);
impl_buffer_element!(u16, Dtype::Uint16);
impl_buffer_element!(u32, Dtype::Uint32);
impl_buffer_element!(u64, Dtype::Uint64);
impl_buffer_element!(i8, Dtype::Int8);
impl_buffer_element!(i16, Dtype::Int16);
impl_buffer_element!(i32, Dtype::Int32);
impl_buffer_element!(i64, Dtype::Int64);
impl_buffer_element!(f32, Dtype::Float32);
impl_buffer_element!(f64, Dtype::Float64);
