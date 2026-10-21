use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::sampling::SamplerConfig;

use crate::sampling::top_k_sampling::config::TopKSamplingBounds;
use crate::sampling::top_k_sampling::config::TopKSamplingShape;

const PARAM_STRIDE: usize = 4;

/// Sampling parameters indexed by stable request slot.
///
/// Sampling-row request slots, positions, and RNG domains are invocation
/// inputs. They are not part of this persistent request-level store.
pub struct SamplingParamsStore {
    bounds: TopKSamplingBounds,
    num_req_slots: u32,
    buffer: Buffer,
}

impl SamplingParamsStore {
    pub fn new(device: &Device, bounds: TopKSamplingBounds, num_req_slots: u32) -> Self {
        bounds.validate();
        assert!(num_req_slots > 0, "sampling parameters require request slots");
        Self {
            bounds,
            num_req_slots,
            buffer: Buffer::new_zeroed_elements(
                device,
                (num_req_slots as usize)
                    .checked_mul(PARAM_STRIDE)
                    .expect("sampling parameter capacity must fit usize"),
                Dtype::Uint32,
            ),
        }
    }

    pub fn set(&self, req_slots: &[u32], configs: &[SamplerConfig]) {
        assert_eq!(
            req_slots.len(),
            configs.len(),
            "sampling parameters require one request slot per config"
        );
        for (&req_slot, config) in req_slots.iter().zip(configs) {
            assert!(
                req_slot < self.num_req_slots,
                "sampling parameter request slot exceeds capacity"
            );
            self.buffer.write_typed(
                req_slot as usize * PARAM_STRIDE,
                &[
                    config.temperature.to_bits(),
                    config.top_p.to_bits(),
                    config.seed(),
                    self.bounds
                        .active_top_k(config)
                        .expect("sampling config must fit sampler bounds"),
                ],
            );
        }
    }

    pub fn active_shape(&self, configs: &[SamplerConfig]) -> TopKSamplingShape {
        self.bounds
            .active_shape(configs)
            .expect("sampling config must fit sampler bounds")
    }

    pub fn bounds(&self) -> TopKSamplingBounds {
        self.bounds
    }

    pub fn num_req_slots(&self) -> u32 {
        self.num_req_slots
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Device;

    use super::*;

    #[test]
    fn test_set_success() {
        let store = SamplingParamsStore::new(
            &Device::system_default(),
            TopKSamplingBounds {
                max_sampling_inputs: 4,
                vocab_size: 128,
                top_k: 16,
            },
            4,
        );
        let configs = [
            SamplerConfig {
                temperature: 0.75,
                top_k: 8,
                top_p: 0.9,
                seed: 11,
            },
            SamplerConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: 7,
            },
        ];

        store.set(&[3, 1], &configs);

        assert_eq!(
            store.buffer().read_typed::<u32>(PARAM_STRIDE, PARAM_STRIDE),
            [0.0_f32.to_bits(), 1.0_f32.to_bits(), 7, 1]
        );
        assert_eq!(
            store.buffer().read_typed::<u32>(3 * PARAM_STRIDE, PARAM_STRIDE),
            [0.75_f32.to_bits(), 0.9_f32.to_bits(), 11, 8]
        );
    }
}
