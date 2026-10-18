use std::cell::Cell;
use std::mem::size_of;

use inference_backend_metal::metal::Buffer;
use inference_backend_metal::metal::Device;
use inference_backend_metal::metal::Dtype;
use inference_executor_core::attn::GDNReplayShape;

/// Capacity-sized GPU metadata and replay shape refreshed during GDN state
/// preparation and shared by all GDN layers.
pub struct GDNMetadataBuffers {
    cu_tokens: Buffer,
    src_recurrent_state_slots: Buffer,
    src_conv_state_slots: Buffer,
    flat_recurrent_state_write_slots: Buffer,
    flat_conv_state_write_slots: Buffer,
    replay_shape: Cell<Option<GDNReplayShape>>,
    num_active_recurrent_requests: Cell<u32>,
}

impl GDNMetadataBuffers {
    pub fn new(device: &Device, max_requests: usize, max_tokens: usize) -> Self {
        assert!(max_requests > 0, "GDN batch metadata requires requests");
        assert!(max_tokens > 0, "GDN batch metadata requires tokens");
        assert!(u32::try_from(max_requests).is_ok(), "GDN request capacity must fit u32");
        assert!(u32::try_from(max_tokens).is_ok(), "GDN token capacity must fit u32");
        Self {
            cu_tokens: Buffer::new_zeroed_elements(
                device,
                max_requests
                    .checked_add(1)
                    .expect("GDN cumulative-token capacity must fit usize"),
                Dtype::Uint32,
            ),
            src_recurrent_state_slots: Buffer::new_zeroed_elements(device, max_requests, Dtype::Uint32),
            src_conv_state_slots: Buffer::new_zeroed_elements(device, max_requests, Dtype::Uint32),
            flat_recurrent_state_write_slots: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            flat_conv_state_write_slots: Buffer::new_zeroed_elements(device, max_tokens, Dtype::Uint32),
            replay_shape: Cell::new(None),
            num_active_recurrent_requests: Cell::new(0),
        }
    }

    pub fn cu_tokens(&self) -> &Buffer {
        &self.cu_tokens
    }

    pub fn num_active_recurrent_requests(&self) -> u32 {
        self.num_active_recurrent_requests.get()
    }

    pub fn src_recurrent_state_slots(&self) -> &Buffer {
        &self.src_recurrent_state_slots
    }

    pub fn src_conv_state_slots(&self) -> &Buffer {
        &self.src_conv_state_slots
    }

    pub fn flat_recurrent_state_write_slots(&self) -> &Buffer {
        &self.flat_recurrent_state_write_slots
    }

    pub fn flat_conv_state_write_slots(&self) -> &Buffer {
        &self.flat_conv_state_write_slots
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &self,
        cu_tokens: &[u32],
        num_active_recurrent_requests: u32,
        src_recurrent_state_slots: &[u32],
        src_conv_state_slots: &[u32],
        flat_recurrent_state_write_slots: &[u32],
        flat_conv_state_write_slots: &[u32],
        num_total_requests: u32,
        num_total_tokens: u32,
    ) -> GDNReplayShape {
        assert!(
            !src_recurrent_state_slots.is_empty(),
            "GDN batch metadata requires requests"
        );
        assert_eq!(src_recurrent_state_slots.len(), src_conv_state_slots.len());
        assert_eq!(cu_tokens.len(), src_recurrent_state_slots.len() + 1);
        assert!(src_recurrent_state_slots.len() < self.cu_tokens.len_bytes() / size_of::<u32>());
        assert_eq!(cu_tokens[0], 0, "GDN batch cu_tokens must start at zero");
        assert!(
            cu_tokens.windows(2).all(|window| window[0] < window[1]),
            "GDN batch cu_tokens must assign at least one token to every request"
        );
        let num_tokens = cu_tokens[cu_tokens.len() - 1];
        let num_tokens_usize = num_tokens as usize;
        assert_eq!(flat_recurrent_state_write_slots.len(), num_tokens_usize);
        assert_eq!(flat_conv_state_write_slots.len(), num_tokens_usize);
        assert!(num_tokens_usize <= self.flat_recurrent_state_write_slots.len_bytes() / size_of::<u32>());
        let num_reqs = src_recurrent_state_slots.len() as u32;
        assert!(
            num_active_recurrent_requests <= num_reqs,
            "GDN recurrent prefix must fit the active requests"
        );
        assert!(
            num_reqs <= num_total_requests,
            "GDN active request count must not exceed the total request count"
        );
        assert!(
            num_total_requests as usize <= self.src_recurrent_state_slots.len_bytes() / size_of::<u32>(),
            "GDN total request count must not exceed the metadata capacity"
        );
        assert!(
            num_tokens <= num_total_tokens,
            "GDN active token count must not exceed the total token count"
        );
        assert!(
            num_total_tokens as usize <= self.flat_recurrent_state_write_slots.len_bytes() / size_of::<u32>(),
            "GDN total token count must not exceed the metadata capacity"
        );
        let replay_shape = GDNReplayShape::new(num_reqs, num_total_requests, num_tokens, num_total_tokens);

        self.cu_tokens.write_typed(0, cu_tokens);
        self.src_recurrent_state_slots.write_typed(0, src_recurrent_state_slots);
        self.src_conv_state_slots.write_typed(0, src_conv_state_slots);
        self.flat_recurrent_state_write_slots
            .write_typed(0, flat_recurrent_state_write_slots);
        self.flat_conv_state_write_slots
            .write_typed(0, flat_conv_state_write_slots);
        self.replay_shape.set(Some(replay_shape));
        self.num_active_recurrent_requests.set(num_active_recurrent_requests);
        replay_shape
    }

    pub fn replay_shape(&self) -> GDNReplayShape {
        self.replay_shape
            .get()
            .expect("GDN batch metadata must be updated before recording")
    }
}

#[cfg(test)]
mod tests {
    use inference_backend_metal::metal::Device;
    use inference_executor_core::attn::GDNReplayShape;

    use super::GDNMetadataBuffers;
    #[test]
    fn test_metadata_preserves_active_work_across_total_capacities() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 6, 8);
        let cu_tokens = [0, 1, 3, 7];
        let src_recurrent_state_slots = [10, 11, 12];
        let src_conv_state_slots = [20, 21, 22];
        let flat_recurrent_state_write_slots = [u32::MAX, 30, 31, u32::MAX, 32, 33, 34];
        let flat_conv_state_write_slots = [u32::MAX, 40, 41, u32::MAX, 42, 43, 44];

        let exact = metadata.update(
            &cu_tokens,
            0,
            &src_recurrent_state_slots,
            &src_conv_state_slots,
            &flat_recurrent_state_write_slots,
            &flat_conv_state_write_slots,
            3,
            7,
        );
        assert_eq!(exact, GDNReplayShape::new(3, 3, 7, 7));

        let padded = metadata.update(
            &cu_tokens,
            0,
            &src_recurrent_state_slots,
            &src_conv_state_slots,
            &flat_recurrent_state_write_slots,
            &flat_conv_state_write_slots,
            4,
            8,
        );
        assert_eq!(padded, GDNReplayShape::new(3, 4, 7, 8));

        let caller_sized = metadata.update(
            &cu_tokens,
            0,
            &src_recurrent_state_slots,
            &src_conv_state_slots,
            &flat_recurrent_state_write_slots,
            &flat_conv_state_write_slots,
            4,
            7,
        );
        assert_eq!(caller_sized, GDNReplayShape::new(3, 4, 7, 7));
        assert_eq!(caller_sized, metadata.replay_shape());
        assert_eq!(metadata.cu_tokens().read_typed::<u32>(0, 4), cu_tokens);
        assert_eq!(
            metadata.src_recurrent_state_slots().read_typed::<u32>(0, 3),
            src_recurrent_state_slots
        );
        assert_eq!(
            metadata.src_conv_state_slots().read_typed::<u32>(0, 3),
            src_conv_state_slots
        );
        assert_eq!(
            metadata.flat_recurrent_state_write_slots().read_typed::<u32>(0, 7),
            flat_recurrent_state_write_slots
        );
        assert_eq!(
            metadata.flat_conv_state_write_slots().read_typed::<u32>(0, 7),
            flat_conv_state_write_slots
        );
    }

    #[test]
    #[should_panic(expected = "GDN active token count must not exceed the total token count")]
    fn test_update_rejects_small_total_token_count() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 1, 8);

        metadata.update(&[0, 7], 0, &[10], &[20], &[u32::MAX; 7], &[u32::MAX; 7], 1, 6);
    }

    #[test]
    #[should_panic(expected = "GDN total token count must not exceed the metadata capacity")]
    fn test_update_rejects_large_total_token_count() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 1, 8);

        metadata.update(&[0, 1], 0, &[10], &[20], &[u32::MAX], &[u32::MAX], 1, 9);
    }

    #[test]
    #[should_panic(expected = "GDN batch cu_tokens must assign at least one token to every request")]
    fn test_update_rejects_empty_request_window() {
        let device = Device::system_default();
        let metadata = GDNMetadataBuffers::new(&device, 2, 2);

        metadata.update(&[0, 1, 1], 0, &[3, 4], &[5, 6], &[7], &[8], 2, 1);
    }
}
