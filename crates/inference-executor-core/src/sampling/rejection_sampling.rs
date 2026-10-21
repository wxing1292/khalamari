pub trait SpecMicrobatch {
    fn num_reqs(&self) -> usize;
    fn is_decode_req(&self, req_index: usize) -> bool;
    fn num_spec_tokens(&self, req_index: usize) -> u32;
    fn req_slots(&self) -> &[u32];
    fn token_indices(&self) -> &[u32];
    fn cu_tokens(&self) -> &[u32];
    fn flat_token_ids(&self) -> &[i32];
}
