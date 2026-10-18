pub mod dflash2;
pub mod layer;
pub mod spec_decode_input;
pub mod state;
pub mod weight;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpecReplayStageEnds {
    pub decode_prepare: Option<usize>,
    pub prefill: usize,
    pub decode: Option<usize>,
}
