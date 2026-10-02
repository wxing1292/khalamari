use crate::components::gqa::split_kv::single_q;
use crate::components::qsa_sparse_gqa;
use crate::metal::Buffer;
use crate::metal::CommandRecorder;
use crate::metal::Operator;

/// Records dense and sparse QSA maps before their output reductions.
/// The caller must partition active query rows into disjoint dense and sparse
/// sets. Each writer scatters to its original output rows. Partial buffers
/// retain their Map-to-Reduce dependencies inside the disjoint output scope.
pub struct Invocation<'a> {
    output: &'a Buffer,
    dense_map: single_q::MapInvocation<'a>,
    sparse_map: qsa_sparse_gqa::MapInvocation<'a>,
    dense_reduce: single_q::MappedReduceInvocation<'a>,
    sparse_reduce: Option<single_q::MappedReduceInvocation<'a>>,
}

impl<'a> Invocation<'a> {
    pub fn new(
        output: &'a Buffer,
        dense_map: single_q::MapInvocation<'a>,
        sparse_map: qsa_sparse_gqa::MapInvocation<'a>,
        dense_reduce: single_q::MappedReduceInvocation<'a>,
        sparse_reduce: Option<single_q::MappedReduceInvocation<'a>>,
    ) -> Self {
        Self {
            output,
            dense_map,
            sparse_map,
            dense_reduce,
            sparse_reduce,
        }
    }
}

impl Operator for Invocation<'_> {
    fn record(self, recorder: &CommandRecorder<'_>) {
        let Self {
            output,
            dense_map,
            sparse_map,
            dense_reduce,
            sparse_reduce,
        } = self;
        recorder.record_disjoint_buffers(&[output], || {
            dense_map.record(recorder);
            sparse_map.record(recorder);
            dense_reduce.record(recorder);
            if let Some(sparse_reduce) = sparse_reduce {
                sparse_reduce.record(recorder);
            }
        });
    }
}
