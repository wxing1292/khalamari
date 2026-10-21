//! Backend-neutral attention model metadata and shape contracts.

pub mod gdn;
pub use gdn::GDNCore;

pub mod gqa;
pub use gqa::BiDiBlockCapacity;
pub use gqa::BiDiBlockGQACore;
pub use gqa::BiDiBlockGQAMetadata;
pub use gqa::GQACore;
pub use gqa::UngatedGQACore;

pub mod qsa_indexer;
