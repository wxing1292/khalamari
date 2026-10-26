use clap::Parser;
use inference_error::Result;

use crate::args::Qwen35Args;
use crate::specialization::SpecializedWorker;

pub mod args;
mod specialization;

pub fn launch_dense() {
    launch_or_exit("qwen3_5_dense", "qwen35::run_dense");
}

pub fn launch_sparse() {
    launch_or_exit("qwen3_5_sparse", "qwen35::run_sparse");
}

pub fn launch_qwen4x() {
    launch_or_exit("qwen4x", "qwen4x::run");
}

fn launch_or_exit(binary: &str, entry: &str) {
    if let Err(error) = launch(binary, entry) {
        eprintln!("unable to start {binary}: {error}");
        std::process::exit(1);
    }
}

fn launch(binary: &str, entry: &str) -> Result<()> {
    let num_cache_lanes = Qwen35Args::parse().num_cache_lanes()?;
    SpecializedWorker::new(
        format!("{binary}_lanes_{num_cache_lanes}"),
        format!("fn main() {{ inference_runtime_service::qwen_server::{entry}::<{num_cache_lanes}>(); }}\n"),
    )
    .exec(std::env::args_os().skip(1))
}
