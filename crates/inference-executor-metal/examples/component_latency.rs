//! Host-only orchestration around the existing component measurement binaries.
#[path = "../benches/components/matrix.rs"]
mod matrix;

fn main() {
    if let Err(error) = matrix::main() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
