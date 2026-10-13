//! Standalone GQA matrix benchmark. The worker shares the original measurement implementation.
#[path = "../benches/gqa/matrix/controller.rs"]
mod controller;
#[path = "../benches/gqa/matrix.rs"]
mod worker;

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--worker") {
        worker::main();
        return;
    }
    if let Err(error) = controller::main() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
