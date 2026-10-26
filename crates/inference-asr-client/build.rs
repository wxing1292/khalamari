use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").unwrap() != "macos" {
        return;
    }
    if env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "aarch64" {
        // The pinned Apple Silicon ORT archive targets macOS 13.4.
        println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
        let deployment = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "13.4".into());
        let mut version = deployment
            .split('.')
            .map(|part| part.parse::<u32>().expect("invalid MACOSX_DEPLOYMENT_TARGET"));
        assert!(
            (version.next().unwrap(), version.next().unwrap_or(0)) >= (13, 4),
            "the static ORT archive requires MACOSX_DEPLOYMENT_TARGET >= 13.4"
        );
        println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET={deployment}");
    }
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    // The static ORT archive uses Objective-C availability checks supplied by Clang's compiler runtime.
    let output = Command::new("xcrun")
        .args(["clang", "--print-file-name=libclang_rt.osx.a"])
        .output()
        .expect("Xcode command-line tools must provide xcrun and clang");
    assert!(output.status.success(), "clang must locate its macOS compiler runtime");
    let library = PathBuf::from(
        String::from_utf8(output.stdout)
            .expect("clang output must be UTF-8")
            .trim(),
    );
    assert!(
        library.is_file(),
        "Clang compiler runtime is missing: {}",
        library.display()
    );
    println!("cargo:rerun-if-changed={}", library.display());
    println!("cargo:rustc-link-search=native={}", library.parent().unwrap().display());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
}
