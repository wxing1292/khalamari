use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

const BUNDLED_MLX_SOURCE_DIR: Option<&str> = option_env!("INFERENCE_BACKEND_METAL_BUNDLED_MLX_SOURCE_DIR");

pub fn find_mlx_metal_header_root(
    required_header: &str,
    is_compatible: impl Fn(&Path) -> bool,
    error_context: &str,
) -> PathBuf {
    let rel_path = format!("mlx/backend/metal/kernels/{required_header}");
    mlx_header_candidates()
        .into_iter()
        .find(|candidate| candidate.join(&rel_path).exists() && is_compatible(candidate))
        .unwrap_or_else(|| {
            panic!(
                "{error_context} cannot find compatible MLX Metal headers. Set INFERENCE_BACKEND_METAL_MLX_SOURCE_DIR \
                 to an MLX source root, or INFERENCE_BACKEND_METAL_MLX_PREFIX to a prefix/include root, containing \
                 {rel_path}"
            )
        })
}

pub fn read_mlx_metal_header(root: &Path, rel_path: &str, included: &mut HashSet<String>) -> String {
    if !included.insert(rel_path.to_string()) {
        return String::new();
    }

    let path = root.join(rel_path);
    let content = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!("cannot read MLX Metal header {}: {err}", path.display());
    });
    let mut output = String::new();
    for line in content.lines() {
        let Some(include_path) = quoted_include_path(line) else {
            if !line.contains("#pragma once") {
                output.push_str(line);
                output.push('\n');
            }
            continue;
        };
        if include_path == "bf16.h" {
            output.push_str(&read_mlx_metal_header(
                root,
                "mlx/backend/metal/kernels/metal_3_1/bf16.h",
                included,
            ));
        } else if include_path.starts_with("mlx/backend/metal/kernels/") {
            output.push_str(&read_mlx_metal_header(root, &include_path, included));
        } else {
            output.push_str(line);
            output.push('\n');
        }
    }
    output
}

/// Shared Steel fragment types for paged attention kernels.
pub fn steel_mma_source() -> &'static str {
    static SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SOURCE.get_or_init(|| {
        let root = find_mlx_metal_header_root("steel/gemm/mma.h", |_| true, "GQA Steel MMA");
        read_mlx_metal_header(&root, "mlx/backend/metal/kernels/steel/gemm/mma.h", &mut HashSet::new())
    })
}

fn mlx_header_candidates() -> Vec<PathBuf> {
    let mut candidates = vec![];
    if let Ok(path) = std::env::var("INFERENCE_BACKEND_METAL_MLX_SOURCE_DIR") {
        push_mlx_header_candidates(&mut candidates, PathBuf::from(path));
    }
    if let Ok(prefix) = std::env::var("INFERENCE_BACKEND_METAL_MLX_PREFIX") {
        push_mlx_header_candidates(&mut candidates, PathBuf::from(prefix));
    }
    if let Some(path) = BUNDLED_MLX_SOURCE_DIR {
        push_mlx_header_candidates(&mut candidates, PathBuf::from(path));
    }
    candidates
}

fn push_mlx_header_candidates(candidates: &mut Vec<PathBuf>, path: PathBuf) {
    candidates.push(path.clone());
    candidates.push(path.join("include"));
}

fn quoted_include_path(line: &str) -> Option<String> {
    let include_pos = line.find("#include \"")?;
    let path_start = include_pos + "#include \"".len();
    let path_end = line[path_start..].find('"')? + path_start;
    Some(line[path_start..path_end].to_string())
}
