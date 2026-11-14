use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn walk(directory: &Path, found: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, found);
            } else if path.is_file() {
                found.push(path);
            }
        }
    }
}

fn main() {
    if env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        return;
    }
    let native = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../native");
    println!("cargo:rerun-if-changed={}", native.join("src").display());
    println!(
        "cargo:rerun-if-changed={}",
        native.join("include/llmbridge.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native.join("CMakeLists.txt").display()
    );
    let out = cmake::Config::new(native)
        .profile("Release")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_BLAS", "OFF")
        // A host-specific -march=native build changes binary mappings and memory
        // admission across runners (including AMX). Use a reproducible baseline.
        .define("GGML_NATIVE", "OFF")
        .define("GGML_SSE42", "OFF")
        .define("GGML_AVX", "OFF")
        .define("GGML_AVX2", "OFF")
        .define("GGML_FMA", "OFF")
        .define("GGML_F16C", "OFF")
        .define("GGML_BMI2", "OFF")
        .build();
    let mut paths = Vec::new();
    walk(&out, &mut paths);
    let extension = if cfg!(windows) { ".lib" } else { ".a" };
    for name in ["llmbridge_native", "llama", "ggml", "ggml-cpu", "ggml-base"] {
        let filename = if cfg!(windows) {
            format!("{name}{extension}")
        } else {
            format!("lib{name}{extension}")
        };
        let path = paths
            .iter()
            .find(|p| p.file_name().is_some_and(|v| v == filename.as_str()))
            .unwrap_or_else(|| {
                panic!(
                    "missing native static library {filename} under {}",
                    out.display()
                )
            });
        println!(
            "cargo:rustc-link-search=native={}",
            path.parent().unwrap().display()
        );
        println!("cargo:rustc-link-lib=static={name}");
    }
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    }
}
