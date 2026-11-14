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
    println!("cargo:rerun-if-env-changed=LLMBRIDGE_CPU_PROFILE");
    let requested = env::var("LLMBRIDGE_CPU_PROFILE").unwrap_or_else(|_| "avx2".into());
    assert!(
        matches!(requested.as_str(), "avx2" | "baseline"),
        "LLMBRIDGE_CPU_PROFILE must be avx2 or baseline"
    );
    let avx2 = requested == "avx2";
    let isa = if avx2 { "ON" } else { "OFF" };
    let architecture = env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo target architecture");
    let profile = match (architecture.as_str(), avx2) {
        ("x86_64" | "x86", true) => "x86-avx2",
        ("x86_64", false) => "x86-baseline",
        _ => "unvalidated",
    };
    println!("cargo:rustc-env=LLMBRIDGE_NATIVE_CPU_PROFILE={profile}");
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
        // admission across runners (including AMX). Keep wrapper and ggml on the same explicit ISA profile.
        .define("GGML_NATIVE", "OFF")
        .define("LLMBRIDGE_X86_AVX2", isa)
        .define("GGML_CPU_REPACK", isa)
        .define("GGML_AVX_VNNI", "OFF")
        .define("GGML_AVX512", "OFF")
        .define("GGML_AVX512_VBMI", "OFF")
        .define("GGML_AVX512_VNNI", "OFF")
        .define("GGML_AVX512_BF16", "OFF")
        .define("GGML_AMX_TILE", "OFF")
        .define("GGML_AMX_INT8", "OFF")
        .define("GGML_AMX_BF16", "OFF")
        .define("GGML_SSE42", isa)
        .define("GGML_AVX", isa)
        .define("GGML_AVX2", isa)
        .define("GGML_FMA", isa)
        .define("GGML_F16C", isa)
        .define("GGML_BMI2", isa)
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
