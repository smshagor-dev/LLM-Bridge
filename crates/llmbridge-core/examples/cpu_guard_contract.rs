//! Run under an x86 emulator whose guest CPU lacks the native build's ISA.
use llmbridge_core::{
    gguf_index::GgufTensorIndex,
    inference, native_cpu,
    storage::BoundedLocalReader,
    streamed::{generate_streamed, generate_streamed_verified, StreamedConfig},
    streamed_plan::{inspect_streamed_shape, inspect_streamed_shape_verified},
};
use std::{path::Path, sync::atomic::AtomicBool};
fn rejected<T: std::fmt::Debug>(result: Result<T, String>) {
    assert!(result.unwrap_err().contains("native CPU incompatible"));
}
fn main() {
    let cpu = native_cpu::compatibility();
    assert!(
        !cpu.compatible(),
        "contract requires an incompatible emulated guest CPU"
    );
    let model = Path::new("/llmbridge-absent-cpu-guard-fixture.gguf");
    let cancelled = AtomicBool::new(false);
    rejected(inference::generate(model, "fixture", 128, 8, 1, |_| {
        panic!("output")
    }));
    rejected(inference::generate_cancellable(
        model,
        "fixture",
        128,
        8,
        1,
        &cancelled,
        |_| panic!("output"),
    ));
    rejected(inference::generate_polled(
        model,
        "fixture",
        128,
        8,
        1,
        &cancelled,
        |_| panic!("output"),
    ));
    rejected(generate_streamed(
        model,
        "fixture",
        StreamedConfig::default(),
        &cancelled,
        |_| panic!("output"),
        |_| panic!("logits"),
    ));
    let digest = "0".repeat(64);
    rejected(generate_streamed_verified(
        model,
        "fixture",
        StreamedConfig::default(),
        &cancelled,
        &digest,
        |_| panic!("output"),
        |_| panic!("logits"),
    ));
    rejected(inspect_streamed_shape(model, 65536, &cancelled));
    rejected(inspect_streamed_shape_verified(
        model,
        65536,
        &cancelled,
        &digest,
        8 * 1024 * 1024,
    ));
    // Caller-owned invalid metadata proves CPU admission precedes native parsing,
    // and the operator performs no range I/O before rejecting the CPU.
    let path = std::env::temp_dir().join(format!("llmbridge-cpu-guard-{}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    std::io::Write::write_all(&mut file, b"fixture").unwrap();
    let mut reader = BoundedLocalReader::open(&path, 65536, 131072).unwrap();
    let index = GgufTensorIndex {
        path: path.clone(),
        file_bytes: 7,
        version: 3,
        alignment: 32,
        data_offset: 7,
        tensors: Vec::new(),
        source_identity: reader.identity().clone(),
    };
    rejected(inference::probe_indexed_gguf(&index, &mut reader));
    rejected(inference::q8_0_matvec(
        &index,
        &mut reader,
        "fixture",
        &[1.0; 32],
        &cancelled,
    ));
    rejected(inference::q8_0_matvec_verified(
        &index,
        &mut reader,
        "fixture",
        &[1.0; 32],
        &cancelled,
        &digest,
    ));
    assert_eq!(reader.disk_bytes_read(), 0);
    drop(reader);
    drop(file);
    std::fs::remove_file(path).unwrap();
    println!("LLMBRIDGE_CPU_GUARD {}", cpu.to_json());
    println!("ten Rust native execution/probe paths rejected before FFI: PASS");
}
