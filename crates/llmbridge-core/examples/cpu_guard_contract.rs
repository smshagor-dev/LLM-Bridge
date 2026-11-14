//! Run under an x86 emulator whose guest CPU lacks the native build's ISA.
use llmbridge_core::{
    gguf_index::GgufTensorIndex,
    inference, native_cpu,
    storage::BoundedLocalReader,
    streamed::{
        generate_streamed, generate_streamed_controlled, generate_streamed_verified, CacheControl,
        StreamedConfig, StreamedControl,
    },
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
    let control = CacheControl::default();
    control.request_release();
    for expected_sha256 in [None, Some(digest.as_str())] {
        rejected(generate_streamed_controlled(
            model,
            "fixture",
            StreamedConfig::default(),
            &cancelled,
            StreamedControl {
                expected_sha256,
                cache: Some(&control),
            },
            |_| panic!("output"),
            |_| panic!("logits"),
        ));
    }
    let budget = llmbridge_core::memory_budget::MemoryBudget::new(1).unwrap();
    let queue = llmbridge_core::admission_queue::AdmissionQueue::new(budget.clone(), 1).unwrap();
    rejected(llmbridge_core::queued_streamed::generate_streamed_queued(
        &queue,
        llmbridge_core::queued_streamed::QueuedStreamedRequest::new(model, "fixture"),
        &cancelled,
        |_| panic!("output"),
        |_| panic!("logits"),
    ));
    assert_eq!(queue.pending(), 0);
    assert_eq!(budget.reserved(), 0);
    let hardware = llmbridge_core::hardware::Hardware {
        logical_cpus: 1,
        total_ram_bytes: Some(1024),
        available_ram_bytes: Some(1024),
    };
    let executor = llmbridge_core::streamed_executor::StreamedExecutor::new(
        &hardware,
        llmbridge_core::streamed_executor::ExecutorConfig {
            requested_workers: 1,
            session_slot_bytes: 1,
            os_reserve_bytes: 0,
            max_pending: 1,
        },
    )
    .unwrap();
    rejected(executor.generate(
        llmbridge_core::queued_streamed::QueuedStreamedRequest::new(model, "fixture"),
        &cancelled,
        |_| panic!("output"),
        |_| panic!("logits"),
    ));
    assert_eq!(executor.pending_sessions(), 0);
    assert_eq!(executor.reserved_bytes(), 0);
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
    println!("fourteen Rust native execution/probe paths rejected before FFI: PASS");
}
