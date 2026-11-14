//! Actual model execution holds reservations; waiting/cancellation performs no model load.
use llmbridge_core::{
    admission_queue::AdmissionQueue,
    memory_budget::MemoryBudget,
    queued_streamed::{generate_streamed_queued, QueuedStreamedRequest},
    streamed::generate_streamed_controlled,
};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("queued_streamed_contract <pinned.gguf> [trusted-sha256]");
    let digest = std::env::args().nth(2).unwrap_or_else(|| {
        "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6".into()
    });
    let mut request = QueuedStreamedRequest::new(Path::new(&model), "The sky is");
    request.config.max_tokens = 4;
    request.control.expected_sha256 = Some(&digest);
    let charge = request.reservation_bytes().unwrap();
    let budget = MemoryBudget::new(charge).unwrap();
    let queue = Arc::new(AdmissionQueue::new(Arc::clone(&budget), 4).unwrap());

    // Holding all capacity must prevent even a nonexistent model from opening.
    let held = budget.try_reserve(charge).unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let worker_queue = Arc::clone(&queue);
    let worker = std::thread::spawn(move || {
        let mut missing =
            QueuedStreamedRequest::new(Path::new("/llmbridge-queue-missing-model.gguf"), "fixture");
        missing.config.max_tokens = 4;
        generate_streamed_queued(
            &worker_queue,
            missing,
            &worker_cancelled,
            |_| panic!("output while waiting"),
            |_| panic!("logits while waiting"),
        )
    });
    let start = Instant::now();
    while queue.pending() != 1 {
        assert!(!worker.is_finished(), "model was opened before admission");
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "waiting request did not enqueue"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(budget.reserved(), charge);
    cancelled.store(true, Ordering::Release);
    assert!(worker
        .join()
        .unwrap()
        .unwrap_err()
        .contains("cancelled while waiting"));
    assert_eq!(queue.pending(), 0);
    assert_eq!(budget.reserved(), charge);
    drop(held);

    let mut direct_bytes = Vec::new();
    let mut direct_logits = Sha256::new();
    generate_streamed_controlled(
        request.model,
        request.prompt,
        request.config,
        &AtomicBool::new(false),
        request.control,
        |b| {
            direct_bytes.extend_from_slice(b);
            true
        },
        |v| {
            for x in v {
                direct_logits.update(x.to_le_bytes());
            }
            true
        },
    )
    .unwrap();
    let mut output = Vec::new();
    let mut logits = Sha256::new();
    let metrics = generate_streamed_queued(
        &queue,
        request,
        &AtomicBool::new(false),
        |b| {
            assert_eq!(budget.reserved(), charge);
            output.extend_from_slice(b);
            true
        },
        |v| {
            assert_eq!(budget.reserved(), charge);
            for x in v {
                logits.update(x.to_le_bytes());
            }
            true
        },
    )
    .unwrap();
    assert_eq!(output, direct_bytes);
    assert_eq!(logits.finalize(), direct_logits.finalize());
    assert!(metrics.inference.authentication.is_some());
    assert_eq!(metrics.reservation_bytes, charge);
    assert_eq!(budget.reserved(), 0);
    assert_eq!(queue.pending(), 0);

    // Both engine rejection and consumer cancellation must release the reservation.
    let mut invalid = request;
    invalid.model = Path::new("/llmbridge-queue-missing-model.gguf");
    assert!(
        generate_streamed_queued(&queue, invalid, &AtomicBool::new(false), |_| true, |_| true)
            .is_err()
    );
    assert_eq!(budget.reserved(), 0);
    let mut tiny = request;
    tiny.config.workspace_bytes = 1;
    let failed_native = generate_streamed_queued(
        &queue,
        tiny,
        &AtomicBool::new(false),
        |_| panic!("output before native workspace rejection"),
        |_| panic!("logits before native workspace rejection"),
    );
    assert!(failed_native
        .unwrap_err()
        .contains("workspace exceeds budget"));
    assert_eq!(budget.reserved(), 0);
    assert_eq!(queue.pending(), 0);
    let cancelled = generate_streamed_queued(
        &queue,
        request,
        &AtomicBool::new(false),
        |_| panic!("output after consumer cancellation"),
        |_| false,
    );
    assert!(cancelled.unwrap_err().contains("cancelled"));
    assert_eq!(budget.reserved(), 0);
    assert_eq!(queue.pending(), 0);
    let undersized = MemoryBudget::new(charge - 1).unwrap();
    let too_small = AdmissionQueue::new(Arc::clone(&undersized), 1).unwrap();
    assert!(generate_streamed_queued(
        &too_small,
        request,
        &AtomicBool::new(false),
        |_| true,
        |_| true
    )
    .unwrap_err()
    .contains("InsufficientCapacity"));
    assert_eq!(undersized.reserved(), 0);
    assert_eq!(too_small.pending(), 0);
    println!("queued native inference: exact bytes/logits, held reservation, waiting cancellation, failure cleanup and insufficient capacity: PASS");
    println!(
        "reservation_bytes={} queue_wait_ns={}",
        metrics.reservation_bytes, metrics.queue_wait_ns
    );
}
