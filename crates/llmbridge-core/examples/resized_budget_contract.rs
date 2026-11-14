//! Real queued generation must retain old reservations across live capacity changes.
use llmbridge_core::{
    admission_queue::AdmissionQueue,
    memory_budget::{BudgetSnapshot, MemoryBudget},
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

fn wait_pending(queue: &AdmissionQueue, done: &AtomicBool) -> bool {
    let start = Instant::now();
    while queue.pending() != 1 {
        if done.load(Ordering::Acquire) || start.elapsed() >= Duration::from_secs(30) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}
fn run(
    queue: &AdmissionQueue,
    budget: &MemoryBudget,
    request: QueuedStreamedRequest<'_>,
    shrink: bool,
) -> Result<(String, String, Option<BudgetSnapshot>), String> {
    let mut output = Sha256::new();
    let mut logits = Sha256::new();
    let mut observed = None;
    generate_streamed_queued(
        queue,
        request,
        &AtomicBool::new(false),
        |b| {
            output.update(b);
            true
        },
        |v| {
            if shrink && observed.is_none() {
                budget.set_capacity(0);
                observed = Some(budget.snapshot());
            }
            for x in v {
                logits.update(x.to_le_bytes());
            }
            true
        },
    )?;
    Ok((
        format!("{:x}", output.finalize()),
        format!("{:x}", logits.finalize()),
        observed,
    ))
}
fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("resized_budget_contract <pinned.gguf> [trusted-sha256]");
    let expected = std::env::args().nth(2).unwrap_or_else(|| {
        "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6".into()
    });
    let mut request = QueuedStreamedRequest::new(Path::new(&model), "The sky is");
    request.config.max_tokens = 4;
    request.control.expected_sha256 = Some(&expected);
    let charge = request.reservation_bytes().unwrap();
    let budget = MemoryBudget::new(charge).unwrap();
    let queue = AdmissionQueue::new(Arc::clone(&budget), 4).unwrap();
    let mut output = Sha256::new();
    let mut logits = Sha256::new();
    generate_streamed_controlled(
        request.model,
        request.prompt,
        request.config,
        &AtomicBool::new(false),
        request.control,
        |b| {
            output.update(b);
            true
        },
        |v| {
            for x in v {
                logits.update(x.to_le_bytes());
            }
            true
        },
    )
    .unwrap();
    let reference = (
        format!("{:x}", output.finalize()),
        format!("{:x}", logits.finalize()),
    );

    let held = budget.try_reserve(charge).unwrap();
    let done = AtomicBool::new(false);
    let result = std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let result = run(&queue, &budget, request, true);
            done.store(true, Ordering::Release);
            result
        });
        let waited = wait_pending(&queue, &done);
        let old_capacity = budget.set_capacity(0);
        let shrinking = budget.snapshot();
        drop(held);
        let released = budget.snapshot();
        let still_pending = queue.pending();
        let still_waiting = !done.load(Ordering::Acquire);
        budget.set_capacity(charge);
        let result = worker.join(); // Restore first, so failure assertions cannot strand a worker.
        assert!(waited && still_waiting);
        assert_eq!(still_pending, 1);
        assert_eq!(old_capacity, charge);
        assert_eq!(shrinking.capacity_bytes, 0);
        assert_eq!(shrinking.reserved_bytes, charge);
        assert_eq!(shrinking.over_capacity_bytes, charge);
        assert_eq!(shrinking.available_bytes, 0);
        assert_eq!(released.reserved_bytes, 0);
        assert_eq!(released.capacity_bytes, 0);
        result.unwrap().unwrap()
    });
    assert_eq!((&result.0, &result.1), (&reference.0, &reference.1));
    let during_native = result.2.unwrap();
    assert_eq!(during_native.capacity_bytes, 0);
    assert_eq!(during_native.reserved_bytes, charge);
    assert_eq!(during_native.over_capacity_bytes, charge);
    assert_eq!(
        budget.snapshot(),
        BudgetSnapshot {
            capacity_bytes: 0,
            reserved_bytes: 0,
            available_bytes: 0,
            over_capacity_bytes: 0
        }
    );
    assert_eq!(queue.pending(), 0);

    let missing =
        QueuedStreamedRequest::new(Path::new("/llmbridge-resized-unopened.gguf"), "fixture");
    assert!(
        generate_streamed_queued(&queue, missing, &AtomicBool::new(false), |_| true, |_| true)
            .unwrap_err()
            .contains("InsufficientCapacity")
    );
    assert_eq!(queue.pending(), 0);
    assert_eq!(budget.reserved(), 0);

    // Existing queued tickets remain cancellable even after capacity reaches zero.
    budget.set_capacity(charge);
    let held = budget.try_reserve(charge).unwrap();
    let cancelled = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let result = generate_streamed_queued(
                &queue,
                missing,
                &cancelled,
                |_| panic!("output before resized admission"),
                |_| panic!("logits before resized admission"),
            );
            done.store(true, Ordering::Release);
            result
        });
        let waited = wait_pending(&queue, &done);
        budget.set_capacity(0);
        drop(held);
        cancelled.store(true, Ordering::Release);
        let result = worker.join();
        assert!(waited);
        assert!(result
            .unwrap()
            .unwrap_err()
            .contains("cancelled while waiting"));
    });
    assert_eq!(queue.pending(), 0);
    assert_eq!(budget.reserved(), 0);
    budget.set_capacity(charge);
    let resumed = run(&queue, &budget, request, false).unwrap();
    assert_eq!((&resumed.0, &resumed.1), (&reference.0, &reference.1));
    assert_eq!(budget.snapshot().available_bytes, charge);
    assert_eq!(budget.reserved(), 0);
    println!("live budget resize: retained native charge, zero-capacity waiting/refusal, growth resume, cancellation and exact bytes/logits: PASS");
    println!(
        "charge_bytes={charge} output_sha256={} logits_sha256={}",
        reference.0, reference.1
    );
}
