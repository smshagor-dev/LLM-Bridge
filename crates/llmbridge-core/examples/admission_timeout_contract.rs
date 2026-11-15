//! Real generation remains correct while bounded admission waits clean up.
use llmbridge_core::{
    admission_queue::AdmissionQueue,
    hardware,
    memory_budget::MemoryBudget,
    memory_pressure::MemoryPressurePolicy,
    queued_streamed::{generate_streamed_queued_with_timeout, QueuedStreamedRequest},
    streamed::generate_streamed_controlled,
    streamed_executor::{ExecutorConfig, StreamedExecutor},
};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{atomic::AtomicBool, mpsc, Arc},
    time::{Duration, Instant},
};
fn wait_pending(
    queue: &AdmissionQueue,
    count: usize,
    worker: &std::thread::ScopedJoinHandle<
        '_,
        Result<llmbridge_core::queued_streamed::QueuedStreamedMetrics, String>,
    >,
) -> bool {
    let start = Instant::now();
    while queue.pending() != count {
        if worker.is_finished() || start.elapsed() > Duration::from_secs(30) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}
fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("admission_timeout_contract <pinned.gguf> [trusted-sha256]");
    let digest = std::env::args().nth(2).unwrap_or_else(|| {
        "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6".into()
    });
    let mut request = QueuedStreamedRequest::new(Path::new(&model), "The sky is");
    request.config.max_tokens = 4;
    request.control.expected_sha256 = Some(&digest);
    let mut missing = request;
    missing.model = Path::new("/llmbridge-timed-unopened.gguf");
    let slot = 64 * 1024 * 1024;
    let budget = MemoryBudget::new(slot).unwrap();
    let queue = AdmissionQueue::new(Arc::clone(&budget), 2).unwrap();
    let held = budget.try_reserve(slot).unwrap();
    let queued_hashes = std::thread::scope(|scope| {
        let head = scope.spawn(|| {
            generate_streamed_queued_with_timeout(
                &queue,
                missing,
                &AtomicBool::new(false),
                Duration::from_millis(500),
                |_| panic!("head output"),
                |_| panic!("head logits"),
            )
        });
        let head_waited = wait_pending(&queue, 1, &head);
        let tail = scope.spawn(|| {
            let mut output = Sha256::new();
            let mut logits = Sha256::new();
            let result = generate_streamed_queued_with_timeout(
                &queue,
                request,
                &AtomicBool::new(false),
                Duration::from_secs(30),
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
            );
            (
                result,
                (
                    format!("{:x}", output.finalize()),
                    format!("{:x}", logits.finalize()),
                ),
            )
        });
        let head_result = head.join();
        let pending_after_head = queue.pending();
        let held_after_head = budget.reserved();
        // Release/join before asserting, so failures cannot strand the tail.
        drop(held);
        let (tail_result, hashes) = tail.join().unwrap();
        assert!(head_waited);
        assert!(head_result
            .unwrap()
            .unwrap_err()
            .contains("admission wait timed out"));
        assert_eq!(pending_after_head, 1);
        assert_eq!(held_after_head, slot);
        tail_result.unwrap();
        hashes
    });
    assert_eq!(queue.pending(), 0);
    assert_eq!(budget.reserved(), 0);
    let config = ExecutorConfig {
        requested_workers: 1,
        session_slot_bytes: slot,
        os_reserve_bytes: 0,
        max_pending: 2,
    };
    let executor = StreamedExecutor::new(&hardware::profile(), config).unwrap();
    executor.set_paused(true);
    let result = executor.generate_with_timeout(
        missing,
        &AtomicBool::new(false),
        Duration::from_millis(20),
        |_| panic!("paused output"),
        |_| panic!("paused logits"),
    );
    assert!(result.unwrap_err().contains("admission wait timed out"));
    assert!(executor.is_paused());
    assert_eq!(executor.pending_sessions(), 0);
    assert_eq!(executor.reserved_bytes(), 0);
    executor.set_paused(false);

    let active_hashes = std::thread::scope(|scope| {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let start = Instant::now();
        let executor_ref = &executor;
        let active = scope.spawn(move || {
            let mut output = Sha256::new();
            let mut logits = Sha256::new();
            let mut first = true;
            let result = executor_ref.generate_with_timeout(
                request,
                &AtomicBool::new(false),
                Duration::from_secs(1),
                |b| {
                    output.update(b);
                    true
                },
                |v| {
                    if first {
                        first = false;
                        ready_tx.send(()).unwrap();
                        release_rx.recv_timeout(Duration::from_secs(30)).unwrap();
                    }
                    for x in v {
                        logits.update(x.to_le_bytes());
                    }
                    true
                },
            );
            (
                result,
                (
                    format!("{:x}", output.finalize()),
                    format!("{:x}", logits.finalize()),
                ),
            )
        });
        ready_rx.recv_timeout(Duration::from_secs(30)).unwrap();
        let result = executor.generate_with_timeout(
            missing,
            &AtomicBool::new(false),
            Duration::from_millis(20),
            |_| panic!("blocked output"),
            |_| panic!("blocked logits"),
        );
        let reserved = executor.reserved_bytes();
        let pending = executor.pending_sessions();
        // Let the active session outlive its admission timeout deliberately.
        std::thread::sleep(Duration::from_millis(1050).saturating_sub(start.elapsed()));
        release_tx.send(()).unwrap();
        let (active_result, hashes) = active.join().unwrap();
        assert!(result.unwrap_err().contains("admission wait timed out"));
        assert_eq!(reserved, slot);
        assert_eq!(pending, 0);
        assert!(start.elapsed() >= Duration::from_secs(1));
        active_result.unwrap();
        hashes
    });
    assert_eq!(executor.reserved_bytes(), 0);

    let pressure = StreamedExecutor::new_with_pressure_sampler(
        &hardware::profile(),
        config,
        MemoryPressurePolicy {
            pause_below_bytes: slot,
            resume_at_bytes: 2 * slot,
            sample_interval: Duration::from_millis(1),
        },
        || None,
    )
    .unwrap();
    let result = pressure.generate_with_timeout(
        missing,
        &AtomicBool::new(false),
        Duration::from_millis(20),
        |_| panic!("pressure output"),
        |_| panic!("pressure logits"),
    );
    assert!(result.unwrap_err().contains("admission wait timed out"));
    let samples = pressure.pressure_snapshot().unwrap();
    assert!(samples.paused && samples.sample_count > 0);
    assert_eq!(pressure.pending_sessions(), 0);
    assert_eq!(pressure.reserved_bytes(), 0);
    let result = pressure.generate_with_timeout(
        missing,
        &AtomicBool::new(false),
        Duration::ZERO,
        |_| panic!("zero output"),
        |_| panic!("zero logits"),
    );
    assert!(result.unwrap_err().contains("admission wait timed out"));
    assert_eq!(pressure.pressure_snapshot().unwrap(), samples);

    let mut output = Sha256::new();
    let mut logits = Sha256::new();
    executor
        .generate_with_timeout(
            request,
            &AtomicBool::new(false),
            Duration::from_secs(30),
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
    let recovered = (
        format!("{:x}", output.finalize()),
        format!("{:x}", logits.finalize()),
    );
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
    let direct = (
        format!("{:x}", output.finalize()),
        format!("{:x}", logits.finalize()),
    );
    assert_eq!(queued_hashes, direct);
    assert_eq!(active_hashes, direct);
    assert_eq!(recovered, direct);
    assert_eq!(executor.pending_sessions(), 0);
    assert_eq!(executor.reserved_bytes(), 0);
    println!("admission timeout: expired FIFO head cleanup, active slot retained beyond wait deadline, paused/unknown-pressure refusal, zero timeout and exact bytes/logits: PASS");
    println!(
        "slot_bytes={slot} output_sha256={} logits_sha256={}",
        direct.0, direct.1
    );
}
