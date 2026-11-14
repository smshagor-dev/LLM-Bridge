//! Exercise shared-executor controls through real queued native generation.
use llmbridge_core::{
    hardware,
    queued_streamed::QueuedStreamedRequest,
    streamed::generate_streamed_controlled,
    streamed_executor::{ExecutorConfig, StreamedExecutor},
};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

fn wait_pending(executor: &StreamedExecutor, count: usize, done: &AtomicBool) -> bool {
    let start = Instant::now();
    while executor.pending_sessions() != count {
        if done.load(Ordering::Acquire) || start.elapsed() > Duration::from_secs(30) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}
fn execute(
    executor: &StreamedExecutor,
    request: QueuedStreamedRequest<'_>,
    shrink: bool,
) -> Result<(String, String), String> {
    let mut output = Sha256::new();
    let mut logits = Sha256::new();
    let mut first = true;
    executor.generate(
        request,
        &AtomicBool::new(false),
        |b| {
            output.update(b);
            true
        },
        |v| {
            if first {
                first = false;
                if shrink {
                    executor.set_paused(true);
                    executor.set_capacity_bytes(0).unwrap();
                    let snapshot = executor.budget_snapshot();
                    assert_eq!(snapshot.reserved_bytes, executor.slot_bytes());
                    assert_eq!(snapshot.over_capacity_bytes, executor.slot_bytes());
                    assert_eq!(executor.active_sessions(), 1);
                }
            }
            for value in v {
                logits.update(value.to_le_bytes());
            }
            true
        },
    )?;
    Ok((
        format!("{:x}", output.finalize()),
        format!("{:x}", logits.finalize()),
    ))
}
fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("executor_controls_contract <pinned.gguf> [trusted-sha256]");
    let digest = std::env::args().nth(2).unwrap_or_else(|| {
        "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6".into()
    });
    let mut request = QueuedStreamedRequest::new(Path::new(&model), "The sky is");
    request.config.max_tokens = 4;
    request.control.expected_sha256 = Some(&digest);
    let slot = 64 * 1024 * 1024;
    let executor = StreamedExecutor::new(
        &hardware::profile(),
        ExecutorConfig {
            requested_workers: 1,
            session_slot_bytes: slot,
            os_reserve_bytes: 0,
            max_pending: 2,
        },
    )
    .unwrap();
    assert_eq!(executor.worker_count(), 1);
    executor.set_paused(true);
    let done_first = AtomicBool::new(false);
    let done_second = AtomicBool::new(false);
    let results = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            let result = execute(&executor, request, true);
            done_first.store(true, Ordering::Release);
            result
        });
        let first_waited = wait_pending(&executor, 1, &done_first);
        let second = scope.spawn(|| {
            let result = execute(&executor, request, false);
            done_second.store(true, Ordering::Release);
            result
        });
        let both_waited = wait_pending(&executor, 2, &done_second);
        let initial = executor.budget_snapshot();
        executor.set_paused(false);
        let first_result = first.join();
        let paused_after_first = executor.is_paused();
        let after_first = executor.budget_snapshot();
        let tail_waited = executor.pending_sessions() == 1 && !done_second.load(Ordering::Acquire);
        executor.set_capacity_bytes(slot).unwrap();
        let pause_retained = executor.is_paused();
        executor.set_capacity_bytes(slot - 1).unwrap();
        executor.set_paused(false);
        let partial = executor.budget_snapshot();
        // Always restore and join before assertions, even if a worker failed.
        executor.set_capacity_bytes(slot).unwrap();
        let second_result = second.join();
        assert!(first_waited && both_waited && tail_waited);
        assert_eq!(initial.reserved_bytes, 0);
        assert!(paused_after_first && pause_retained);
        assert_eq!(
            (after_first.capacity_bytes, after_first.reserved_bytes),
            (0, 0)
        );
        assert_eq!(after_first.over_capacity_bytes, 0);
        assert_eq!(
            (partial.capacity_bytes, partial.reserved_bytes),
            (slot - 1, 0)
        );
        (
            first_result.unwrap().unwrap(),
            second_result.unwrap().unwrap(),
        )
    });
    assert_eq!(executor.reserved_bytes(), 0);
    assert_eq!(executor.pending_sessions(), 0);
    let before = executor.budget_snapshot();
    assert!(executor.set_capacity_bytes(slot + 1).is_err());
    assert_eq!(executor.budget_snapshot(), before);
    assert_eq!(executor.max_capacity_bytes(), slot);

    // Paused pending work cancels without opening a nonexistent model.
    executor.set_paused(true);
    let cancelled = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    let mut missing = request;
    missing.model = Path::new("/llmbridge-controlled-executor-unopened.gguf");
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let result = executor.generate(
                missing,
                &cancelled,
                |_| panic!("unexpected output"),
                |_| panic!("unexpected logits"),
            );
            done.store(true, Ordering::Release);
            result
        });
        let waited = wait_pending(&executor, 1, &done);
        executor.set_capacity_bytes(0).unwrap();
        cancelled.store(true, Ordering::Release);
        let result = worker.join();
        assert!(waited);
        assert!(result
            .unwrap()
            .unwrap_err()
            .contains("cancelled while waiting"));
    });
    assert!(executor.is_paused());
    executor.set_paused(false);
    assert!(executor
        .generate(missing, &AtomicBool::new(false), |_| true, |_| true)
        .unwrap_err()
        .contains("InsufficientCapacity"));
    assert_eq!(executor.pending_sessions(), 0);
    assert_eq!(executor.budget_snapshot().reserved_bytes, 0);
    executor.set_capacity_bytes(slot).unwrap();
    let recovered = execute(&executor, request, false).unwrap();

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
            for value in v {
                logits.update(value.to_le_bytes());
            }
            true
        },
    )
    .unwrap();
    let direct = (
        format!("{:x}", output.finalize()),
        format!("{:x}", logits.finalize()),
    );
    assert_eq!(results.0, direct);
    assert_eq!(results.1, direct);
    assert_eq!(recovered, direct);
    assert_eq!(executor.pending_sessions(), 0);
    assert_eq!(executor.reserved_bytes(), 0);
    println!("executor controls: pause, retained running slot, shrink/growth, ceiling refusal, pending cancellation and exact bytes/logits: PASS");
    println!(
        "slot_bytes={slot} output_sha256={} logits_sha256={}",
        direct.0, direct.1
    );
}
