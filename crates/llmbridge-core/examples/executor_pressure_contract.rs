//! Real inference with deterministic headroom signals; no physical pressure claim.
use llmbridge_core::{
    hardware,
    memory_pressure::MemoryPressurePolicy,
    queued_streamed::QueuedStreamedRequest,
    streamed::generate_streamed_controlled,
    streamed_executor::{ExecutorConfig, StreamedExecutor},
};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

fn wait_until(mut condition: impl FnMut() -> bool, done: &AtomicBool) -> bool {
    let start = Instant::now();
    while !condition() {
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
    drop_headroom: Option<&AtomicU64>,
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
                if let Some(headroom) = drop_headroom {
                    headroom.store(0, Ordering::Release);
                    let start = Instant::now();
                    loop {
                        executor.refresh_pressure();
                        if executor.pressure_snapshot().unwrap().paused {
                            break;
                        }
                        assert!(start.elapsed() < Duration::from_secs(30));
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    assert_eq!(executor.reserved_bytes(), executor.slot_bytes());
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
        .expect("executor_pressure_contract <pinned.gguf> [trusted-sha256]");
    let digest = std::env::args().nth(2).unwrap_or_else(|| {
        "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6".into()
    });
    let mut request = QueuedStreamedRequest::new(Path::new(&model), "The sky is");
    request.config.max_tokens = 4;
    request.control.expected_sha256 = Some(&digest);
    let slot = 64 * 1024 * 1024;
    let config = ExecutorConfig {
        requested_workers: 1,
        session_slot_bytes: slot,
        os_reserve_bytes: 0,
        max_pending: 4,
    };
    let policy = MemoryPressurePolicy {
        pause_below_bytes: slot,
        resume_at_bytes: 2 * slot,
        sample_interval: Duration::from_millis(1),
    };
    // Application-supplied source: sentinel means unknown, not unbounded memory.
    let headroom = Arc::new(AtomicU64::new(u64::MAX));
    let source = Arc::clone(&headroom);
    let executor = StreamedExecutor::new_with_pressure_sampler(
        &hardware::profile(),
        config,
        policy,
        move || match source.load(Ordering::Acquire) {
            u64::MAX => None,
            n => Some(n),
        },
    )
    .unwrap();
    let done_first = AtomicBool::new(false);
    let done_second = AtomicBool::new(false);
    let results = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            let result = execute(&executor, request, Some(&headroom));
            done_first.store(true, Ordering::Release);
            result
        });
        let unknown_waited = wait_until(
            || {
                executor.pending_sessions() == 1
                    && executor.pressure_snapshot().unwrap().sample_count > 0
            },
            &done_first,
        );
        let unknown = executor.pressure_snapshot().unwrap();
        let second = scope.spawn(|| {
            let result = execute(&executor, request, None);
            done_second.store(true, Ordering::Release);
            result
        });
        let both_waited = wait_until(|| executor.pending_sessions() == 2, &done_second);
        headroom.store(2 * slot, Ordering::Release); // Waiting poll resumes automatically.
        let first_result = first.join();
        let pressured = executor.pressure_snapshot().unwrap();
        let tail_waited = executor.pending_sessions() == 1 && !done_second.load(Ordering::Acquire);
        let empty = executor.reserved_bytes();
        executor.set_paused(true);
        headroom.store(slot + 1, Ordering::Release); // Hysteresis band cannot reopen.
        let band_observed = wait_until(
            || executor.pressure_snapshot().unwrap().last_available_bytes == Some(slot + 1),
            &done_second,
        );
        let band_paused = executor.pressure_snapshot().unwrap().paused;
        headroom.store(2 * slot, Ordering::Release);
        let recovered = wait_until(
            || !executor.pressure_snapshot().unwrap().paused,
            &done_second,
        );
        let manual_retained = executor.is_paused()
            && executor.pending_sessions() == 1
            && executor.reserved_bytes() == 0;
        // Always resume/join before assertions.
        executor.set_paused(false);
        let second_result = second.join();
        assert!(
            unknown_waited
                && both_waited
                && tail_waited
                && band_observed
                && recovered
                && manual_retained
        );
        assert!(unknown.paused && unknown.last_available_bytes.is_none());
        assert!(pressured.paused && band_paused);
        assert_eq!(empty, 0);
        (
            first_result.unwrap().unwrap(),
            second_result.unwrap().unwrap(),
        )
    });
    assert_eq!(executor.pending_sessions(), 0);
    assert_eq!(executor.reserved_bytes(), 0);

    // Exercise the production OS/cgroup sampler. The impossibly high threshold
    // forces refusal without claiming this machine is physically under pressure.
    let os = StreamedExecutor::new_with_pressure(
        &hardware::profile(),
        config,
        MemoryPressurePolicy {
            pause_below_bytes: u64::MAX - 1,
            resume_at_bytes: u64::MAX,
            ..policy
        },
    )
    .unwrap();
    let cancelled = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    let mut missing = request;
    missing.model = Path::new("/llmbridge-pressure-unopened.gguf");
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let result = os.generate(
                missing,
                &cancelled,
                |_| panic!("unexpected output"),
                |_| panic!("unexpected logits"),
            );
            done.store(true, Ordering::Release);
            result
        });
        let waited = wait_until(
            || os.pending_sessions() == 1 && os.pressure_snapshot().unwrap().sample_count > 0,
            &done,
        );
        let measured = os.pressure_snapshot().unwrap();
        cancelled.store(true, Ordering::Release);
        let result = worker.join();
        assert!(waited && measured.paused);
        assert!(result
            .unwrap()
            .unwrap_err()
            .contains("cancelled while waiting"));
        println!(
            "production_os_sample_available_bytes={:?} samples={}",
            measured.last_available_bytes, measured.sample_count
        );
    });
    assert_eq!(os.pending_sessions(), 0);
    assert_eq!(os.reserved_bytes(), 0);

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
    println!("executor pressure: unknown refusal, active-slot retention, automatic queued recovery, hysteresis, independent manual pause, OS sampling/cancellation and exact bytes/logits: PASS");
    println!(
        "slot_bytes={slot} output_sha256={} logits_sha256={}",
        direct.0, direct.1
    );
}
