//! One absolute deadline spans queued waiting, authenticated native work and cleanup.
use llmbridge_core::{
    admission_queue::AdmissionQueue,
    hardware,
    memory_budget::MemoryBudget,
    memory_pressure::MemoryPressurePolicy,
    queued_streamed::{
        generate_streamed_queued_with_limits, QueuedExecutionLimits, QueuedStreamedRequest,
    },
    streamed::generate_streamed_controlled,
    streamed_executor::{ExecutorConfig, StreamedExecutor},
};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};
type Hashes = (String, String);
fn run(
    executor: &StreamedExecutor,
    request: QueuedStreamedRequest<'_>,
    cancelled: &AtomicBool,
    limits: QueuedExecutionLimits,
    mut on_first: impl FnMut(),
    pieces: &AtomicUsize,
) -> Result<Hashes, String> {
    let mut output = Sha256::new();
    let mut logits = Sha256::new();
    let mut first = true;
    let metrics = executor.generate_with_limits(
        request,
        cancelled,
        limits,
        |b| {
            pieces.fetch_add(1, Ordering::Relaxed);
            output.update(b);
            true
        },
        |v| {
            if first {
                first = false;
                on_first();
            }
            for x in v {
                logits.update(x.to_le_bytes());
            }
            true
        },
    )?;
    assert_eq!(metrics.reservation_bytes, executor.slot_bytes());
    Ok((
        format!("{:x}", output.finalize()),
        format!("{:x}", logits.finalize()),
    ))
}
fn pending(executor: &StreamedExecutor, count: usize, mut finished: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while executor.pending_sessions() != count {
        if finished() || start.elapsed() > Duration::from_secs(30) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}
fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("queued_deadline_contract <pinned.gguf> [trusted-sha256]");
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
    let executor = StreamedExecutor::new(&hardware::profile(), config).unwrap();
    let flag = AtomicBool::new(false);
    let head_flag = AtomicBool::new(false);
    let ordinary_pieces = AtomicUsize::new(0);
    let expired_pieces = AtomicUsize::new(0);
    let (blocker_hashes, tail_hashes) = std::thread::scope(|scope| {
        let (ready0, recv0) = mpsc::channel();
        let (release0, wait0) = mpsc::channel();
        let e = &executor;
        let f = &flag;
        let pieces = &ordinary_pieces;
        let blocker = scope.spawn(move || {
            run(
                e,
                request,
                f,
                QueuedExecutionLimits::default(),
                || {
                    ready0.send(()).unwrap();
                    wait0.recv_timeout(Duration::from_secs(30)).unwrap();
                },
                pieces,
            )
        });
        recv0.recv_timeout(Duration::from_secs(30)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let limits = QueuedExecutionLimits {
            wait_timeout: Some(Duration::from_secs(30)),
            deadline: Some(deadline),
        };
        let (ready1, recv1) = mpsc::channel();
        let (release1, wait1) = mpsc::channel();
        let h = &head_flag;
        let pieces = &expired_pieces;
        let head = scope.spawn(move || {
            run(
                e,
                request,
                h,
                limits,
                || {
                    ready1.send(()).unwrap();
                    wait1.recv_timeout(Duration::from_secs(30)).unwrap();
                },
                pieces,
            )
        });
        let head_waited = pending(&executor, 1, || head.is_finished());
        let pieces = &ordinary_pieces;
        let tail = scope.spawn(move || {
            run(
                e,
                request,
                f,
                QueuedExecutionLimits {
                    deadline: Some(Instant::now() + Duration::from_secs(30)),
                    wait_timeout: None,
                },
                || {},
                pieces,
            )
        });
        let both_waited = pending(&executor, 2, || tail.is_finished());
        let blocker_charge = executor.reserved_bytes();
        // Consume part of the head's original budget while it is still queued.
        std::thread::sleep(Duration::from_millis(500));
        release0.send(()).unwrap();
        let blocker_result = blocker.join();
        recv1.recv_timeout(Duration::from_secs(30)).unwrap();
        let head_charge = executor.reserved_bytes();
        let tail_pending = executor.pending_sessions();
        std::thread::sleep(
            deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(5),
        );
        release1.send(()).unwrap();
        let head_result = head.join();
        let tail_result = tail.join();
        assert!(head_waited && both_waited);
        assert_eq!((blocker_charge, head_charge, tail_pending), (slot, slot, 1));
        assert!(head_result
            .unwrap()
            .unwrap_err()
            .contains("deadline expired"));
        assert_eq!(expired_pieces.load(Ordering::Relaxed), 0);
        assert!(!head_flag.load(Ordering::Acquire));
        (
            blocker_result.unwrap().unwrap(),
            tail_result.unwrap().unwrap(),
        )
    });
    assert_eq!(
        (executor.pending_sessions(), executor.reserved_bytes()),
        (0, 0)
    );

    let mut missing = request;
    missing.model = Path::new("/llmbridge-queued-deadline-unopened.gguf");
    executor.set_paused(true);
    let error = executor
        .generate_with_limits(
            missing,
            &flag,
            QueuedExecutionLimits {
                wait_timeout: Some(Duration::from_millis(20)),
                deadline: Some(Instant::now() + Duration::from_secs(30)),
            },
            |_| panic!("paused piece"),
            |_| panic!("paused logits"),
        )
        .unwrap_err();
    assert!(error.contains("admission wait timed out"));
    assert!(executor.is_paused());
    assert_eq!(
        (executor.pending_sessions(), executor.reserved_bytes()),
        (0, 0)
    );
    executor.set_paused(false);
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
    let error = pressure
        .generate_with_limits(
            missing,
            &flag,
            QueuedExecutionLimits {
                wait_timeout: Some(Duration::from_secs(30)),
                deadline: Some(Instant::now() + Duration::from_millis(20)),
            },
            |_| panic!("pressure piece"),
            |_| panic!("pressure logits"),
        )
        .unwrap_err();
    assert!(error.contains("deadline expired"));
    assert!(pressure.pressure_snapshot().unwrap().paused);
    assert_eq!(
        (pressure.pending_sessions(), pressure.reserved_bytes()),
        (0, 0)
    );
    assert!(!flag.load(Ordering::Acquire));
    let recovered = run(
        &executor,
        request,
        &flag,
        QueuedExecutionLimits {
            wait_timeout: Some(Duration::from_secs(30)),
            deadline: Some(Instant::now() + Duration::from_secs(30)),
        },
        || {},
        &ordinary_pieces,
    )
    .unwrap();

    let budget = MemoryBudget::new(slot).unwrap();
    let queue = AdmissionQueue::new(Arc::clone(&budget), 2).unwrap();
    let mut output = Sha256::new();
    let mut logits = Sha256::new();
    generate_streamed_queued_with_limits(
        &queue,
        request,
        &flag,
        QueuedExecutionLimits {
            deadline: Some(Instant::now() + Duration::from_secs(30)),
            wait_timeout: None,
        },
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
    let queued_hashes = (
        format!("{:x}", output.finalize()),
        format!("{:x}", logits.finalize()),
    );
    assert_eq!((queue.pending(), budget.reserved()), (0, 0));
    let mut output = Sha256::new();
    let mut logits = Sha256::new();
    generate_streamed_controlled(
        request.model,
        request.prompt,
        request.config,
        &flag,
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
    for result in [blocker_hashes, tail_hashes, recovered, queued_hashes] {
        assert_eq!(result, direct);
    }
    assert_eq!(
        (executor.pending_sessions(), executor.reserved_bytes()),
        (0, 0)
    );
    println!("queued deadline: queue time not reset, active slot retention/release, FIFO tail progress, independent waiting timeout, pressure cleanup and exact bytes/logits recovery: PASS");
    println!("expired_session_piece_callbacks={} final_pending={} final_reserved={} output_sha256={} logits_sha256={}", expired_pieces.load(Ordering::Relaxed), executor.pending_sessions(), executor.reserved_bytes(), direct.0, direct.1);
}
