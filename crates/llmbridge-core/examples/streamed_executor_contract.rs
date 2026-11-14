//! Real concurrent sessions must obey fixed worker slots and preserve direct results.
use llmbridge_core::{
    hardware,
    queued_streamed::QueuedStreamedRequest,
    streamed::generate_streamed_controlled,
    streamed_executor::{ExecutorConfig, StreamedExecutor},
};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};

fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("streamed_executor_contract <pinned.gguf> [trusted-sha256]");
    let digest = std::env::args().nth(2).unwrap_or_else(|| {
        "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6".into()
    });
    let mut request = QueuedStreamedRequest::new(Path::new(&model), "The sky is");
    request.control.expected_sha256 = Some(&digest);
    request.config.max_tokens = 4;
    let hardware = hardware::profile();
    let config = ExecutorConfig {
        requested_workers: 2,
        session_slot_bytes: 64 * 1024 * 1024,
        os_reserve_bytes: 0,
        max_pending: 4,
    };
    let executor = StreamedExecutor::new(&hardware, config).unwrap();
    assert_eq!(
        executor.worker_count(),
        2,
        "real concurrency contract requires two admitted CPU workers"
    );
    // Both native sessions pause at their first logits, proving actual overlap.
    let concurrent = std::thread::scope(|scope| {
        let (ready_tx, ready_rx) = mpsc::channel();
        let mut release = Vec::new();
        let mut workers = Vec::new();
        for index in 0..2 {
            let (release_tx, release_rx) = mpsc::channel();
            release.push(release_tx);
            let ready = ready_tx.clone();
            let executor = &executor;
            workers.push(scope.spawn(move || {
                let mut output = Sha256::new();
                let mut logits = Sha256::new();
                let mut first = true;
                let result = executor
                    .generate(
                        request,
                        &AtomicBool::new(false),
                        |b| {
                            output.update(b);
                            true
                        },
                        |v| {
                            if first {
                                first = false;
                                ready.send(index).unwrap();
                                release_rx.recv_timeout(Duration::from_secs(30)).unwrap();
                            }
                            for x in v {
                                logits.update(x.to_le_bytes());
                            }
                            true
                        },
                    )
                    .unwrap();
                assert_eq!(result.reservation_bytes, executor.slot_bytes());
                (
                    format!("{:x}", output.finalize()),
                    format!("{:x}", logits.finalize()),
                )
            }));
        }
        let a = ready_rx.recv_timeout(Duration::from_secs(30)).unwrap();
        let b = ready_rx.recv_timeout(Duration::from_secs(30)).unwrap();
        assert_ne!(a, b);
        assert_eq!(executor.active_sessions(), 2);
        assert_eq!(executor.reserved_bytes(), 2 * executor.slot_bytes());
        for sender in release {
            sender.send(()).unwrap();
        }
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(executor.reserved_bytes(), 0);
    assert_eq!(executor.pending_sessions(), 0);

    let mut direct_output = Sha256::new();
    let mut direct_logits = Sha256::new();
    generate_streamed_controlled(
        request.model,
        request.prompt,
        request.config,
        &AtomicBool::new(false),
        request.control,
        |b| {
            direct_output.update(b);
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
    let direct = (
        format!("{:x}", direct_output.finalize()),
        format!("{:x}", direct_logits.finalize()),
    );

    for result in concurrent {
        assert_eq!(result, direct);
    }

    // With one worker, a much smaller request must still wait rather than use
    // leftover byte capacity to start a second native execution.
    let serial = StreamedExecutor::new(
        &hardware,
        ExecutorConfig {
            requested_workers: 1,
            ..config
        },
    )
    .unwrap();
    std::thread::scope(|scope| {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let serial_ref = &serial;
        let first = scope.spawn(move || {
            let mut first = true;
            serial_ref.generate(
                request,
                &AtomicBool::new(false),
                |_| true,
                |_| {
                    if first {
                        first = false;
                        ready_tx.send(()).unwrap();
                        release_rx.recv_timeout(Duration::from_secs(30)).unwrap();
                    }
                    true
                },
            )
        });
        ready_rx.recv_timeout(Duration::from_secs(30)).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let mut small = request;
        small.model = Path::new("/llmbridge-executor-unopened-model.gguf");
        small.additional_reserve_bytes = 0;
        small.config.workspace_bytes = 4 * 1024 * 1024;
        small.config.authentication_budget_bytes = 1024 * 1024;
        assert!(
            small.reservation_bytes().unwrap()
                < serial.slot_bytes() - request.reservation_bytes().unwrap()
        );
        let second = scope.spawn(move || {
            serial_ref.generate(
                small,
                &worker_cancelled,
                |_| panic!("second session output while worker occupied"),
                |_| panic!("second session logits while worker occupied"),
            )
        });
        let start = Instant::now();
        while serial.pending_sessions() != 1 {
            assert!(
                !second.is_finished(),
                "smaller request opened its model before worker admission"
            );
            assert!(start.elapsed() < Duration::from_secs(30));
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(serial.active_sessions(), 1);
        assert_eq!(serial.reserved_bytes(), serial.slot_bytes());
        cancelled.store(true, Ordering::Release);
        assert!(second
            .join()
            .unwrap()
            .unwrap_err()
            .contains("cancelled while waiting"));
        assert_eq!(serial.pending_sessions(), 0);
        assert_eq!(serial.active_sessions(), 1);
        release_tx.send(()).unwrap();
        first.join().unwrap().unwrap();
    });
    assert_eq!(serial.reserved_bytes(), 0);
    let mut too_large = request;
    too_large.model = Path::new("/llmbridge-executor-oversized-unopened.gguf");
    too_large.additional_reserve_bytes = config.session_slot_bytes;
    assert!(serial
        .generate(too_large, &AtomicBool::new(false), |_| true, |_| true)
        .unwrap_err()
        .contains("exceeds executor slot"));
    assert_eq!(serial.reserved_bytes(), 0);
    assert_eq!(serial.pending_sessions(), 0);
    println!("streamed executor: two overlapping real sessions match direct bytes/logits; fixed-slot CPU cap, queued cancellation and zero final reservation: PASS");
    println!(
        "workers={} slot_bytes={} output_sha256={} logits_sha256={}",
        executor.worker_count(),
        executor.slot_bytes(),
        direct.0,
        direct.1
    );
}
