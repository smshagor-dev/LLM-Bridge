//! Pause/resume gates actual admission while retaining running native sessions.
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

fn wait_pending(queue: &AdmissionQueue, count: usize, done: &AtomicBool) -> bool {
    let start = Instant::now();
    while queue.pending() != count {
        if done.load(Ordering::Acquire) || start.elapsed() >= Duration::from_secs(30) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}
fn execute(
    queue: &AdmissionQueue,
    request: QueuedStreamedRequest<'_>,
    pause_at_logits: bool,
) -> Result<(String, String), String> {
    let mut output = Sha256::new();
    let mut logits = Sha256::new();
    let mut first = true;
    generate_streamed_queued(
        queue,
        request,
        &AtomicBool::new(false),
        |b| {
            output.update(b);
            true
        },
        |v| {
            if first {
                first = false;
                if pause_at_logits {
                    queue.set_paused(true);
                }
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
    ))
}
fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("paused_admission_contract <pinned.gguf> [trusted-sha256]");
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

    assert!(!queue.set_paused(true));
    let done_first = AtomicBool::new(false);
    let done_second = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            let r = execute(&queue, request, true);
            done_first.store(true, Ordering::Release);
            r
        });
        let first_waited = wait_pending(&queue, 1, &done_first);
        let second = scope.spawn(|| {
            let r = execute(&queue, request, false);
            done_second.store(true, Ordering::Release);
            r
        });
        let both_waited = wait_pending(&queue, 2, &done_second);
        let paused_bytes = budget.reserved();
        queue.set_paused(false);
        let first_result = first.join();
        let paused_after_first = queue.is_paused();
        let pending_after_first = queue.pending();
        let bytes_after_first = budget.reserved();
        // Always resume/join before assertions, including on a failed first call.
        queue.set_paused(false);
        let second_result = second.join();
        assert!(first_waited && both_waited);
        assert_eq!(paused_bytes, 0);
        assert!(paused_after_first);
        assert_eq!(pending_after_first, 1);
        assert_eq!(bytes_after_first, 0);
        assert_eq!(first_result.unwrap().unwrap(), reference);
        assert_eq!(second_result.unwrap().unwrap(), reference);
    });
    assert_eq!(queue.pending(), 0);
    assert_eq!(budget.reserved(), 0);

    // With all capacity free, pause alone must block a missing-model request.
    queue.set_paused(true);
    let cancelled = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let missing =
                QueuedStreamedRequest::new(Path::new("/llmbridge-paused-unopened.gguf"), "fixture");
            let result = generate_streamed_queued(
                &queue,
                missing,
                &cancelled,
                |_| panic!("paused output"),
                |_| panic!("paused logits"),
            );
            done.store(true, Ordering::Release);
            result
        });
        let waited = wait_pending(&queue, 1, &done);
        let bytes_while_paused = budget.reserved();
        cancelled.store(true, Ordering::Release);
        let result = worker.join();
        assert!(waited);
        assert_eq!(bytes_while_paused, 0);
        assert!(result
            .unwrap()
            .unwrap_err()
            .contains("cancelled while waiting"));
    });
    assert!(queue.is_paused());
    assert_eq!(queue.pending(), 0);
    assert_eq!(budget.reserved(), 0);
    assert!(queue.set_paused(false));
    println!("paused admission: bounded waiting without model open, FIFO resume, running native completion, exact bytes/logits and cancelled-ticket cleanup: PASS");
    println!(
        "reservation_bytes={charge} output_sha256={} logits_sha256={}",
        reference.0, reference.1
    );
}
