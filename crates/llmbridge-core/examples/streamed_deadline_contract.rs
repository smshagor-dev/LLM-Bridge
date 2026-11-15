//! Real native cancellation at a cooperative deadline, with exact recovery parity.
use llmbridge_core::streamed::{
    generate_streamed_controlled, generate_streamed_with_deadline, StreamedConfig, StreamedControl,
    StreamedDeadlineControl,
};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
fn hashes(model: &Path, digest: &str, timed: bool) -> (String, String) {
    let config = StreamedConfig {
        max_tokens: 4,
        cache_bytes: 1024 * 1024,
        ..StreamedConfig::default()
    };
    let control = StreamedControl {
        expected_sha256: Some(digest),
        cache: None,
    };
    let flag = AtomicBool::new(false);
    let mut output = Sha256::new();
    let mut logits = Sha256::new();
    let piece = |v: &[u8]| {
        output.update(v);
        true
    };
    let logit = |v: &[f32]| {
        for x in v {
            logits.update(x.to_le_bytes());
        }
        true
    };
    if timed {
        generate_streamed_with_deadline(
            model,
            "The sky is",
            config,
            &flag,
            StreamedDeadlineControl {
                control,
                deadline: Instant::now() + Duration::from_secs(30),
            },
            piece,
            logit,
        )
        .unwrap();
    } else {
        generate_streamed_controlled(model, "The sky is", config, &flag, control, piece, logit)
            .unwrap();
    }
    assert!(!flag.load(Ordering::Acquire));
    (
        format!("{:x}", output.finalize()),
        format!("{:x}", logits.finalize()),
    )
}
fn descriptors() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_dir("/proc/self/fd")
            .ok()
            .map(|entries| entries.count())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("streamed_deadline_contract <pinned.gguf> [trusted-sha256]");
    let digest = std::env::args().nth(2).unwrap_or_else(|| {
        "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6".into()
    });
    let model = Path::new(&path);
    let direct = hashes(model, &digest, false);
    assert_eq!(hashes(model, &digest, true), direct);
    let config = StreamedConfig {
        max_tokens: 4,
        ..StreamedConfig::default()
    };
    let control = StreamedControl {
        expected_sha256: Some(&digest),
        cache: None,
    };
    let cancelled = AtomicBool::new(false);
    let error = generate_streamed_with_deadline(
        Path::new("/llmbridge-deadline-unopened.gguf"),
        "fixture",
        config,
        &cancelled,
        StreamedDeadlineControl {
            control,
            deadline: Instant::now(),
        },
        |_| panic!("expired piece"),
        |_| panic!("expired logits"),
    )
    .unwrap_err();
    assert!(error.contains("deadline expired"));
    assert!(!cancelled.load(Ordering::Acquire));

    let before = descriptors();
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut pieces = 0;
    let mut logits = 0;
    let error = generate_streamed_with_deadline(
        model,
        "The sky is",
        config,
        &cancelled,
        StreamedDeadlineControl { control, deadline },
        |_| {
            pieces += 1;
            true
        },
        |_| {
            logits += 1;
            // Deliberately hold the first native consumer past the deadline. No
            // preemption is claimed; the wrapper must reject on callback return.
            std::thread::sleep(
                deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(5),
            );
            true
        },
    )
    .unwrap_err();
    let after = descriptors();
    assert!(error.contains("deadline expired"));
    assert_eq!((pieces, logits), (0, 1));
    assert!(!cancelled.load(Ordering::Acquire));
    if let (Some(before), Some(after)) = (before, after) {
        assert_eq!(before, after);
    }
    assert_eq!(hashes(model, &digest, false), direct);

    let wrong = "0".repeat(64);
    let error = generate_streamed_with_deadline(
        model,
        "The sky is",
        config,
        &cancelled,
        StreamedDeadlineControl {
            control: StreamedControl {
                expected_sha256: Some(&wrong),
                cache: None,
            },
            deadline: Instant::now() + Duration::from_secs(30),
        },
        |_| panic!("untrusted piece"),
        |_| panic!("untrusted logits"),
    )
    .unwrap_err();
    assert!(error.contains("SHA-256"), "{error}");
    let error = generate_streamed_with_deadline(
        model,
        "The sky is",
        config,
        &cancelled,
        StreamedDeadlineControl {
            control,
            deadline: Instant::now() + Duration::from_secs(30),
        },
        |_| true,
        |_| {
            cancelled.store(true, Ordering::Release);
            true
        },
    )
    .unwrap_err();
    assert!(error.contains("cancelled"));
    assert!(cancelled.load(Ordering::Acquire));
    assert_eq!(hashes(model, &digest, true), direct);
    println!("streamed deadline: expired pre-open refusal, running first-logits expiry, caller cancellation, trust refusal, descriptor cleanup and exact bytes/logits recovery: PASS");
    println!("expired_session_piece_callbacks={pieces} logits_callbacks={logits} descriptors_before={before:?} descriptors_after={after:?}");
    println!("output_sha256={} logits_sha256={}", direct.0, direct.1);
}
