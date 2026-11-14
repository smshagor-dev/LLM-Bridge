//! Real-model cache release must preserve bytes/logits and authenticated reads.
use llmbridge_core::streamed::{
    generate_streamed_controlled, generate_streamed_verified, CacheControl, StreamedConfig,
    StreamedControl,
};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

fn main() {
    let model_arg = std::env::args()
        .nth(1)
        .expect("cache_release_contract <pinned.gguf> [trusted-sha256]");
    let expected = std::env::args().nth(2).unwrap_or_else(|| {
        "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6".into()
    });
    let model = Path::new(&model_arg);
    let cancelled = AtomicBool::new(false);
    let config = StreamedConfig {
        max_tokens: 8,
        cache_bytes: 1024 * 1024,
        ..Default::default()
    };
    let mut reference_output = Vec::new();
    let mut reference_logits = Sha256::new();
    let reference = generate_streamed_verified(
        model,
        "The sky is",
        config,
        &cancelled,
        &expected,
        |b| {
            reference_output.extend_from_slice(b);
            true
        },
        |v| {
            for x in v {
                reference_logits.update(x.to_le_bytes());
            }
            true
        },
    )
    .unwrap();
    let control = CacheControl::default();
    let mut output = Vec::new();
    let mut logits = Sha256::new();
    let mut steps = 0;
    let released = generate_streamed_controlled(
        model,
        "The sky is",
        config,
        &cancelled,
        StreamedControl {
            expected_sha256: Some(&expected),
            cache: Some(&control),
        },
        |b| {
            output.extend_from_slice(b);
            true
        },
        |v| {
            steps += 1;
            for x in v {
                logits.update(x.to_le_bytes());
            }
            control.request_release();
            true
        },
    )
    .unwrap();
    assert_eq!(output, reference_output);
    assert_eq!(logits.finalize(), reference_logits.finalize());
    assert!(steps > 1 && released.authentication.is_some());
    assert_eq!(released.cache.allocated_bytes, 0);
    assert_eq!(
        released.cache.peak_allocated_bytes,
        reference.cache.allocated_bytes
    );
    assert_eq!(
        released.cache.released_bytes,
        reference.cache.allocated_bytes
    );
    assert_eq!(released.cache.release_events, 1);
    // Authenticated sessions deliberately bypass the optional LRU: releasing its
    // unused reservation must not change the freshly verified read schedule.
    assert_eq!(released.cache.block_hits, 0);
    assert_eq!(reference.cache.block_hits, 0);
    assert_eq!(
        released.cache.source_read_calls,
        reference.cache.source_read_calls
    );
    assert_eq!(released.cache.source_read_calls, released.weight_read_calls);
    assert!(
        released.authentication.as_ref().unwrap().blocks_verified >= released.weight_read_calls
    );

    let cancelled_control = CacheControl::default();
    let mut pieces = 0;
    let rejected = generate_streamed_controlled(
        model,
        "The sky is",
        config,
        &cancelled,
        StreamedControl {
            expected_sha256: Some(&expected),
            cache: Some(&cancelled_control),
        },
        |_| {
            pieces += 1;
            true
        },
        |_| {
            cancelled_control.request_release();
            false
        },
    );
    assert!(rejected.unwrap_err().contains("cancelled"));
    assert_eq!(pieces, 0);

    struct CopyGuard(PathBuf);
    impl Drop for CopyGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let copy_path = std::env::temp_dir().join(format!(
        "bridge-cache-release-mutation-{}",
        std::process::id()
    ));
    let mut target = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&copy_path)
        .unwrap();
    let copy = CopyGuard(copy_path);
    std::io::copy(&mut std::fs::File::open(model).unwrap(), &mut target).unwrap();
    drop(target);
    let metadata = std::fs::metadata(&copy.0).unwrap();
    let modified = metadata.modified().unwrap();
    let mutation_control = CacheControl::default();
    let mut observed_logits = 0;
    let tampered = generate_streamed_controlled(
        &copy.0,
        "The sky is",
        config,
        &cancelled,
        StreamedControl {
            expected_sha256: Some(&expected),
            cache: Some(&mutation_control),
        },
        |_| true,
        |_| {
            observed_logits += 1;
            if observed_logits == 1 {
                mutation_control.request_release();
                let mut file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&copy.0)
                    .unwrap();
                file.seek(SeekFrom::Start(metadata.len() - 1)).unwrap();
                let mut old = [0u8; 1];
                file.read_exact(&mut old).unwrap();
                file.seek(SeekFrom::Start(metadata.len() - 1)).unwrap();
                file.write_all(&[old[0] ^ 0xff]).unwrap();
                file.set_times(std::fs::FileTimes::new().set_modified(modified))
                    .unwrap();
                assert_eq!(file.metadata().unwrap().len(), metadata.len());
            }
            true
        },
    );
    assert!(tampered
        .unwrap_err()
        .contains("authenticated model block changed"));
    assert_eq!(observed_logits, 1);
    println!("cache release output/logit parity, zero final allocation, cancellation and late-mutation rejection: PASS");
    println!(
        "released_bytes={} release_events={} current_cache_bytes={}",
        released.cache.released_bytes,
        released.cache.release_events,
        released.cache.allocated_bytes
    );
}
