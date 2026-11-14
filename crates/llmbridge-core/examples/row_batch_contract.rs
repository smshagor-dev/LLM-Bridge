//! Real-model chunk geometry must preserve all output bytes and FP32 logits.
use llmbridge_core::streamed::{generate_streamed_verified, StreamedConfig};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::atomic::AtomicBool};

fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("row_batch_contract <pinned.gguf> [trusted-sha256]");
    let expected = std::env::args().nth(2).unwrap_or_else(|| {
        "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6".into()
    });
    let mut reference = None;
    for cache_bytes in [0, 1024 * 1024] {
        let mut small_calls = 0;
        for chunk_bytes in [4096, 65536, 65535] {
            let mut output = Sha256::new();
            let mut logits = Sha256::new();
            let mut steps = 0;
            let metrics = generate_streamed_verified(
                Path::new(&model),
                "The sky is",
                StreamedConfig {
                    chunk_bytes,
                    cache_bytes,
                    max_tokens: 4,
                    ..Default::default()
                },
                &AtomicBool::new(false),
                &expected,
                |b| {
                    output.update(b);
                    true
                },
                |values| {
                    steps += 1;
                    for x in values {
                        logits.update(x.to_le_bytes());
                    }
                    true
                },
            )
            .unwrap();
            let output_sha = format!("{:x}", output.finalize());
            let logits_sha = format!("{:x}", logits.finalize());
            let identity = (
                output_sha.clone(),
                logits_sha.clone(),
                steps,
                metrics.weight_bytes_requested,
            );
            if let Some(expected) = &reference {
                assert_eq!(&identity, expected);
            } else {
                reference = Some(identity);
            }
            assert!(metrics.max_range_read_bytes <= chunk_bytes);
            assert_eq!(metrics.reader_buffer_bytes, chunk_bytes);
            assert_eq!(metrics.cache.block_hits, 0); // No authenticated LRU reuse.
            assert_eq!(metrics.cache.source_read_calls, metrics.weight_read_calls);
            assert!(
                metrics.authentication.as_ref().unwrap().blocks_verified
                    >= metrics.weight_read_calls
            );
            if chunk_bytes == 4096 {
                small_calls = metrics.weight_read_calls;
            } else {
                assert!(metrics.weight_read_calls < small_calls / 4);
            }
            println!("{{\"chunk_bytes\":{chunk_bytes},\"cache_budget_bytes\":{cache_bytes},\"weight_read_calls\":{},\"weight_bytes_requested\":{},\"max_range_read_bytes\":{},\"logits_steps\":{steps},\"output_sha256\":\"{output_sha}\",\"logits_sha256\":\"{logits_sha}\"}}", metrics.weight_read_calls, metrics.weight_bytes_requested, metrics.max_range_read_bytes);
        }
    }
    println!("bounded row batches: identical bytes/logits, exact logical reads, authenticated LRU bypass and chunk bounds: PASS");
}
