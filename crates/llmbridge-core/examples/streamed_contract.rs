//! Real-model rejection and cancellation checks for the full streamed decoder.
use llmbridge_core::streamed::{generate_streamed, StreamedConfig};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: streamed_contract <real.gguf>");
    let model = Path::new(&path);
    for cache_bytes in [0, 2 * 1024 * 1024] {
        let config = StreamedConfig {
            cache_bytes,
            ..Default::default()
        };
        let stopped = AtomicBool::new(false);
        let mut pieces = 0;
        let result = generate_streamed(
            model,
            "The sky is",
            config,
            &stopped,
            |_| {
                pieces += 1;
                stopped.store(true, Ordering::Release);
                true
            },
            |_| true,
        );
        assert_eq!(pieces, 1);
        assert!(result.unwrap_err().contains("cancelled"));
        let tiny = StreamedConfig {
            workspace_bytes: 1,
            ..config
        };
        let result = generate_streamed(
            model,
            "The sky is",
            tiny,
            &AtomicBool::new(false),
            |_| panic!("output before workspace admission"),
            |_| panic!("logits before admission"),
        );
        assert!(result.unwrap_err().contains("workspace exceeds budget"));
        let result = generate_streamed(
            model,
            &"The sky is ".repeat(300),
            config,
            &AtomicBool::new(false),
            |_| panic!("oversized prompt output"),
            |_| true,
        );
        assert!(result.unwrap_err().contains("context"));
        let result = generate_streamed(
            model,
            "The sky is",
            config,
            &AtomicBool::new(false),
            |_| panic!("token after logits callback cancellation"),
            |_| false,
        );
        assert!(result.unwrap_err().contains("cancelled"));
    }
    println!("cached/uncached streamed real-model cancellation, workspace and context gates: PASS");
}
