//! Real-model integrity admission and post-verification mutation regression.
#[cfg(target_os = "linux")]
fn main() {
    use llmbridge_core::streamed::{generate_streamed_verified, StreamedConfig};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicBool;
    const SHA: &str = "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6";
    let model_arg = std::env::args()
        .nth(1)
        .expect("usage: verified_contract <pinned-stories15M-q8_0.gguf>");
    let model = Path::new(&model_arg);
    let config = StreamedConfig {
        max_tokens: 2,
        ..Default::default()
    };
    let bad = generate_streamed_verified(
        model,
        "The sky is",
        config,
        &AtomicBool::new(false),
        &"0".repeat(64),
        |_| panic!("token from failed hash"),
        |_| panic!("logits from failed hash"),
    );
    assert!(bad.unwrap_err().contains("SHA-256 mismatch"));
    let tiny = StreamedConfig {
        authentication_budget_bytes: 1,
        ..config
    };
    assert!(generate_streamed_verified(
        model,
        "The sky is",
        tiny,
        &AtomicBool::new(false),
        SHA,
        |_| panic!("token before integrity admission"),
        |_| true
    )
    .unwrap_err()
    .contains("overhead budget"));
    let mut pieces = 0;
    let verified = generate_streamed_verified(
        model,
        "The sky is",
        config,
        &AtomicBool::new(false),
        SHA,
        |_| {
            pieces += 1;
            false
        },
        |_| true,
    );
    assert_eq!(pieces, 1);
    assert!(verified.unwrap_err().contains("cancelled"));
    struct CopyGuard(PathBuf);
    impl Drop for CopyGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let copy_path = std::env::temp_dir().join(format!(
        "llmbridge-auth-mutation-{}.gguf",
        std::process::id()
    ));
    // create_new avoids replacing another file, including a hostile symlink.
    let mut target = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&copy_path)
        .unwrap();
    let copy = CopyGuard(copy_path);
    std::io::copy(&mut std::fs::File::open(model).unwrap(), &mut target).unwrap();
    drop(target);
    let modified = std::fs::metadata(&copy.0).unwrap().modified().unwrap();
    let file_bytes = std::fs::metadata(&copy.0).unwrap().len();
    let mut first = true;
    let rejected = generate_streamed_verified(
        &copy.0,
        "The sky is",
        config,
        &AtomicBool::new(false),
        SHA,
        |_| true,
        |_| {
            if first {
                first = false;
                let mut source = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&copy.0)
                    .unwrap();
                source.seek(SeekFrom::Start(file_bytes - 1)).unwrap();
                let mut original = [0u8; 1];
                source.read_exact(&mut original).unwrap();
                source.seek(SeekFrom::Start(file_bytes - 1)).unwrap();
                source.write_all(&[original[0] ^ 0xff]).unwrap();
                source
                    .set_times(std::fs::FileTimes::new().set_modified(modified))
                    .unwrap();
            }
            true
        },
    );
    assert!(rejected
        .unwrap_err()
        .contains("authenticated model block changed"));
    println!("verified hash admission, budget, cancellation and same-identity late mutation: PASS");
}
#[cfg(not(target_os = "linux"))]
fn main() {
    panic!("sealed tokenizer verification requires Linux");
}
