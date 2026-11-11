use llmbridge_core::{hardware, models};
#[cfg(feature = "native")]
use std::io::{self, Write};
#[cfg(feature = "native")]
use std::path::Path;
#[cfg(feature = "native")]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn main() {
    if let Err(error) = run(std::env::args().skip(1).collect()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
fn usage() -> &'static str {
    "Usage:\n  llmbridge doctor\n  llmbridge hardware --json\n  llmbridge models inspect <path.gguf>\n  llmbridge models verify <path.gguf> <sha256hex>\n  llmbridge chat <path.gguf> --prompt <text> [--max-tokens 16] [--ctx 128] [--threads 2] [--stream]"
}
fn json_number(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |x| x.to_string())
}
fn run(args: Vec<String>) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("doctor") if args.len() == 1 => {
            let h = hardware::profile();
            println!("LLM Bridge M1 — real CPU GGUF baseline; not out-of-core inference");
            println!("logical CPUs: {}", h.logical_cpus);
            println!("total RAM bytes: {:?}", h.total_ram_bytes);
            println!("available RAM bytes: {:?}", h.available_ram_bytes);
            Ok(())
        }
        Some("hardware") if args.len() == 2 && args[1] == "--json" => {
            let h = hardware::profile();
            println!(
                "{{\"logical_cpus\":{},\"total_ram_bytes\":{},\"available_ram_bytes\":{}}}",
                h.logical_cpus,
                json_number(h.total_ram_bytes),
                json_number(h.available_ram_bytes)
            );
            Ok(())
        }
        Some("models") if args.len() == 4 && args[1] == "verify" => {
            let mut reader = llmbridge_core::storage::BoundedLocalReader::open(
                &args[2],
                256 * 1024,
                2 * 1024 * 1024,
            )
            .map_err(|e| e.to_string())?;
            let digest = reader.verify_sha256(&args[3]).map_err(|e| e.to_string())?;
            println!("sha256: {digest}");
            println!("file bytes: {}", reader.file_bytes());
            println!("allocated chunk bytes: {}", reader.allocated_chunk_bytes());
            println!("disk bytes read: {}", reader.disk_bytes_read());
            println!("status: entire local file SHA-256 verified with bounded reusable buffer; not an executable model proof");
            Ok(())
        }
        Some("models") if args.len() == 3 && args[1] == "inspect" => {
            let h = models::inspect_gguf(&args[2]).map_err(|e| e.to_string())?;
            println!("path: {}", h.path.display());
            println!("file bytes: {}", h.file_bytes);
            println!("GGUF version: {}", h.format_version);
            println!("declared tensors: {}", h.tensor_count);
            println!("metadata entries: {}", h.metadata_entries);
            println!(
                "status: header inspected only (payload and executable compatibility unverified)"
            );
            Ok(())
        }
        Some("chat") => chat(&args[1..]),
        _ => Err(usage().to_string()),
    }
}

#[cfg(feature = "native")]
fn chat(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err(usage().to_owned());
    }
    let model_path = Path::new(&args[0]);
    let mut prompt: Option<String> = None;
    let mut max_tokens = 16u32;
    let mut ctx = 128u32;
    let mut threads = 2i32;
    let mut pos = 1usize;
    while pos < args.len() {
        if args[pos] == "--stream" {
            pos += 1;
            continue;
        }
        let flag = args[pos].as_str();
        let value = args
            .get(pos + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--prompt" => prompt = Some(value.clone()),
            "--max-tokens" => {
                max_tokens = value.parse::<u32>().map_err(|_| "invalid max-tokens")?
            }
            "--ctx" => ctx = value.parse::<u32>().map_err(|_| "invalid ctx")?,
            "--threads" => threads = value.parse::<i32>().map_err(|_| "invalid threads")?,
            _ => return Err(format!("unknown flag {flag}")),
        }
        pos += 2;
    }
    let prompt = prompt.ok_or("--prompt is required")?;
    if prompt.is_empty() {
        return Err("empty prompt".to_string());
    }
    if !(64..=8192).contains(&ctx)
        || !(1..=2048).contains(&max_tokens)
        || !(1..=64).contains(&threads)
    {
        return Err("context, token or thread request exceeds M1 bounds".to_string());
    }
    let info =
        models::inspect_gguf(model_path).map_err(|e| format!("GGUF inspection failed: {e}"))?;
    let h = hardware::profile();
    if let Some(available) = h.available_ram_bytes {
        // This is deliberately conservative pre-load admission, not an enforceable RSS cap.
        let os_reserve = h
            .total_ram_bytes
            .unwrap_or(0)
            .saturating_div(4)
            .max(512 * 1024 * 1024);
        let scratch_estimate = 256_u64 * 1024 * 1024 + u64::from(ctx) * 128 * 1024;
        let estimate = info.file_bytes.saturating_add(scratch_estimate);
        hardware::admit(Some(available), os_reserve, estimate)?;
    } else {
        eprintln!("warning: available RAM not measurable on this host; M1 has no enforced process memory limit");
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    let stop = Arc::clone(&cancelled);
    ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
        .map_err(|e| format!("unable to install cancellation handler: {e}"))?;
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let generated = llmbridge_core::inference::generate(
        model_path,
        &prompt,
        ctx,
        max_tokens,
        threads,
        |bytes| {
            if cancelled.load(Ordering::SeqCst) {
                return false;
            }
            handle.write_all(bytes).and_then(|_| handle.flush()).is_ok()
        },
    );
    writeln!(handle).map_err(|e| format!("stdout error: {e}"))?;
    generated
}

#[cfg(not(feature = "native"))]
fn chat(_args: &[String]) -> Result<(), String> {
    Err("this binary has no native CPU backend: rebuild with --features native".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_gguf_input_returns_error() {
        assert!(run(vec![
            "models".into(),
            "inspect".into(),
            "/does-not-exist.gguf".into()
        ])
        .is_err());
    }
    #[test]
    fn hardware_json_uses_json_null_for_unknown() {
        assert_eq!(json_number(None), "null");
        assert_eq!(json_number(Some(1_024)), "1024");
    }
    #[test]
    fn missing_chat_model_is_an_error() {
        assert!(chat(&[]).is_err());
    }
}
