use llmbridge_core::{gguf_index::GgufTensorIndex, hardware, models};
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
    "Usage:\n  llmbridge doctor\n  llmbridge hardware --json\n  llmbridge models inspect <path.gguf>\n  llmbridge models verify <path.gguf> <sha256hex>\n  llmbridge models tensor-index <path.gguf>\n  llmbridge models tensor-matvec <path.gguf> <q8-0-tensor-name> [expected-sha256]\n  llmbridge models native-index-probe <path.gguf>\n  llmbridge models tensor-sha256 <path.gguf> <tensor-name> [expected-sha256]\n  llmbridge chat <path.gguf> --prompt <text> [--max-tokens 16] [--ctx 128] [--threads 2] [--stream] [--metrics]"
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

        #[cfg(feature = "native")]
        Some("models") if args.len() == 3 && args[1] == "native-index-probe" => {
            let index = GgufTensorIndex::inspect(&args[2]).map_err(|e| e.to_string())?;
            let mut reader =
                llmbridge_core::storage::BoundedLocalReader::open(&args[2], 64 * 1024, 128 * 1024)
                    .map_err(|e| e.to_string())?;
            let probe = llmbridge_core::inference::probe_indexed_gguf(&index, &mut reader)?;
            println!("native tensor count: {}", probe.tensor_count);
            println!("native tensor data offset: {}", probe.data_offset);
            println!("bounded source bytes read: {}", reader.disk_bytes_read());
            println!(
                "bounded callback buffer bytes: {}",
                reader.allocated_chunk_bytes()
            );
            println!(
                "status: native GGUF metadata agrees with Rust index; NOT out-of-core inference"
            );
            Ok(())
        }

        #[cfg(feature = "native")]
        Some("models") if (args.len() == 4 || args.len() == 5) && args[1] == "tensor-matvec" => {
            let index = GgufTensorIndex::inspect(&args[2]).map_err(|e| e.to_string())?;
            let tensor = index.tensor(&args[3]).ok_or("tensor does not exist")?;
            let cols = usize::try_from(*tensor.dimensions.first().ok_or("empty shape")?)
                .map_err(|_| "invalid tensor shape")?;
            if cols > 16384 || tensor.ggml_type != 8 || tensor.dimensions.len() != 2 {
                return Err("requires rank-2 GGML Q8_0 with <=16384 columns".to_string());
            }
            let mut reader =
                llmbridge_core::storage::BoundedLocalReader::open(&args[2], 64 * 1024, 128 * 1024)
                    .map_err(|e| e.to_string())?;
            let cancelled = AtomicBool::new(false);
            let ones = vec![1.0f32; cols];
            let output = if args.len() == 5 {
                llmbridge_core::inference::q8_0_matvec_verified(
                    &index,
                    &mut reader,
                    &args[3],
                    &ones,
                    &cancelled,
                    &args[4],
                )?
            } else {
                llmbridge_core::inference::q8_0_matvec(
                    &index,
                    &mut reader,
                    &args[3],
                    &ones,
                    &cancelled,
                )?
            };
            println!("tensor: {}", args[3]);
            println!("rows: {}", output.len());
            println!("columns: {cols}");
            for (row, value) in output.iter().enumerate() {
                println!("row {row} {value:.8}");
            }
            println!("weight bytes read: {}", reader.disk_bytes_read());
            println!("native scratch: one quantized row + one dequantized row");
            println!(
                "integrity: {}",
                if args.len() == 5 {
                    "SHA-256 verified against caller-supplied digest"
                } else {
                    "NOT independently verified"
                }
            );
            println!("status: real bounded tensor math; not full transformer inference");
            Ok(())
        }
        Some("models") if args.len() == 3 && args[1] == "tensor-index" => {
            let index = GgufTensorIndex::inspect(&args[2]).map_err(|e| e.to_string())?;
            println!("GGUF version: {}", index.version);
            println!("tensor count: {}", index.tensors.len());
            println!("tensor data offset: {}", index.data_offset);
            println!("file bytes: {}", index.file_bytes);
            for tensor in &index.tensors {
                println!(
                    "{} type={} dims={:?} offset={} size={}",
                    tensor.name,
                    tensor.ggml_type,
                    tensor.dimensions,
                    tensor.file_offset,
                    tensor.byte_len
                );
            }
            println!("status: validated indexed byte ranges; execution not memory-bounded");
            Ok(())
        }
        Some("models") if (args.len() == 4 || args.len() == 5) && args[1] == "tensor-sha256" => {
            let index = GgufTensorIndex::inspect(&args[2]).map_err(|e| e.to_string())?;
            let mut reader =
                llmbridge_core::storage::BoundedLocalReader::open(&args[2], 64 * 1024, 128 * 1024)
                    .map_err(|e| e.to_string())?;
            let digest = if args.len() == 5 {
                index.verify_tensor_sha256(&mut reader, &args[3], &args[4])
            } else {
                index.sha256_tensor(&mut reader, &args[3])
            }
            .map_err(|e| e.to_string())?;
            println!("tensor: {}", args[3]);
            println!("sha256: {digest}");
            println!("allocated chunk bytes: {}", reader.allocated_chunk_bytes());
            println!("disk bytes read: {}", reader.disk_bytes_read());
            println!("status: tensor byte stream verified locally; not model-execution parity");
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
    let mut show_metrics = false;
    let mut pos = 1usize;
    while pos < args.len() {
        if args[pos] == "--metrics" {
            show_metrics = true;
            pos += 1;
            continue;
        }
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

    if show_metrics {
        let usage = hardware::process_memory();
        // VmHWM is a process-lifetime peak; the other values are post-run samples.
        eprintln!(
            "LLMBRIDGE_MEMORY {{\"peak_rss_bytes\":{},\"rss_after_run_bytes\":{},\"pss_after_run_bytes\":{},\"cgroup_current_after_run_bytes\":{},\"cgroup_limit_bytes\":{}}}",
            json_number(usage.peak_rss_bytes),
            json_number(usage.rss_bytes),
            json_number(usage.pss_bytes),
            json_number(usage.cgroup_current_bytes),
            json_number(usage.cgroup_limit_bytes)
        );
    }
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
