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
    "Usage:\n  llmbridge doctor\n  llmbridge hardware --json\n  llmbridge models inspect <path.gguf>\n  llmbridge models verify <path.gguf> <sha256hex>\n  llmbridge models tensor-index <path.gguf>\n  llmbridge models tensor-matvec <path.gguf> <q8-0-tensor-name> [expected-sha256]\n  llmbridge models native-index-probe <path.gguf>\n  llmbridge models tensor-sha256 <path.gguf> <tensor-name> [expected-sha256]\n  llmbridge plan <path.gguf> [--ctx 128] [--workspace-mb 8] [--max-ram-mb 64] [--auto-context] [--json]\n  llmbridge chat <path.gguf> --prompt <text> [--max-tokens 16] [--ctx 128] [--threads 2] [--stream] [--metrics] [--backend resident|streamed] [--workspace-mb 8] [--cache-mb 0] [--sha256 trusted-digest] [--verification-mb 8] [--address-space-mb 24] [--auto-context] [--max-ram-mb 64] [--logits path.f32]\n  Planning max-ram-mb is an incremental estimate budget; address-space-mb enforces a separate Linux process limit."
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
        Some("plan") => plan(&args[1..]),
        _ => Err(usage().to_string()),
    }
}

#[cfg(feature = "native")]
fn plan(args: &[String]) -> Result<(), String> {
    use llmbridge_core::streamed_plan::{
        inspect_streamed_shape, plan_streamed_shape, StreamedPlanRequest,
    };
    let model = args.first().ok_or("plan requires a GGUF path")?;
    let mut request = StreamedPlanRequest::default();
    let mut verification_mb = 8u64;
    let mut reserve_verification = false;
    let mut pos = 1;
    while pos < args.len() {
        match args[pos].as_str() {
            "--auto-context" => {
                request.allow_context_reduction = true;
                pos += 1;
                continue;
            }
            "--reserve-verification" => {
                reserve_verification = true;
                pos += 1;
                continue;
            }
            "--json" => {
                pos += 1;
                continue;
            }
            _ => {}
        }
        let value = args.get(pos + 1).ok_or("missing plan option value")?;
        let mb = |text: &str| -> Result<u64, String> {
            text.parse::<u64>()
                .map_err(|_| "invalid MiB value".into())
                .and_then(|n| n.checked_mul(1024 * 1024).ok_or("MiB overflow".into()))
        };
        match args[pos].as_str() {
            "--ctx" => request.context_tokens = value.parse().map_err(|_| "invalid ctx")?,
            "--workspace-mb" => request.workspace_budget_bytes = mb(value)?,
            "--cache-mb" => request.cache_budget_bytes = mb(value)?,
            "--max-ram-mb" => request.max_additional_bytes = Some(mb(value)?),
            "--os-reserve-mb" => request.os_reserve_bytes = mb(value)?,
            "--runtime-reserve-mb" => request.runtime_reserve_bytes = mb(value)?,
            "--verification-mb" => {
                verification_mb = value.parse().map_err(|_| "invalid verification-mb")?
            }
            "--backend" if value == "cpu" || value == "streamed" => {}
            _ => return Err("unknown/unsupported plan option".into()),
        }
        pos += 2;
    }
    if reserve_verification {
        if !(1..=64).contains(&verification_mb) {
            return Err("verification-mb must be 1..64".into());
        }
        request.verification_budget_bytes = verification_mb * 1024 * 1024;
    }
    let shape = inspect_streamed_shape(Path::new(model), 64 * 1024, &AtomicBool::new(false))?;
    let plan = plan_streamed_shape(shape, &hardware::profile(), request)?;
    println!("{}", plan.to_json(&shape, &request));
    Ok(())
}
#[cfg(not(feature = "native"))]
fn plan(_args: &[String]) -> Result<(), String> {
    Err("plan requires native streamed backend".into())
}

#[cfg(feature = "native")]
fn create_logit_trace(path: String) -> std::io::Result<std::fs::File> {
    // Never truncate an existing model, trace or symlink supplied as --logits.
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

/// Restricts this standalone CLI process, never a library host or parent process.
#[cfg(all(feature = "native", target_os = "linux"))]
fn restrict_address_space(bytes: u64) -> Result<(), String> {
    if bytes == 0 {
        return Err("address-space limit must be positive".into());
    }
    let mut previous = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit/setrlimit operate on the live stack rlimit and this CLI only.
    if unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut previous) } != 0 {
        return Err(format!(
            "cannot read address-space limit: {}",
            io::Error::last_os_error()
        ));
    }
    let requested = libc::rlim_t::try_from(bytes).map_err(|_| "address-space limit too large")?;
    let limit = requested.min(previous.rlim_cur).min(previous.rlim_max);
    let next = libc::rlimit {
        rlim_cur: limit,
        rlim_max: limit,
    };
    if unsafe { libc::setrlimit(libc::RLIMIT_AS, &next) } != 0 {
        return Err(format!(
            "cannot enforce address-space limit: {}",
            io::Error::last_os_error()
        ));
    }
    Ok(())
}
#[cfg(all(feature = "native", not(target_os = "linux")))]
fn restrict_address_space(_bytes: u64) -> Result<(), String> {
    Err("address-space enforcement currently requires Linux".into())
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
    let mut backend = String::from("resident");
    let mut workspace_mb = 8u64;
    let mut cache_mb = 0usize;
    let mut verification_mb = 8usize;
    let mut auto_context = false;
    let mut max_admission_mb: Option<u64> = None;
    let mut expected_hash: Option<String> = None;
    let mut address_space_mb: Option<u64> = None;
    let mut logits_path: Option<String> = None;
    let mut pos = 1usize;
    while pos < args.len() {
        if args[pos] == "--auto-context" {
            auto_context = true;
            pos += 1;
            continue;
        }
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
            "--backend" => backend = value.clone(),
            "--max-ram-mb" => {
                max_admission_mb = Some(value.parse().map_err(|_| "invalid max-ram-mb")?)
            }
            "--sha256" => expected_hash = Some(value.clone()),
            "--verification-mb" => {
                verification_mb = value.parse().map_err(|_| "invalid verification-mb")?
            }
            "--cache-mb" => cache_mb = value.parse().map_err(|_| "invalid cache-mb")?,
            "--address-space-mb" => {
                address_space_mb = Some(value.parse().map_err(|_| "invalid address-space-mb")?)
            }
            "--workspace-mb" => workspace_mb = value.parse().map_err(|_| "invalid workspace-mb")?,
            "--logits" => logits_path = Some(value.clone()),
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
    if backend != "resident" && backend != "streamed" {
        return Err("backend must be resident or streamed".into());
    }
    if backend != "streamed" && (auto_context || max_admission_mb.is_some()) {
        return Err("auto-context/admission budgeting requires streamed backend".into());
    }
    if verification_mb == 0
        || verification_mb > 64
        || (backend == "resident" && expected_hash.is_some())
    {
        return Err("verification requires streamed backend with verification-mb in 1..64".into());
    }
    if expected_hash
        .as_ref()
        .is_some_and(|hash| hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return Err("expected SHA-256 must be 64 hex digits".into());
    }
    if cache_mb > 32 || (backend == "resident" && cache_mb != 0) {
        return Err("cache-mb requires streamed backend and must be 0..32".into());
    }
    if backend == "resident" && logits_path.is_some() {
        return Err("--logits requires streamed backend".into());
    }
    if let Some(megabytes) = address_space_mb {
        restrict_address_space(
            megabytes
                .checked_mul(1024 * 1024)
                .ok_or("address-space limit overflow")?,
        )?;
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
        let estimate = if backend == "streamed" {
            workspace_mb
                .checked_mul(1024 * 1024)
                .and_then(|n| {
                    n.checked_add(
                        (32 + cache_mb as u64
                            + if expected_hash.is_some() {
                                verification_mb as u64
                            } else {
                                0
                            })
                            * 1024
                            * 1024,
                    )
                })
                .ok_or("workspace overflow")?
        } else {
            info.file_bytes.saturating_add(scratch_estimate)
        };
        hardware::admit(Some(available), os_reserve, estimate)?;
    } else {
        return Err(
            "available RAM not measurable; refusing unverified inference admission on this host"
                .into(),
        );
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    let stop = Arc::clone(&cancelled);
    ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
        .map_err(|e| format!("unable to install cancellation handler: {e}"))?;
    if backend == "streamed" && (auto_context || max_admission_mb.is_some()) {
        use llmbridge_core::streamed_plan::{
            inspect_streamed_shape, inspect_streamed_shape_verified, plan_streamed_shape,
            StreamedPlanRequest,
        };
        let shape = if let Some(hash) = expected_hash.as_deref() {
            inspect_streamed_shape_verified(
                model_path,
                64 * 1024,
                &cancelled,
                hash,
                verification_mb * 1024 * 1024,
            )?
        } else {
            inspect_streamed_shape(model_path, 64 * 1024, &cancelled)?
        };
        let process_headroom = address_space_mb
            .map(|mb| -> Result<u64, String> {
                let limit = mb
                    .checked_mul(1024 * 1024)
                    .ok_or("address-space MiB overflow")?;
                let current = hardware::process_memory()
                    .virtual_bytes
                    .ok_or("virtual memory unknown; cannot plan within address-space cap")?;
                Ok(limit.saturating_sub(current))
            })
            .transpose()?;
        let requested_budget = max_admission_mb
            .map(|mb| mb.checked_mul(1024 * 1024).ok_or("admission MiB overflow"))
            .transpose()?;
        let max_additional = match (requested_budget, process_headroom) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let request = StreamedPlanRequest {
            context_tokens: ctx,
            workspace_budget_bytes: workspace_mb
                .checked_mul(1024 * 1024)
                .ok_or("workspace overflow")?,
            max_additional_bytes: max_additional,
            cache_budget_bytes: (cache_mb as u64) * 1024 * 1024,
            verification_budget_bytes: if expected_hash.is_some() {
                (verification_mb as u64) * 1024 * 1024
            } else {
                0
            },
            allow_context_reduction: auto_context,
            ..Default::default()
        };
        let plan = plan_streamed_shape(shape, &hardware::profile(), request)?;
        eprintln!("LLMBRIDGE_PLAN {}", plan.to_json(&shape, &request));
        ctx = plan.selected_context;
    }
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let mut logits_file = logits_path
        .map(create_logit_trace)
        .transpose()
        .map_err(|e| e.to_string())?;
    let generated = if backend == "streamed" {
        let config = llmbridge_core::streamed::StreamedConfig {
            context_tokens: ctx,
            max_tokens,
            cache_bytes: cache_mb * 1024 * 1024,
            authentication_budget_bytes: verification_mb * 1024 * 1024,
            workspace_bytes: workspace_mb
                .checked_mul(1024 * 1024)
                .ok_or("workspace overflow")?,
            ..Default::default()
        };
        let emit = |bytes: &[u8]| handle.write_all(bytes).and_then(|_| handle.flush()).is_ok();
        let logits = |values: &[f32]| {
            if let Some(file) = logits_file.as_mut() {
                for value in values {
                    if file.write_all(&value.to_le_bytes()).is_err() {
                        return false;
                    }
                }
            }
            true
        };
        let execution = if let Some(hash) = expected_hash.as_ref() {
            llmbridge_core::streamed::generate_streamed_verified(
                model_path, &prompt, config, &cancelled, hash, emit, logits,
            )
        } else {
            llmbridge_core::streamed::generate_streamed(
                model_path, &prompt, config, &cancelled, emit, logits,
            )
        };
        execution
        .map(|metrics| {
            if show_metrics {
                if let Some(authentication) = metrics.authentication.as_ref() {
                    eprintln!("LLMBRIDGE_INTEGRITY verified=1 accounted_bytes={} blocks_verified={} sha256={}", authentication.accounted_bytes, authentication.blocks_verified, authentication.expected_sha256);
                } else { eprintln!("LLMBRIDGE_INTEGRITY verified=0"); }
                eprintln!(
                    "LLMBRIDGE_STREAMED source_bytes_read={} reader_buffer_bytes={} source_read_calls={} cache_allocated_bytes={} cache_block_hits={} cache_block_misses={} cache_evictions={}",
                    metrics.source_bytes_read, metrics.reader_buffer_bytes, metrics.cache.source_read_calls, metrics.cache.allocated_bytes, metrics.cache.block_hits, metrics.cache.block_misses, metrics.cache.evictions
                );
            }
        })
    } else {
        llmbridge_core::inference::generate_polled(
            model_path,
            &prompt,
            ctx,
            max_tokens,
            threads,
            &cancelled,
            |bytes| {
                if cancelled.load(Ordering::SeqCst) {
                    return false;
                }
                handle.write_all(bytes).and_then(|_| handle.flush()).is_ok()
            },
        )
    };

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
    #[cfg(feature = "native")]
    #[test]
    fn logits_export_never_truncates_an_existing_source() {
        let path = std::env::temp_dir().join(format!(
            "llmbridge-logits-protection-{}",
            std::process::id()
        ));
        std::fs::write(&path, b"preserve-model-or-existing-trace").unwrap();
        assert!(create_logit_trace(path.to_string_lossy().into_owned()).is_err());
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"preserve-model-or-existing-trace"
        );
        std::fs::remove_file(path).unwrap();
    }

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
