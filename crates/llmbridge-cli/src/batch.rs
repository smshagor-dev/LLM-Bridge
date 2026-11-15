use llmbridge_core::{
    hardware,
    memory_pressure::MemoryPressurePolicy,
    queued_streamed::QueuedStreamedRequest,
    streamed_executor::{ExecutorConfig, StreamedExecutor},
};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

struct Options {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    workers: usize,
    slot: u64,
    reserve: u64,
    wait: Duration,
    line_bytes: usize,
    tokens: u32,
    context: u32,
    digest: Option<String>,
    pressure: Option<MemoryPressurePolicy>,
}
fn options(args: &[String]) -> Result<Options, String> {
    let model = args.first().ok_or("batch requires a GGUF model path")?;
    let mut input = None;
    let mut output = None;
    let mut workers = 1;
    let mut slot = 64;
    let mut reserve = 512;
    let mut wait = 30_000;
    let mut line_bytes = llmbridge_core::streamed::MAX_STREAMED_PROMPT_BYTES;
    let mut tokens = 16;
    let mut context = 128;
    let mut digest = None;
    let mut low = None;
    let mut high = None;
    for pair in args[1..].chunks(2) {
        let value = pair.get(1).ok_or("missing batch option value")?;
        let number = || {
            value
                .parse::<u64>()
                .map_err(|_| format!("invalid value for {}", pair[0]))
        };
        match pair[0].as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--output-dir" => output = Some(PathBuf::from(value)),
            "--workers" => workers = usize::try_from(number()?).map_err(|_| "workers too large")?,
            "--slot-mb" => slot = number()?,
            "--os-reserve-mb" => reserve = number()?,
            "--wait-timeout-ms" => wait = number()?,
            "--max-prompt-bytes" => {
                line_bytes = usize::try_from(number()?).map_err(|_| "prompt limit too large")?
            }
            "--max-tokens" => {
                tokens = u32::try_from(number()?).map_err(|_| "token count too large")?
            }
            "--ctx" => context = u32::try_from(number()?).map_err(|_| "context too large")?,
            "--sha256" => digest = Some(value.clone()),
            "--pressure-pause-mb" => low = Some(number()?),
            "--pressure-resume-mb" => high = Some(number()?),
            flag => return Err(format!("unknown batch option {flag}")),
        }
    }
    let mb = |n: u64| n.checked_mul(1024 * 1024).ok_or("MiB overflow".to_string());
    if workers == 0
        || slot == 0
        || line_bytes == 0
        || line_bytes > llmbridge_core::streamed::MAX_STREAMED_PROMPT_BYTES
        || !(1..=2048).contains(&tokens)
        || !(64..=8192).contains(&context)
    {
        return Err("invalid batch worker/slot/prompt/token/context limit".into());
    }
    if digest
        .as_ref()
        .is_some_and(|d| d.len() != 64 || !d.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return Err("sha256 must contain 64 hexadecimal digits".into());
    }
    let pressure =
        match (low, high) {
            (None, None) => None,
            (Some(low), Some(high)) if high > low => Some(MemoryPressurePolicy {
                pause_below_bytes: mb(low)?,
                resume_at_bytes: mb(high)?,
                sample_interval: Duration::from_millis(100),
            }),
            _ => return Err(
                "pressure pause/resume must be supplied together with resume greater than pause"
                    .into(),
            ),
        };
    Ok(Options {
        model: model.into(),
        input: input.ok_or("batch requires --input")?,
        output: output.ok_or("batch requires --output-dir")?,
        workers,
        slot: mb(slot)?,
        reserve: mb(reserve)?,
        wait: Duration::from_millis(wait),
        line_bytes,
        tokens,
        context,
        digest,
        pressure,
    })
}

fn next_prompt(reader: &mut impl BufRead, limit: usize) -> Result<Option<String>, String> {
    let mut bytes = Vec::new();
    reader
        .take(limit as u64 + 1)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| format!("prompt read failed: {e}"))?;
    if bytes.len() > limit {
        return Err("prompt exceeds --max-prompt-bytes (including line ending)".into());
    }
    if bytes.is_empty() {
        return Ok(None);
    }
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
    }
    if bytes.is_empty() {
        return Err("empty prompt line".into());
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| "prompt input must be UTF-8".into())
}

pub fn run(args: &[String]) -> Result<(), String> {
    let options = options(args)?;
    llmbridge_core::native_cpu::ensure_compatible()?;
    let config = ExecutorConfig {
        requested_workers: options.workers,
        session_slot_bytes: options.slot,
        os_reserve_bytes: options.reserve,
        max_pending: options.workers,
    };
    let profile = hardware::profile();
    let executor = match options.pressure {
        Some(policy) => StreamedExecutor::new_with_pressure(&profile, config, policy)?,
        None => StreamedExecutor::new(&profile, config)?,
    };
    let mut template = QueuedStreamedRequest::new(&options.model, "validation");
    template.config.max_tokens = options.tokens;
    template.config.context_tokens = options.context;
    template.control.expected_sha256 = options.digest.as_deref();
    if template.reservation_bytes()? > executor.slot_bytes() {
        return Err("batch request exceeds session slot".into());
    }
    let file = File::open(&options.input).map_err(|e| format!("cannot open prompt input: {e}"))?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&cancelled);
    ctrlc::set_handler(move || signal.store(true, Ordering::Release))
        .map_err(|e| format!("cannot install cancellation handler: {e}"))?;
    // Never reuse/truncate an existing output directory or prompt/model path.
    fs::create_dir(&options.output)
        .map_err(|e| format!("cannot create new output directory: {e}"))?;
    let input = Mutex::new((BufReader::new(file), 0u64));
    let completed = AtomicU64::new(0);
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let result = std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for index in 0..executor.worker_count() {
            let worker = std::thread::Builder::new()
                .name(format!("batch-{index}"))
                .spawn_scoped(scope, || -> Result<(), String> {
                    let process = || -> Result<(), String> {
                        loop {
                            if cancelled.load(Ordering::Acquire) {
                                return Err("batch cancelled".into());
                            }
                            let item = {
                                let mut input =
                                    input.lock().map_err(|_| "batch input lock poisoned")?;
                                if cancelled.load(Ordering::Acquire) {
                                    return Err("batch cancelled".into());
                                }
                                match next_prompt(&mut input.0, options.line_bytes)? {
                                    None => None,
                                    Some(prompt) => {
                                        input.1 = input
                                            .1
                                            .checked_add(1)
                                            .ok_or("batch job counter overflow")?;
                                        Some((input.1, prompt))
                                    }
                                }
                            };
                            let Some((id, prompt)) = item else {
                                return Ok(());
                            };
                            let partial = options.output.join(format!("job-{id:08}.partial"));
                            let target = options.output.join(format!("job-{id:08}.txt"));
                            let mut output = OpenOptions::new()
                                .write(true)
                                .create_new(true)
                                .open(&partial)
                                .map_err(|e| format!("job {id}: cannot create output: {e}"))?;
                            let mut request = template;
                            request.prompt = &prompt;
                            let mut write_error = None;
                            let generated = executor.generate_with_timeout(
                                request,
                                &cancelled,
                                options.wait,
                                |bytes| match output.write_all(bytes) {
                                    Ok(()) => true,
                                    Err(error) => {
                                        write_error = Some(error);
                                        false
                                    }
                                },
                                |_| true,
                            );
                            if let Some(error) = write_error {
                                return Err(format!("job {id}: output write failed: {error}"));
                            }
                            let metrics = generated.map_err(|e| format!("job {id}: {e}"))?;
                            output
                                .flush()
                                .map_err(|e| format!("job {id}: output flush failed: {e}"))?;
                            drop(output);
                            // Atomic no-replace publication; preserve partial output on error.
                            fs::hard_link(&partial, &target)
                                .map_err(|e| format!("job {id}: cannot publish result: {e}"))?;
                            fs::remove_file(&partial).map_err(|e| {
                                format!("job {id}: cannot remove partial name: {e}")
                            })?;
                            completed.fetch_add(1, Ordering::Relaxed);
                            eprintln!(
                                "job={id} status=complete reservation_bytes={} queue_wait_ns={}",
                                metrics.reservation_bytes, metrics.queue_wait_ns
                            );
                        }
                    };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(process))
                        .unwrap_or_else(|_| Err("batch worker panicked".into()));
                    if let Err(error) = &result {
                        failure
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .get_or_insert_with(|| error.clone());
                        cancelled.store(true, Ordering::Release);
                    }
                    result
                });
            match worker {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    failure
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .get_or_insert_with(|| format!("cannot start batch worker: {error}"));
                    cancelled.store(true, Ordering::Release);
                    break;
                }
            }
        }
        for worker in workers {
            if worker.join().is_err() {
                failure
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_or_insert_with(|| "batch worker panicked".into());
                cancelled.store(true, Ordering::Release);
            }
        }
        failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .map_or(Ok(()), Err)
    });
    println!(
        "completed={} workers={} pending={} reserved_bytes={}",
        completed.load(Ordering::Relaxed),
        executor.worker_count(),
        executor.pending_sessions(),
        executor.reserved_bytes()
    );
    result?;
    if cancelled.load(Ordering::Acquire) {
        return Err("batch cancelled".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn input_is_bounded_and_handles_utf8_crlf_and_final_line() {
        let mut input = &b"one\r\n\xd0\xb4\xd0\xb2\xd0\xb0"[..];
        assert_eq!(next_prompt(&mut input, 6).unwrap().unwrap(), "one");
        assert_eq!(next_prompt(&mut input, 6).unwrap().unwrap(), "два");
        assert!(next_prompt(&mut input, 6).unwrap().is_none());
        assert!(next_prompt(&mut &b"abcd\n"[..], 4).is_err());
        assert!(next_prompt(&mut &b"\xff\n"[..], 4).is_err());
        assert!(next_prompt(&mut &b"\n"[..], 4).is_err());
    }
    #[test]
    fn malformed_options_reject_before_io() {
        let args = |extra: &[&str]| {
            let mut a = vec!["model", "--input", "in", "--output-dir", "out"];
            a.extend_from_slice(extra);
            a.into_iter().map(str::to_string).collect::<Vec<_>>()
        };
        assert!(options(&args(&[])).is_ok());
        for flags in [
            vec!["--workers", "0"],
            vec!["--slot-mb", "18446744073709551615"],
            vec!["--pressure-pause-mb", "64"],
            vec!["--sha256", "bad"],
            vec!["--max-prompt-bytes", "0"],
            vec!["--max-prompt-bytes", "65537"],
            vec!["--ctx", "1"],
        ] {
            assert!(options(&args(&flags)).is_err());
        }
    }
}
