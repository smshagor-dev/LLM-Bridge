mod checkpoint;
use llmbridge_core::{
    hardware,
    memory_pressure::MemoryPressurePolicy,
    queued_streamed::{QueuedExecutionLimits, QueuedStreamedRequest},
    streamed_executor::{ExecutorConfig, StreamedExecutor},
};
use sha2::{Digest, Sha256};
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
    deadline_ms: Option<u64>,
    line_bytes: usize,
    tokens: u32,
    context: u32,
    digest: Option<String>,
    pressure: Option<MemoryPressurePolicy>,
    resumable: bool,
    resume: bool,
}
fn options(args: &[String]) -> Result<Options, String> {
    let model = args.first().ok_or("batch requires a GGUF model path")?;
    let mut input = None;
    let mut output = None;
    let mut workers = 1;
    let mut slot = 64;
    let mut reserve = 512;
    let mut wait = 30_000;
    let mut deadline_ms = None;
    let mut line_bytes = llmbridge_core::streamed::MAX_STREAMED_PROMPT_BYTES;
    let mut tokens = 16;
    let mut context = 128;
    let mut digest = None;
    let mut low = None;
    let mut high = None;
    let mut resumable = false;
    let mut resume = false;
    let mut position = 1;
    while position < args.len() {
        match args[position].as_str() {
            "--resumable" => {
                resumable = true;
                position += 1;
                continue;
            }
            "--resume" => {
                resume = true;
                position += 1;
                continue;
            }
            _ => {}
        }
        let pair = &args[position..args.len().min(position + 2)];
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
            "--deadline-ms" => deadline_ms = Some(number()?),
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
        position += 2;
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
    if (resumable && resume) || ((resumable || resume) && digest.is_none()) {
        return Err("choose --resumable or --resume, with a trusted --sha256".into());
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
        deadline_ms,
        line_bytes,
        tokens,
        context,
        digest,
        pressure,
        resumable,
        resume,
    })
}

fn next_prompt(
    reader: &mut impl BufRead,
    limit: usize,
    hash: Option<&mut Sha256>,
) -> Result<Option<String>, String> {
    let mut bytes = Vec::new();
    reader
        .take(limit as u64 + 1)
        .read_until(b'\n', &mut bytes)
        .map_err(|e| format!("prompt read failed: {e}"))?;
    if let Some(hash) = hash {
        hash.update(&bytes);
    }
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
    let mut file =
        File::open(&options.input).map_err(|e| format!("cannot open prompt input: {e}"))?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&cancelled);
    ctrlc::set_handler(move || signal.store(true, Ordering::Release))
        .map_err(|e| format!("cannot install cancellation handler: {e}"))?;
    let journal = if options.resumable || options.resume {
        Some(checkpoint::Journal::open(&options, &mut file, &cancelled)?)
    } else {
        // Ordinary batches still require a new output directory.
        fs::create_dir(&options.output)
            .map_err(|e| format!("cannot create new output directory: {e}"))?;
        None
    };
    let input = Mutex::new((
        BufReader::new(file),
        0u64,
        journal.as_ref().map(|_| Sha256::new()),
    ));
    let skipped = AtomicU64::new(0);
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
                                let (reader, counter, hash) = &mut *input;
                                match next_prompt(reader, options.line_bytes, hash.as_mut())? {
                                    None => {
                                        if let (Some(journal), Some(hash)) = (&journal, hash) {
                                            if format!("{:x}", hash.clone().finalize())
                                                != journal.input_sha
                                            {
                                                return Err(
                                                    "batch input changed while processing".into()
                                                );
                                            }
                                        }
                                        None
                                    }
                                    Some(prompt) => {
                                        *counter = counter
                                            .checked_add(1)
                                            .ok_or("batch job counter overflow")?;
                                        Some((*counter, prompt))
                                    }
                                }
                            };
                            let Some((id, prompt)) = item else {
                                return Ok(());
                            };
                            if let Some(journal) = &journal {
                                if journal.before_job(id, &prompt, &cancelled)? {
                                    skipped.fetch_add(1, Ordering::Relaxed);
                                    eprintln!("job={id} status=verified-skip");
                                    continue;
                                }
                            }
                            let partial = options.output.join(format!("job-{id:08}.partial"));
                            let target = options.output.join(format!("job-{id:08}.txt"));
                            let mut output = OpenOptions::new()
                                .read(journal.is_some())
                                .write(true)
                                .create_new(true)
                                .open(&partial)
                                .map_err(|e| format!("job {id}: cannot create output: {e}"))?;
                            let mut request = template;
                            request.prompt = &prompt;
                            let mut write_error = None;
                            let limits = QueuedExecutionLimits {
                                wait_timeout: Some(options.wait),
                                deadline: super::execution_deadline(options.deadline_ms)?,
                            };
                            let generated = executor.generate_with_limits(
                                request,
                                &cancelled,
                                limits,
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
                            let receipt = journal
                                .as_ref()
                                .map(|journal| {
                                    journal.result_record(id, &prompt, &mut output, &cancelled)
                                })
                                .transpose()?;
                            drop(output);
                            // Atomic no-replace publication; preserve partial output on error.
                            fs::hard_link(&partial, &target)
                                .map_err(|e| format!("job {id}: cannot publish result: {e}"))?;
                            fs::remove_file(&partial).map_err(|e| {
                                format!("job {id}: cannot remove partial name: {e}")
                            })?;
                            if let (Some(journal), Some(receipt)) = (&journal, receipt) {
                                journal.complete(id, &receipt)?;
                            }
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
        "completed={} workers={} pending={} reserved_bytes={}{}",
        completed.load(Ordering::Relaxed),
        executor.worker_count(),
        executor.pending_sessions(),
        executor.reserved_bytes(),
        if journal.is_some() {
            format!(" skipped={}", skipped.load(Ordering::Relaxed))
        } else {
            String::new()
        }
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
    fn checkpoint_modes_require_trust_and_are_exclusive() {
        let args = |flags: &[&str]| {
            let mut values = vec!["model", "--input", "in", "--output-dir", "out"];
            values.extend_from_slice(flags);
            values.into_iter().map(str::to_string).collect::<Vec<_>>()
        };
        assert!(options(&args(&["--resumable"])).is_err());
        assert!(options(&args(&["--resume"])).is_err());
        let digest = "0".repeat(64);
        assert!(options(&args(&["--resumable", "--resume", "--sha256", &digest])).is_err());
        assert!(
            options(&args(&["--resumable", "--sha256", &digest]))
                .unwrap()
                .resumable
        );
        assert!(
            options(&args(&["--sha256", &digest, "--resume"]))
                .unwrap()
                .resume
        );
    }
    #[test]
    fn checkpoint_input_hash_includes_exact_line_endings() {
        let bytes = b"one\r\ntwo";
        let mut reader = &bytes[..];
        let mut hash = Sha256::new();
        assert_eq!(
            next_prompt(&mut reader, 8, Some(&mut hash))
                .unwrap()
                .unwrap(),
            "one"
        );
        assert_eq!(
            next_prompt(&mut reader, 8, Some(&mut hash))
                .unwrap()
                .unwrap(),
            "two"
        );
        assert!(next_prompt(&mut reader, 8, Some(&mut hash))
            .unwrap()
            .is_none());
        assert_eq!(hash.finalize(), Sha256::digest(bytes));
    }
    #[test]
    fn input_is_bounded_and_handles_utf8_crlf_and_final_line() {
        let mut input = &b"one\r\n\xd0\xb4\xd0\xb2\xd0\xb0"[..];
        assert_eq!(next_prompt(&mut input, 6, None).unwrap().unwrap(), "one");
        assert_eq!(next_prompt(&mut input, 6, None).unwrap().unwrap(), "два");
        assert!(next_prompt(&mut input, 6, None).unwrap().is_none());
        assert!(next_prompt(&mut &b"abcd\n"[..], 4, None).is_err());
        assert!(next_prompt(&mut &b"\xff\n"[..], 4, None).is_err());
        assert!(next_prompt(&mut &b"\n"[..], 4, None).is_err());
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
            vec!["--deadline-ms", "-1"],
            vec!["--deadline-ms", "18446744073709551616"],
        ] {
            assert!(options(&args(&flags)).is_err());
        }
    }
}
