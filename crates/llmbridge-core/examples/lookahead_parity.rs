#[cfg(target_os = "linux")]
use llmbridge_core::streamed::generate_streamed_verified;
use llmbridge_core::streamed::{
    generate_streamed, generate_streamed_with_lookahead, LookaheadLimits, StreamedConfig,
    StreamedControl, StreamedDeadlineControl, StreamedLookaheadControl,
};
use std::{
    env, fs,
    path::PathBuf,
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};

const MODEL_SHA256: &str = "2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: lookahead_parity MODEL.gguf OUTPUT_DIR [EXPIRY_SECONDS]".into());
    }
    let model = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    let expiry_seconds = args
        .get(2)
        .map(|value| value.to_string_lossy().parse::<u64>())
        .transpose()?
        .unwrap_or(15);
    if expiry_seconds == 0 {
        return Err("expiry seconds must be nonzero".into());
    }
    fs::create_dir_all(&out)?;
    let config = StreamedConfig {
        context_tokens: 64,
        max_tokens: 8,
        chunk_bytes: 64 * 1024,
        workspace_bytes: 8 * 1024 * 1024,
        cache_bytes: 0,
        authentication_budget_bytes: 8 * 1024 * 1024,
        collect_performance: true,
    };
    let cancelled = AtomicBool::new(false);
    let mut sync_text = Vec::new();
    let mut sync_logits = Vec::new();
    #[cfg(target_os = "linux")]
    let sync = generate_streamed_verified(
        &model,
        "Once upon a time",
        config,
        &cancelled,
        MODEL_SHA256,
        |p| {
            sync_text.extend_from_slice(p);
            true
        },
        |l| {
            sync_logits.extend_from_slice(l);
            true
        },
    )?;
    // Sealed tokenizer verification currently depends on Linux memfd seals.
    // Windows NTFS runs this protocol with the exact model hash checked by
    // the harness before launch, but without claiming sealed-source auth.
    #[cfg(target_os = "windows")]
    let sync = generate_streamed(
        &model,
        "Once upon a time",
        config,
        &cancelled,
        |p| {
            sync_text.extend_from_slice(p);
            true
        },
        |l| {
            sync_logits.extend_from_slice(l);
            true
        },
    )?;
    let mut ahead_text = Vec::new();
    let mut ahead_logits = Vec::new();
    let limits = LookaheadLimits {
        staging_bytes: 4 * 1024 * 1024,
        max_outstanding_reads: 2,
        fixed_overhead_bytes: 2 * 1024 * 1024,
        worker_stack_bytes: 1024 * 1024,
    };
    let ahead = generate_streamed_with_lookahead(
        &model,
        "Once upon a time",
        config,
        &cancelled,
        StreamedDeadlineControl {
            control: StreamedControl {
                expected_sha256: if cfg!(target_os = "linux") {
                    Some(MODEL_SHA256)
                } else {
                    None
                },
                cache: None,
            },
            deadline: Instant::now() + Duration::from_secs(300),
        },
        StreamedLookaheadControl { limits },
        |p| {
            ahead_text.extend_from_slice(p);
            true
        },
        |l| {
            ahead_logits.extend_from_slice(l);
            true
        },
    )?;
    if sync_text != ahead_text || sync_logits != ahead_logits {
        return Err("lookahead and synchronous outputs/logits differ".into());
    }
    fs::write(out.join("lookahead.stdout"), &ahead_text)?;
    fs::write(
        out.join("lookahead.logits.f32"),
        ahead_logits
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    fs::write(out.join("synchronous.stdout"), &sync_text)?;
    fs::write(
        out.join("synchronous.logits.f32"),
        sync_logits
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    println!(
        "lookahead scheduled={} consumed={} worker_terminated={} final_reserved={} outstanding={} worker_source_bytes={}",
        ahead.lookahead_scheduled_reads,
        ahead.lookahead_consumed_reads,
        ahead.lookahead_worker_terminated,
        ahead.lookahead_final_reserved_bytes,
        ahead.lookahead_final_outstanding_reads,
        ahead.lookahead_source_bytes_read
    );
    if ahead.lookahead_scheduled_reads == 0
        || ahead.lookahead_consumed_reads == 0
        || !ahead.lookahead_worker_terminated
        || ahead.lookahead_final_reserved_bytes != 0
        || ahead.lookahead_final_outstanding_reads != 0
    {
        return Err("lookahead integration/cleanup was not observed".into());
    }
    println!("synchronous and lookahead complete output/logit parity: PASS; source_bytes_sync={} source_bytes_lookahead={}",
        sync.source_bytes_read, ahead.source_bytes_read);
    // Partial output before expiry is permitted. Record each consumer entry
    // boundary and prove none starts after the deadline is observed.
    let expiry = Instant::now() + Duration::from_secs(expiry_seconds);
    let late_delivery = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let expiry_observed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observer_flag = expiry_observed.clone();
    let observer = std::thread::spawn(move || {
        while Instant::now() < expiry {
            std::thread::yield_now();
        }
        observer_flag.store(true, std::sync::atomic::Ordering::Release);
    });
    let late_delivery_flag = late_delivery.clone();
    let expiry_observed_text = expiry_observed.clone();
    let mut deadline_config = config;
    deadline_config.context_tokens = 128;
    deadline_config.max_tokens = 64;
    let mut partial_bytes = 0usize;
    let deadline_error = generate_streamed_with_lookahead(
        &model,
        "Once upon a time",
        deadline_config,
        &cancelled,
        StreamedDeadlineControl {
            control: StreamedControl::default(),
            deadline: expiry,
        },
        StreamedLookaheadControl { limits },
        |piece| {
            if expiry_observed_text.load(std::sync::atomic::Ordering::Acquire) {
                late_delivery_flag.store(true, std::sync::atomic::Ordering::Release);
            }
            partial_bytes += piece.len();
            true
        },
        |_| {
            if expiry_observed_text.load(std::sync::atomic::Ordering::Acquire) {
                late_delivery_flag.store(true, std::sync::atomic::Ordering::Release);
            }
            true
        },
    )
    .unwrap_err();
    observer.join().map_err(|_| "deadline observer panicked")?;
    if !deadline_error.contains("deadline expired")
        || !deadline_error.contains("worker_terminated=true")
        || !deadline_error.contains("reserved=0")
        || !deadline_error.contains("outstanding=0")
        || deadline_error.contains("source_bytes=0")
        || late_delivery.load(std::sync::atomic::Ordering::Acquire)
    {
        return Err(format!("deadline delivery/cleanup contract failed: {deadline_error}").into());
    }
    println!("deadline expired; partial_output_bytes={partial_bytes}; late_delivery=false; cleanup={deadline_error}");
    let mut recovery_text = Vec::new();
    let mut recovery_logits = Vec::new();
    let recovery = generate_streamed_with_lookahead(
        &model,
        "Once upon a time",
        config,
        &cancelled,
        StreamedDeadlineControl {
            control: StreamedControl::default(),
            deadline: Instant::now() + Duration::from_secs(300),
        },
        StreamedLookaheadControl { limits },
        |piece| {
            recovery_text.extend_from_slice(piece);
            true
        },
        |logits| {
            recovery_logits.extend_from_slice(logits);
            true
        },
    )?;
    if recovery_text != sync_text
        || recovery_logits != sync_logits
        || recovery.lookahead_scheduled_reads == 0
        || recovery.lookahead_consumed_reads == 0
        || !recovery.lookahead_worker_terminated
        || recovery.lookahead_final_reserved_bytes != 0
        || recovery.lookahead_final_outstanding_reads != 0
    {
        return Err("successful recovery inference/cleanup failed".into());
    }
    println!("post-expiry recovery inference: PASS; scheduled={} consumed={} worker_terminated={} reserved={} outstanding={}",
        recovery.lookahead_scheduled_reads, recovery.lookahead_consumed_reads,
        recovery.lookahead_worker_terminated, recovery.lookahead_final_reserved_bytes,
        recovery.lookahead_final_outstanding_reads);
    // Interleaved repeats share the same file, prompt, context, token budget,
    // native build and OS cache policy. We report rather than infer speedup.
    let mut benchmark = Vec::<String>::new();
    // Four pairs balance which path runs first (two lookahead-first, two
    // synchronous-first) while keeping each pair adjacent in time.
    for trial in 0..4 {
        for lookahead_enabled in [trial % 2 == 0, trial % 2 != 0] {
            let mut text = Vec::new();
            let mut logits = Vec::new();
            let started = Instant::now();
            let metrics = if lookahead_enabled {
                generate_streamed_with_lookahead(
                    &model,
                    "Once upon a time",
                    config,
                    &cancelled,
                    StreamedDeadlineControl {
                        control: StreamedControl::default(),
                        deadline: Instant::now() + Duration::from_secs(300),
                    },
                    StreamedLookaheadControl { limits },
                    |piece| {
                        text.extend_from_slice(piece);
                        true
                    },
                    |values| {
                        logits.extend_from_slice(values);
                        true
                    },
                )?
            } else {
                generate_streamed(
                    &model,
                    "Once upon a time",
                    config,
                    &cancelled,
                    |piece| {
                        text.extend_from_slice(piece);
                        true
                    },
                    |values| {
                        logits.extend_from_slice(values);
                        true
                    },
                )?
            };
            if text != sync_text || logits != sync_logits {
                return Err("benchmark output/logit parity failed".into());
            }
            let performance = metrics.performance.ok_or("performance metrics missing")?;
            let first_token_ms = performance
                .first_piece_ns
                .map_or_else(|| "null".into(), |n| (n as f64 / 1e6).to_string());
            let decode_rate = performance
                .decode_steps_per_second()
                .map_or_else(|| "null".into(), |n| n.to_string());
            benchmark.push(format!(
                r#"{{"trial":{},"mode":"{}","wall_ns":{},"first_token_ms":{},"decode_steps_per_second":{},"range_read_calls":{},"source_bytes_read":{},"lookahead_scheduled_reads":{},"lookahead_consumed_reads":{},"lookahead_source_bytes_read":{},"lookahead_queue_wait_ns":{},"lookahead_worker_read_ns":{},"lookahead_staging_copy_ns":{},"lookahead_consumer_wait_ns":{},"lookahead_revalidation_ns":{},"lookahead_peak_staging_capacity_bytes":{},"time_outside_read_and_consumer_callbacks_ns":{},"lookahead_total_admission_limit_bytes":{},"fixed_overhead_allowance_bytes":{},"worker_stack_allowance_bytes":{},"worker_terminated":{},"final_reserved_bytes":{},"final_outstanding_reads":{}}}"#,
                trial + 1,
                if lookahead_enabled { "lookahead" } else { "synchronous" },
                started.elapsed().as_nanos(),
                first_token_ms,
                decode_rate,
                metrics.range_read_calls,
                metrics.source_bytes_read,
                metrics.lookahead_scheduled_reads,
                metrics.lookahead_consumed_reads,
                metrics.lookahead_source_bytes_read,
                metrics.lookahead_queue_wait_ns,
                metrics.lookahead_worker_read_ns,
                metrics.lookahead_staging_copy_ns,
                metrics.lookahead_consumer_wait_ns,
                metrics.lookahead_revalidation_ns,
                metrics.lookahead_peak_staging_capacity_bytes,
                performance.total_ns
                    .saturating_sub(performance.range_callback_ns)
                    .saturating_sub(performance.consumer_callback_ns),
                if lookahead_enabled { limits.staging_bytes } else { 0 },
                if lookahead_enabled { limits.fixed_overhead_bytes } else { 0 },
                if lookahead_enabled { limits.worker_stack_bytes } else { 0 },
                metrics.lookahead_worker_terminated,
                metrics.lookahead_final_reserved_bytes,
                metrics.lookahead_final_outstanding_reads
            ));
        }
    }
    let benchmark_report = format!(
        "{{\"model_sha256\":\"{}\",\"model_bytes\":{},\"filesystem\":\"Windows NTFS local file; OS cache state uncontrolled, repeated trials progressively warm\",\"integrity\":\"unsealed local source; same policy for synchronous and lookahead\",\"prompt\":\"Once upon a time\",\"context_tokens\":{},\"max_tokens\":{},\"chunk_bytes\":{},\"workspace_bytes\":{},\"deadline_protocol_seconds\":{},\"lookahead_limits\":{{\"total_admission_limit_bytes\":{},\"max_outstanding_reads\":{},\"fixed_overhead_allowance_bytes\":{},\"worker_stack_allowance_bytes\":{}}},\"build\":\"cargo release; native MSVC; pinned llama.cpp cd959bae99ff4bed22bef2604f45241c09bbb8c8; CPU profile from build configuration\",\"trials_interleaved\":[{}]}}",
        MODEL_SHA256,
        fs::metadata(&model)?.len(),
        config.context_tokens,
        config.max_tokens,
        config.chunk_bytes,
        config.workspace_bytes,
        expiry_seconds,
        limits.staging_bytes,
        limits.max_outstanding_reads,
        limits.fixed_overhead_bytes,
        limits.worker_stack_bytes,
        benchmark.join(",")
    );
    fs::write(out.join("ntfs_benchmark.json"), benchmark_report)?;
    println!(
        "NTFS benchmark: 4 balanced interleaved synchronous/lookahead pairs recorded to {}",
        out.join("ntfs_benchmark.json").display()
    );
    Ok(())
}
