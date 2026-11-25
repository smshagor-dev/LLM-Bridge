//! Experimental complete Llama decoder backed by bounded local reads. Admits
//! F32, the 32-element quants Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, and the K-quant
//! superblocks Q4_K/Q5_K/Q6_K; all dequant/dot kernels are pinned ggml traits.
//! The explicit workspace budget is not an enforced process RSS limit.
use crate::{
    gguf_index::GgufTensorIndex,
    local_lookahead::LocalLookahead,
    read_cache::{CachedLocalReader, ReadCacheMetrics},
    session_performance::{PerformanceClock, SessionPerformance},
    storage::BoundedLocalReader,
};
use std::collections::VecDeque;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub use crate::local_lookahead::LookaheadLimits;

/// Maximum UTF-8 prompt bytes accepted by the streamed native wrapper.
pub const MAX_STREAMED_PROMPT_BYTES: usize = 65536;

/// One cooperative release signal per session; repeated requests coalesce.
/// This never grows a cache or bypasses source/integrity checks.
#[derive(Debug, Default)]
pub struct CacheControl {
    release: AtomicBool,
}
impl CacheControl {
    pub fn request_release(&self) {
        self.release.store(true, Ordering::Release);
    }
    fn take_release(&self) -> bool {
        self.release.swap(false, Ordering::AcqRel)
    }
}
/// Optional independently trusted digest and session-local cache signal.
/// No digest means explicitly unverified execution, exactly as generate_streamed.
#[derive(Clone, Copy, Default)]
pub struct StreamedControl<'a> {
    pub expected_sha256: Option<&'a str>,
    pub cache: Option<&'a CacheControl>,
}

/// Opt-in absolute monotonic deadline, observed cooperatively at safe boundaries.
/// Existing StreamedControl literals and untimed entry points remain unchanged.
#[derive(Clone, Copy)]
pub struct StreamedDeadlineControl<'a> {
    pub control: StreamedControl<'a>,
    pub deadline: Instant,
}
/// Explicit opt-in bounded local read lookahead. The existing entry points
/// remain synchronous. `fixed_overhead_bytes` must cover the worker stack,
/// source buffer, bounded channels and descriptors; allocator/kernel overhead
/// cannot be measured exactly by this reservation.
#[derive(Debug, Clone, Copy)]
pub struct StreamedLookaheadControl {
    pub limits: LookaheadLimits,
}
struct ExecutionControl<'a> {
    session: StreamedControl<'a>,
    deadline: Option<Instant>,
    lookahead: Option<LookaheadLimits>,
}
impl<'a> From<StreamedControl<'a>> for ExecutionControl<'a> {
    fn from(session: StreamedControl<'a>) -> Self {
        Self {
            session,
            deadline: None,
            lookahead: None,
        }
    }
}
pub(crate) fn deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
}
pub(crate) fn interrupted(cancelled: &AtomicBool, deadline: Option<Instant>) -> bool {
    cancelled.load(Ordering::Acquire) || deadline_expired(deadline)
}
fn ensure_active(cancelled: &AtomicBool, deadline: Option<Instant>) -> Result<(), String> {
    if cancelled.load(Ordering::Acquire) {
        Err("streamed generation cancelled".into())
    } else if deadline_expired(deadline) {
        Err("streamed generation deadline expired".into())
    } else {
        Ok(())
    }
}
fn interruption_reason(cancelled: &AtomicBool, deadline: Option<Instant>) -> &'static str {
    if cancelled.load(Ordering::Acquire) {
        "streamed generation cancelled"
    } else if deadline_expired(deadline) {
        "streamed generation deadline expired"
    } else {
        "streamed I/O interrupted"
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StreamedConfig {
    pub context_tokens: u32,
    pub max_tokens: u32,
    pub chunk_bytes: usize,
    pub workspace_bytes: u64,
    pub cache_bytes: usize,
    pub authentication_budget_bytes: usize,
    /// Adds constant-space wall-clock observations; disabled by default.
    pub collect_performance: bool,
    /// Opt into Mode B authenticated-snapshot range reuse (content-addressed
    /// manifest verification through the bounded cache). Off by default, so the
    /// decoder keeps the mutable-source, mutation-detecting read path. Has effect
    /// only on an authenticated source with a nonzero `cache_bytes` budget.
    pub snapshot_reuse: bool,
}
impl Default for StreamedConfig {
    fn default() -> Self {
        Self {
            context_tokens: 128,
            max_tokens: 16,
            chunk_bytes: 64 * 1024,
            workspace_bytes: 8 * 1024 * 1024,
            cache_bytes: 0,
            authentication_budget_bytes: 8 * 1024 * 1024,
            collect_performance: false,
            snapshot_reuse: false,
        }
    }
}
#[derive(Debug)]
pub struct StreamedMetrics {
    /// Native range callback attempts, including metadata and weight batches.
    pub range_read_calls: u64,
    pub range_bytes_requested: u64,
    pub weight_read_calls: u64,
    pub weight_bytes_requested: u64,
    pub max_range_read_bytes: usize,
    pub source_bytes_read: u64,
    pub reader_buffer_bytes: usize,
    pub cache: ReadCacheMetrics,
    pub authentication: Option<crate::storage::AuthenticationMetrics>,
    pub performance: Option<SessionPerformance>,
    pub lookahead_scheduled_reads: u64,
    pub lookahead_consumed_reads: u64,
    pub lookahead_worker_terminated: bool,
    pub lookahead_final_reserved_bytes: usize,
    pub lookahead_final_outstanding_reads: usize,
    pub lookahead_source_bytes_read: u64,
    pub lookahead_queue_wait_ns: u64,
    pub lookahead_worker_read_ns: u64,
    pub lookahead_staging_copy_ns: u64,
    pub lookahead_consumer_wait_ns: u64,
    pub lookahead_revalidation_ns: u64,
    pub lookahead_peak_staging_capacity_bytes: usize,
}

extern "C" {
    fn llmbridge_abi_version() -> u32;
    fn llmbridge_generate_streamed_cpu_v2(
        path: *const c_char,
        prompt: *const c_char,
        context: u32,
        max_tokens: u32,
        file_bytes: u64,
        max_chunk: usize,
        workspace: u64,
        tokenizer_prefix: u64,
        reader: extern "C" fn(*mut c_void, *mut c_void, u64, usize) -> usize,
        emit: extern "C" fn(*const c_char, usize, *mut c_void) -> i32,
        logits: extern "C" fn(*const f32, usize, *mut c_void) -> i32,
        poll: extern "C" fn(*mut c_void) -> i32,
        state: *mut c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn llmbridge_generate_streamed_cpu_v3(
        path: *const c_char,
        prompt: *const c_char,
        context: u32,
        max_tokens: u32,
        file_bytes: u64,
        max_chunk: usize,
        workspace: u64,
        tokenizer_prefix: u64,
        reader: extern "C" fn(*mut c_void, *mut c_void, u64, usize) -> usize,
        schedule: extern "C" fn(*mut c_void, u64, usize, u64, u64) -> i32,
        emit: extern "C" fn(*const c_char, usize, *mut c_void) -> i32,
        logits: extern "C" fn(*const f32, usize, *mut c_void) -> i32,
        poll: extern "C" fn(*mut c_void) -> i32,
        state: *mut c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn llmbridge_streamed_forced_logits_v1(
        path: *const c_char,
        prompt: *const c_char,
        forced_tokens: *const i32,
        forced_count: u32,
        context: u32,
        file_bytes: u64,
        max_chunk: usize,
        workspace: u64,
        tokenizer_prefix: u64,
        reader: extern "C" fn(*mut c_void, *mut c_void, u64, usize) -> usize,
        logits: extern "C" fn(*const f32, usize, *mut c_void) -> i32,
        poll: extern "C" fn(*mut c_void) -> i32,
        state: *mut c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
}

/// Native operation selector for `generate_streamed_internal`. `Generate` runs
/// greedy generation (the proven path); `ForcedLogits` feeds caller-supplied
/// continuation tokens and emits per-position logits only (teacher forcing).
enum StreamedOp<'a> {
    Generate,
    ForcedLogits(&'a [i32]),
}
#[derive(Debug, Default)]
struct RangeReads {
    calls: u64,
    bytes: u64,
    maximum: usize,
    weight_calls: u64,
    weight_bytes: u64,
}
impl RangeReads {
    fn record(&mut self, bytes: usize, weight: bool) {
        self.calls = self.calls.saturating_add(1);
        self.bytes = self.bytes.saturating_add(bytes as u64);
        self.maximum = self.maximum.max(bytes);
        if weight {
            self.weight_calls = self.weight_calls.saturating_add(1);
            self.weight_bytes = self.weight_bytes.saturating_add(bytes as u64);
        }
    }
}
struct State<'a, F, L> {
    ranges: RangeReads,
    sealed: Option<&'a crate::storage::SealedMetadata>,
    weight_offset: u64,
    max_chunk: usize,
    reader: CachedLocalReader,
    cancelled: &'a AtomicBool,
    deadline: Option<Instant>,
    cache_control: Option<&'a CacheControl>,
    emit: F,
    logits: L,
    failure: Option<String>,
    performance: Option<PerformanceClock>,
    lookahead: Option<LocalLookahead>,
    scheduled: VecDeque<(u64, usize)>,
    lookahead_scheduled: u64,
    lookahead_consumed: u64,
    lookahead_revalidation_ns: u64,
}
fn apply_cache_control<F, L>(state: &mut State<'_, F, L>) {
    if state.cache_control.is_some_and(CacheControl::take_release) {
        state.reader.release_cache();
    }
}
fn contained<T: Default>(call: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).unwrap_or_default()
}
fn contained_session<F, L, T: Default>(
    pointer: *mut c_void,
    message: &'static str,
    call: impl FnOnce() -> T,
) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)) {
        Ok(value) => value,
        Err(_) => {
            if !pointer.is_null() {
                // The session remains borrowed by the synchronous native call.
                let state = unsafe { &mut *(pointer.cast::<State<'_, F, L>>()) };
                state.failure.get_or_insert_with(|| message.to_owned());
            }
            T::default()
        }
    }
}
extern "C" fn poll<F, L>(state: *mut c_void) -> i32 {
    contained(|| {
        if state.is_null() {
            return 0;
        }
        // SAFETY: native borrows this State synchronously, never stores its pointer.
        let state = unsafe { &*(state.cast::<State<'_, F, L>>()) };
        i32::from(!interrupted(state.cancelled, state.deadline) && state.failure.is_none())
    })
}
extern "C" fn read<F, L>(
    state: *mut c_void,
    output: *mut c_void,
    offset: u64,
    length: usize,
) -> usize {
    contained_session::<F, L, usize>(state, "panic in streamed range callback", || {
        if state.is_null() || output.is_null() || length == 0 {
            return 0;
        }
        // SAFETY: sole mutable synchronous callback; C++ supplies length writable bytes.
        let state = unsafe { &mut *(state.cast::<State<'_, F, L>>()) };
        if interrupted(state.cancelled, state.deadline) || state.failure.is_some() {
            return 0;
        }
        let start = state.performance.as_ref().map(|_| Instant::now());
        if length > state.max_chunk {
            state.failure = Some("native read exceeds configured chunk".into());
            return 0;
        }
        state.ranges.record(length, offset >= state.weight_offset);
        // Native owns a copy of previous row bytes; no cache borrow survives
        // a synchronous callback, so releasing here cannot invalidate native data.
        apply_cache_control(state);
        let flag = state.cancelled;
        let deadline = state.deadline;
        let copied = (|| {
            if let Some(lookahead) = state.lookahead.as_ref() {
                if state
                    .scheduled
                    .front()
                    .is_some_and(|&(at, bytes)| at == offset && bytes == length)
                {
                    let (expected_offset, expected_length) = state.scheduled.pop_front().unwrap();
                    let staged = lookahead
                        .take_exact(expected_offset, expected_length, || {
                            interrupted(flag, deadline)
                        })
                        .inspect_err(|error| {
                            state.failure.get_or_insert_with(|| {
                                if error.kind() == std::io::ErrorKind::Interrupted {
                                    interruption_reason(flag, deadline).to_owned()
                                } else {
                                    error.to_string()
                                }
                            });
                        })?;
                    if interrupted(flag, deadline) {
                        state.failure.get_or_insert_with(|| {
                            "lookahead delivery cancelled or deadline expired".into()
                        });
                        return Ok::<usize, std::io::Error>(0);
                    }
                    let verification_started = Instant::now();
                    let verification =
                        state
                            .reader
                            .verify_staged_cancellable(offset, staged.bytes(), || {
                                interrupted(flag, deadline)
                            });
                    state.lookahead_revalidation_ns =
                        state.lookahead_revalidation_ns.saturating_add(
                            verification_started
                                .elapsed()
                                .as_nanos()
                                .min(u128::from(u64::MAX)) as u64,
                        );
                    verification?;
                    if interrupted(flag, deadline) {
                        state.failure.get_or_insert_with(|| {
                            "lookahead delivery cancelled or deadline expired".into()
                        });
                        return Ok(0);
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            staged.bytes().as_ptr(),
                            output.cast(),
                            length,
                        );
                    }
                    state.lookahead_consumed = state.lookahead_consumed.saturating_add(1);
                    return Ok(length);
                }
            }
            // Authenticated metadata comes from the already sealed trusted prefix.
            // This avoids rehashing a source block for every small GGUF field.
            if let Some(metadata) = state.sealed.filter(|m| offset < m.prefix_bytes) {
                if interrupted(flag, deadline) {
                    return Ok(0);
                }
                // SAFETY: the parser supplies a writable bounded destination.
                return match unsafe { metadata.read_into_native(offset, output.cast(), length) } {
                    Ok(()) if !interrupted(flag, deadline) => Ok(length),
                    Ok(()) => Ok(0),
                    Err(error) => Err(error),
                };
            }
            match state
                .reader
                .read_range_cancellable(offset, length, || interrupted(flag, deadline))
            {
                Ok(bytes) => {
                    if interrupted(flag, deadline) {
                        return Ok(0);
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.cast(), bytes.len());
                    }
                    Ok(bytes.len())
                }
                Err(error) => Err(error),
            }
        })();
        if let (Some(clock), Some(start)) = (&mut state.performance, start) {
            clock.range(start.elapsed());
        }
        match copied {
            Ok(count) => count,
            Err(error) => {
                state.failure.get_or_insert_with(|| {
                    if error.kind() == std::io::ErrorKind::Interrupted {
                        interruption_reason(state.cancelled, state.deadline).to_owned()
                    } else {
                        error.to_string()
                    }
                });
                0
            }
        }
    })
}
extern "C" fn schedule<F, L>(
    state: *mut c_void,
    offset: u64,
    length: usize,
    tensor: u64,
    bytes: u64,
) -> i32 {
    contained_session::<F, L, i32>(state, "panic in lookahead scheduling callback", || {
        if state.is_null() {
            return 0;
        }
        let state = unsafe { &mut *(state.cast::<State<'_, F, L>>()) };
        if interrupted(state.cancelled, state.deadline) || state.failure.is_some() {
            return 0;
        }
        let Some(lookahead) = state.lookahead.as_ref() else {
            return 0;
        };
        match lookahead.schedule(offset, length, tensor, bytes, state.deadline) {
            Ok(()) => {
                state.scheduled.push_back((offset, length));
                state.lookahead_scheduled = state.lookahead_scheduled.saturating_add(1);
                1
            }
            Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => 0,
            Err(error) => {
                state.failure.get_or_insert_with(|| error.to_string());
                0
            }
        }
    })
}
extern "C" fn emit<F: FnMut(&[u8]) -> bool, L>(
    bytes: *const c_char,
    count: usize,
    state: *mut c_void,
) -> i32 {
    contained_session::<F, L, i32>(state, "panic in streamed token consumer callback", || {
        if state.is_null() || bytes.is_null() || count > 4096 {
            return 0;
        }
        let state = unsafe { &mut *(state.cast::<State<'_, F, L>>()) };
        if interrupted(state.cancelled, state.deadline) || state.failure.is_some() {
            return 0;
        }
        let piece = unsafe { std::slice::from_raw_parts(bytes.cast(), count) };
        if let Some(clock) = state.performance.as_mut() {
            clock.piece(count);
        }
        let start = state.performance.as_ref().map(|_| Instant::now());
        let accepted = (state.emit)(piece);
        if let (Some(clock), Some(start)) = (&mut state.performance, start) {
            clock.consumer(start.elapsed());
        }
        i32::from(accepted && !interrupted(state.cancelled, state.deadline))
    })
}
extern "C" fn logits<F, L: FnMut(&[f32]) -> bool>(
    values: *const f32,
    count: usize,
    state: *mut c_void,
) -> i32 {
    contained_session::<F, L, i32>(state, "panic in streamed logits consumer callback", || {
        if state.is_null() || values.is_null() || count == 0 || count > 262144 {
            return 0;
        }
        let state = unsafe { &mut *(state.cast::<State<'_, F, L>>()) };
        if interrupted(state.cancelled, state.deadline) || state.failure.is_some() {
            return 0;
        }
        apply_cache_control(state);
        let values = unsafe { std::slice::from_raw_parts(values, count) };
        if let Some(clock) = state.performance.as_mut() {
            clock.logits();
        }
        let start = state.performance.as_ref().map(|_| Instant::now());
        let accepted = (state.logits)(values);
        if let (Some(clock), Some(start)) = (&mut state.performance, start) {
            clock.consumer(start.elapsed());
        }
        apply_cache_control(state);
        i32::from(accepted && !interrupted(state.cancelled, state.deadline))
    })
}

/// Runs every decoder layer. Optional caching reads bounded adjacent blocks ahead.
/// Tokenizer metadata uses upstream vocabulary-only loading. No mmap weight loader is used.
/// Unknown architecture options/tensor layouts fail; logits are exposed before greedy sampling.
/// Callbacks may see prior tokens before a later read fails, as with other streaming APIs.
pub fn generate_streamed<F: FnMut(&[u8]) -> bool, L: FnMut(&[f32]) -> bool>(
    model: &Path,
    prompt: &str,
    config: StreamedConfig,
    cancelled: &AtomicBool,
    on_piece: F,
    on_logits: L,
) -> Result<StreamedMetrics, String> {
    generate_streamed_internal(
        model,
        prompt,
        config,
        cancelled,
        StreamedControl::default().into(),
        StreamedOp::Generate,
        on_piece,
        on_logits,
    )
}

/// Teacher-forced streamed logits: prefill `prompt`, then feed the fixed
/// `forced_tokens` continuation, invoking `on_logits` once at the final prompt
/// position and once after each fed token (`forced_tokens.len() + 1` calls).
/// Samples nothing, so each emitted position shares a prefix with an
/// independently decoded reference at the same index. Synchronous; no lookahead.
pub fn generate_streamed_forced_logits<L: FnMut(&[f32]) -> bool>(
    model: &Path,
    prompt: &str,
    config: StreamedConfig,
    cancelled: &AtomicBool,
    forced_tokens: &[i32],
    on_logits: L,
) -> Result<StreamedMetrics, String> {
    generate_streamed_internal(
        model,
        prompt,
        config,
        cancelled,
        StreamedControl::default().into(),
        StreamedOp::ForcedLogits(forced_tokens),
        |_: &[u8]| true,
        on_logits,
    )
}

/// Opt-in authenticated generation: expected whole-file hash must be independently trusted.
/// Sealed tokenizer metadata and per-block verification prevent a hash/load source race.
pub fn generate_streamed_verified<F: FnMut(&[u8]) -> bool, L: FnMut(&[f32]) -> bool>(
    model: &Path,
    prompt: &str,
    config: StreamedConfig,
    cancelled: &AtomicBool,
    expected_sha256: &str,
    on_piece: F,
    on_logits: L,
) -> Result<StreamedMetrics, String> {
    generate_streamed_internal(
        model,
        prompt,
        config,
        cancelled,
        StreamedControl {
            expected_sha256: Some(expected_sha256),
            cache: None,
        }
        .into(),
        StreamedOp::Generate,
        on_piece,
        on_logits,
    )
}
/// Advanced synchronous session API. Release signals are honored at safe bounded-read
/// and logits boundaries; all borrowed control references end with this call.
/// Use one CacheControl per session. Cache release is monotonic and does not
/// release KV/activation memory or the authenticated source manifest.
pub fn generate_streamed_controlled<F: FnMut(&[u8]) -> bool, L: FnMut(&[f32]) -> bool>(
    model: &Path,
    prompt: &str,
    config: StreamedConfig,
    cancelled: &AtomicBool,
    control: StreamedControl<'_>,
    on_piece: F,
    on_logits: L,
) -> Result<StreamedMetrics, String> {
    generate_streamed_internal(
        model,
        prompt,
        config,
        cancelled,
        control.into(),
        StreamedOp::Generate,
        on_piece,
        on_logits,
    )
}

/// Cooperatively bound this synchronous session, including authentication and
/// native execution. In-flight I/O/native work/consumers cannot be preempted.
/// Deadline expiry never writes the caller's cancellation flag.
pub fn generate_streamed_with_deadline<F: FnMut(&[u8]) -> bool, L: FnMut(&[f32]) -> bool>(
    model: &Path,
    prompt: &str,
    config: StreamedConfig,
    cancelled: &AtomicBool,
    control: StreamedDeadlineControl<'_>,
    on_piece: F,
    on_logits: L,
) -> Result<StreamedMetrics, String> {
    generate_streamed_internal(
        model,
        prompt,
        config,
        cancelled,
        ExecutionControl {
            session: control.control,
            deadline: Some(control.deadline),
            lookahead: None,
        },
        StreamedOp::Generate,
        on_piece,
        on_logits,
    )
}
/// Runs an authenticated or ordinary streamed session with explicit bounded
/// lookahead. Without calling this function, execution is always synchronous.
#[allow(clippy::too_many_arguments)]
pub fn generate_streamed_with_lookahead<F: FnMut(&[u8]) -> bool, L: FnMut(&[f32]) -> bool>(
    model: &Path,
    prompt: &str,
    config: StreamedConfig,
    cancelled: &AtomicBool,
    deadline: StreamedDeadlineControl<'_>,
    lookahead: StreamedLookaheadControl,
    on_piece: F,
    on_logits: L,
) -> Result<StreamedMetrics, String> {
    generate_streamed_internal(
        model,
        prompt,
        config,
        cancelled,
        ExecutionControl {
            session: deadline.control,
            deadline: Some(deadline.deadline),
            lookahead: Some(lookahead.limits),
        },
        StreamedOp::Generate,
        on_piece,
        on_logits,
    )
}
#[allow(clippy::too_many_arguments)]
fn generate_streamed_internal<F: FnMut(&[u8]) -> bool, L: FnMut(&[f32]) -> bool>(
    model: &Path,
    prompt: &str,
    config: StreamedConfig,
    cancelled: &AtomicBool,
    execution: ExecutionControl<'_>,
    op: StreamedOp<'_>,
    on_piece: F,
    on_logits: L,
) -> Result<StreamedMetrics, String> {
    let ExecutionControl {
        session: control,
        deadline,
        lookahead: lookahead_limits,
    } = execution;
    let expected_sha256 = control.expected_sha256;
    let performance = config.collect_performance.then(PerformanceClock::start);
    if cancelled.load(Ordering::Acquire) {
        return Err("streamed generation cancelled before admission".into());
    }
    ensure_active(cancelled, deadline)?;
    if !(64..=8192).contains(&config.context_tokens)
        || !(1..=2048).contains(&config.max_tokens)
        || !(34..=1024 * 1024).contains(&config.chunk_bytes)
        || config.cache_bytes > 32 * 1024 * 1024
        || config.authentication_budget_bytes == 0
        || config.authentication_budget_bytes > 64 * 1024 * 1024
        || config.workspace_bytes == 0
        || config.workspace_bytes > 1024 * 1024 * 1024
        || prompt.is_empty()
        || prompt.len() > MAX_STREAMED_PROMPT_BYTES
    {
        return Err("invalid streamed generation configuration".into());
    }
    if expected_sha256
        .is_some_and(|hash| hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return Err("expected SHA-256 must be 64 hex digits".into());
    }
    crate::native_cpu::ensure_compatible()?;
    let prompt = CString::new(prompt).map_err(|_| "NUL in prompt")?;
    ensure_active(cancelled, deadline)?;
    let index = GgufTensorIndex::inspect_cancellable(model, || interrupted(cancelled, deadline));
    ensure_active(cancelled, deadline)?;
    let GgufTensorIndex {
        file_bytes,
        data_offset,
        source_identity,
        ..
    } = index.map_err(|e| e.to_string())?;
    let mut reader = BoundedLocalReader::open(model, config.chunk_bytes, config.chunk_bytes * 2)
        .map_err(|e| e.to_string())?;
    ensure_active(cancelled, deadline)?;
    if reader.identity() != &source_identity {
        return Err("streamed model changed after index".into());
    }
    let sealed = expected_sha256
        .map(|hash| {
            reader.authenticate_model(
                hash,
                data_offset,
                config.authentication_budget_bytes,
                || interrupted(cancelled, deadline),
            )
        })
        .transpose();
    ensure_active(cancelled, deadline)?;
    let sealed = sealed.map_err(|e| e.to_string())?;
    let lookahead = if let Some(limits) = lookahead_limits {
        let worker_reader =
            BoundedLocalReader::open(model, config.chunk_bytes, config.chunk_bytes * 2)
                .map_err(|e| e.to_string())?;
        if worker_reader.identity() != &source_identity {
            return Err("streamed model changed before lookahead worker start".into());
        }
        Some(LocalLookahead::start(worker_reader, limits).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let tokenizer_path = sealed.as_ref().map_or(model, |snapshot| snapshot.path());
    let path = CString::new(tokenizer_path.to_str().ok_or("model path must be UTF-8")?)
        .map_err(|_| "NUL in model path")?;
    // SAFETY: constant ABI query; all pointers below remain live for this synchronous call.
    if unsafe { llmbridge_abi_version() } != 1 {
        return Err("streamed native ABI mismatch".into());
    }
    let mut cached_reader =
        CachedLocalReader::new(reader, config.cache_bytes).map_err(|e| e.to_string())?;
    if config.snapshot_reuse {
        // Opt-in Mode B: authenticated range reads reuse manifest-verified bytes
        // through the bounded cache. No effect on an unauthenticated source or a
        // zero cache budget. The default path is untouched.
        cached_reader.set_read_policy(crate::read_cache::ReadPolicy::AuthenticatedSnapshot);
    }
    let mut state = State {
        ranges: RangeReads::default(),
        sealed: sealed.as_ref(),
        weight_offset: data_offset,
        max_chunk: config.chunk_bytes,
        reader: cached_reader,
        cancelled,
        deadline,
        cache_control: control.cache,
        emit: on_piece,
        logits: on_logits,
        failure: None,
        performance,
        lookahead,
        scheduled: VecDeque::new(),
        lookahead_scheduled: 0,
        lookahead_consumed: 0,
        lookahead_revalidation_ns: 0,
    };
    let mut error = [0 as c_char; 1024];
    let state_ptr = (&mut state as *mut State<'_, F, L>).cast();
    let status = unsafe {
        match op {
            StreamedOp::ForcedLogits(forced) => llmbridge_streamed_forced_logits_v1(
                path.as_ptr(),
                prompt.as_ptr(),
                forced.as_ptr(),
                forced.len() as u32,
                config.context_tokens,
                file_bytes,
                config.chunk_bytes,
                config.workspace_bytes,
                sealed.as_ref().map_or(0, |snapshot| snapshot.prefix_bytes),
                read::<F, L>,
                logits::<F, L>,
                poll::<F, L>,
                state_ptr,
                error.as_mut_ptr(),
                error.len(),
            ),
            StreamedOp::Generate if state.lookahead.is_some() => {
                llmbridge_generate_streamed_cpu_v3(
                    path.as_ptr(),
                    prompt.as_ptr(),
                    config.context_tokens,
                    config.max_tokens,
                    file_bytes,
                    config.chunk_bytes,
                    config.workspace_bytes,
                    sealed.as_ref().map_or(0, |snapshot| snapshot.prefix_bytes),
                    read::<F, L>,
                    schedule::<F, L>,
                    emit::<F, L>,
                    logits::<F, L>,
                    poll::<F, L>,
                    state_ptr,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
            StreamedOp::Generate => llmbridge_generate_streamed_cpu_v2(
                path.as_ptr(),
                prompt.as_ptr(),
                config.context_tokens,
                config.max_tokens,
                file_bytes,
                config.chunk_bytes,
                config.workspace_bytes,
                sealed.as_ref().map_or(0, |snapshot| snapshot.prefix_bytes),
                read::<F, L>,
                emit::<F, L>,
                logits::<F, L>,
                poll::<F, L>,
                state_ptr,
                error.as_mut_ptr(),
                error.len(),
            ),
        }
    };
    let lookahead_finish = state.lookahead.as_mut().map(LocalLookahead::shutdown);
    let cleanup = lookahead_finish
        .as_ref()
        .map(|finish| {
            format!(
                "; lookahead worker_terminated={} reserved={} outstanding={} source_bytes={} worker_panic={}",
                finish.worker_terminated,
                finish.reserved_bytes,
                finish.outstanding_reads,
                finish.source_bytes_read,
                finish.worker_panic.as_deref().unwrap_or("none")
            )
        })
        .unwrap_or_default();
    if let Some(failure) = state.failure.as_ref() {
        if failure == "lookahead worker stopped" {
            if let Some(panic) = lookahead_finish
                .as_ref()
                .and_then(|finish| finish.worker_panic.as_ref())
            {
                return Err(format!("lookahead worker panicked: {panic}{cleanup}"));
            }
        }
        return Err(format!("streamed generation failed: {failure}{cleanup}"));
    }
    if let Some(panic) = lookahead_finish
        .as_ref()
        .and_then(|finish| finish.worker_panic.as_ref())
    {
        return Err(format!("lookahead worker panicked: {panic}{cleanup}"));
    }
    if let Err(error) = ensure_active(cancelled, deadline) {
        return Err(format!("{error}{cleanup}"));
    }
    if status == 2 {
        return Err(format!("streamed generation cancelled{cleanup}"));
    }
    if status != 0 {
        let diagnostic = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        return Err(format!(
            "streamed generation failed: {diagnostic}; source: {}{cleanup}",
            state.failure.as_deref().unwrap_or("no read error")
        ));
    }
    Ok(StreamedMetrics {
        range_read_calls: state.ranges.calls,
        range_bytes_requested: state.ranges.bytes,
        weight_read_calls: state.ranges.weight_calls,
        weight_bytes_requested: state.ranges.weight_bytes,
        max_range_read_bytes: state.ranges.maximum,
        source_bytes_read: state.reader.source_bytes_read(),
        reader_buffer_bytes: config.chunk_bytes,
        cache: state.reader.metrics(),
        authentication: state.reader.authentication_metrics(),
        performance: state.performance.map(PerformanceClock::finish),
        lookahead_scheduled_reads: state.lookahead_scheduled,
        lookahead_consumed_reads: state.lookahead_consumed,
        lookahead_worker_terminated: lookahead_finish
            .as_ref()
            .is_none_or(|f| f.worker_terminated),
        lookahead_final_reserved_bytes: lookahead_finish.as_ref().map_or(0, |f| f.reserved_bytes),
        lookahead_final_outstanding_reads: lookahead_finish
            .as_ref()
            .map_or(0, |f| f.outstanding_reads),
        lookahead_source_bytes_read: lookahead_finish.as_ref().map_or(0, |f| f.source_bytes_read),
        lookahead_queue_wait_ns: lookahead_finish.as_ref().map_or(0, |f| f.queue_wait_ns),
        lookahead_worker_read_ns: lookahead_finish.as_ref().map_or(0, |f| f.worker_read_ns),
        lookahead_staging_copy_ns: lookahead_finish.as_ref().map_or(0, |f| f.staging_copy_ns),
        lookahead_consumer_wait_ns: lookahead_finish.as_ref().map_or(0, |f| f.consumer_wait_ns),
        lookahead_revalidation_ns: state.lookahead_revalidation_ns,
        lookahead_peak_staging_capacity_bytes: lookahead_finish
            .as_ref()
            .map_or(0, |f| f.peak_actual_staging_bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expired_deadline_refuses_before_open_and_preserves_caller_flag() {
        let flag = AtomicBool::new(false);
        let control = StreamedDeadlineControl {
            control: StreamedControl::default(),
            deadline: Instant::now(),
        };
        let error = generate_streamed_with_deadline(
            Path::new("/absent-deadline.gguf"),
            "fixture",
            StreamedConfig::default(),
            &flag,
            control,
            |_| panic!("piece"),
            |_| panic!("logits"),
        )
        .unwrap_err();
        assert!(error.contains("deadline expired"));
        assert!(!flag.load(Ordering::Acquire));
        flag.store(true, Ordering::Release);
        let error = generate_streamed_with_deadline(
            Path::new("/absent-deadline.gguf"),
            "fixture",
            StreamedConfig::default(),
            &flag,
            control,
            |_| panic!("piece"),
            |_| panic!("logits"),
        )
        .unwrap_err();
        assert!(error.contains("cancelled before admission"));
    }
    #[test]
    fn expired_callbacks_never_read_or_deliver_native_data() {
        static DELIVERED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        fn no_piece(_: &[u8]) -> bool {
            DELIVERED.fetch_add(1, Ordering::Relaxed);
            true
        }
        fn no_logits(_: &[f32]) -> bool {
            DELIVERED.fetch_add(1, Ordering::Relaxed);
            true
        }
        DELIVERED.store(0, Ordering::Relaxed);
        type Piece = fn(&[u8]) -> bool;
        type Logits = fn(&[f32]) -> bool;
        let path = std::env::temp_dir().join(format!(
            "llmbridge-deadline-callback-{}",
            std::process::id()
        ));
        std::fs::write(&path, [1u8; 64]).unwrap();
        let reader = BoundedLocalReader::open(&path, 64, 128).unwrap();
        let flag = AtomicBool::new(false);
        let mut state = State {
            ranges: RangeReads::default(),
            sealed: None,
            weight_offset: 0,
            max_chunk: 64,
            reader: CachedLocalReader::new(reader, 0).unwrap(),
            cancelled: &flag,
            deadline: Some(Instant::now()),
            cache_control: None,
            emit: no_piece as Piece,
            logits: no_logits as Logits,
            failure: None,
            performance: None,
            lookahead: None,
            scheduled: VecDeque::new(),
            lookahead_scheduled: 0,
            lookahead_consumed: 0,
            lookahead_revalidation_ns: 0,
        };
        let pointer = (&mut state as *mut State<'_, Piece, Logits>).cast();
        let mut output = [0xa5u8; 16];
        assert_eq!(poll::<Piece, Logits>(pointer), 0);
        assert_eq!(
            read::<Piece, Logits>(pointer, output.as_mut_ptr().cast(), 0, 16),
            0
        );
        assert_eq!(emit::<Piece, Logits>(output.as_ptr().cast(), 1, pointer), 0);
        assert_eq!(logits::<Piece, Logits>([0.0f32].as_ptr(), 1, pointer), 0);
        assert_eq!(output, [0xa5; 16]);
        assert_eq!(state.ranges.calls, 0);
        assert_eq!(state.reader.metrics().source_read_calls, 0);
        assert!(!flag.load(Ordering::Acquire));
        assert_eq!(DELIVERED.load(Ordering::Relaxed), 0);
        state.deadline = None;
        let pointer = (&mut state as *mut State<'_, Piece, Logits>).cast();
        assert_eq!(poll::<Piece, Logits>(pointer), 1);
        assert_eq!(emit::<Piece, Logits>(output.as_ptr().cast(), 1, pointer), 1);
        assert_eq!(logits::<Piece, Logits>([0.0f32].as_ptr(), 1, pointer), 1);
        assert_eq!(DELIVERED.load(Ordering::Relaxed), 2);
        assert_eq!(
            read::<Piece, Logits>(pointer, output.as_mut_ptr().cast(), 0, 16),
            16
        );
        assert_eq!(output, [1; 16]);
        drop(state);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn range_metrics_count_calls_bytes_and_maximum_without_overflow() {
        let mut ranges = RangeReads::default();
        ranges.record(34, false);
        ranges.record(65536, true);
        assert_eq!(
            (ranges.calls, ranges.bytes, ranges.maximum),
            (2, 65570, 65536)
        );
        assert_eq!((ranges.weight_calls, ranges.weight_bytes), (1, 65536));
        ranges.calls = u64::MAX;
        ranges.bytes = u64::MAX;
        ranges.record(1, false);
        assert_eq!(
            (ranges.calls, ranges.bytes, ranges.maximum),
            (u64::MAX, u64::MAX, 65536)
        );
    }
    #[test]
    fn cache_release_requests_are_one_shot_and_coalesced() {
        let control = CacheControl::default();
        assert!(!control.take_release());
        control.request_release();
        control.request_release();
        assert!(control.take_release());
        assert!(!control.take_release());
    }
    #[test]
    fn cache_signal_can_be_sent_from_another_thread() {
        let control = CacheControl::default();
        std::thread::scope(|scope| {
            scope.spawn(|| control.request_release()).join().unwrap();
        });
        assert!(control.take_release());
    }
    #[test]
    fn precancelled_streaming_never_opens_model_or_delivers_output() {
        let result = generate_streamed(
            Path::new("/absent.gguf"),
            "Hello",
            StreamedConfig::default(),
            &AtomicBool::new(true),
            |_| panic!("token delivered"),
            |_| panic!("logits delivered"),
        );
        assert!(result.unwrap_err().contains("cancelled before admission"));
    }
    #[test]
    fn invalid_budget_or_context_fails_before_model_open() {
        for config in [
            StreamedConfig {
                workspace_bytes: 0,
                ..StreamedConfig::default()
            },
            StreamedConfig {
                context_tokens: 1,
                ..StreamedConfig::default()
            },
        ] {
            assert!(generate_streamed(
                Path::new("/absent.gguf"),
                "Hello",
                config,
                &AtomicBool::new(false),
                |_| true,
                |_| true
            )
            .unwrap_err()
            .contains("configuration"));
        }
    }
}
