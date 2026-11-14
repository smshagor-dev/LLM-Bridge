//! Experimental complete Llama F32/Q8_0/Q4_0 decoder backed by bounded local reads.
//! The explicit workspace budget is not an enforced process RSS limit.
use crate::{
    gguf_index::GgufTensorIndex,
    read_cache::{CachedLocalReader, ReadCacheMetrics},
    session_performance::{PerformanceClock, SessionPerformance},
    storage::BoundedLocalReader,
};
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

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
        }
    }
}
#[derive(Debug)]
pub struct StreamedMetrics {
    pub source_bytes_read: u64,
    pub reader_buffer_bytes: usize,
    pub cache: ReadCacheMetrics,
    pub authentication: Option<crate::storage::AuthenticationMetrics>,
    pub performance: Option<SessionPerformance>,
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
}
struct State<'a, F, L> {
    reader: CachedLocalReader,
    cancelled: &'a AtomicBool,
    emit: F,
    logits: L,
    failure: Option<String>,
    performance: Option<PerformanceClock>,
}
fn contained<T: Default>(call: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).unwrap_or_default()
}
extern "C" fn poll<F, L>(state: *mut c_void) -> i32 {
    contained(|| {
        if state.is_null() {
            return 0;
        }
        // SAFETY: native borrows this State synchronously, never stores its pointer.
        let state = unsafe { &*(state.cast::<State<'_, F, L>>()) };
        i32::from(!state.cancelled.load(Ordering::Acquire) && state.failure.is_none())
    })
}
extern "C" fn read<F, L>(
    state: *mut c_void,
    output: *mut c_void,
    offset: u64,
    length: usize,
) -> usize {
    contained(|| {
        if state.is_null() || output.is_null() || length == 0 {
            return 0;
        }
        // SAFETY: sole mutable synchronous callback; C++ supplies length writable bytes.
        let state = unsafe { &mut *(state.cast::<State<'_, F, L>>()) };
        let flag = state.cancelled;
        let start = state.performance.as_ref().map(|_| Instant::now());
        let copied = (|| match state
            .reader
            .read_range_cancellable(offset, length, || flag.load(Ordering::Acquire))
        {
            Ok(bytes) => {
                if flag.load(Ordering::Acquire) {
                    return 0;
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.cast(), bytes.len());
                }
                bytes.len()
            }
            Err(error) => {
                state.failure = Some(error.to_string());
                0
            }
        })();
        if let (Some(clock), Some(start)) = (&mut state.performance, start) {
            clock.range(start.elapsed());
        }
        copied
    })
}
extern "C" fn emit<F: FnMut(&[u8]) -> bool, L>(
    bytes: *const c_char,
    count: usize,
    state: *mut c_void,
) -> i32 {
    contained(|| {
        if state.is_null() || bytes.is_null() || count > 4096 {
            return 0;
        }
        let state = unsafe { &mut *(state.cast::<State<'_, F, L>>()) };
        if state.cancelled.load(Ordering::Acquire) || state.failure.is_some() {
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
        i32::from(accepted && !state.cancelled.load(Ordering::Acquire))
    })
}
extern "C" fn logits<F, L: FnMut(&[f32]) -> bool>(
    values: *const f32,
    count: usize,
    state: *mut c_void,
) -> i32 {
    contained(|| {
        if state.is_null() || values.is_null() || count == 0 || count > 262144 {
            return 0;
        }
        let state = unsafe { &mut *(state.cast::<State<'_, F, L>>()) };
        if state.cancelled.load(Ordering::Acquire) || state.failure.is_some() {
            return 0;
        }
        let values = unsafe { std::slice::from_raw_parts(values, count) };
        if let Some(clock) = state.performance.as_mut() {
            clock.logits();
        }
        let start = state.performance.as_ref().map(|_| Instant::now());
        let accepted = (state.logits)(values);
        if let (Some(clock), Some(start)) = (&mut state.performance, start) {
            clock.consumer(start.elapsed());
        }
        i32::from(accepted && !state.cancelled.load(Ordering::Acquire))
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
    generate_streamed_internal(model, prompt, config, cancelled, None, on_piece, on_logits)
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
        Some(expected_sha256),
        on_piece,
        on_logits,
    )
}
fn generate_streamed_internal<F: FnMut(&[u8]) -> bool, L: FnMut(&[f32]) -> bool>(
    model: &Path,
    prompt: &str,
    config: StreamedConfig,
    cancelled: &AtomicBool,
    expected_sha256: Option<&str>,
    on_piece: F,
    on_logits: L,
) -> Result<StreamedMetrics, String> {
    let performance = config.collect_performance.then(PerformanceClock::start);
    if cancelled.load(Ordering::Acquire) {
        return Err("streamed generation cancelled before admission".into());
    }
    if !(64..=8192).contains(&config.context_tokens)
        || !(1..=2048).contains(&config.max_tokens)
        || !(34..=1024 * 1024).contains(&config.chunk_bytes)
        || config.cache_bytes > 32 * 1024 * 1024
        || config.authentication_budget_bytes == 0
        || config.authentication_budget_bytes > 64 * 1024 * 1024
        || config.workspace_bytes == 0
        || config.workspace_bytes > 1024 * 1024 * 1024
        || prompt.is_empty()
        || prompt.len() > 65536
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
    let index = GgufTensorIndex::inspect(model).map_err(|e| e.to_string())?;
    let mut reader = BoundedLocalReader::open(model, config.chunk_bytes, config.chunk_bytes * 2)
        .map_err(|e| e.to_string())?;
    if reader.identity() != &index.source_identity {
        return Err("streamed model changed after index".into());
    }
    let sealed = expected_sha256
        .map(|hash| {
            reader.authenticate_model(
                hash,
                index.data_offset,
                config.authentication_budget_bytes,
                || cancelled.load(Ordering::Acquire),
            )
        })
        .transpose()
        .map_err(|e| e.to_string())?;
    let tokenizer_path = sealed.as_ref().map_or(model, |snapshot| snapshot.path());
    let path = CString::new(tokenizer_path.to_str().ok_or("model path must be UTF-8")?)
        .map_err(|_| "NUL in model path")?;
    // SAFETY: constant ABI query; all pointers below remain live for this synchronous call.
    if unsafe { llmbridge_abi_version() } != 1 {
        return Err("streamed native ABI mismatch".into());
    }
    let mut state = State {
        reader: CachedLocalReader::new(reader, config.cache_bytes).map_err(|e| e.to_string())?,
        cancelled,
        emit: on_piece,
        logits: on_logits,
        failure: None,
        performance,
    };
    let mut error = [0 as c_char; 1024];
    let status = unsafe {
        llmbridge_generate_streamed_cpu_v2(
            path.as_ptr(),
            prompt.as_ptr(),
            config.context_tokens,
            config.max_tokens,
            index.file_bytes,
            config.chunk_bytes,
            config.workspace_bytes,
            sealed.as_ref().map_or(0, |snapshot| snapshot.prefix_bytes),
            read::<F, L>,
            emit::<F, L>,
            logits::<F, L>,
            poll::<F, L>,
            (&mut state as *mut State<'_, F, L>).cast(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if status == 2 || cancelled.load(Ordering::Acquire) {
        return Err("streamed generation cancelled".into());
    }
    if status != 0 {
        let diagnostic = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        return Err(format!(
            "streamed generation failed: {diagnostic}; source: {}",
            state.failure.as_deref().unwrap_or("no read error")
        ));
    }
    Ok(StreamedMetrics {
        source_bytes_read: state.reader.source_bytes_read(),
        reader_buffer_bytes: config.chunk_bytes,
        cache: state.reader.metrics(),
        authentication: state.reader.authentication_metrics(),
        performance: state.performance.map(PerformanceClock::finish),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
