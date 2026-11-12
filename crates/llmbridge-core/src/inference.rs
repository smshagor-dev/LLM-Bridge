use crate::gguf_index::GgufTensorIndex;
use crate::storage::BoundedLocalReader;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::slice;

type NativeCallback = extern "C" fn(*const c_char, usize, *mut c_void) -> i32;
type RangeCallback = extern "C" fn(*mut c_void, *mut c_void, u64, usize) -> usize;

extern "C" {
    fn llmbridge_abi_version() -> u32;
    fn llmbridge_probe_gguf_source(
        file_bytes: u64,
        max_chunk_bytes: usize,
        callback: RangeCallback,
        userdata: *mut c_void,
        tensor_count: *mut u64,
        tensor_data_offset: *mut u64,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
    fn llmbridge_generate_cpu(
        path: *const c_char,
        prompt: *const c_char,
        context_tokens: u32,
        max_tokens: u32,
        threads: c_int,
        callback: NativeCallback,
        user_data: *mut c_void,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}
extern "C" fn dispatch<F: FnMut(&[u8]) -> bool>(
    data: *const c_char,
    length: usize,
    user_data: *mut c_void,
) -> i32 {
    // A callback panic must never cross a C ABI; cancel instead.
    let invoked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if data.is_null() || user_data.is_null() || length > 1_048_576 {
            return false;
        }
        // SAFETY: native bridge promises data is valid for this synchronous callback,
        // user_data is the borrowed F for the duration of llmbridge_generate_cpu.
        let bytes = unsafe { slice::from_raw_parts(data.cast::<u8>(), length) };
        let callback = unsafe { &mut *(user_data.cast::<F>()) };
        callback(bytes)
    }));
    if matches!(invoked, Ok(true)) {
        1
    } else {
        0
    }
}

/// Real CPU-only llama.cpp GGUF generation; callback receives raw token bytes.
/// No hard resident-memory bound or trusted model verification is claimed for M1.
pub fn generate<F: FnMut(&[u8]) -> bool>(
    model: &Path,
    prompt: &str,
    context_tokens: u32,
    max_tokens: u32,
    threads: i32,
    mut on_piece: F,
) -> Result<(), String> {
    let model = model.to_str().ok_or("non-UTF8 model path not supported")?;
    let model = CString::new(model).map_err(|_| "invalid model path")?;
    let prompt = CString::new(prompt).map_err(|_| "prompt contains NUL byte")?;
    // SAFETY: ABI version is constant and the function has no input pointers.
    let abi = unsafe { llmbridge_abi_version() };
    if abi != 1 {
        return Err(format!("native ABI mismatch: expected 1, found {abi}"));
    }
    let mut error = [0 as c_char; 1024];
    // SAFETY: borrowed CStrings and closure remain live until call returns;
    // native writes NUL-terminated diagnostics into the bounded buffer.
    let status = unsafe {
        llmbridge_generate_cpu(
            model.as_ptr(),
            prompt.as_ptr(),
            context_tokens,
            max_tokens,
            threads,
            dispatch::<F>,
            (&mut on_piece as *mut F).cast(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    match status {
        0 => Ok(()),
        2 => Err("generation cancelled".into()),
        _ => {
            // SAFETY: the native wrapper ensures this buffer is NUL terminated.
            let msg = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            Err(format!("native CPU inference failed: {msg}"))
        }
    }
}

struct ReaderState<'a> {
    reader: &'a mut BoundedLocalReader,
    failure: Option<String>,
}

/// A panic must never cross the C ABI; a failed range fetch is reported as a
/// short read so the upstream parser rejects the file.
extern "C" fn read_range_callback(
    userdata: *mut c_void,
    output: *mut c_void,
    file_offset: u64,
    requested: usize,
) -> usize {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if userdata.is_null() || output.is_null() || requested == 0 {
            return 0;
        }
        // SAFETY: caller passes a live borrowed ReaderState for the synchronous
        // probe call only. The callback is not retained by C++.
        let state = unsafe { &mut *(userdata.cast::<ReaderState<'_>>()) };
        match state.reader.read_range(file_offset, requested) {
            Ok(chunk) => {
                // SAFETY: native callback contract guarantees an output region
                // of at least requested bytes, not overlapping the Rust buffer.
                unsafe {
                    std::ptr::copy_nonoverlapping(chunk.as_ptr(), output.cast::<u8>(), chunk.len());
                }
                chunk.len()
            }
            Err(e) => {
                state.failure = Some(e.to_string());
                0
            }
        }
    }));
    result.unwrap_or(0)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeGgufProbe {
    pub tensor_count: u64,
    pub data_offset: u64,
}

/// Real native ggml metadata probe driven entirely by the Rust bounded
/// source callback. This does NOT hook the weight reader into model execution.
pub fn probe_indexed_gguf(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
) -> Result<NativeGgufProbe, String> {
    if index.file_bytes != reader.file_bytes() {
        return Err("source size changed between GGUF indexing and native probe".to_string());
    }
    let mut state = ReaderState {
        reader,
        failure: None,
    };
    let mut count = 0u64;
    let mut offset = 0u64;
    let mut error = [0 as c_char; 1024];
    // SAFETY: synchronous call; ReaderState and output buffers live for the
    // entire call. Native validates pointers and traps C++ exceptions.
    let status = unsafe {
        llmbridge_probe_gguf_source(
            index.file_bytes,
            state.reader.allocated_chunk_bytes(),
            read_range_callback,
            (&mut state as *mut ReaderState<'_>).cast(),
            &mut count,
            &mut offset,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if status != 0 {
        // SAFETY: error buffer is guaranteed NUL-terminated by native wrapper.
        let native_error = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        return Err(format!(
            "bounded native GGUF probe failed: {}; source: {}",
            native_error,
            state.failure.as_deref().unwrap_or("no range error")
        ));
    }
    if count != index.tensors.len() as u64 || offset != index.data_offset {
        return Err(format!(
            "native/Rust tensor metadata mismatch: native {count} tensors @ {offset}, Rust {} tensors @ {}",
            index.tensors.len(),
            index.data_offset
        ));
    }
    Ok(NativeGgufProbe {
        tensor_count: count,
        data_offset: offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_abi_matches() {
        assert_eq!(unsafe { llmbridge_abi_version() }, 1);
    }
    #[test]
    fn malformed_model_is_rejected_without_generating_tokens() {
        let mut count = 0;
        let result = generate(
            Path::new("/llmbridge/nonexistent/does-not-exist.gguf"),
            "test",
            128,
            8,
            1,
            |_| {
                count += 1;
                true
            },
        );
        assert!(result.is_err());
        assert_eq!(count, 0);
    }
}
