use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::slice;

type NativeCallback = extern "C" fn(*const c_char, usize, *mut c_void) -> i32;

extern "C" {
    fn llmbridge_abi_version() -> u32;
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
    if matches!(invoked, Ok(true)) { 1 } else { 0 }
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
        let result = generate(Path::new("/llmbridge/nonexistent/does-not-exist.gguf"), "test", 128, 8, 1, |_| {
            count += 1;
            true
        });
        assert!(result.is_err());
        assert_eq!(count, 0);
    }
}
