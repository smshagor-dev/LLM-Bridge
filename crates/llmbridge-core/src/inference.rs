use crate::gguf_index::GgufTensorIndex;
use crate::storage::BoundedLocalReader;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};

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

    fn llmbridge_q8_0_matvec(
        tensor_offset: u64,
        tensor_bytes: u64,
        columns: u32,
        rows: u32,
        max_chunk_bytes: usize,
        callback: RangeCallback,
        userdata: *mut c_void,
        input: *const f32,
        output: *mut f32,
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
    cancelled: Option<&'a AtomicBool>,
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
        if state
            .cancelled
            .is_some_and(|flag| flag.load(Ordering::Acquire))
        {
            state.failure = Some("tensor operator cancelled".to_string());
            return 0;
        }
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
    if reader.identity() != &index.source_identity {
        return Err("source identity changed between GGUF indexing and native probe".to_string());
    }
    let mut state = ReaderState {
        reader,
        failure: None,
        cancelled: None,
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

/// Bounded Q8_0 matrix-vector operator using real GGUF tensor weights and
/// pinned native ggml dequantization. The native operator reads one row at a
/// time through the Rust range callback; it never materializes the matrix.
///
/// This is **one CPU operator**, not a transformer graph or full out-of-core
/// text generation. The reader buffer and native row scratch are bounded, but
/// OS page cache and process resident memory are not capped here.
pub fn q8_0_matvec(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    tensor_name: &str,
    activation: &[f32],
    cancelled: &AtomicBool,
) -> Result<Vec<f32>, String> {
    q8_0_matvec_with_budget(
        index,
        reader,
        tensor_name,
        activation,
        cancelled,
        1024 * 1024,
    )
}

/// Caps explicitly managed tensor-operator data (source buffer, input,
/// output, native packed/dequantized row and reserved allocator overhead).
/// This is not an OS-level RSS/cgroup limit or a graph-wide budget.
pub fn q8_0_matvec_with_budget(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    tensor_name: &str,
    activation: &[f32],
    cancelled: &AtomicBool,
    workspace_budget_bytes: usize,
) -> Result<Vec<f32>, String> {
    if reader.identity() != &index.source_identity {
        return Err("indexed model source identity changed".to_string());
    }
    let region = index
        .tensor(tensor_name)
        .ok_or("named tensor not indexed")?;
    if region.ggml_type != 8 || region.dimensions.len() != 2 {
        return Err("operator requires a rank-2 Q8_0 GGUF tensor".to_string());
    }
    let columns = u32::try_from(region.dimensions[0]).map_err(|_| "too many tensor columns")?;
    let rows = u32::try_from(region.dimensions[1]).map_err(|_| "too many tensor rows")?;
    if columns == 0
        || columns > 16384
        || columns % 32 != 0
        || rows == 0
        || rows > 65536
        || activation.len() != columns as usize
        || !activation.iter().all(|f| f.is_finite())
    {
        return Err("invalid Q8_0 operator dimensions or activation".to_string());
    }
    let row_bytes = usize::try_from(region.byte_len / u64::from(rows))
        .map_err(|_| "tensor row does not fit address space")?;
    // Account for our explicitly managed buffers before making any I/O.
    // Actual process RSS includes metadata, C++ runtime, allocator, OS
    // page cache and other allocations and is NOT constrained by this check.
    let operand_bytes = (columns as usize)
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("activation allocation size overflow")?;
    let result_bytes = (rows as usize)
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("result allocation size overflow")?;
    let scratch_bytes = reader
        .allocated_chunk_bytes()
        .checked_add(row_bytes)
        .and_then(|sum| sum.checked_add(operand_bytes))
        .and_then(|sum| sum.checked_add(operand_bytes))
        .and_then(|sum| sum.checked_add(result_bytes))
        .and_then(|sum| sum.checked_add(64 * 1024))
        .ok_or("operator workspace arithmetic overflow")?;
    if scratch_bytes > workspace_budget_bytes
        || row_bytes > reader.allocated_chunk_bytes()
        || cancelled.load(Ordering::Acquire)
    {
        return Err("operator workspace exceeds budget or request cancelled".to_string());
    }
    let mut output = vec![0f32; rows as usize];
    let mut state = ReaderState {
        reader,
        failure: None,
        cancelled: Some(cancelled),
    };
    let mut diagnostic = [0 as c_char; 1024];
    // SAFETY: all borrowed buffers and callback state outlive this synchronous
    // C call. Native checks bounds before writes and contains all exceptions.
    // An error discards output instead of exposing partially written rows.
    let status = unsafe {
        llmbridge_q8_0_matvec(
            region.file_offset,
            region.byte_len,
            columns,
            rows,
            state.reader.allocated_chunk_bytes(),
            read_range_callback,
            (&mut state as *mut ReaderState<'_>).cast(),
            activation.as_ptr(),
            output.as_mut_ptr(),
            diagnostic.as_mut_ptr(),
            diagnostic.len(),
        )
    };
    if status != 0 || cancelled.load(Ordering::Acquire) {
        // SAFETY: native writes a terminated buffer when it reports an error.
        let native = unsafe { CStr::from_ptr(diagnostic.as_ptr()) }.to_string_lossy();
        return Err(format!(
            "bounded Q8_0 operator failed: {native}; source: {}",
            state.failure.as_deref().unwrap_or("no reader failure")
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_operator_computes_from_real_file_bytes_and_refuses_cancel() {
        use std::io::Write;
        use std::sync::atomic::AtomicU64;
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "llmbridge-q8-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        let mut gguf = b"GGUF".to_vec();
        gguf.extend_from_slice(&3u32.to_le_bytes()); // version
        gguf.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        gguf.extend_from_slice(&0u64.to_le_bytes()); // no metadata
        let name = "weights";
        gguf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        gguf.extend_from_slice(name.as_bytes());
        gguf.extend_from_slice(&2u32.to_le_bytes()); // rank two
        gguf.extend_from_slice(&32u64.to_le_bytes());
        gguf.extend_from_slice(&2u64.to_le_bytes());
        gguf.extend_from_slice(&8u32.to_le_bytes()); // ggml Q8_0
        gguf.extend_from_slice(&0u64.to_le_bytes());
        while gguf.len() % 32 != 0 {
            gguf.push(0);
        }
        // GGML Q8_0 block: little-endian F16 scale, then 32 signed bytes.
        gguf.extend_from_slice(&[0x00, 0x38]); // scale 0.5
        gguf.extend(1u8..=32u8);
        gguf.extend_from_slice(&[0x00, 0x38]);
        gguf.extend((1i8..=32).map(|v| (-v) as u8));
        file.write_all(&gguf).unwrap();
        drop(file);

        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 64, 128).unwrap();
        let cancel = AtomicBool::new(false);
        let actual = q8_0_matvec(&index, &mut reader, "weights", &[1f32; 32], &cancel).unwrap();
        assert_eq!(actual, vec![264.0, -264.0]);
        assert_eq!(reader.disk_bytes_read(), 68);
        assert!(
            q8_0_matvec_with_budget(&index, &mut reader, "weights", &[1f32; 32], &cancel, 64)
                .is_err()
        );
        assert_eq!(reader.disk_bytes_read(), 68);
        cancel.store(true, Ordering::Release);
        assert!(q8_0_matvec(&index, &mut reader, "weights", &[1f32; 32], &cancel).is_err());
        assert_eq!(reader.disk_bytes_read(), 68);
        cancel.store(false, Ordering::Release);
        assert!(q8_0_matvec(&index, &mut reader, "weights", &[f32::NAN; 32], &cancel).is_err());
        assert!(q8_0_matvec(&index, &mut reader, "weights", &[1f32; 31], &cancel).is_err());
        std::fs::remove_file(path).unwrap();
    }

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
