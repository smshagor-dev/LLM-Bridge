//! Bounded F32 GGUF matrix-vector primitive. This is not a full model graph.
use crate::gguf_index::GgufTensorIndex;
use crate::storage::BoundedLocalReader;
use std::io::{self, ErrorKind};
use std::sync::atomic::{AtomicBool, Ordering};

fn reject(kind: ErrorKind, message: &'static str) -> io::Error {
    io::Error::new(kind, message)
}

/// Computes y = W*x for a rank-2 GGUF F32 matrix (dimensions [columns, rows]).
/// Output is caller-owned; the only tensor staging memory is the reader's bounded
/// reusable chunk. No full weight matrix or output matrix is allocated.
///
/// Output is written incrementally and must be discarded on error/cancellation.
/// Accumulation is in f32 and is not a claim of bitwise parity with ggml SIMD.
pub fn f32_matvec_cancellable(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    tensor_name: &str,
    input: &[f32],
    output: &mut [f32],
    cancelled: &AtomicBool,
    workspace_budget_bytes: usize,
) -> io::Result<()> {
    let tensor = index
        .tensor(tensor_name)
        .ok_or_else(|| reject(ErrorKind::NotFound, "GGUF tensor does not exist"))?;
    if tensor.ggml_type != 0 || tensor.dimensions.len() != 2 {
        return Err(reject(
            ErrorKind::InvalidInput,
            "expected rank-2 F32 tensor",
        ));
    }
    let cols = usize::try_from(tensor.dimensions[0])
        .map_err(|_| reject(ErrorKind::InvalidInput, "column count exceeds usize"))?;
    let rows = usize::try_from(tensor.dimensions[1])
        .map_err(|_| reject(ErrorKind::InvalidInput, "row count exceeds usize"))?;
    let row_bytes = cols
        .checked_mul(4)
        .ok_or_else(|| reject(ErrorKind::InvalidInput, "row size overflow"))?;
    if input.len() != cols || output.len() != rows || input.iter().any(|x| !x.is_finite()) {
        return Err(reject(
            ErrorKind::InvalidInput,
            "activation or output shape invalid",
        ));
    }
    if reader.allocated_chunk_bytes() < 4 || reader.allocated_chunk_bytes() > workspace_budget_bytes
    {
        return Err(reject(
            ErrorKind::InvalidInput,
            "insufficient bounded staging budget",
        ));
    }
    if reader.identity() != &index.source_identity {
        return Err(reject(
            ErrorKind::InvalidData,
            "GGUF source identity mismatch",
        ));
    }
    if cancelled.load(Ordering::Acquire) {
        return Err(reject(ErrorKind::Interrupted, "matrix operation cancelled"));
    }
    let row_bytes_u64 = u64::try_from(row_bytes)
        .map_err(|_| reject(ErrorKind::InvalidInput, "row size exceeds u64"))?;
    let total = row_bytes_u64
        .checked_mul(tensor.dimensions[1])
        .ok_or_else(|| reject(ErrorKind::InvalidInput, "tensor length overflow"))?;
    if total != tensor.byte_len {
        return Err(reject(
            ErrorKind::InvalidData,
            "indexed F32 tensor length mismatch",
        ));
    }
    let chunk = reader.allocated_chunk_bytes() / 4 * 4;
    for (row, destination) in output.iter_mut().enumerate() {
        let row_start = tensor
            .file_offset
            .checked_add(
                (row as u64)
                    .checked_mul(row_bytes_u64)
                    .ok_or_else(|| reject(ErrorKind::InvalidData, "row offset overflow"))?,
            )
            .ok_or_else(|| reject(ErrorKind::InvalidData, "row offset overflow"))?;
        let mut consumed = 0usize;
        let mut sum = 0f32;
        while consumed < row_bytes {
            if cancelled.load(Ordering::Acquire) {
                return Err(reject(ErrorKind::Interrupted, "matrix operation cancelled"));
            }
            let count = (row_bytes - consumed).min(chunk);
            let offset = row_start
                .checked_add(consumed as u64)
                .ok_or_else(|| reject(ErrorKind::InvalidData, "chunk offset overflow"))?;
            let bytes = reader.read_range(offset, count)?;
            for (j, encoded) in bytes.chunks_exact(4).enumerate() {
                let value = f32::from_le_bytes(encoded.try_into().expect("four bytes"));
                if !value.is_finite() {
                    return Err(reject(ErrorKind::InvalidData, "non-finite F32 tensor weight"));
                }
                let column = consumed / 4 + j;
                sum += value * input[column];
            }
            consumed += count;
        }
        if cancelled.load(Ordering::Acquire) {
            return Err(reject(ErrorKind::Interrupted, "matrix operation cancelled"));
        }
        if !sum.is_finite() {
            return Err(reject(ErrorKind::InvalidData, "non-finite F32 matrix result"));
        }
        *destination = sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn nonfinite_weights_and_accumulation_fail_closed() {
        fn check(values: [f32; 2], input: f32) {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend_from_slice(&3u32.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.push(b'w');
            bytes.extend_from_slice(&2u32.to_le_bytes());
            bytes.extend_from_slice(&2u64.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            while bytes.len() % 32 != 0 { bytes.push(0); }
            for value in values { bytes.extend_from_slice(&value.to_le_bytes()); }
            let path = std::env::temp_dir().join(format!(
                "llmbridge-f32-invalid-{}-{}", std::process::id(),
                NEXT.fetch_add(1, AtomicOrdering::Relaxed)
            ));
            File::create(&path).unwrap().write_all(&bytes).unwrap();
            let index = GgufTensorIndex::inspect(&path).unwrap();
            let mut reader = BoundedLocalReader::open(&path, 8, 16).unwrap();
            let mut output = [123.0f32];
            let error = f32_matvec_cancellable(
                &index, &mut reader, "w", &[input, input], &mut output,
                &AtomicBool::new(false), 8,
            ).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidData);
            assert_eq!(output, [123.0]);
            std::fs::remove_file(path).unwrap();
        }
        check([f32::NAN, 1.0], 1.0);
        check([f32::INFINITY, 1.0], 1.0);
        check([f32::MAX, f32::MAX], f32::MAX);
    }

    #[test]
    fn streamed_f32_matches_scalar_and_rejects_without_io() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        bytes.extend_from_slice(&0u64.to_le_bytes()); // no kv
        let name = b"matrix";
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes()); // columns
        bytes.extend_from_slice(&2u64.to_le_bytes()); // rows
        bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
        bytes.extend_from_slice(&0u64.to_le_bytes()); // relative offset
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        for value in [1f32, 2., 3., 4., 5., 6.] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let path = std::env::temp_dir().join(format!(
            "llmbridge-f32-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        ));
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 5, 10).unwrap();
        let cancel = AtomicBool::new(false);
        let mut out = [0f32; 2];
        assert!(f32_matvec_cancellable(
            &index,
            &mut reader,
            "matrix",
            &[1., 1., 1.],
            &mut out,
            &cancel,
            4
        )
        .is_err());
        assert_eq!(reader.disk_bytes_read(), 0);
        f32_matvec_cancellable(
            &index,
            &mut reader,
            "matrix",
            &[1., 2., 3.],
            &mut out,
            &cancel,
            5,
        )
        .unwrap();
        assert_eq!(out, [14., 32.]);
        assert_eq!(reader.disk_bytes_read(), 24);
        cancel.store(true, Ordering::Release);
        assert_eq!(
            f32_matvec_cancellable(
                &index,
                &mut reader,
                "matrix",
                &[1., 2., 3.],
                &mut out,
                &cancel,
                5
            )
            .unwrap_err()
            .kind(),
            ErrorKind::Interrupted
        );
        assert_eq!(reader.disk_bytes_read(), 24);
        std::fs::remove_file(path).unwrap();
    }
}
