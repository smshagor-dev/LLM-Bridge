//! Bounded F32 matrix-vector followed by a residual addition.
//! A graph operator building block, not a complete transformer inference graph.
use crate::f32_stream::f32_matvec_cancellable;
use crate::gguf_index::GgufTensorIndex;
use crate::storage::BoundedLocalReader;
use std::io::{self, ErrorKind};
use std::sync::atomic::{AtomicBool, Ordering};

/// Computes output = W * input + residual.
/// Scratch is caller-owned and may be modified on failure; output is only
/// committed after all reads, arithmetic and cancellation checks succeed.
/// This does not guarantee ggml floating-point/bitwise parity.
#[allow(clippy::too_many_arguments)] // Preserve the existing caller-owned buffer API.
pub fn f32_matvec_residual_cancellable(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    tensor_name: &str,
    input: &[f32],
    residual: &[f32],
    scratch: &mut [f32],
    output: &mut [f32],
    cancelled: &AtomicBool,
    staging_budget_bytes: usize,
) -> io::Result<()> {
    if residual.len() != output.len()
        || scratch.len() != output.len()
        || residual.iter().any(|v| !v.is_finite())
    {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "residual shape or values invalid",
        ));
    }
    // Validate the indexed output shape before any tensor I/O or scratch writes.
    // Otherwise the matrix primitive can read rows and partially overwrite scratch
    // before discovering a caller-supplied residual/output length mismatch.
    let tensor = index
        .tensor(tensor_name)
        .ok_or_else(|| io::Error::new(ErrorKind::NotFound, "GGUF tensor does not exist"))?;
    if tensor.ggml_type != 0 || tensor.dimensions.len() != 2 {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "expected rank-2 F32 projection",
        ));
    }
    let rows = usize::try_from(tensor.dimensions[1]).map_err(|_| {
        io::Error::new(
            ErrorKind::InvalidInput,
            "projection row count exceeds usize",
        )
    })?;
    if rows != output.len() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "projection and residual dimensions differ",
        ));
    }
    // Fail before touching scratch when the activation or bounded reader is
    // invalid; the underlying projection would otherwise discover this later.
    let columns = usize::try_from(tensor.dimensions[0]).map_err(|_| {
        io::Error::new(
            ErrorKind::InvalidInput,
            "projection column count exceeds usize",
        )
    })?;
    if input.len() != columns || input.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "projection input shape or values invalid",
        ));
    }
    if reader.identity() != &index.source_identity {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "GGUF source identity mismatch",
        ));
    }
    if reader.allocated_chunk_bytes() < 4 || reader.allocated_chunk_bytes() > staging_budget_bytes {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "insufficient residual staging budget",
        ));
    }
    if cancelled.load(Ordering::Acquire) {
        return Err(io::Error::new(
            ErrorKind::Interrupted,
            "residual operation cancelled",
        ));
    }
    f32_matvec_cancellable(
        index,
        reader,
        tensor_name,
        input,
        scratch,
        cancelled,
        staging_budget_bytes,
    )?;
    // Validate all arithmetic before writing the caller-visible output.
    for (value, bias) in scratch.iter().zip(residual.iter()) {
        if cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(
                ErrorKind::Interrupted,
                "residual operation cancelled",
            ));
        }
        if !value.is_finite() || !(*value + *bias).is_finite() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "non-finite residual result",
            ));
        }
    }
    if cancelled.load(Ordering::Acquire) {
        return Err(io::Error::new(
            ErrorKind::Interrupted,
            "residual operation cancelled",
        ));
    }
    for ((destination, value), bias) in output.iter_mut().zip(scratch.iter()).zip(residual.iter()) {
        *destination = *value + *bias;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn fixture(weights: [f32; 4]) -> std::path::PathBuf {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let name = b"projection";
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        for weight in weights {
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
        let path = std::env::temp_dir().join(format!(
            "llmbridge-residual-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        ));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        path
    }

    #[test]
    fn streams_projection_and_commits_exact_residual_values() {
        let path = fixture([1., 2., 3., 4.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut scratch = [0.; 2];
        let mut output = [-1.; 2];
        f32_matvec_residual_cancellable(
            &index,
            &mut reader,
            "projection",
            &[2., 3.],
            &[10., -2.],
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap();
        assert_eq!(output, [18., 16.]);
        assert_eq!(reader.disk_bytes_read(), 16);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_residual_fails_before_disk_io() {
        let path = fixture([1., 2., 3., 4.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut scratch = [0.; 2];
        let mut output = [77.; 2];
        let err = f32_matvec_residual_cancellable(
            &index,
            &mut reader,
            "projection",
            &[2., 3.],
            &[f32::NAN, 0.],
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(output, [77.; 2]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn indexed_output_shape_mismatch_fails_before_io_and_scratch_mutation() {
        let path = fixture([1., 2., 3., 4.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut scratch = [55.; 1];
        let mut output = [77.; 1];
        let err = f32_matvec_residual_cancellable(
            &index,
            &mut reader,
            "projection",
            &[2., 3.],
            &[0.],
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(scratch, [55.]);
        assert_eq!(output, [77.]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn missing_tensor_fails_before_io_and_scratch_mutation() {
        let path = fixture([1., 2., 3., 4.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut scratch = [55.; 2];
        let mut output = [77.; 2];
        let err = f32_matvec_residual_cancellable(
            &index,
            &mut reader,
            "not_present",
            &[2., 3.],
            &[0., 0.],
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(scratch, [55.; 2]);
        assert_eq!(output, [77.; 2]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn nonfinite_projection_input_rejects_without_io_or_scratch_mutation() {
        let path = fixture([1., 2., 3., 4.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut scratch = [55.; 2];
        let mut output = [77.; 2];
        let err = f32_matvec_residual_cancellable(
            &index,
            &mut reader,
            "projection",
            &[f32::NAN, 3.],
            &[0., 0.],
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(scratch, [55.; 2]);
        assert_eq!(output, [77.; 2]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn residual_staging_budget_rejects_before_projection_reads() {
        let path = fixture([1., 2., 3., 4.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 8, 16).unwrap();
        let mut scratch = [55.; 2];
        let mut output = [77.; 2];
        let err = f32_matvec_residual_cancellable(
            &index,
            &mut reader,
            "projection",
            &[2., 3.],
            &[0., 0.],
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(scratch, [55.; 2]);
        assert_eq!(output, [77.; 2]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn residual_overflow_does_not_commit_output() {
        let path = fixture([1., 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut scratch = [0.; 2];
        let mut output = [77.; 2];
        let err = f32_matvec_residual_cancellable(
            &index,
            &mut reader,
            "projection",
            &[f32::MAX, 1.],
            &[f32::MAX, 0.],
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert_eq!(output, [77.; 2]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn pre_cancelled_execution_never_reads_or_commits() {
        let path = fixture([1., 2., 3., 4.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut scratch = [0.; 2];
        let mut output = [77.; 2];
        let err = f32_matvec_residual_cancellable(
            &index,
            &mut reader,
            "projection",
            &[2., 3.],
            &[0., 0.],
            &mut scratch,
            &mut output,
            &AtomicBool::new(true),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Interrupted);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(output, [77.; 2]);
        std::fs::remove_file(path).unwrap();
    }
}
