//! Two-stage bounded F32 tensor execution with caller-owned activation buffers.
//! A small graph-composition checkpoint, not a transformer runtime or RSS cap.
use crate::f32_stream::f32_matvec_cancellable;
use crate::gguf_index::GgufTensorIndex;
use crate::storage::BoundedLocalReader;
use std::io::{self, ErrorKind};
use std::sync::atomic::{AtomicBool, Ordering};

/// Execute output = second * (first * input), with no tensor-sized allocation.
/// Both activation workspaces are caller-owned and must not alias the output.
/// The output is committed only if both stages succeed. The intermediate and
/// final scratch may contain partial results after an error and must be discarded.
#[allow(clippy::too_many_arguments)] // Stable caller-owned activation API; keep compatibility.
pub fn f32_two_stage_cancellable(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    first: &str,
    second: &str,
    input: &[f32],
    intermediate: &mut [f32],
    final_scratch: &mut [f32],
    output: &mut [f32],
    cancelled: &AtomicBool,
    staging_budget_bytes: usize,
) -> io::Result<()> {
    if output.len() != final_scratch.len() || input.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "invalid input activation or final scratch/output shape",
        ));
    }
    if cancelled.load(Ordering::Acquire) {
        return Err(io::Error::new(
            ErrorKind::Interrupted,
            "two-stage operation cancelled",
        ));
    }
    // Preflight both matrix shapes before reading either tensor.
    let first_tensor = index
        .tensor(first)
        .ok_or_else(|| io::Error::new(ErrorKind::NotFound, "first tensor missing"))?;
    let second_tensor = index
        .tensor(second)
        .ok_or_else(|| io::Error::new(ErrorKind::NotFound, "second tensor missing"))?;
    if first_tensor.ggml_type != 0
        || second_tensor.ggml_type != 0
        || first_tensor.dimensions.len() != 2
        || second_tensor.dimensions.len() != 2
        || first_tensor.dimensions[0] != input.len() as u64
        || first_tensor.dimensions[1] != intermediate.len() as u64
        || second_tensor.dimensions[0] != intermediate.len() as u64
        || second_tensor.dimensions[1] != output.len() as u64
    {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "incompatible F32 graph shapes",
        ));
    }
    // Validate shared reader admission before the first stage modifies any
    // intermediate buffer. Both stages use the same bounded staging allocation.
    if reader.identity() != &index.source_identity {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "GGUF source identity mismatch",
        ));
    }
    if reader.allocated_chunk_bytes() < 4 || reader.allocated_chunk_bytes() > staging_budget_bytes {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "insufficient chain staging budget",
        ));
    }
    f32_matvec_cancellable(
        index,
        reader,
        first,
        input,
        intermediate,
        cancelled,
        staging_budget_bytes,
    )?;
    f32_matvec_cancellable(
        index,
        reader,
        second,
        intermediate,
        final_scratch,
        cancelled,
        staging_budget_bytes,
    )?;
    if cancelled.load(Ordering::Acquire) {
        return Err(io::Error::new(
            ErrorKind::Interrupted,
            "two-stage operation cancelled",
        ));
    }
    output.copy_from_slice(final_scratch);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::manual_is_multiple_of)] // is_multiple_of requires newer Rust than MSRV.
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn fixture(second_value: f32) -> std::path::PathBuf {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        for (name, columns, rows, offset) in [("first", 2u64, 2u64, 0u64), ("second", 2, 1, 32)] {
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&2u32.to_le_bytes());
            bytes.extend_from_slice(&columns.to_le_bytes());
            bytes.extend_from_slice(&rows.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        for value in [1f32, 2., 3., 4.] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0u8; 16]);
        for value in [second_value, 2.] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let path = std::env::temp_dir().join(format!(
            "llmbridge-f32-chain-{}-{}",
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
    fn composes_two_real_indexed_matrices_without_loading_weights() {
        let path = fixture(1.);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut hidden = [0f32; 2];
        let mut scratch = [0f32; 1];
        let mut result = [-1f32; 1];
        f32_two_stage_cancellable(
            &index,
            &mut reader,
            "first",
            "second",
            &[2., 3.],
            &mut hidden,
            &mut scratch,
            &mut result,
            &AtomicBool::new(false),
            4,
        )
        .unwrap();
        assert_eq!(hidden, [8., 18.]);
        assert_eq!(result, [44.]);
        assert_eq!(reader.disk_bytes_read(), 24);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn preflight_rejects_bad_shapes_without_disk_io() {
        let path = fixture(1.);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut hidden = [0f32; 3];
        let mut scratch = [0f32; 1];
        let mut output = [77f32; 1];
        let err = f32_two_stage_cancellable(
            &index,
            &mut reader,
            "first",
            "second",
            &[2., 3.],
            &mut hidden,
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(output, [77.]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn nonfinite_input_is_rejected_before_either_stage_reads() {
        let path = fixture(1.);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut hidden = [55f32; 2];
        let mut scratch = [66f32; 1];
        let mut output = [77f32; 1];
        let err = f32_two_stage_cancellable(
            &index,
            &mut reader,
            "first",
            "second",
            &[f32::NAN, 3.],
            &mut hidden,
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(hidden, [55.; 2]);
        assert_eq!(scratch, [66.]);
        assert_eq!(output, [77.]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_second_stage_weights_leave_final_output_uncommitted() {
        let path = fixture(f32::NAN);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut hidden = [0f32; 2];
        let mut scratch = [99f32; 1];
        let mut output = [77f32; 1];
        let err = f32_two_stage_cancellable(
            &index,
            &mut reader,
            "first",
            "second",
            &[2., 3.],
            &mut hidden,
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert_eq!(hidden, [8., 18.]);
        assert_eq!(output, [77.]);
        assert_eq!(reader.disk_bytes_read(), 20);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn insufficient_shared_staging_budget_preserves_all_buffers_without_io() {
        let path = fixture(1.);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 8, 16).unwrap();
        let mut hidden = [55f32; 2];
        let mut scratch = [66f32; 1];
        let mut output = [77f32; 1];
        let err = f32_two_stage_cancellable(
            &index,
            &mut reader,
            "first",
            "second",
            &[2., 3.],
            &mut hidden,
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(hidden, [55.; 2]);
        assert_eq!(scratch, [66.]);
        assert_eq!(output, [77.]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn second_stage_failure_never_commits_partial_output() {
        let path = fixture(1.);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut hidden = [0f32; 2];
        let mut scratch = [0f32; 1];
        let mut output = [77f32; 1];
        let err = f32_two_stage_cancellable(
            &index,
            &mut reader,
            "first",
            "second",
            &[f32::MAX, f32::MAX],
            &mut hidden,
            &mut scratch,
            &mut output,
            &AtomicBool::new(false),
            4,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert_eq!(output, [77.]);
        std::fs::remove_file(path).unwrap();
    }
}
