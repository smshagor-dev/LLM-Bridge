//! A two-stage native Q8_0 tensor composition backed by indexed GGUF ranges.
//!
//! Each projection uses the native ggml Q8_0 dequantizer through the existing
//! bounded Rust reader. This is a small graph execution slice: it is not
//! transformer prefill/decode, a process-RSS limit, or logit parity evidence.

use crate::gguf_index::{GgufTensorIndex};
use crate::inference::q8_0_matvec_with_budget;
use crate::storage::BoundedLocalReader;
use std::sync::atomic::{AtomicBool, Ordering};

fn q8_shape(index: &GgufTensorIndex, name: &str) -> Result<(usize, usize), String> {
    let tensor = index.tensor(name).ok_or_else(|| format!("missing Q8_0 tensor: {name}"))?;
    if tensor.ggml_type != 8 || tensor.dimensions.len() != 2 {
        return Err(format!("{name} must be a rank-2 Q8_0 GGUF tensor"));
    }
    let columns = usize::try_from(tensor.dimensions[0])
        .map_err(|_| format!("{name} columns exceed the address space"))?;
    let rows = usize::try_from(tensor.dimensions[1])
        .map_err(|_| format!("{name} rows exceed the address space"))?;
    if columns == 0 || columns > 16_384 || columns % 32 != 0 || rows == 0 || rows > 65_536 {
        return Err(format!("{name} dimensions exceed native Q8_0 operator limits"));
    }
    let row_bytes = (columns / 32)
        .checked_mul(34)
        .ok_or_else(|| format!("{name} row-byte arithmetic overflow"))?;
    let tensor_bytes = row_bytes
        .checked_mul(rows)
        .ok_or_else(|| format!("{name} tensor-byte arithmetic overflow"))?;
    if u64::try_from(tensor_bytes).ok() != Some(tensor.byte_len) {
        return Err(format!("{name} indexed byte length disagrees with Q8_0 shape"));
    }
    Ok((columns, rows))
}

/// Execute second * (first * input) using two actual native ggml Q8_0 operators.
///
/// Both weights are indexed in the same GGUF and read through bounded callbacks,
/// rather than being materialized as matrices. A failed second stage returns
/// no output and drops the first activation. Cancellation is checked between
/// the two stages and inside each native range callback.
///
/// The graph budget conservatively reserves one native operator workspace plus
/// the first stage's retained activation. It bounds only accounted application
/// buffers, not the allocator, OS page cache, native model graph, or process RSS.
pub fn q8_0_two_stage_with_budget(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    first_name: &str,
    second_name: &str,
    input: &[f32],
    cancelled: &AtomicBool,
    operator_budget_bytes: usize,
    graph_budget_bytes: usize,
) -> Result<Vec<f32>, String> {
    if cancelled.load(Ordering::Acquire) {
        return Err("Q8_0 graph cancelled before admission".into());
    }
    if reader.identity() != &index.source_identity {
        return Err("GGUF reader identity does not match graph index".into());
    }
    let (first_columns, first_rows) = q8_shape(index, first_name)?;
    let (second_columns, _second_rows) = q8_shape(index, second_name)?;
    if first_name == second_name {
        return Err("a two-stage graph requires distinct tensor names".into());
    }
    if input.len() != first_columns || first_rows != second_columns
        || input.iter().any(|value| !value.is_finite())
    {
        return Err("incompatible Q8_0 graph shapes or non-finite input".into());
    }
    let first_activation_bytes = first_rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("first activation size overflow")?;
    let needed = operator_budget_bytes
        .checked_add(first_activation_bytes)
        .ok_or("graph workspace budget overflow")?;
    if needed > graph_budget_bytes {
        return Err("insufficient Q8_0 graph budget for retained activation".into());
    }
    // These checks precede the first tensor read or any activation allocation.
    if reader.allocated_chunk_bytes() < 34
        || reader.allocated_chunk_bytes() > operator_budget_bytes
    {
        return Err("invalid Q8_0 reader chunk budget".into());
    }

    let intermediate = q8_0_matvec_with_budget(
        index, reader, first_name, input, cancelled, operator_budget_bytes,
    )?;
    if cancelled.load(Ordering::Acquire) {
        return Err("Q8_0 graph cancelled after first stage".into());
    }
    // The retained intermediate was accounted above. The native call returns
    // an owned output only on success; no partially computed output escapes.
    let final_output = q8_0_matvec_with_budget(
        index, reader, second_name, &intermediate, cancelled, operator_budget_bytes,
    )?;
    if cancelled.load(Ordering::Acquire) {
        return Err("Q8_0 graph cancelled before output commit".into());
    }
    Ok(final_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    const OP_BUDGET: usize = 128 * 1024;
    const GRAPH_BUDGET: usize = OP_BUDGET + 32 * 4;

    fn fixture(second_columns: u64, corrupt_second: bool) -> std::path::PathBuf {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        for (name, columns, rows, offset) in [
            ("first", 32u64, 32u64, 0u64),
            ("second", second_columns, 1u64, 32 * 34),
        ] {
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&2u32.to_le_bytes());
            bytes.extend_from_slice(&columns.to_le_bytes());
            bytes.extend_from_slice(&rows.to_le_bytes());
            bytes.extend_from_slice(&8u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        // First: each row picks input[0] with scale 1.0 (F16 = 0x3c00).
        for _ in 0..32 {
            bytes.extend_from_slice(&[0x00, 0x3c]);
            bytes.push(1);
            bytes.extend_from_slice(&[0u8; 31]);
        }
        // Second: sum all intermediate activations; optionally corrupt the
        // scale to a F16 NaN, which native finite-output checks reject.
        for block in 0..(second_columns / 32) {
            bytes.extend_from_slice(if corrupt_second && block == 0 {
                &[0x00, 0x7e]
            } else {
                &[0x00, 0x3c]
            });
            bytes.extend_from_slice(&[1u8; 32]);
        }
        let path = std::env::temp_dir().join(format!(
            "llmbridge-q8-chain-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        ));
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
        path
    }

    #[test]
    fn composes_two_real_native_q8_tensors_without_loading_matrices() {
        let path = fixture(32, false);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 64, 128).unwrap();
        let output = q8_0_two_stage_with_budget(
            &index, &mut reader, "first", "second", &[2f32; 32],
            &AtomicBool::new(false), OP_BUDGET, GRAPH_BUDGET,
        ).unwrap();
        assert_eq!(output, vec![64f32]);
        assert_eq!(reader.disk_bytes_read(), 32 * 34 + 34);
        assert_eq!(reader.allocated_chunk_bytes(), 64);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mismatched_graph_dimensions_fail_before_any_tensor_io() {
        let path = fixture(64, false);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 64, 128).unwrap();
        let err = q8_0_two_stage_with_budget(
            &index, &mut reader, "first", "second", &[2f32; 32],
            &AtomicBool::new(false), OP_BUDGET, GRAPH_BUDGET,
        ).unwrap_err();
        assert!(err.contains("shape"));
        assert_eq!(reader.disk_bytes_read(), 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn budget_and_cancellation_reject_without_tensor_io() {
        let path = fixture(32, false);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 64, 128).unwrap();
        let err = q8_0_two_stage_with_budget(
            &index, &mut reader, "first", "second", &[2f32; 32],
            &AtomicBool::new(false), OP_BUDGET, GRAPH_BUDGET - 1,
        ).unwrap_err();
        assert!(err.contains("graph budget"));
        assert_eq!(reader.disk_bytes_read(), 0);
        let cancelled = AtomicBool::new(true);
        let err = q8_0_two_stage_with_budget(
            &index, &mut reader, "first", "second", &[2f32; 32],
            &cancelled, OP_BUDGET, GRAPH_BUDGET,
        ).unwrap_err();
        assert!(err.contains("cancelled"));
        assert_eq!(reader.disk_bytes_read(), 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn failed_second_native_stage_never_returns_partial_graph_output() {
        let path = fixture(32, true);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 64, 128).unwrap();
        let err = q8_0_two_stage_with_budget(
            &index, &mut reader, "first", "second", &[2f32; 32],
            &AtomicBool::new(false), OP_BUDGET, GRAPH_BUDGET,
        ).unwrap_err();
        assert!(err.contains("non-finite") || err.contains("failed"));
        assert_eq!(reader.disk_bytes_read(), 32 * 34 + 34);
        std::fs::remove_file(path).unwrap();
    }
}
