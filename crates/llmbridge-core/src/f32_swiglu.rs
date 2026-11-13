//! Bounded F32 SwiGLU feed-forward operator over three indexed GGUF tensors.
//! This is a real multi-tensor computation, not full transformer inference,
//! exact ggml parity, a native graph connection or a process RSS guarantee.

use crate::f32_stream::f32_matvec_cancellable;
use crate::gguf_index::GgufTensorIndex;
use crate::storage::BoundedLocalReader;
use std::io::{self, ErrorKind};
use std::sync::atomic::{AtomicBool, Ordering};

fn rejected(kind: ErrorKind, message: &'static str) -> io::Error {
    io::Error::new(kind, message)
}

fn f32_shape(index: &GgufTensorIndex, name: &str) -> io::Result<(usize, usize)> {
    let tensor = index
        .tensor(name)
        .ok_or_else(|| rejected(ErrorKind::NotFound, "feed-forward tensor missing"))?;
    if tensor.ggml_type != 0 || tensor.dimensions.len() != 2 {
        return Err(rejected(ErrorKind::InvalidInput, "expected rank-2 F32 feed-forward tensor"));
    }
    let cols = usize::try_from(tensor.dimensions[0])
        .map_err(|_| rejected(ErrorKind::InvalidInput, "F32 tensor column count overflow"))?;
    let rows = usize::try_from(tensor.dimensions[1])
        .map_err(|_| rejected(ErrorKind::InvalidInput, "F32 tensor row count overflow"))?;
    Ok((cols, rows))
}

/// Compute down * (SiLU(gate * input) elementwise-multiplied by (up * input)).
///
/// Gate, up, and final_scratch are caller-owned, independent activation
/// buffers. The caller-visible output is only committed after all three
/// indexed F32 weight projections and nonlinear arithmetic have succeeded.
/// Workspaces may contain partial data after a failure and must be discarded.
///
/// Staging budget bounds the reusable reader chunk. Activation budget charges
/// all supplied input, intermediate, scratch and output slices; it does not
/// reserve physical memory, account for unrelated allocations, or bound RSS.
/// No weight matrix is resident in these workspaces. Arithmetic uses f32 and
/// does not claim numerical parity with ggml SIMD or other quantized kernels.
#[allow(clippy::too_many_arguments)]
pub fn f32_swiglu_cancellable(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    gate_tensor: &str,
    up_tensor: &str,
    down_tensor: &str,
    input: &[f32],
    gate: &mut [f32],
    up: &mut [f32],
    final_scratch: &mut [f32],
    output: &mut [f32],
    cancelled: &AtomicBool,
    staging_budget_bytes: usize,
    activation_budget_bytes: usize,
) -> io::Result<()> {
    f32_swiglu_internal(
        index, reader, gate_tensor, up_tensor, down_tensor, input, None,
        gate, up, final_scratch, output, cancelled,
        staging_budget_bytes, activation_budget_bytes,
    )
}

/// Compute a full gated feed-forward residual: output = input_residual + down(SwiGLU).
/// This version validates the residual before any tensor I/O and includes its
/// slice in the caller-owned activation budget. The output stays untouched on
/// late non-finite arithmetic, tensor errors, or cancellation.
#[allow(clippy::too_many_arguments)]
pub fn f32_swiglu_residual_cancellable(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    gate_tensor: &str,
    up_tensor: &str,
    down_tensor: &str,
    input: &[f32],
    residual: &[f32],
    gate: &mut [f32],
    up: &mut [f32],
    final_scratch: &mut [f32],
    output: &mut [f32],
    cancelled: &AtomicBool,
    staging_budget_bytes: usize,
    activation_budget_bytes: usize,
) -> io::Result<()> {
    f32_swiglu_internal(
        index, reader, gate_tensor, up_tensor, down_tensor, input,
        Some(residual), gate, up, final_scratch, output, cancelled,
        staging_budget_bytes, activation_budget_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
fn f32_swiglu_internal(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    gate_tensor: &str,
    up_tensor: &str,
    down_tensor: &str,
    input: &[f32],
    residual: Option<&[f32]>,
    gate: &mut [f32],
    up: &mut [f32],
    final_scratch: &mut [f32],
    output: &mut [f32],
    cancelled: &AtomicBool,
    staging_budget_bytes: usize,
    activation_budget_bytes: usize,
) -> io::Result<()> {
    // Validate all three tensor shapes, input values and budgets before
    // executing any projection or modifying caller-owned activation buffers.
    let gate_shape = f32_shape(index, gate_tensor)?;
    let up_shape = f32_shape(index, up_tensor)?;
    let down_shape = f32_shape(index, down_tensor)?;
    if input.is_empty()
        || gate.is_empty()
        || output.is_empty()
        || gate.len() != up.len()
        || final_scratch.len() != output.len()
        || gate_shape != (input.len(), gate.len())
        || up_shape != gate_shape
        || down_shape != (gate.len(), output.len())
        || input.iter().any(|value| !value.is_finite())
    {
        return Err(rejected(ErrorKind::InvalidInput, "incompatible SwiGLU shapes or inputs"));
    }
    if let Some(values) = residual {
        if values.len() != output.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(rejected(ErrorKind::InvalidInput, "invalid SwiGLU residual values or shape"));
        }
    }
    let elements = [
        input.len(), gate.len(), up.len(), final_scratch.len(),
        output.len(), residual.map_or(0, <[f32]>::len),
    ]
        .into_iter()
        .try_fold(0usize, |sum, len| sum.checked_add(len))
        .ok_or_else(|| rejected(ErrorKind::InvalidInput, "activation size overflow"))?;
    let required = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| rejected(ErrorKind::InvalidInput, "activation budget overflow"))?;
    if required > activation_budget_bytes {
        return Err(rejected(ErrorKind::InvalidInput, "SwiGLU activation budget exceeded"));
    }
    if reader.identity() != &index.source_identity {
        return Err(rejected(ErrorKind::InvalidData, "GGUF source identity mismatch"));
    }
    if reader.allocated_chunk_bytes() < 4
        || reader.allocated_chunk_bytes() > staging_budget_bytes
    {
        return Err(rejected(ErrorKind::InvalidInput, "insufficient bounded reader budget"));
    }
    if cancelled.load(Ordering::Acquire) {
        return Err(rejected(ErrorKind::Interrupted, "SwiGLU cancelled"));
    }

    f32_matvec_cancellable(
        index, reader, gate_tensor, input, gate, cancelled, staging_budget_bytes,
    )?;
    f32_matvec_cancellable(
        index, reader, up_tensor, input, up, cancelled, staging_budget_bytes,
    )?;

    // Reuse gate in place as the gated activation. No vector allocation.
    for (gate_value, up_value) in gate.iter_mut().zip(up.iter()) {
        if cancelled.load(Ordering::Acquire) {
            return Err(rejected(ErrorKind::Interrupted, "SwiGLU cancelled"));
        }
        let sigmoid = 1.0f32 / (1.0f32 + (-*gate_value).exp());
        let gated = (*gate_value * sigmoid) * *up_value;
        if !gated.is_finite() {
            return Err(rejected(ErrorKind::InvalidData, "non-finite SwiGLU activation"));
        }
        *gate_value = gated;
    }
    if cancelled.load(Ordering::Acquire) {
        return Err(rejected(ErrorKind::Interrupted, "SwiGLU cancelled"));
    }
    f32_matvec_cancellable(
        index, reader, down_tensor, gate, final_scratch, cancelled, staging_budget_bytes,
    )?;
    if let Some(values) = residual {
        for (computed, addition) in final_scratch.iter_mut().zip(values.iter()) {
            if cancelled.load(Ordering::Acquire) {
                return Err(rejected(ErrorKind::Interrupted, "SwiGLU cancelled"));
            }
            let sum = *computed + *addition;
            if !sum.is_finite() {
                return Err(rejected(ErrorKind::InvalidData, "non-finite SwiGLU residual result"));
            }
            *computed = sum;
        }
    }
    if cancelled.load(Ordering::Acquire) {
        return Err(rejected(ErrorKind::Interrupted, "SwiGLU cancelled"));
    }
    output.copy_from_slice(final_scratch);
    Ok(())
}

/// Compute residual + down(SiLU(gate(input)) * up(input)) with atomic output.
///
/// The extra caller-owned `projection` workspace holds the finished SwiGLU
/// result while residual arithmetic is checked. The caller-visible `output`
/// is never used as scratch and remains unchanged on any failure. All six
/// activation slices (including input and residual) are charged to the
/// explicit budget. This does not enforce process RSS or guarantee ggml parity.
#[allow(clippy::too_many_arguments)]
pub fn f32_swiglu_residual_cancellable(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    gate_tensor: &str,
    up_tensor: &str,
    down_tensor: &str,
    input: &[f32],
    residual: &[f32],
    gate: &mut [f32],
    up: &mut [f32],
    final_scratch: &mut [f32],
    projection: &mut [f32],
    output: &mut [f32],
    cancelled: &AtomicBool,
    staging_budget_bytes: usize,
    activation_budget_bytes: usize,
) -> io::Result<()> {
    if residual.len() != output.len()
        || projection.len() != output.len()
        || residual.iter().any(|v| !v.is_finite())
    {
        return Err(rejected(ErrorKind::InvalidInput, "invalid SwiGLU residual shape or values"));
    }
    let extra_bytes = residual.len()
        .checked_add(projection.len())
        .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| rejected(ErrorKind::InvalidInput, "residual activation size overflow"))?;
    let inner_budget = activation_budget_bytes
        .checked_sub(extra_bytes)
        .ok_or_else(|| rejected(ErrorKind::InvalidInput, "residual activation budget exceeded"))?;
    f32_swiglu_cancellable(
        index, reader, gate_tensor, up_tensor, down_tensor, input,
        gate, up, final_scratch, projection, cancelled,
        staging_budget_bytes, inner_budget,
    )?;
    // Preflight every result before writing the caller-visible output.
    for (&value, &bias) in projection.iter().zip(residual.iter()) {
        if cancelled.load(Ordering::Acquire) {
            return Err(rejected(ErrorKind::Interrupted, "SwiGLU residual cancelled"));
        }
        if !(value + bias).is_finite() {
            return Err(rejected(ErrorKind::InvalidData, "non-finite SwiGLU residual result"));
        }
    }
    if cancelled.load(Ordering::Acquire) {
        return Err(rejected(ErrorKind::Interrupted, "SwiGLU residual cancelled"));
    }
    for ((destination, &value), &bias) in output.iter_mut().zip(projection.iter()).zip(residual.iter()) {
        *destination = value + bias;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn fixture(down: [f32; 4]) -> std::path::PathBuf {
        let mut gguf = b"GGUF".to_vec();
        gguf.extend_from_slice(&3u32.to_le_bytes());
        gguf.extend_from_slice(&3u64.to_le_bytes());
        gguf.extend_from_slice(&0u64.to_le_bytes());
        for (n, name) in ["gate", "up", "down"].iter().enumerate() {
            gguf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            gguf.extend_from_slice(name.as_bytes());
            gguf.extend_from_slice(&2u32.to_le_bytes());
            gguf.extend_from_slice(&2u64.to_le_bytes());
            gguf.extend_from_slice(&2u64.to_le_bytes());
            gguf.extend_from_slice(&0u32.to_le_bytes());
            gguf.extend_from_slice(&((n as u64) * 32).to_le_bytes());
        }
        while gguf.len() % 32 != 0 {
            gguf.push(0);
        }
        for weights in [[1.0f32, 0., 0., 1.], [1., 0., 0., 2.], down] {
            for value in weights {
                gguf.extend_from_slice(&value.to_le_bytes());
            }
            gguf.extend_from_slice(&[0u8; 16]);
        }
        let path = std::env::temp_dir().join(format!(
            "llmbridge-swiglu-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        ));
        File::create(&path).unwrap().write_all(&gguf).unwrap();
        path
    }

    #[test]
    fn streams_three_tensors_and_matches_scalar_reference() {
        let path = fixture([1., 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut gate = [0f32; 2];
        let mut up = [0f32; 2];
        let mut scratch = [0f32; 2];
        let mut output = [-1f32; 2];
        f32_swiglu_cancellable(
            &index, &mut reader, "gate", "up", "down", &[2., 3.],
            &mut gate, &mut up, &mut scratch, &mut output,
            &AtomicBool::new(false), 4, 40,
        ).unwrap();
        for (observed, (g, u)) in output.iter().zip([(2.0f64, 2.0), (3.0, 6.0)]) {
            let expected = (g / (1.0 + (-g).exp())) * u;
            assert!((*observed as f64 - expected).abs() < 0.00001);
        }
        assert_eq!(up, [2., 6.]);
        assert_eq!(reader.disk_bytes_read(), 48);
        assert_eq!(reader.allocated_chunk_bytes(), 4);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_bad_shapes_and_budgets_before_any_tensor_io() {
        let path = fixture([1., 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut gate = [44f32; 2];
        let mut up = [55f32; 2];
        let mut scratch = [66f32; 2];
        let mut output = [77f32; 2];
        let error = f32_swiglu_cancellable(
            &index, &mut reader, "gate", "up", "down", &[2., 3.],
            &mut gate, &mut up, &mut scratch, &mut output,
            &AtomicBool::new(false), 4, 39,
        ).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        let error = f32_swiglu_cancellable(
            &index, &mut reader, "gate", "up", "absent", &[2., 3.],
            &mut gate, &mut up, &mut scratch, &mut output,
            &AtomicBool::new(false), 4, 40,
        ).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(gate, [44.; 2]);
        assert_eq!(up, [55.; 2]);
        assert_eq!(scratch, [66.; 2]);
        assert_eq!(output, [77.; 2]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn late_down_projection_failure_never_commits_output() {
        let path = fixture([f32::NAN, 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut gate = [0f32; 2];
        let mut up = [0f32; 2];
        let mut scratch = [66f32; 2];
        let mut output = [77f32; 2];
        let error = f32_swiglu_cancellable(
            &index, &mut reader, "gate", "up", "down", &[2., 3.],
            &mut gate, &mut up, &mut scratch, &mut output,
            &AtomicBool::new(false), 4, 40,
        ).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(reader.disk_bytes_read(), 36);
        assert_eq!(output, [77.; 2]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn streamed_swiglu_residual_matches_independent_scalar_reference() {
        let path = fixture([1., 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut gate = [0f32; 2];
        let mut up = [0f32; 2];
        let mut scratch = [0f32; 2];
        let mut output = [-1f32; 2];
        f32_swiglu_residual_cancellable(
            &index, &mut reader, "gate", "up", "down", &[2., 3.], &[10., -2.],
            &mut gate, &mut up, &mut scratch, &mut output,
            &AtomicBool::new(false), 4, 48,
        ).unwrap();
        for (observed, (g, u, residual)) in output.iter()
            .zip([(2.0f64, 2.0, 10.0), (3.0, 6.0, -2.0)])
        {
            let expected = (g / (1.0 + (-g).exp())) * u + residual;
            assert!((*observed as f64 - expected).abs() < 0.00001);
        }
        assert_eq!(reader.disk_bytes_read(), 48);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn residual_overflow_preserves_output_after_all_tensor_reads() {
        let path = fixture([3e37, 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut gate = [0f32; 2];
        let mut up = [0f32; 2];
        let mut scratch = [0f32; 2];
        let mut output = [77f32; 2];
        let error = f32_swiglu_residual_cancellable(
            &index, &mut reader, "gate", "up", "down", &[2., 3.],
            &[f32::MAX, 0.], &mut gate, &mut up,
            &mut scratch, &mut output, &AtomicBool::new(false), 4, 48,
        ).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(reader.disk_bytes_read(), 48);
        assert_eq!(output, [77.; 2]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn residual_invalid_or_underbudget_rejected_without_file_reads() {
        let path = fixture([1., 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut gate = [44f32; 2];
        let mut up = [55f32; 2];
        let mut scratch = [66f32; 2];
        let mut output = [77f32; 2];
        for (residual, budget) in [
            (&[1.0f32, f32::NAN][..], 48usize),
            (&[1.0f32, 0.0][..], 47usize),
            (&[1.0f32][..], 48usize),
        ] {
            let error = f32_swiglu_residual_cancellable(
                &index, &mut reader, "gate", "up", "down", &[2., 3.],
                residual, &mut gate, &mut up, &mut scratch, &mut output,
                &AtomicBool::new(false), 4, budget,
            ).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
        }
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(gate, [44.; 2]);
        assert_eq!(up, [55.; 2]);
        assert_eq!(scratch, [66.; 2]);
        assert_eq!(output, [77.; 2]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn pre_cancelled_execution_preserves_all_buffers() {
        let path = fixture([1., 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let mut gate = [44f32; 2];
        let mut up = [55f32; 2];
        let mut scratch = [66f32; 2];
        let mut output = [77f32; 2];
        let error = f32_swiglu_cancellable(
            &index, &mut reader, "gate", "up", "down", &[2., 3.],
            &mut gate, &mut up, &mut scratch, &mut output,
            &AtomicBool::new(true), 4, 40,
        ).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Interrupted);
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(gate, [44.; 2]);
        assert_eq!(up, [55.; 2]);
        assert_eq!(scratch, [66.; 2]);
        assert_eq!(output, [77.; 2]);
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn residual_swiglu_streams_and_commits_reference_values() {
        let path = fixture([1., 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let (mut gate, mut up, mut scratch, mut projection, mut output) =
            ([0.; 2], [0.; 2], [0.; 2], [0.; 2], [77.; 2]);
        f32_swiglu_residual_cancellable(
            &index, &mut reader, "gate", "up", "down",
            &[2., 3.], &[10., -2.], &mut gate, &mut up,
            &mut scratch, &mut projection, &mut output,
            &AtomicBool::new(false), 4, 56,
        ).unwrap();
        for (observed, (g, u, bias)) in output.iter().zip([
            (2.0f64, 2.0f64, 10.0f64), (3.0, 6.0, -2.0),
        ]) {
            let expected = (g / (1.0 + (-g).exp())) * u + bias;
            assert!((*observed as f64 - expected).abs() < 0.00001);
        }
        assert_eq!(reader.disk_bytes_read(), 48);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn residual_budget_and_nonfinite_inputs_fail_before_reads() {
        let path = fixture([1., 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let (mut gate, mut up, mut scratch, mut projection, mut output) =
            ([44.; 2], [55.; 2], [66.; 2], [88.; 2], [77.; 2]);
        for (residual, budget) in [([1., 2.], 55), ([f32::NAN, 2.], 56)] {
            assert!(f32_swiglu_residual_cancellable(
                &index, &mut reader, "gate", "up", "down",
                &[2., 3.], &residual, &mut gate, &mut up,
                &mut scratch, &mut projection, &mut output,
                &AtomicBool::new(false), 4, budget,
            ).is_err());
        }
        assert_eq!(reader.disk_bytes_read(), 0);
        assert_eq!(gate, [44.; 2]);
        assert_eq!(up, [55.; 2]);
        assert_eq!(scratch, [66.; 2]);
        assert_eq!(projection, [88.; 2]);
        assert_eq!(output, [77.; 2]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn residual_overflow_does_not_commit_partial_output() {
        let path = fixture([1., 0., 0., 1.]);
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 4, 8).unwrap();
        let (mut gate, mut up, mut scratch, mut projection, mut output) =
            ([0.; 2], [0.; 2], [0.; 2], [0.; 2], [77.; 2]);
        let err = f32_swiglu_residual_cancellable(
            &index, &mut reader, "gate", "up", "down",
            &[1.0e19, 1.], &[f32::MAX, 0.], &mut gate, &mut up,
            &mut scratch, &mut projection, &mut output,
            &AtomicBool::new(false), 4, 56,
        ).unwrap_err();
        // The first projected result is finite (~1e38), but adding
        // f32::MAX overflows. Neither output element may be committed.
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert_eq!(output, [77.; 2]);
        assert_eq!(reader.disk_bytes_read(), 48);
        fs::remove_file(path).unwrap();
    }

}
