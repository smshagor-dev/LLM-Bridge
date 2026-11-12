//! Sequential multi-vector execution using the existing bounded native Q8_0 operator.
//!
//! Results are delivered one at a time rather than accumulated in a matrix.
//! This is not a complete transformer graph, an RSS cap, or an out-of-core model.

use crate::gguf_index::GgufTensorIndex;
use crate::inference::q8_0_matvec_with_budget;
use crate::storage::BoundedLocalReader;
use std::sync::atomic::{AtomicBool, Ordering};

/// Execute a sequence of input vectors with a fixed operator workspace budget.
///
/// The callback must consume each output before the next operator invocation.
/// The function does not retain outputs or input vectors; the caller owns the
/// activation storage. The budget is enforced per invocation by the native
/// operator, not across the caller's allocations or the entire process.
///
/// A callback returning false requests cancellation. Completed callbacks
/// cannot be rolled back; no subsequent vector is evaluated.
pub fn q8_0_matvec_each<I, F>(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    tensor_name: &str,
    activations: I,
    cancelled: &AtomicBool,
    workspace_budget_bytes: usize,
    mut on_result: F,
) -> Result<usize, String>
where
    I: IntoIterator,
    I::Item: AsRef<[f32]>,
    F: FnMut(usize, &[f32]) -> bool,
{
    let mut completed = 0usize;
    for activation in activations {
        if cancelled.load(Ordering::Acquire) {
            return Err("batched Q8_0 operator cancelled".into());
        }
        let result = q8_0_matvec_with_budget(
            index,
            reader,
            tensor_name,
            activation.as_ref(),
            cancelled,
            workspace_budget_bytes,
        )?;
        // Do not expose a result after cancellation, even if the native
        // operator completed between its final check and this callback.
        if cancelled.load(Ordering::Acquire) {
            return Err("batched Q8_0 operator cancelled".into());
        }
        let next = completed
            .checked_add(1)
            .ok_or("batch result counter overflow")?;
        if !on_result(completed, &result) {
            return Err("batched Q8_0 operator stopped by result callback".into());
        }
        completed = next;
    }
    Ok(completed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::AtomicU64;
    static SERIAL: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn sequential_outputs_and_early_stop_have_exact_read_accounting() {
        let path = std::env::temp_dir().join(format!(
            "llmbridge-q8-batch-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.push(b'w');
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&32u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0, 0x38]); // f16 scale 0.5
        bytes.extend(1u8..=32u8);
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 64, 128).unwrap();
        let cancelled = AtomicBool::new(false);
        let vectors = vec![vec![1.0f32; 32], vec![2.0; 32], vec![3.0; 32]];
        let mut observed = Vec::new();
        let error = q8_0_matvec_each(
            &index, &mut reader, "w", &vectors, &cancelled, 256 * 1024,
            |i, result| {
                observed.push((i, result.to_vec()));
                i == 0
            },
        ).unwrap_err();
        assert!(error.contains("callback"));
        assert_eq!(observed, vec![(0, vec![264.0]), (1, vec![528.0])]);
        assert_eq!(reader.disk_bytes_read(), 68);

        cancelled.store(true, Ordering::Release);
        assert!(q8_0_matvec_each(
            &index, &mut reader, "w", &vectors, &cancelled, 256 * 1024,
            |_, _| true,
        ).is_err());
        assert_eq!(reader.disk_bytes_read(), 68);
        cancelled.store(false, Ordering::Release);
        assert!(q8_0_matvec_each(
            &index, &mut reader, "w", &vectors, &cancelled, 1,
            |_, _| true,
        ).is_err());
        assert_eq!(reader.disk_bytes_read(), 68);
        std::fs::remove_file(path).unwrap();
    }

}
