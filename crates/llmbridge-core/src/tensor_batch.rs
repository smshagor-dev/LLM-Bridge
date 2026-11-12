//! Preflight admission and sequential bounded streaming of indexed GGUF tensors.
//! The budget caps logical bytes read, NOT process RSS or OS page cache.
use crate::gguf_index::GgufTensorIndex;
use crate::storage::BoundedLocalReader;
use std::collections::HashSet;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

/// Validates the complete batch before any tensor bytes are read.
/// Repeated tensor names are rejected to prevent accidental double execution.
pub fn stream_tensor_batch<F>(
    index: &GgufTensorIndex,
    reader: &mut BoundedLocalReader,
    names: &[&str],
    max_total_bytes: u64,
    cancelled: &AtomicBool,
    mut consume: F,
) -> io::Result<u64>
where
    F: FnMut(&str, &[u8]) -> io::Result<()>,
{
    if cancelled.load(Ordering::Acquire) {
        return Err(io::Error::new(io::ErrorKind::Interrupted, "batch cancelled"));
    }
    if reader.identity() != &index.source_identity {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "GGUF source identity mismatch"));
    }
    let mut seen = HashSet::new();
    let mut total = 0u64;
    for name in names {
        if !seen.insert(*name) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "duplicate tensor in batch"));
        }
        let tensor = index.tensor(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "tensor missing from GGUF index")
        })?;
        total = total.checked_add(tensor.byte_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "tensor batch size overflow")
        })?;
        if total > max_total_bytes {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "tensor batch exceeds byte budget"));
        }
    }
    for name in names {
        if cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "batch cancelled"));
        }
        index.stream_tensor_cancellable(reader, name, cancelled, |chunk| consume(name, chunk))?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_index::GgufTensorIndex;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> std::path::PathBuf {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensors
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata
        let name = b"weights";
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&1u32.to_le_bytes()); // rank
        bytes.extend_from_slice(&4u64.to_le_bytes()); // four elements
        bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
        bytes.extend_from_slice(&0u64.to_le_bytes()); // relative offset
        while bytes.len() % 32 != 0 { bytes.push(0); }
        bytes.extend_from_slice(&[7u8; 16]);
        let path = std::env::temp_dir().join(format!(
            "llmbridge-batch-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
        path
    }

    #[test]
    fn preflight_rejects_invalid_batches_without_io() {
        let path = fixture();
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 5, 10).unwrap();
        let cancel = AtomicBool::new(false);
        for names in [&["weights", "weights"][..], &["weights", "missing"][..]] {
            assert!(stream_tensor_batch(&index, &mut reader, names, 32, &cancel, |_, _| Ok(())).is_err());
        }
        assert!(stream_tensor_batch(&index, &mut reader, &["weights"], 15, &cancel, |_, _| Ok(())).is_err());
        assert_eq!(reader.disk_bytes_read(), 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn streams_exact_bytes_and_honors_cancellation() {
        let path = fixture();
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let mut reader = BoundedLocalReader::open(&path, 5, 10).unwrap();
        let cancel = AtomicBool::new(false);
        let mut chunks = Vec::new();
        let total = stream_tensor_batch(&index, &mut reader, &["weights"], 16, &cancel, |name, chunk| {
            assert_eq!(name, "weights");
            chunks.extend_from_slice(chunk);
            Ok(())
        }).unwrap();
        assert_eq!(total, 16);
        assert_eq!(chunks, vec![7u8; 16]);
        assert_eq!(reader.disk_bytes_read(), 16);
        cancel.store(true, Ordering::Release);
        assert_eq!(stream_tensor_batch(&index, &mut reader, &["weights"], 16, &cancel, |_, _| Ok(())).unwrap_err().kind(), io::ErrorKind::Interrupted);
        assert_eq!(reader.disk_bytes_read(), 16);
        std::fs::remove_file(path).unwrap();
    }
}
