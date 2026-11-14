//! Exact on-disk GGUF tensor coverage accounting for indexed layouts.
//! This is a metadata-only audit, not a resident-memory or runtime guarantee.
use crate::gguf_index::TensorRegion;
use std::io::{self, ErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorLayoutAudit {
    pub tensor_count: usize,
    pub payload_bytes: u64,
    pub inter_tensor_padding_bytes: u64,
    pub trailing_bytes: u64,
    pub data_region_bytes: u64,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

/// Audit tensor byte coverage without allocating a tensor-sized buffer.
///
/// Input regions must already come from a trusted GGUF index. Sorting a small
/// offset table permits arbitrary declaration order; the function rejects
/// overlaps, inconsistent relative offsets, and out-of-file spans. Trailing
/// bytes are reported, not assumed to be a tensor or safe to execute.
pub fn audit_tensor_layout(
    regions: &[TensorRegion],
    data_offset: u64,
    file_bytes: u64,
) -> io::Result<TensorLayoutAudit> {
    if data_offset > file_bytes {
        return Err(invalid("GGUF tensor data begins beyond EOF"));
    }
    let mut spans = Vec::new();
    spans.try_reserve_exact(regions.len()).map_err(|_| {
        io::Error::new(
            ErrorKind::OutOfMemory,
            "tensor audit offset table allocation failed",
        )
    })?;
    for region in regions {
        if region.byte_len == 0 {
            return Err(invalid("zero-length GGUF tensor"));
        }
        let expected = data_offset
            .checked_add(region.relative_offset)
            .ok_or_else(|| invalid("tensor relative offset overflow"))?;
        if region.file_offset != expected {
            return Err(invalid("tensor absolute and relative offsets disagree"));
        }
        let end = expected
            .checked_add(region.byte_len)
            .ok_or_else(|| invalid("tensor byte range overflow"))?;
        if end > file_bytes {
            return Err(invalid("GGUF tensor extends beyond EOF"));
        }
        spans.push((expected, end));
    }
    spans.sort_unstable();
    let mut cursor = data_offset;
    let mut payload_bytes = 0u64;
    let mut inter_tensor_padding_bytes = 0u64;
    for (start, end) in spans {
        if start < cursor {
            return Err(invalid("GGUF tensor regions overlap"));
        }
        inter_tensor_padding_bytes = inter_tensor_padding_bytes
            .checked_add(start - cursor)
            .ok_or_else(|| invalid("GGUF padding total overflow"))?;
        payload_bytes = payload_bytes
            .checked_add(end - start)
            .ok_or_else(|| invalid("GGUF tensor payload total overflow"))?;
        cursor = end;
    }
    Ok(TensorLayoutAudit {
        tensor_count: regions.len(),
        payload_bytes,
        inter_tensor_padding_bytes,
        trailing_bytes: file_bytes - cursor,
        data_region_bytes: file_bytes - data_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(name: &str, relative_offset: u64, byte_len: u64) -> TensorRegion {
        TensorRegion {
            name: name.to_owned(),
            dimensions: vec![byte_len],
            ggml_type: 0,
            relative_offset,
            file_offset: 64 + relative_offset,
            byte_len,
        }
    }

    #[test]
    fn unsorted_regions_count_payload_padding_and_trailer() {
        let regions = [region("second", 32, 16), region("first", 0, 16)];
        let result = audit_tensor_layout(&regions, 64, 128).unwrap();
        assert_eq!(result.tensor_count, 2);
        assert_eq!(result.payload_bytes, 32);
        assert_eq!(result.inter_tensor_padding_bytes, 16);
        assert_eq!(result.trailing_bytes, 16);
        assert_eq!(result.data_region_bytes, 64);
        assert_eq!(
            result.payload_bytes + result.inter_tensor_padding_bytes + result.trailing_bytes,
            result.data_region_bytes
        );
    }

    #[test]
    fn overlap_and_mismatched_offsets_are_rejected() {
        assert_eq!(
            audit_tensor_layout(&[region("a", 0, 20), region("b", 16, 4)], 64, 128)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
        let mut bad = region("a", 0, 4);
        bad.file_offset = 65;
        assert_eq!(
            audit_tensor_layout(&[bad], 64, 128).unwrap_err().kind(),
            ErrorKind::InvalidData
        );
    }

    #[test]
    fn eof_overflow_and_zero_length_fail_closed() {
        assert!(audit_tensor_layout(&[region("a", 64, 1)], 64, 128).is_err());
        assert!(audit_tensor_layout(&[region("a", 0, 0)], 64, 128).is_err());
        assert!(audit_tensor_layout(&[], 129, 128).is_err());
        let mut bad = region("overflow", 0, 1);
        bad.relative_offset = u64::MAX;
        bad.file_offset = 0;
        assert!(audit_tensor_layout(&[bad], 64, u64::MAX).is_err());
    }

    #[test]
    fn empty_inventory_reports_only_trailing_data() {
        let result = audit_tensor_layout(&[], 64, 100).unwrap();
        assert_eq!(result.payload_bytes, 0);
        assert_eq!(result.trailing_bytes, 36);
        assert_eq!(result.inter_tensor_padding_bytes, 0);
    }
}
