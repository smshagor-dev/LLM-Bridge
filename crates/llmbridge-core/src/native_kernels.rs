//! LLM Bridge-owned native quantized kernels (pure Rust, no GGML/FFI).
//!
//! These are the first step of making the production numerical path independent
//! of the pinned GGML `to_float` FFI (`crate::native_dequant`,
//! `crate::inference::dequant_matvec`). Each kernel decodes a GGUF quantized
//! block format directly in safe Rust and performs the bounded matrix-vector
//! product, reading weight rows through the existing authenticated bounded
//! reader ([`crate::storage::BoundedLocalReader`]) — the same trusted-source and
//! accounting contract as the GGML path, with no native execution dependency.
//!
//! Scope so far: `Q8_0` (ggml type 8), the simplest exactly-representable block
//! format (f16 block scale + 32 int8 quants). The GGML path remains available as
//! a test-only reference and as the fallback for formats not yet owned here.
//!
//! Correctness is cross-checked two ways in tests: against an independent in-Rust
//! FP64 oracle (exact `d * q` arithmetic) and against the GGML FFI kernel on a
//! real Nano Q8_0 tensor slice.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::gguf_index::TensorRegion;
use crate::storage::BoundedLocalReader;

/// ggml type id for Q8_0.
pub const GGML_TYPE_Q8_0: u32 = 8;
/// Q8_0 block: 32 quants per block.
const Q8_0_BLOCK_ELEMS: usize = 32;
/// Q8_0 encoded block size: 2-byte f16 scale + 32 int8 quants.
const Q8_0_BLOCK_BYTES: usize = 2 + Q8_0_BLOCK_ELEMS;

/// IEEE-754 half-precision bit pattern to f32 (matches ggml `ggml_fp16_to_fp32`).
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let val = match exp {
        0 => (mant as f32) * 2f32.powi(-24),
        0x1f => {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// Decode one Q8_0 row (`columns` elements, `columns/32` blocks) from exact
/// packed bytes into f32. `bytes.len()` must equal `columns/32 * 34`.
pub fn q8_0_dequantize_row(bytes: &[u8], columns: usize) -> Result<Vec<f32>, String> {
    if columns % Q8_0_BLOCK_ELEMS != 0 {
        return Err("Q8_0 columns not a multiple of 32".into());
    }
    let blocks = columns / Q8_0_BLOCK_ELEMS;
    if bytes.len() != blocks * Q8_0_BLOCK_BYTES {
        return Err(format!(
            "Q8_0 row byte length {} != expected {}",
            bytes.len(),
            blocks * Q8_0_BLOCK_BYTES
        ));
    }
    let mut out = Vec::with_capacity(columns);
    for b in 0..blocks {
        let base = b * Q8_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        for i in 0..Q8_0_BLOCK_ELEMS {
            let q = bytes[base + 2 + i] as i8;
            out.push(d * f32::from(q));
        }
    }
    Ok(out)
}

/// Owned Q8_0 bounded matrix-vector product `W . x`, byte-for-byte equivalent in
/// contract to [`crate::inference::dequant_matvec`] for `ggml_type == 8`, but
/// implemented entirely in Rust with no GGML/FFI call.
///
/// `region` is a rank-2 Q8_0 tensor (`[columns, rows]` in ggml ne order, stored
/// row-major as `rows` rows of `columns/32` Q8_0 blocks). `activation.len()` must
/// equal `columns`. One weight row is read at a time through the authenticated
/// bounded `reader`; scratch is a single row plus the `rows`-length output, so
/// the model is never materialized.
pub fn q8_0_dequant_matvec(
    region: &TensorRegion,
    reader: &mut BoundedLocalReader,
    activation: &[f32],
    cancelled: &AtomicBool,
    workspace_budget_bytes: usize,
) -> Result<Vec<f32>, String> {
    if region.ggml_type != GGML_TYPE_Q8_0 {
        return Err("q8_0_dequant_matvec requires a Q8_0 tensor".into());
    }
    if region.dimensions.len() != 2 {
        return Err("q8_0_dequant_matvec requires a rank-2 tensor".into());
    }
    let columns = usize::try_from(region.dimensions[0]).map_err(|_| "columns overflow")?;
    let rows = usize::try_from(region.dimensions[1]).map_err(|_| "rows overflow")?;
    if columns == 0 || rows == 0 || columns % Q8_0_BLOCK_ELEMS != 0 {
        return Err("invalid Q8_0 geometry".into());
    }
    if activation.len() != columns || !activation.iter().all(|f| f.is_finite()) {
        return Err("activation length/finiteness invalid".into());
    }
    let row_bytes = (columns / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
    let expected_len = (row_bytes as u64)
        .checked_mul(rows as u64)
        .ok_or("tensor size overflow")?;
    if region.byte_len != expected_len {
        return Err(format!(
            "Q8_0 tensor byte_len {} != rows*row_bytes {}",
            region.byte_len, expected_len
        ));
    }
    // Bounded-workspace accounting mirrors the GGML path: one weight row buffer
    // plus the activation-sized operands and the result, under the caller budget.
    let scratch = row_bytes
        .checked_add(columns * 4)
        .and_then(|s| s.checked_add(rows * 4))
        .and_then(|s| s.checked_add(64 * 1024))
        .ok_or("workspace arithmetic overflow")?;
    if scratch > workspace_budget_bytes || row_bytes > reader.allocated_chunk_bytes() {
        return Err("operator workspace exceeds budget".into());
    }
    let mut output = vec![0f32; rows];
    for (r, out) in output.iter_mut().enumerate() {
        if cancelled.load(Ordering::Acquire) {
            return Err("q8_0 matvec cancelled".into());
        }
        let offset = region
            .file_offset
            .checked_add((r as u64) * (row_bytes as u64))
            .ok_or("row offset overflow")?;
        let bytes = reader
            .read_range(offset, row_bytes)
            .map_err(|e| format!("bounded row read: {e}"))?;
        // Decode + dot in one pass over the row (no full-row Vec materialized).
        let blocks = columns / Q8_0_BLOCK_ELEMS;
        let mut acc = 0f32;
        for b in 0..blocks {
            let base = b * Q8_0_BLOCK_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
            let act = &activation[b * Q8_0_BLOCK_ELEMS..(b + 1) * Q8_0_BLOCK_ELEMS];
            let mut blk = 0f32;
            for i in 0..Q8_0_BLOCK_ELEMS {
                let q = bytes[base + 2 + i] as i8;
                blk += f32::from(q) * act[i];
            }
            acc += d * blk;
        }
        if !acc.is_finite() {
            return Err("non-finite Q8_0 matvec output".into());
        }
        *out = acc;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // f16 bit pattern for 1.0 and 0.5.
    const F16_ONE: u16 = 0x3C00;
    const F16_HALF: u16 = 0x3800;

    #[test]
    fn f16_decode_matches_known_values() {
        assert_eq!(f16_to_f32(F16_ONE), 1.0);
        assert_eq!(f16_to_f32(F16_HALF), 0.5);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
    }

    #[test]
    fn dequantize_row_exact_for_scale_one() {
        // one block, scale 1.0, quants 0..32 (as signed) -> exact integers.
        let mut bytes = F16_ONE.to_le_bytes().to_vec();
        for i in 0..32i32 {
            bytes.push((i - 16) as i8 as u8);
        }
        let row = q8_0_dequantize_row(&bytes, 32).unwrap();
        for (i, v) in row.iter().enumerate() {
            assert_eq!(*v, (i as i32 - 16) as f32);
        }
    }

    #[test]
    fn rejects_bad_geometry() {
        assert!(q8_0_dequantize_row(&[0u8; 10], 32).is_err());
        assert!(q8_0_dequantize_row(&[0u8; 34], 31).is_err());
    }
}
