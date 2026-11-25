//! Safe wrapper over the native `llmbridge_dequantize_rows_v1` FFI entry, which
//! decodes quantized GGUF tensor rows to FP32 using the pinned ggml `to_float`
//! trait (the same reference kernel llama.cpp uses). This is weight decoding, not
//! model execution: it does not run attention, SSM or MoE layers.
//!
//! The i-quant types in the real Nemotron UD-IQ2_XXS shards (IQ2_XXS, IQ4_NL) are
//! decoded here via the pinned kernel; `IQ4_NL` additionally has an independent
//! in-crate reference used by the tests to cross-check the native path.

use std::os::raw::c_char;

extern "C" {
    fn llmbridge_dequantize_rows_v1(
        ggml_type: u32,
        columns: u32,
        rows: u32,
        src: *const u8,
        src_len: u64,
        dst: *mut f32,
        dst_count: u64,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
}

/// Dequantize `rows` rows of `columns` elements of the given ggml tensor type
/// from `src` (exact packed bytes) into a fresh FP32 vector. The native side
/// validates geometry against the pinned ggml block/row size and rejects
/// non-finite output, so a wrong length or an undecodable type returns `Err`.
pub fn dequantize_rows(
    ggml_type: u32,
    columns: u32,
    rows: u32,
    src: &[u8],
) -> Result<Vec<f32>, String> {
    let dst_count = u64::from(columns)
        .checked_mul(u64::from(rows))
        .ok_or_else(|| "dequantize output count overflow".to_string())?;
    let mut dst = vec![0f32; dst_count as usize];
    let mut error = [0u8; 256];
    let rc = unsafe {
        llmbridge_dequantize_rows_v1(
            ggml_type,
            columns,
            rows,
            src.as_ptr(),
            src.len() as u64,
            dst.as_mut_ptr(),
            dst_count,
            error.as_mut_ptr() as *mut c_char,
            error.len(),
        )
    };
    if rc == 0 {
        Ok(dst)
    } else {
        let end = error.iter().position(|&b| b == 0).unwrap_or(0);
        let message = String::from_utf8_lossy(&error[..end]).into_owned();
        Err(if message.is_empty() {
            format!("native dequantize failed (rc={rc})")
        } else {
            message
        })
    }
}

/// The pinned ggml IQ4_NL codebook (`kvalues_iq4nl`). Non-linear 16-entry table.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Independent reference dequantization of one 32-element IQ4_NL block, matching
/// ggml's `dequantize_row_iq4_nl`: low nibbles fill y[0..16], high nibbles fill
/// y[16..32], each scaled by the block's FP32 scale `d`.
pub fn iq4nl_reference_block(d: f32, qs: &[u8; 16]) -> [f32; 32] {
    let mut y = [0f32; 32];
    for j in 0..16 {
        y[j] = d * f32::from(KVALUES_IQ4NL[(qs[j] & 0x0f) as usize]);
        y[j + 16] = d * f32::from(KVALUES_IQ4NL[(qs[j] >> 4) as usize]);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    // f16 bit pattern for exactly 1.0 (sign 0, exp 15, mantissa 0) -> 0x3C00.
    const F16_ONE: [u8; 2] = [0x00, 0x3C];

    /// Build `n` IQ4_NL blocks (18 bytes each: f16 scale + 16 packed bytes) with
    /// scale exactly 1.0 and varied nibbles, returning (encoded, expected_f32).
    fn synth_iq4nl(n: usize) -> (Vec<u8>, Vec<f32>) {
        let mut encoded = Vec::new();
        let mut expected = Vec::new();
        for b in 0..n {
            let mut qs = [0u8; 16];
            for (j, q) in qs.iter_mut().enumerate() {
                *q = (((b * 31 + j * 17) & 0xff) as u8).rotate_left(1);
            }
            encoded.extend_from_slice(&F16_ONE);
            encoded.extend_from_slice(&qs);
            expected.extend_from_slice(&iq4nl_reference_block(1.0, &qs));
        }
        (encoded, expected)
    }

    #[test]
    fn iq4nl_native_matches_independent_reference() {
        // Three 32-element blocks => 96 columns, one row.
        let (encoded, expected) = synth_iq4nl(3);
        let decoded = dequantize_rows(20, 96, 1, &encoded).expect("IQ4_NL dequant");
        assert_eq!(decoded.len(), expected.len());
        let mut max_abs = 0f32;
        for (a, b) in decoded.iter().zip(&expected) {
            assert!(a.is_finite());
            max_abs = max_abs.max((a - b).abs());
        }
        // Same arithmetic (scale 1.0 * integer codebook) on both sides: exact.
        assert!(
            max_abs < 1e-6,
            "IQ4_NL native vs reference max abs = {max_abs}"
        );
    }

    #[test]
    fn iq4nl_is_deterministic_across_calls() {
        let (encoded, _) = synth_iq4nl(2);
        let a = dequantize_rows(20, 64, 1, &encoded).unwrap();
        let b = dequantize_rows(20, 64, 1, &encoded).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn dequant_rejects_malformed_geometry_and_unknown_types() {
        let (encoded, _) = synth_iq4nl(1); // 18 bytes = exactly one block
                                           // columns not a whole number of 32-element blocks.
        assert!(dequantize_rows(20, 31, 1, &encoded).is_err());
        // src length disagrees with declared geometry (claim 2 rows).
        assert!(dequantize_rows(20, 32, 2, &encoded).is_err());
        // Unknown/undecodable ggml type id.
        assert!(dequantize_rows(9999, 32, 1, &encoded).is_err());
        // Correct call still succeeds (control).
        assert!(dequantize_rows(20, 32, 1, &encoded).is_ok());
    }

    #[test]
    fn iq2xxs_block_size_round_trips_finite_for_synthetic_zeros() {
        // IQ2_XXS block = 256 elements / 66 bytes. An all-zero encoded block is a
        // valid bit pattern (scale 0 + grid index 0); the pinned kernel must decode
        // it to finite values. This exercises the IQ2_XXS path without the model.
        let encoded = vec![0u8; 66]; // one 256-element block
        let decoded = dequantize_rows(16, 256, 1, &encoded).expect("IQ2_XXS dequant");
        assert_eq!(decoded.len(), 256);
        assert!(decoded.iter().all(|v| v.is_finite()));
    }

    // Real-model validation: decode genuine IQ2_XXS bytes from the downloaded
    // Nemotron shards. Gated on LLMBRIDGE_NEMOTRON_DIR so CI without the 49 GB
    // model skips it rather than failing.
    #[test]
    fn real_nemotron_iq2xxs_expert_row_dequantizes_finite() {
        use crate::gguf_index::SplitGgufIndex;
        use crate::storage::BoundedLocalReader;
        use std::sync::atomic::AtomicBool;

        let dir = match std::env::var("LLMBRIDGE_NEMOTRON_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: LLMBRIDGE_NEMOTRON_DIR unset (real-model IQ2_XXS test)");
                return;
            }
        };
        let shards: Vec<String> = (1..=3)
            .map(|i| {
                format!("{dir}/NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ2_XXS-0000{i}-of-00003.gguf")
            })
            .collect();
        let index = SplitGgufIndex::inspect(&shards).expect("real split index");
        let name = "blk.83.ffn_up_exps.weight";
        let (member, region) = index.tensor(name).expect("expert tensor present");
        assert_eq!(region.ggml_type, 16, "expected IQ2_XXS expert weights");
        let columns = region.dimensions[0] as u32; // 1024
        let row_bytes = u64::from(columns) / 256 * 66; // 264 bytes
        let gi = member.index.as_ref().expect("data shard index");
        let mut reader = BoundedLocalReader::open(&member.path, 4096, 1 << 20).unwrap();
        let stop = AtomicBool::new(false);
        let mut bytes = Vec::new();
        gi.stream_tensor_slice_cancellable(&mut reader, name, 0, row_bytes, &stop, |chunk| {
            bytes.extend_from_slice(chunk);
            Ok(())
        })
        .expect("bounded read of one expert row");
        assert_eq!(bytes.len() as u64, row_bytes);
        let decoded = dequantize_rows(16, columns, 1, &bytes).expect("real IQ2_XXS dequant");
        assert_eq!(decoded.len(), columns as usize);
        assert!(decoded.iter().all(|v| v.is_finite()));
        assert!(
            decoded.iter().any(|&v| v != 0.0),
            "real expert row should not decode to all zeros"
        );
    }
}
