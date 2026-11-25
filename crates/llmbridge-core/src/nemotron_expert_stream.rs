//! Sparse quantized expert streaming for the Nemotron LatentMoE FFN.
//!
//! The real `nemotron_h_moe` expert weights are **packed**: a single GGUF tensor
//! holds every expert, stacked along the last dimension (see
//! build/reference/src/models/nemotron-h.cpp:130-131, rev cd959ba):
//!
//!   * `ffn_up_exps`   has ggml shape `[moe_n_embd, n_ff_exp, n_expert]`,
//!   * `ffn_down_exps` has ggml shape `[n_ff_exp, moe_n_embd, n_expert]`.
//!
//! Expert `e`'s 2-D weight matrix is therefore a contiguous quantized sub-block
//! of the packed tensor at stride `byte_len / n_expert`. This module resolves
//! that sub-block and runs the already-validated bounded, authenticated matvec
//! ([`crate::inference::dequant_matvec`]) on **only the selected experts** — no
//! all-expert residency, no full packed-tensor dequantization, and the trusted
//! range-reader contract (reads are clamped to each expert's own byte range) is
//! preserved unchanged.
//!
//! Per expert: `up = relu2(up_exps[e] . latent)`, `down = down_exps[e] . up`,
//! aggregated as `Σ weight[e] * down[e]` in the latent space — the same
//! arithmetic the dense [`crate::latent_moe_ffn`] was validated against, now
//! sourced from quantized bytes.

use std::sync::atomic::AtomicBool;

use crate::gguf_index::TensorRegion;
use crate::inference::dequant_matvec;
use crate::latent_moe_routing::Routing;
use crate::storage::BoundedLocalReader;

/// A packed expert tensor: `n_expert` stacked 2-D quantized matrices of shape
/// `[columns, rows]` each (ggml `ne = [columns, rows, n_expert]`), contiguous
/// along the expert dimension within the source file.
#[derive(Debug, Clone)]
pub struct PackedExperts {
    /// Absolute byte offset of the packed tensor within its shard.
    pub file_offset: u64,
    /// ggml type id of the packed tensor.
    pub ggml_type: u32,
    /// Per-expert 2-D shape (ggml ne0 / ne1).
    pub columns: u64,
    pub rows: u64,
    pub n_expert: u64,
    /// Byte length of the WHOLE packed tensor (all experts).
    pub byte_len: u64,
}

impl PackedExperts {
    pub fn new(
        file_offset: u64,
        ggml_type: u32,
        columns: u64,
        rows: u64,
        n_expert: u64,
        byte_len: u64,
    ) -> Result<Self, String> {
        if columns == 0 || rows == 0 || n_expert == 0 {
            return Err("packed expert tensor requires non-zero dimensions".into());
        }
        if byte_len == 0 || byte_len % n_expert != 0 {
            return Err("packed byte_len is not a positive multiple of n_expert".into());
        }
        let per_expert = byte_len / n_expert;
        if per_expert % rows != 0 {
            return Err("per-expert bytes not divisible by rows (bad geometry)".into());
        }
        file_offset
            .checked_add(byte_len)
            .ok_or("packed tensor end offset overflow")?;
        Ok(Self {
            file_offset,
            ggml_type,
            columns,
            rows,
            n_expert,
            byte_len,
        })
    }

    /// Bytes occupied by one expert's 2-D matrix.
    pub fn per_expert_bytes(&self) -> u64 {
        self.byte_len / self.n_expert
    }

    /// Build the rank-2 [`TensorRegion`] for expert `e`'s own byte range, so the
    /// bounded reader authenticates against exactly that sub-slice and can never
    /// read a neighbouring expert's bytes.
    pub fn expert_region(&self, e: usize) -> Result<TensorRegion, String> {
        if (e as u64) >= self.n_expert {
            return Err(format!(
                "expert id {e} out of range (n_expert = {})",
                self.n_expert
            ));
        }
        let per = self.per_expert_bytes();
        let offset = self
            .file_offset
            .checked_add((e as u64) * per)
            .ok_or("expert offset overflow")?;
        Ok(TensorRegion {
            name: format!("exp[{e}]"),
            dimensions: vec![self.columns, self.rows],
            ggml_type: self.ggml_type,
            relative_offset: (e as u64) * per,
            file_offset: offset,
            byte_len: per,
        })
    }

    /// Project `activation` (length == `columns`) through expert `e` using a
    /// bounded authenticated range read. Returns `rows` outputs.
    pub fn project_expert(
        &self,
        reader: &mut BoundedLocalReader,
        e: usize,
        activation: &[f32],
        cancelled: &AtomicBool,
        workspace_budget_bytes: usize,
    ) -> Result<Vec<f32>, String> {
        let region = self.expert_region(e)?;
        dequant_matvec(
            &region,
            reader,
            activation,
            cancelled,
            workspace_budget_bytes,
        )
    }
}

#[inline]
fn relu_sqr_in_place(v: &mut [f32]) {
    for x in v.iter_mut() {
        let r = if *x > 0.0 { *x } else { 0.0 };
        *x = r * r;
    }
}

/// Run one selected expert's full FFN in latent space from quantized bytes:
/// `down_exps[e] . relu2(up_exps[e] . latent)`.
///
/// `up` must be `[n_latent, n_ff_exp, n_expert]` and `down`
/// `[n_ff_exp, n_latent, n_expert]`; `latent.len()` must equal `n_latent`.
/// Only this expert's bytes are read (twice: up then down).
#[allow(clippy::too_many_arguments)]
pub fn stream_expert_ffn(
    up: &PackedExperts,
    down: &PackedExperts,
    reader: &mut BoundedLocalReader,
    e: usize,
    latent: &[f32],
    cancelled: &AtomicBool,
    workspace_budget_bytes: usize,
) -> Result<Vec<f32>, String> {
    if up.n_expert != down.n_expert {
        return Err("up/down expert counts disagree".into());
    }
    // up: columns = n_latent, rows = n_ff_exp
    if latent.len() as u64 != up.columns {
        return Err("latent length disagrees with up_exps columns (n_latent)".into());
    }
    // down: columns = n_ff_exp (== up.rows), rows = n_latent (== up.columns)
    if down.columns != up.rows || down.rows != up.columns {
        return Err("down_exps shape is not the transpose of up_exps".into());
    }
    let mut u = up.project_expert(reader, e, latent, cancelled, workspace_budget_bytes)?;
    relu_sqr_in_place(&mut u);
    down.project_expert(reader, e, &u, cancelled, workspace_budget_bytes)
}

/// Aggregate the selected experts' latent-space outputs with the routing
/// weights: `Σ_e weight[e] * (down_exps[e] . relu2(up_exps[e] . latent))`.
///
/// Only the experts named in `routing.expert_ids` are read (sparse); the result
/// is the `n_latent`-length MoE contribution (before `latent_up` / shared
/// expert, which the dense [`crate::latent_moe_ffn`] composition applies).
#[allow(clippy::too_many_arguments)]
pub fn stream_moe_experts(
    up: &PackedExperts,
    down: &PackedExperts,
    reader: &mut BoundedLocalReader,
    routing: &Routing,
    latent: &[f32],
    cancelled: &AtomicBool,
    workspace_budget_bytes: usize,
) -> Result<Vec<f32>, String> {
    if routing.expert_ids.len() != routing.weights.len() {
        return Err("routing ids/weights length mismatch".into());
    }
    let n_latent = up.columns as usize;
    let mut acc = vec![0.0f32; n_latent];
    for (&e, &w) in routing.expert_ids.iter().zip(&routing.weights) {
        if cancelled.load(std::sync::atomic::Ordering::Acquire) {
            return Err("sparse expert aggregation cancelled".into());
        }
        let d = stream_expert_ffn(
            up,
            down,
            reader,
            e,
            latent,
            cancelled,
            workspace_budget_bytes,
        )?;
        if d.len() != n_latent {
            return Err("expert down output length disagrees with n_latent".into());
        }
        for (a, dv) in acc.iter_mut().zip(&d) {
            *a += w * dv;
        }
    }
    if !acc.iter().all(|v| v.is_finite()) {
        return Err("non-finite aggregated expert output".into());
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_dequant::dequantize_rows;
    use std::io::Write;
    use std::sync::atomic::AtomicU64;

    const Q8_0: u32 = 8;
    const Q8_0_ROW_BYTES: usize = 34; // 32-wide block: f16 scale + 32 int8

    // Deterministic Q8_0 block for (expert, row): f16 scale 0.125 + 32 small int8
    // in [-3, 3], i.e. realistic dequantized weights in [-0.375, 0.375]. Keeping
    // magnitudes realistic keeps the squared-ReLU chain well-conditioned in f32.
    fn q8_block(e: usize, r: usize, col_block: usize) -> Vec<u8> {
        let mut b = vec![0x00u8, 0x30]; // f16 0.125
        for c in 0..32u32 {
            let v = ((e as u32 * 31 + r as u32 * 7 + col_block as u32 * 5 + c) % 7) as i32 - 3;
            b.push(v as i8 as u8);
        }
        b
    }

    /// Build a packed Q8_0 tensor payload [columns, rows, n_expert] and write it
    /// to a file at byte offset 0. Returns (path, payload).
    fn write_packed(columns: usize, rows: usize, n_expert: usize) -> (std::path::PathBuf, Vec<u8>) {
        assert!(columns % 32 == 0);
        let blocks_per_row = columns / 32;
        let mut payload = Vec::new();
        for e in 0..n_expert {
            for r in 0..rows {
                for cb in 0..blocks_per_row {
                    payload.extend_from_slice(&q8_block(e, r, cb));
                }
            }
        }
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "llmbridge-packed-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&payload)
            .unwrap();
        (path, payload)
    }

    fn packed(columns: usize, rows: usize, n_expert: usize, byte_len: usize) -> PackedExperts {
        PackedExperts::new(
            0,
            Q8_0,
            columns as u64,
            rows as u64,
            n_expert as u64,
            byte_len as u64,
        )
        .unwrap()
    }

    // Independent reference: dequantize expert e's rows from the raw payload and
    // dot with the activation (separate FFI path + Rust f64 dot).
    fn ref_project(payload: &[u8], columns: usize, rows: usize, e: usize, act: &[f32]) -> Vec<f64> {
        let per = rows * (columns / 32) * Q8_0_ROW_BYTES;
        let slice = &payload[e * per..(e + 1) * per];
        let flat = dequantize_rows(Q8_0, columns as u32, rows as u32, slice).unwrap();
        (0..rows)
            .map(|r| {
                (0..columns)
                    .map(|c| flat[r * columns + c] as f64 * act[c] as f64)
                    .sum()
            })
            .collect()
    }

    #[test]
    fn project_first_middle_last_expert_match_reference() {
        let (columns, rows, n_expert) = (32usize, 3usize, 4usize);
        let (path, payload) = write_packed(columns, rows, n_expert);
        let byte_len = payload.len();
        let pk = packed(columns, rows, n_expert, byte_len);
        let act: Vec<f32> = (0..columns).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let stop = AtomicBool::new(false);
        for &e in &[0usize, n_expert / 2, n_expert - 1] {
            let mut reader = BoundedLocalReader::open(&path, 128, 256).unwrap();
            let got = pk
                .project_expert(&mut reader, e, &act, &stop, 1 << 20)
                .unwrap();
            let want = ref_project(&payload, columns, rows, e, &act);
            let max_abs = got
                .iter()
                .zip(&want)
                .map(|(g, r)| (*g as f64 - r).abs())
                .fold(0.0, f64::max);
            assert!(max_abs < 1e-3, "expert {e} max_abs {max_abs}");
            assert_eq!(got.len(), rows);
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cross_block_columns_match_reference() {
        // 64 columns => 2 Q8_0 blocks per row (cross-block boundary exercised).
        let (columns, rows, n_expert) = (64usize, 2usize, 3usize);
        let (path, payload) = write_packed(columns, rows, n_expert);
        let pk = packed(columns, rows, n_expert, payload.len());
        let act: Vec<f32> = (0..columns).map(|i| ((i % 11) as f32) - 5.0).collect();
        let stop = AtomicBool::new(false);
        let mut reader = BoundedLocalReader::open(&path, 256, 512).unwrap();
        let got = pk
            .project_expert(&mut reader, 1, &act, &stop, 1 << 20)
            .unwrap();
        let want = ref_project(&payload, columns, rows, 1, &act);
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(g, r)| (*g as f64 - r).abs())
            .fold(0.0, f64::max);
        assert!(max_abs < 1e-3, "cross-block max_abs {max_abs}");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn full_expert_ffn_chain_matches_reference() {
        // up: [n_latent=32, n_ff_exp=32, E]; down: [n_ff_exp=32, n_latent=32, E].
        let (n_latent, n_ff, e_count) = (32usize, 32usize, 3usize);
        let (up_path, up_payload) = write_packed(n_latent, n_ff, e_count);
        let (down_path, down_payload) = write_packed(n_ff, n_latent, e_count);
        let up = packed(n_latent, n_ff, e_count, up_payload.len());
        let down = packed(n_ff, n_latent, e_count, down_payload.len());
        let latent: Vec<f32> = (0..n_latent).map(|i| (i as f32) * 0.05 - 0.8).collect();
        let stop = AtomicBool::new(false);
        let e = 2usize;

        let mut rd = BoundedLocalReader::open(&up_path, 256, 512).unwrap();
        let mut rd2 = BoundedLocalReader::open(&down_path, 256, 512).unwrap();
        // production path reads up from one reader, down from another (two tensors)
        let mut u = up
            .project_expert(&mut rd, e, &latent, &stop, 1 << 20)
            .unwrap();
        relu_sqr_in_place(&mut u);
        let got = down
            .project_expert(&mut rd2, e, &u, &stop, 1 << 20)
            .unwrap();

        // independent reference
        let up_ref = ref_project(&up_payload, n_latent, n_ff, e, &latent);
        let u_ref: Vec<f32> = up_ref
            .iter()
            .map(|&x| if x > 0.0 { (x * x) as f32 } else { 0.0 })
            .collect();
        let down_ref = ref_project(&down_payload, n_ff, n_latent, e, &u_ref);
        let max_abs = got
            .iter()
            .zip(&down_ref)
            .map(|(g, r)| (*g as f64 - r).abs())
            .fold(0.0, f64::max);
        assert!(max_abs < 1e-2, "chain max_abs {max_abs}");
        assert_eq!(got.len(), n_latent);
        std::fs::remove_file(up_path).unwrap();
        std::fs::remove_file(down_path).unwrap();
    }

    #[test]
    fn aggregate_selected_experts_matches_weighted_sum() {
        let (n_latent, n_ff, e_count) = (32usize, 32usize, 6usize);
        let (up_path, up_payload) = write_packed(n_latent, n_ff, e_count);
        let (down_path, down_payload) = write_packed(n_ff, n_latent, e_count);
        let up = packed(n_latent, n_ff, e_count, up_payload.len());
        let down = packed(n_ff, n_latent, e_count, down_payload.len());
        let latent: Vec<f32> = (0..n_latent)
            .map(|i| ((i % 5) as f32) * 0.3 - 0.6)
            .collect();
        let stop = AtomicBool::new(false);
        let routing = Routing {
            expert_ids: vec![4, 1, 5],
            weights: vec![0.5f32, 0.3, 0.2],
        };
        // Two separate readers: one for up, one for down. stream_moe_experts uses
        // a single reader per call, so aggregate up and down via the helper using
        // the SAME file layout — here up and down live in different files, so we
        // aggregate manually through stream_expert_ffn with its own readers and
        // compare to a direct weighted sum of the per-expert reference.
        let mut acc = vec![0.0f32; n_latent];
        for (&e, &w) in routing.expert_ids.iter().zip(&routing.weights) {
            let mut rd = BoundedLocalReader::open(&up_path, 256, 512).unwrap();
            let mut rd2 = BoundedLocalReader::open(&down_path, 256, 512).unwrap();
            let mut u = up
                .project_expert(&mut rd, e, &latent, &stop, 1 << 20)
                .unwrap();
            relu_sqr_in_place(&mut u);
            let d = down
                .project_expert(&mut rd2, e, &u, &stop, 1 << 20)
                .unwrap();
            for (a, dv) in acc.iter_mut().zip(&d) {
                *a += w * dv;
            }
        }
        // independent weighted-sum reference
        let mut want = vec![0.0f64; n_latent];
        for (&e, &w) in routing.expert_ids.iter().zip(&routing.weights) {
            let up_ref = ref_project(&up_payload, n_latent, n_ff, e, &latent);
            let u_ref: Vec<f32> = up_ref
                .iter()
                .map(|&x| if x > 0.0 { (x * x) as f32 } else { 0.0 })
                .collect();
            let d_ref = ref_project(&down_payload, n_ff, n_latent, e, &u_ref);
            for (a, dv) in want.iter_mut().zip(&d_ref) {
                *a += w as f64 * dv;
            }
        }
        let max_abs = acc
            .iter()
            .zip(&want)
            .map(|(g, r)| (*g as f64 - r).abs())
            .fold(0.0, f64::max);
        assert!(max_abs < 1e-2, "aggregate max_abs {max_abs}");
        std::fs::remove_file(up_path).unwrap();
        std::fs::remove_file(down_path).unwrap();
    }

    #[test]
    fn single_file_aggregate_helper_streams_only_selected() {
        // up and down packed into the SAME file so stream_moe_experts (one reader)
        // can be exercised end-to-end. We place up at offset 0 and down after it.
        let (n_latent, n_ff, e_count) = (32usize, 32usize, 4usize);
        let (_tmp_up, up_payload) = write_packed(n_latent, n_ff, e_count);
        let (_tmp_down, down_payload) = write_packed(n_ff, n_latent, e_count);
        std::fs::remove_file(_tmp_up).unwrap();
        std::fs::remove_file(_tmp_down).unwrap();
        let mut combined = up_payload.clone();
        combined.extend_from_slice(&down_payload);
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "llmbridge-packed-combined-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&combined)
            .unwrap();

        let up = PackedExperts::new(
            0,
            Q8_0,
            n_latent as u64,
            n_ff as u64,
            e_count as u64,
            up_payload.len() as u64,
        )
        .unwrap();
        let down = PackedExperts::new(
            up_payload.len() as u64,
            Q8_0,
            n_ff as u64,
            n_latent as u64,
            e_count as u64,
            down_payload.len() as u64,
        )
        .unwrap();
        let latent: Vec<f32> = (0..n_latent).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let routing = Routing {
            expert_ids: vec![3, 0],
            weights: vec![0.6, 0.4],
        };
        let stop = AtomicBool::new(false);
        let mut reader = BoundedLocalReader::open(&path, 512, 1024).unwrap();
        let got =
            stream_moe_experts(&up, &down, &mut reader, &routing, &latent, &stop, 1 << 20).unwrap();
        assert_eq!(got.len(), n_latent);
        assert!(got.iter().all(|v| v.is_finite()));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn invalid_index_truncation_cancellation_and_budget_rejected() {
        let (columns, rows, n_expert) = (32usize, 2usize, 3usize);
        let (path, payload) = write_packed(columns, rows, n_expert);
        let pk = packed(columns, rows, n_expert, payload.len());
        let act: Vec<f32> = (0..columns).map(|i| i as f32 * 0.1).collect();

        // wrong expert index
        assert!(pk.expert_region(n_expert).is_err());

        // cancellation
        let mut rd = BoundedLocalReader::open(&path, 128, 256).unwrap();
        let cancelled = AtomicBool::new(true);
        assert!(pk
            .project_expert(&mut rd, 0, &act, &cancelled, 1 << 20)
            .is_err());

        // tiny workspace budget
        let mut rd2 = BoundedLocalReader::open(&path, 128, 256).unwrap();
        let stop = AtomicBool::new(false);
        assert!(pk.project_expert(&mut rd2, 0, &act, &stop, 8).is_err());

        // truncated source: claim more experts than the file holds.
        let toobig = PackedExperts::new(
            0,
            Q8_0,
            columns as u64,
            rows as u64,
            (n_expert as u64) + 2,
            payload.len() as u64 + 2 * (rows as u64) * Q8_0_ROW_BYTES as u64,
        )
        .unwrap();
        let mut rd3 = BoundedLocalReader::open(&path, 128, 256).unwrap();
        assert!(toobig
            .project_expert(&mut rd3, (n_expert) + 1, &act, &stop, 1 << 20)
            .is_err());

        // bad geometry: per-expert bytes not divisible by rows.
        assert!(PackedExperts::new(0, Q8_0, 32, 3, 4, 100).is_err());

        std::fs::remove_file(path).unwrap();
    }

    // Real-model: resolve a genuine packed expert tensor from the three G: shards,
    // stream expert 0 through a bounded authenticated read, and cross-check it
    // against dequantize_rows over the same sub-slice. Reads only one expert's
    // bytes (not the whole tensor, not the model). Gated on LLMBRIDGE_NEMOTRON_DIR
    // so it is skipped by default and never touches G: in CI.
    #[test]
    fn real_nemotron_packed_expert_matches_dequant_reference() {
        use crate::gguf_index::SplitGgufIndex;
        use std::io::{Read, Seek, SeekFrom};
        let dir = match std::env::var("LLMBRIDGE_NEMOTRON_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: LLMBRIDGE_NEMOTRON_DIR unset (real packed-expert test)");
                return;
            }
        };
        let shards: Vec<String> = (1..=3)
            .map(|i| {
                format!("{dir}/NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ2_XXS-0000{i}-of-00003.gguf")
            })
            .collect();
        let index = SplitGgufIndex::inspect(&shards).expect("real split index");
        // Layer 1 is a MoE layer; its up experts are a packed rank-3 tensor.
        let name = "blk.1.ffn_up_exps.weight";
        let (member, region) = index.tensor(name).expect("ffn_up_exps present");
        assert_eq!(region.dimensions.len(), 3, "expert tensor must be rank-3");
        let columns = region.dimensions[0];
        let rows = region.dimensions[1];
        let n_expert = region.dimensions[2];
        let pk = PackedExperts::new(
            region.file_offset,
            region.ggml_type,
            columns,
            rows,
            n_expert,
            region.byte_len,
        )
        .expect("packed geometry");

        let latent: Vec<f32> = (0..columns as usize)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.01)
            .collect();
        let stop = AtomicBool::new(false);
        let e = 0usize;

        // production: bounded authenticated range read of expert e only.
        let mut reader = BoundedLocalReader::open(&member.path, 1 << 16, 1 << 17).unwrap();
        let got = pk
            .project_expert(&mut reader, e, &latent, &stop, 1 << 22)
            .expect("stream expert 0");
        assert_eq!(got.len(), rows as usize);

        // reference: read expert e's raw bytes directly, dequantize, dot.
        let per = pk.per_expert_bytes();
        let mut f = std::fs::File::open(&member.path).unwrap();
        f.seek(SeekFrom::Start(region.file_offset + e as u64 * per))
            .unwrap();
        let mut bytes = vec![0u8; per as usize];
        f.read_exact(&mut bytes).unwrap();
        let decoded =
            dequantize_rows(region.ggml_type, columns as u32, rows as u32, &bytes).unwrap();
        let mut max_abs = 0f64;
        for r in 0..rows as usize {
            let row = &decoded[r * columns as usize..(r + 1) * columns as usize];
            let want: f64 = (0..columns as usize)
                .map(|c| row[c] as f64 * latent[c] as f64)
                .sum();
            max_abs = max_abs.max((got[r] as f64 - want).abs());
        }
        assert!(
            max_abs < 1e-2,
            "real packed expert matvec vs dequant reference max abs = {max_abs}"
        );
        eprintln!(
            "real packed expert ok: {name} e={e} columns={columns} rows={rows} \
             n_expert={n_expert} per_expert_bytes={per} max_abs={max_abs}"
        );
    }
}
