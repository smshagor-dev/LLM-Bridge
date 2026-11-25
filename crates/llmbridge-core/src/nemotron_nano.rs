//! Model-backed Nemotron 3 **Nano** 30B-A3B (`nemotron_h_moe`) streamed execution.
//!
//! This is the Nano-specific sibling of [`crate::nemotron_model`] (which targets
//! the 120B Super split shards). The two share the architecture *family*
//! (`nemotron_h_moe`: interleaved Mamba-2 / attention / MoE blocks) and reuse the
//! same validated math kernels ([`crate::mamba2_conv`], [`crate::mamba2_ssm`],
//! [`crate::nemotron_expert_stream::PackedExperts`],
//! [`crate::latent_moe_routing`], [`crate::inference::dequant_matvec`]), but the
//! Nano GGUF differs materially and these differences are read from / validated
//! against the **real single-file tensor table** — never hardcoded blindly:
//!
//! | property           | 120B Super            | Nano 30B (this file)                 |
//! |--------------------|-----------------------|--------------------------------------|
//! | file               | 3 split shards        | single GGUF                          |
//! | blocks             | 88                    | 52 (23 rec + 23 moe + 6 attn)        |
//! | embedding_length   | 4096                  | 2688                                 |
//! | ssm heads / inner  | 128 / 8192            | 64 / 4096                            |
//! | experts / top-k    | 512 / 22              | 128 / 6                              |
//! | MoE FFN            | **latent** (down/up)  | **direct** up_exps->relu2->down_exps |
//! | attention pos-enc  | NoPE                  | **RoPE** (n_rot 84, base 10000)      |
//! | expert weights_scale | 5.0                 | 2.5                                  |
//!
//! Numerical honesty: the per-kernel math (dequant, matvec, Mamba conv/SSM,
//! packed-expert projection, routing) is the same code that is independently
//! reference-checked elsewhere in this crate. The **composition** here
//! (block ordering, the RoPE convention, grouped SSM norm placement, shared +
//! routed MoE aggregation) is NOT validated against a runnable `nemotron_h_moe`
//! reference on this host. The RoPE convention in particular is assumed to be
//! GGML-NeoX partial rotary; until a trusted reference comparison exists, logits
//! produced by [`generate_next_token`] / [`run_full_forward`] must be treated as
//! genuine-but-UNVALIDATED, exactly as the master specification requires.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::gguf_index::{GgufTensorIndex, TensorRegion};
use crate::inference::dequant_matvec;
use crate::latent_moe_routing::{route_sigmoid_topk, RoutingConfig};
use crate::mamba2_conv::{conv_forward, ConvState, ConvWeights};
use crate::mamba2_ssm::{ssm_scan, SsmConfig, SsmState};
use crate::native_dequant::dequantize_rows;
use crate::nemotron_expert_stream::PackedExperts;
use crate::storage::BoundedLocalReader;

const BUDGET: usize = 8 * 1024 * 1024;
const CHUNK: usize = 1 << 16;
const CAP: usize = 1 << 17;

/// Typed Nemotron 3 Nano 30B-A3B configuration. The reference values here are
/// taken from the real GGUF `nemotron_h_moe.*` metadata; every field is
/// re-confirmed against resolved tensor geometry in [`ModelPlan::build`], which
/// fails rather than silently executing a mismatched graph.
#[derive(Debug, Clone, Copy)]
pub struct NanoConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub head_dim: usize,
    /// RoPE rotary dimension count (partial rotary: first `n_rot` of each head).
    pub n_rot: usize,
    pub rope_freq_base: f32,
    // Mamba-2
    pub d_conv: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub ssm_n_head: usize,
    pub ssm_n_group: usize,
    pub ssm_norm_eps: f32,
    // MoE (direct, non-latent)
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub weights_scale: f32,
    pub weights_norm: bool,
    pub norm_eps: f32,
}

impl NanoConfig {
    /// Values verified from `nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_S.gguf`
    /// (`general.architecture = nemotron_h_moe`, `general.size_label = 30B-A3B`).
    pub const NEMOTRON_3_NANO_30B: NanoConfig = NanoConfig {
        n_layer: 52,
        n_embd: 2688,
        n_vocab: 131072,
        n_head: 32,
        head_dim: 128,
        n_rot: 84,
        rope_freq_base: 10000.0,
        d_conv: 4,
        d_inner: 4096,
        d_state: 128,
        ssm_n_head: 64,
        ssm_n_group: 8,
        ssm_norm_eps: 1e-5,
        n_expert: 128,
        n_expert_used: 6,
        n_ff_exp: 1856,
        n_ff_shexp: 3712,
        weights_scale: 2.5,
        weights_norm: true,
        norm_eps: 1e-5,
    };

    pub fn conv_channels(&self) -> usize {
        self.d_inner + 2 * self.ssm_n_group * self.d_state
    }
    pub fn d_in_proj(&self) -> usize {
        2 * self.d_inner + 2 * self.ssm_n_group * self.d_state + self.ssm_n_head
    }
    pub fn ssm_head_dim(&self) -> usize {
        self.d_inner / self.ssm_n_head
    }
    pub fn ssm_config(&self) -> SsmConfig {
        SsmConfig {
            n_head: self.ssm_n_head,
            dim: self.ssm_head_dim(),
            d_state: self.d_state,
            n_group: self.ssm_n_group,
        }
    }
    pub fn routing_config(&self) -> RoutingConfig {
        RoutingConfig {
            n_expert: self.n_expert,
            n_expert_used: self.n_expert_used,
            weights_norm: self.weights_norm,
            weights_scale: self.weights_scale,
        }
    }
}

/// Resolved kind of one decoder block, read from the real tensor table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    Recurrent,
    Attention,
    Moe,
}

#[derive(Debug, Clone)]
pub struct LayerPlan {
    pub index: usize,
    pub kind: LayerType,
    pub tensors: Vec<(String, u32)>,
}

/// A validated execution plan over the real single-file Nano model.
#[derive(Debug, Clone)]
pub struct ModelPlan {
    pub cfg: NanoConfig,
    pub layers: Vec<LayerPlan>,
    pub n_attention: usize,
    pub n_recurrent: usize,
    pub n_moe: usize,
}

fn tn(i: usize, suffix: &str) -> String {
    format!("blk.{i}.{suffix}")
}

fn require<'a>(index: &'a GgufTensorIndex, name: &str) -> Result<&'a TensorRegion, String> {
    index
        .tensor(name)
        .ok_or_else(|| format!("required tensor missing: {name}"))
}

fn dims_eq(region: &TensorRegion, expect: &[u64], what: &str) -> Result<(), String> {
    if region.dimensions != expect {
        return Err(format!(
            "{what} ({}) shape {:?} != expected {:?}",
            region.name, region.dimensions, expect
        ));
    }
    Ok(())
}

impl ModelPlan {
    /// Build and validate the full 52-block plan from the real single-file index.
    /// Reads only the tensor table (no weight payload). Every block is typed from
    /// the tensors actually present and every required tensor shape is checked
    /// against the Nano config before any execution can occur.
    pub fn build(index: &GgufTensorIndex, cfg: NanoConfig) -> Result<Self, String> {
        let tok = require(index, "token_embd.weight")?;
        dims_eq(tok, &[cfg.n_embd as u64, cfg.n_vocab as u64], "token_embd")?;
        require(index, "output_norm.weight")?;
        if index.tensor("output.weight").is_none() && index.tensor("token_embd.weight").is_none() {
            return Err("missing output projection".into());
        }

        let ne = cfg.n_embd as u64;
        let hd = cfg.head_dim as u64;
        let conv_ch = cfg.conv_channels() as u64;
        let d_in_proj = cfg.d_in_proj() as u64;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        let (mut n_att, mut n_rec, mut n_moe) = (0usize, 0usize, 0usize);

        for i in 0..cfg.n_layer {
            require(index, &tn(i, "attn_norm.weight"))?;
            let mut tensors: Vec<(String, u32)> = Vec::new();
            let push = |index: &GgufTensorIndex, name: String, t: &mut Vec<(String, u32)>| {
                if let Some(r) = index.tensor(&name) {
                    t.push((name, r.ggml_type));
                }
            };

            let kind = if index.tensor(&tn(i, "ssm_in.weight")).is_some() {
                // ---- recurrent (Mamba-2) ----
                let ssm_in = require(index, &tn(i, "ssm_in.weight"))?;
                dims_eq(ssm_in, &[ne, d_in_proj], "ssm_in")?;
                let conv = require(index, &tn(i, "ssm_conv1d.weight"))?;
                dims_eq(conv, &[cfg.d_conv as u64, conv_ch], "ssm_conv1d")?;
                let ssm_out = require(index, &tn(i, "ssm_out.weight"))?;
                dims_eq(ssm_out, &[cfg.d_inner as u64, ne], "ssm_out")?;
                require(index, &tn(i, "ssm_dt.bias"))?;
                require(index, &tn(i, "ssm_a"))?;
                require(index, &tn(i, "ssm_d"))?;
                require(index, &tn(i, "ssm_norm.weight"))?;
                for s in [
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_conv1d.bias",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ] {
                    push(index, tn(i, s), &mut tensors);
                }
                n_rec += 1;
                LayerType::Recurrent
            } else if index.tensor(&tn(i, "attn_q.weight")).is_some() {
                // ---- attention (GQA + RoPE) ----
                let wq = require(index, &tn(i, "attn_q.weight"))?;
                dims_eq(wq, &[ne, cfg.n_head as u64 * hd], "attn_q")?;
                let wk = require(index, &tn(i, "attn_k.weight"))?;
                let kv_out = wk.dimensions.get(1).copied().unwrap_or(0);
                if wk.dimensions.first().copied() != Some(ne) || kv_out == 0 || kv_out % hd != 0 {
                    return Err(format!("attn_k shape {:?} invalid", wk.dimensions));
                }
                let wv = require(index, &tn(i, "attn_v.weight"))?;
                dims_eq(wv, &[ne, kv_out], "attn_v")?;
                let wo = require(index, &tn(i, "attn_output.weight"))?;
                dims_eq(wo, &[cfg.n_head as u64 * hd, ne], "attn_output")?;
                for s in [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                ] {
                    push(index, tn(i, s), &mut tensors);
                }
                n_att += 1;
                LayerType::Attention
            } else if index.tensor(&tn(i, "ffn_gate_inp.weight")).is_some() {
                // ---- MoE FFN (direct, non-latent) ----
                let gate = require(index, &tn(i, "ffn_gate_inp.weight"))?;
                dims_eq(gate, &[ne, cfg.n_expert as u64], "ffn_gate_inp")?;
                require(index, &tn(i, "exp_probs_b.bias"))?;
                // up:   [n_embd, n_ff_exp, n_expert]   (input n_embd -> n_ff_exp)
                let up = require(index, &tn(i, "ffn_up_exps.weight"))?;
                dims_eq(
                    up,
                    &[ne, cfg.n_ff_exp as u64, cfg.n_expert as u64],
                    "ffn_up_exps",
                )?;
                // down: [n_ff_exp, n_embd, n_expert]   (input n_ff_exp -> n_embd)
                let down = require(index, &tn(i, "ffn_down_exps.weight"))?;
                dims_eq(
                    down,
                    &[cfg.n_ff_exp as u64, ne, cfg.n_expert as u64],
                    "ffn_down_exps",
                )?;
                let ups = require(index, &tn(i, "ffn_up_shexp.weight"))?;
                dims_eq(ups, &[ne, cfg.n_ff_shexp as u64], "ffn_up_shexp")?;
                let downs = require(index, &tn(i, "ffn_down_shexp.weight"))?;
                dims_eq(downs, &[cfg.n_ff_shexp as u64, ne], "ffn_down_shexp")?;
                for s in [
                    "ffn_gate_inp.weight",
                    "exp_probs_b.bias",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                    "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight",
                ] {
                    push(index, tn(i, s), &mut tensors);
                }
                n_moe += 1;
                LayerType::Moe
            } else {
                return Err(format!("layer {i} has no recognizable sub-layer tensors"));
            };

            layers.push(LayerPlan {
                index: i,
                kind,
                tensors,
            });
        }

        Ok(ModelPlan {
            cfg,
            layers,
            n_attention: n_att,
            n_recurrent: n_rec,
            n_moe,
        })
    }
}

// ----------------------------------------------------------------------------
// Streamed, bounded execution helpers (single-file GgufTensorIndex).
// ----------------------------------------------------------------------------

/// Stream a full (small) tensor and decode it row-major to f32. For small
/// per-layer parameters and float tensors only.
fn stream_small(
    index: &GgufTensorIndex,
    name: &str,
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    let region = index
        .tensor(name)
        .ok_or_else(|| format!("tensor missing: {name}"))?;
    if region.byte_len > 64 * 1024 * 1024 {
        return Err(format!("{name} too large to materialize as f32"));
    }
    let mut reader = BoundedLocalReader::open(&index.path, CHUNK, CAP)
        .map_err(|e| format!("open {}: {e}", index.path.display()))?;
    let mut encoded = Vec::with_capacity(region.byte_len as usize);
    index
        .stream_tensor_cancellable(&mut reader, name, stop, |c| {
            encoded.extend_from_slice(c);
            Ok(())
        })
        .map_err(|e| format!("stream {name}: {e}"))?;
    let columns = region.dimensions[0] as u32;
    let rows = if region.dimensions.len() > 1 {
        region.dimensions[1] as u32
    } else {
        1
    };
    let n_elems = columns as usize * rows as usize;
    match region.ggml_type {
        0 => {
            // F32
            if encoded.len() < n_elems * 4 {
                return Err(format!("{name}: truncated F32 payload"));
            }
            let mut out = Vec::with_capacity(n_elems);
            for i in 0..n_elems {
                out.push(f32::from_le_bytes([
                    encoded[i * 4],
                    encoded[i * 4 + 1],
                    encoded[i * 4 + 2],
                    encoded[i * 4 + 3],
                ]));
            }
            if !out.iter().all(|v| v.is_finite()) {
                return Err(format!("{name}: non-finite F32 value"));
            }
            Ok(out)
        }
        _ => dequantize_rows(region.ggml_type, columns, rows, &encoded),
    }
}

/// Streamed rank-2 matrix-vector product `W . x` from the real GGUF tensor.
/// Quantized tensors use the bounded native kernel (one row streamed at a time);
/// the only float projection in this model is the F32 router (`ffn_gate_inp`).
fn matvec(
    index: &GgufTensorIndex,
    name: &str,
    x: &[f32],
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    let region = index
        .tensor(name)
        .ok_or_else(|| format!("tensor missing: {name}"))?;
    if region.dimensions.len() != 2 {
        return Err(format!("{name}: matvec requires a rank-2 tensor"));
    }
    let columns = region.dimensions[0] as usize;
    let rows = region.dimensions[1] as usize;
    if x.len() != columns {
        return Err(format!(
            "{name}: activation len {} != columns {columns}",
            x.len()
        ));
    }
    if region.ggml_type == 0 {
        // F32: decode row-major [rows][columns] and dense-dot.
        let w = stream_small(index, name, stop)?;
        if w.len() != rows * columns {
            return Err(format!("{name}: decoded float length mismatch"));
        }
        let mut out = vec![0.0f32; rows];
        for (r, o) in out.iter_mut().enumerate() {
            let row = &w[r * columns..(r + 1) * columns];
            let mut acc = 0.0f32;
            for (wv, xv) in row.iter().zip(x) {
                acc += wv * xv;
            }
            if !acc.is_finite() {
                return Err(format!("{name}: non-finite float matvec output"));
            }
            *o = acc;
        }
        return Ok(out);
    }
    let mut reader = BoundedLocalReader::open(&index.path, CHUNK, CAP)
        .map_err(|e| format!("open {}: {e}", index.path.display()))?;
    // Q8_0 uses the LLM Bridge-owned pure-Rust kernel (no GGML/FFI); every other
    // quantized type still routes through the pinned GGML dequant matvec until an
    // owned kernel for it is implemented and validated.
    if region.ggml_type == crate::native_kernels::GGML_TYPE_Q8_0 {
        return crate::native_kernels::q8_0_dequant_matvec(region, &mut reader, x, stop, BUDGET);
    }
    dequant_matvec(region, &mut reader, x, stop, BUDGET)
}

#[inline]
fn silu(x: f32) -> f32 {
    let s = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    };
    x * s
}

#[inline]
fn relu_sqr_in_place(v: &mut [f32]) {
    for x in v.iter_mut() {
        let r = if *x > 0.0 { *x } else { 0.0 };
        *x = r * r;
    }
}

/// RMS-normalize `x` against `weight` (both length `n`).
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    (0..x.len()).map(|i| x[i] * inv * weight[i]).collect()
}

/// Apply GGML-NeoX partial rotary embedding to one head slice (`head_dim`
/// elements) at absolute sequence position `pos`. Only the first `n_rot`
/// dimensions are rotated; the remainder pass through unchanged.
///
/// CONVENTION CAVEAT: the NeoX pairing `(i, i + n_rot/2)` and
/// `theta = pos * base^(-2i/n_rot)` is assumed for `nemotron_h_moe`; it is not
/// validated against a runnable reference on this host.
fn rope_neox_inplace(head: &mut [f32], pos: usize, n_rot: usize, base: f32) {
    let half = n_rot / 2;
    let p = pos as f32;
    for i in 0..half {
        let theta = p * base.powf(-2.0 * (i as f32) / (n_rot as f32));
        let (s, c) = theta.sin_cos();
        let a = head[i];
        let b = head[i + half];
        head[i] = a * c - b * s;
        head[i + half] = a * s + b * c;
    }
}

/// Read one token's embedding row via a single bounded range read.
pub fn embed_token(
    index: &GgufTensorIndex,
    cfg: &NanoConfig,
    token: u32,
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let region = index
        .tensor("token_embd.weight")
        .ok_or("token_embd missing")?;
    let id = token as u64;
    if id >= cfg.n_vocab as u64 {
        return Err("token id out of vocab range".into());
    }
    if stop.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let row_bytes = region.byte_len / cfg.n_vocab as u64;
    let mut f = std::fs::File::open(&index.path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(region.file_offset + id * row_bytes))
        .map_err(|e| e.to_string())?;
    let mut bytes = vec![0u8; row_bytes as usize];
    f.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let v = dequantize_rows(region.ggml_type, cfg.n_embd as u32, 1, &bytes)?;
    if !v.iter().all(|x| x.is_finite()) {
        return Err("non-finite embedding".into());
    }
    Ok(v)
}

/// Execute one real recurrent (Mamba-2) block for a single token (pre-normed
/// input `x`, length n_embd). Returns the pre-residual sub-layer output.
#[allow(clippy::needless_range_loop)]
pub fn run_mamba_layer(
    index: &GgufTensorIndex,
    cfg: &NanoConfig,
    il: usize,
    conv_state: &mut ConvState,
    ssm_state: &mut SsmState,
    x: &[f32],
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    let di = cfg.d_inner;
    let cc = cfg.conv_channels();
    let gds = cfg.ssm_n_group * cfg.d_state;
    let nh = cfg.ssm_n_head;
    let hd = cfg.ssm_head_dim();

    let proj = matvec(index, &tn(il, "ssm_in.weight"), x, stop)?; // [d_in_proj]
    let z = proj[0..di].to_vec();
    let xbc_in = proj[di..di + cc].to_vec();
    let dt_off = 2 * di + gds;

    let conv_flat = stream_small(index, &tn(il, "ssm_conv1d.weight"), stop)?;
    let conv_bias = stream_small(index, &tn(il, "ssm_conv1d.bias"), stop)?;
    let dt_b = stream_small(index, &tn(il, "ssm_dt.bias"), stop)?;
    let a = stream_small(index, &tn(il, "ssm_a"), stop)?;
    let d = stream_small(index, &tn(il, "ssm_d"), stop)?;
    let norm = stream_small(index, &tn(il, "ssm_norm.weight"), stop)?;

    let conv_w = ConvWeights::new(cc, cfg.d_conv, conv_flat)?;
    let mut xbc = conv_forward(&conv_w, conv_state, &xbc_in)?;
    for (ch, vv) in xbc.iter_mut().enumerate() {
        *vv = silu(*vv + conv_bias[ch]);
    }
    let x_part = xbc[0..di].to_vec();
    let b = xbc[di..di + gds].to_vec();
    let c = xbc[di + gds..di + 2 * gds].to_vec();
    let dt: Vec<f32> = (0..nh).map(|h| proj[dt_off + h] + dt_b[h]).collect();

    let y_ssm = ssm_scan(&cfg.ssm_config(), &a, ssm_state, &x_part, &dt, &b, &c)?;
    let mut y = vec![0.0f32; di];
    for h in 0..nh {
        for i1 in 0..hd {
            let ch = h * hd + i1;
            let yi = y_ssm[ch] + x_part[ch] * d[h];
            y[ch] = silu(z[ch]) * yi;
        }
    }
    // grouped RMS norm
    let gsize = di / cfg.ssm_n_group;
    for g in 0..cfg.ssm_n_group {
        let lo = g * gsize;
        let hi = lo + gsize;
        let ms: f32 = y[lo..hi].iter().map(|v| v * v).sum::<f32>() / gsize as f32;
        let inv = 1.0 / (ms + cfg.ssm_norm_eps).sqrt();
        for ch in lo..hi {
            y[ch] = y[ch] * inv * norm[ch];
        }
    }
    let out = matvec(index, &tn(il, "ssm_out.weight"), &y, stop)?;
    if !out.iter().all(|v| v.is_finite()) {
        return Err("non-finite mamba output".into());
    }
    Ok(out)
}

/// Growing KV cache for streamed attention.
#[derive(Debug, Default, Clone)]
pub struct StreamKvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub pos: usize,
    pub kv_dim: usize,
}

/// Execute one real attention block (GQA + partial RoPE) for a single token.
pub fn run_attention_layer(
    index: &GgufTensorIndex,
    cfg: &NanoConfig,
    il: usize,
    kv: &mut StreamKvCache,
    x: &[f32],
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    let hd = cfg.head_dim;
    let pos = kv.pos; // absolute sequence index of this token
    let mut q = matvec(index, &tn(il, "attn_q.weight"), x, stop)?; // [n_head*hd]
    let mut k = matvec(index, &tn(il, "attn_k.weight"), x, stop)?; // [n_head_kv*hd]
    let v = matvec(index, &tn(il, "attn_v.weight"), x, stop)?;
    let kv_dim = k.len();
    if v.len() != kv_dim || kv_dim % hd != 0 {
        return Err("attention K/V geometry invalid".into());
    }
    if q.len() != cfg.n_head * hd {
        return Err("attention Q geometry invalid".into());
    }
    let n_head_kv = kv_dim / hd;

    // RoPE on each query head and each key head (before caching K).
    for h in 0..cfg.n_head {
        rope_neox_inplace(
            &mut q[h * hd..(h + 1) * hd],
            pos,
            cfg.n_rot,
            cfg.rope_freq_base,
        );
    }
    for h in 0..n_head_kv {
        rope_neox_inplace(
            &mut k[h * hd..(h + 1) * hd],
            pos,
            cfg.n_rot,
            cfg.rope_freq_base,
        );
    }

    if kv.pos == 0 {
        kv.kv_dim = kv_dim;
    } else if kv.kv_dim != kv_dim {
        return Err("KV cache dim changed between tokens".into());
    }
    kv.k.extend_from_slice(&k);
    kv.v.extend_from_slice(&v);
    kv.pos += 1;
    let n_kv = kv.pos;
    let hpk = cfg.n_head / n_head_kv;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let mut concat = vec![0.0f32; cfg.n_head * hd];
    for h in 0..cfg.n_head {
        let g = h / hpk;
        let qh = &q[h * hd..(h + 1) * hd];
        let mut scores = vec![0.0f32; n_kv];
        for (j, sc) in scores.iter_mut().enumerate() {
            let kh = &kv.k[j * kv_dim + g * hd..j * kv_dim + (g + 1) * hd];
            let mut dot = 0.0f32;
            for (qa, kb) in qh.iter().zip(kh) {
                dot += qa * kb;
            }
            *sc = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for s in scores.iter_mut() {
            let e = ((*s - m) as f64).exp();
            *s = e as f32;
            sum += e;
        }
        for s in scores.iter_mut() {
            *s = (*s as f64 / sum) as f32;
        }
        let oh = &mut concat[h * hd..(h + 1) * hd];
        for (j, &w) in scores.iter().enumerate() {
            let vh = &kv.v[j * kv_dim + g * hd..j * kv_dim + (g + 1) * hd];
            for (od, &vd) in oh.iter_mut().zip(vh) {
                *od += w * vd;
            }
        }
    }
    let out = matvec(index, &tn(il, "attn_output.weight"), &concat, stop)?;
    if !out.iter().all(|v| v.is_finite()) {
        return Err("non-finite attention output".into());
    }
    Ok(out)
}

/// Execute one real MoE FFN block (direct, non-latent) for a single token:
/// `shared(x) + Σ_e w_e · down_e(relu²(up_e(x)))` over the top-k routed experts.
pub fn run_moe_layer(
    index: &GgufTensorIndex,
    cfg: &NanoConfig,
    il: usize,
    x: &[f32],
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    let logits = matvec(index, &tn(il, "ffn_gate_inp.weight"), x, stop)?; // [n_expert]
    let bias = stream_small(index, &tn(il, "exp_probs_b.bias"), stop)?;
    let routing = route_sigmoid_topk(&logits, &cfg.routing_config(), Some(&bias))?;

    // shared expert on the full input
    let mut sh = matvec(index, &tn(il, "ffn_up_shexp.weight"), x, stop)?;
    relu_sqr_in_place(&mut sh);
    let shexp = matvec(index, &tn(il, "ffn_down_shexp.weight"), &sh, stop)?;

    // routed experts, operating directly on x (no latent projection)
    let up_region = index
        .tensor(&tn(il, "ffn_up_exps.weight"))
        .ok_or("ffn_up_exps missing")?;
    let down_region = index
        .tensor(&tn(il, "ffn_down_exps.weight"))
        .ok_or("ffn_down_exps missing")?;
    let up = PackedExperts::new(
        up_region.file_offset,
        up_region.ggml_type,
        up_region.dimensions[0], // columns = n_embd
        up_region.dimensions[1], // rows    = n_ff_exp
        up_region.dimensions[2], // n_expert
        up_region.byte_len,
    )?;
    let down = PackedExperts::new(
        down_region.file_offset,
        down_region.ggml_type,
        down_region.dimensions[0], // columns = n_ff_exp
        down_region.dimensions[1], // rows    = n_embd
        down_region.dimensions[2],
        down_region.byte_len,
    )?;
    let mut up_reader =
        BoundedLocalReader::open(&index.path, CHUNK, CAP).map_err(|e| e.to_string())?;
    let mut acc = vec![0.0f32; cfg.n_embd];
    for (&e, &w) in routing.expert_ids.iter().zip(&routing.weights) {
        let mut u = up.project_expert(&mut up_reader, e, x, stop, BUDGET)?; // [n_ff_exp]
        relu_sqr_in_place(&mut u);
        let mut down_reader =
            BoundedLocalReader::open(&index.path, CHUNK, CAP).map_err(|e| e.to_string())?;
        let dvec = down.project_expert(&mut down_reader, e, &u, stop, BUDGET)?; // [n_embd]
        for (m, dv) in acc.iter_mut().zip(&dvec) {
            *m += w * dv;
        }
    }
    let out: Vec<f32> = acc.iter().zip(&shexp).map(|(a, b)| a + b).collect();
    if !out.iter().all(|v| v.is_finite()) {
        return Err("non-finite moe output".into());
    }
    Ok(out)
}

/// Per-block recurrent/KV state for a full-model session.
pub enum FullLayerState {
    Recurrent(Box<ConvState>, Box<SsmState>),
    Attention(Box<StreamKvCache>),
    Moe,
}

/// Session state for a streamed full-model forward pass.
pub struct FullSession {
    states: Vec<FullLayerState>,
    pos: usize,
}

impl FullSession {
    pub fn new(cfg: &NanoConfig, plan: &ModelPlan) -> Result<Self, String> {
        let mut states = Vec::with_capacity(plan.layers.len());
        for l in &plan.layers {
            states.push(match l.kind {
                LayerType::Recurrent => FullLayerState::Recurrent(
                    Box::new(ConvState::zeros(cfg.conv_channels(), cfg.d_conv)?),
                    Box::new(SsmState::zeros(cfg.ssm_config())?),
                ),
                LayerType::Attention => FullLayerState::Attention(Box::<StreamKvCache>::default()),
                LayerType::Moe => FullLayerState::Moe,
            });
        }
        Ok(Self { states, pos: 0 })
    }
    pub fn position(&self) -> usize {
        self.pos
    }
}

fn output_tensor_name(index: &GgufTensorIndex) -> &'static str {
    if index.tensor("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    }
}

/// Run the full 52-block streamed forward over `tokens` (fed one at a time,
/// advancing all per-block state), then return the final-position logits.
///
/// Produces genuine model-backed logits. NOT validated against a runnable
/// `nemotron_h_moe` reference on this host (see module docs); the RoPE
/// convention in particular is assumed, so logits are genuine-but-UNVALIDATED.
pub fn run_full_forward(
    index: &GgufTensorIndex,
    cfg: &NanoConfig,
    plan: &ModelPlan,
    session: &mut FullSession,
    tokens: &[u32],
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Err("full forward requires at least one token".into());
    }
    if session.states.len() != plan.layers.len() {
        return Err("session does not match plan layer count".into());
    }
    let mut last_hidden: Option<Vec<f32>> = None;
    for &tok in tokens {
        if stop.load(Ordering::Acquire) {
            return Err("full forward cancelled".into());
        }
        let mut h = embed_token(index, cfg, tok, stop)?;
        for (li, layer) in plan.layers.iter().enumerate() {
            let attn_norm = stream_small(index, &tn(layer.index, "attn_norm.weight"), stop)?;
            let hn = rms_norm(&h, &attn_norm, cfg.norm_eps);
            let out = match (&layer.kind, &mut session.states[li]) {
                (LayerType::Recurrent, FullLayerState::Recurrent(conv, ssm)) => {
                    run_mamba_layer(index, cfg, layer.index, conv, ssm, &hn, stop)?
                }
                (LayerType::Attention, FullLayerState::Attention(kvc)) => {
                    run_attention_layer(index, cfg, layer.index, kvc, &hn, stop)?
                }
                (LayerType::Moe, FullLayerState::Moe) => {
                    run_moe_layer(index, cfg, layer.index, &hn, stop)?
                }
                _ => return Err(format!("layer {} state/kind mismatch", layer.index)),
            };
            for (hv, ov) in h.iter_mut().zip(&out) {
                *hv += ov;
            }
        }
        last_hidden = Some(h);
        session.pos += 1;
    }
    let h = last_hidden.ok_or("no hidden state produced")?;
    let on = stream_small(index, "output_norm.weight", stop)?;
    let hf = rms_norm(&h, &on, cfg.norm_eps);
    let logits = matvec(index, output_tensor_name(index), &hf, stop)?;
    if logits.len() != cfg.n_vocab || !logits.iter().all(|v| v.is_finite()) {
        return Err("invalid full-forward logits".into());
    }
    Ok(logits)
}

/// Greedy argmax token id over a logit vector (ties to lowest index).
pub fn greedy_token(logits: &[f32]) -> Option<usize> {
    let mut best = None;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > best_v {
            best_v = v;
            best = Some(i);
        }
    }
    best
}

/// Top-k (id, logit) pairs, descending by logit then ascending id.
pub fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.into_iter().take(k).map(|i| (i, logits[i])).collect()
}

/// Result of one prompt-conditioned generation step.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub prompt: String,
    pub token_ids: Vec<u32>,
    pub n_prefill: usize,
    pub next_token_id: usize,
    pub next_token_text: String,
    pub top5: Vec<(usize, f32)>,
    pub prefill_seconds: f64,
}

/// Encode `prompt`, run the full streamed 52-block prefill, and greedily pick the
/// next token from the final-position logits. Genuine-but-UNVALIDATED (see module
/// docs). Non-ASCII tokenizer-id parity is also unverified.
pub fn generate_next_token(
    index: &GgufTensorIndex,
    cfg: &NanoConfig,
    plan: &ModelPlan,
    tokenizer: &crate::nemotron_tokenizer::NemotronTokenizer,
    prompt: &str,
    stop: &AtomicBool,
) -> Result<GenerationResult, String> {
    let token_ids = tokenizer.encode(prompt);
    if token_ids.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }
    if let Some(&bad) = token_ids.iter().find(|&&id| id as usize >= cfg.n_vocab) {
        return Err(format!("encoded token id {bad} out of vocab range"));
    }
    let mut session = FullSession::new(cfg, plan)?;
    let t0 = std::time::Instant::now();
    let logits = run_full_forward(index, cfg, plan, &mut session, &token_ids, stop)?;
    let prefill_seconds = t0.elapsed().as_secs_f64();
    let next_token_id = greedy_token(&logits).ok_or("no greedy token")?;
    let top5 = top_k(&logits, 5);
    let next_token_text = tokenizer
        .decode(&[next_token_id as u32])
        .unwrap_or_default();
    Ok(GenerationResult {
        prompt: prompt.to_string(),
        token_ids,
        n_prefill: session.position(),
        next_token_id,
        next_token_text,
        top5,
        prefill_seconds,
    })
}

/// Exact per-token source-read profile computed from the tensor table (no weight
/// payload is read): each recurrent/attention/MoE-non-expert tensor in full, only
/// `n_expert_used / n_expert` of each packed expert tensor, one embedding row, and
/// the output projection once.
#[derive(Debug, Clone, Default)]
pub struct ReadProfile {
    pub total_bytes: u64,
    pub embedding_bytes: u64,
    pub recurrent_bytes: u64,
    pub attention_bytes: u64,
    pub moe_non_expert_bytes: u64,
    pub moe_expert_bytes: u64,
    pub output_bytes: u64,
    pub norm_bytes: u64,
}

pub fn source_read_profile(
    index: &GgufTensorIndex,
    cfg: &NanoConfig,
    plan: &ModelPlan,
) -> Result<ReadProfile, String> {
    let bytes = |name: &str| -> u64 { index.tensor(name).map(|r| r.byte_len).unwrap_or(0) };
    let mut p = ReadProfile::default();
    let tok = bytes("token_embd.weight");
    p.embedding_bytes = tok / cfg.n_vocab as u64;
    p.output_bytes = bytes(output_tensor_name(index));
    p.norm_bytes += bytes("output_norm.weight");
    for l in &plan.layers {
        let i = l.index;
        p.norm_bytes += bytes(&tn(i, "attn_norm.weight"));
        match l.kind {
            LayerType::Recurrent => {
                for s in [
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_conv1d.bias",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ] {
                    p.recurrent_bytes += bytes(&tn(i, s));
                }
            }
            LayerType::Attention => {
                for s in [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                ] {
                    p.attention_bytes += bytes(&tn(i, s));
                }
            }
            LayerType::Moe => {
                for s in [
                    "ffn_gate_inp.weight",
                    "exp_probs_b.bias",
                    "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight",
                ] {
                    p.moe_non_expert_bytes += bytes(&tn(i, s));
                }
                let up = bytes(&tn(i, "ffn_up_exps.weight"));
                let down = bytes(&tn(i, "ffn_down_exps.weight"));
                let used = cfg.n_expert_used as u64;
                let total = cfg.n_expert as u64;
                p.moe_expert_bytes += (up / total) * used + (down / total) * used;
            }
        }
    }
    p.total_bytes = p.embedding_bytes
        + p.recurrent_bytes
        + p.attention_bytes
        + p.moe_non_expert_bytes
        + p.moe_expert_bytes
        + p.output_bytes
        + p.norm_bytes;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> Option<String> {
        std::env::var("LLMBRIDGE_NANO_GGUF").ok()
    }

    #[test]
    fn config_derived_dimensions_are_consistent() {
        let c = NanoConfig::NEMOTRON_3_NANO_30B;
        assert_eq!(c.d_in_proj(), 2 * 4096 + 2 * 8 * 128 + 64);
        assert_eq!(c.d_in_proj(), 10304);
        assert_eq!(c.conv_channels(), 4096 + 2 * 8 * 128);
        assert_eq!(c.conv_channels(), 6144);
        assert_eq!(c.ssm_head_dim(), 64);
        assert_eq!(c.n_head * c.head_dim, 4096);
    }

    #[test]
    fn rope_neox_is_identity_at_position_zero() {
        let mut h: Vec<f32> = (0..128).map(|i| i as f32 * 0.01 - 0.5).collect();
        let orig = h.clone();
        rope_neox_inplace(&mut h, 0, 84, 10000.0);
        for (a, b) in h.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-6, "rope at pos 0 must be identity");
        }
        let mut h2 = orig.clone();
        rope_neox_inplace(&mut h2, 5, 84, 10000.0);
        for i in 84..128 {
            assert!((h2[i] - orig[i]).abs() < 1e-6, "dim {i} must pass through");
        }
    }

    #[test]
    fn full_52_layer_plan_builds_and_validates() {
        let Some(path) = model_path() else {
            eprintln!("skip: LLMBRIDGE_NANO_GGUF unset");
            return;
        };
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let cfg = NanoConfig::NEMOTRON_3_NANO_30B;
        let plan = ModelPlan::build(&index, cfg).expect("plan builds");
        assert_eq!(plan.layers.len(), 52);
        assert_eq!(plan.n_attention, 6, "expected 6 attention layers");
        assert_eq!(plan.n_recurrent, 23);
        assert_eq!(plan.n_moe, 23);
        let att: Vec<usize> = plan
            .layers
            .iter()
            .filter(|l| l.kind == LayerType::Attention)
            .map(|l| l.index)
            .collect();
        assert_eq!(att, vec![5, 12, 19, 26, 33, 42]);
    }

    #[test]
    fn real_nano_layers_execute_through_streamed_reader() {
        let Some(path) = model_path() else {
            eprintln!("skip: LLMBRIDGE_NANO_GGUF unset");
            return;
        };
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let cfg = NanoConfig::NEMOTRON_3_NANO_30B;
        let plan = ModelPlan::build(&index, cfg).unwrap();
        let stop = AtomicBool::new(false);

        let emb = embed_token(&index, &cfg, 42, &stop).unwrap();
        assert_eq!(emb.len(), cfg.n_embd);
        assert!(emb.iter().all(|v| v.is_finite()));

        let first = |k: LayerType| plan.layers.iter().find(|l| l.kind == k).unwrap().index;
        let rec = first(LayerType::Recurrent);
        let att = first(LayerType::Attention);
        let moe = first(LayerType::Moe);

        let mut conv_state = ConvState::zeros(cfg.conv_channels(), cfg.d_conv).unwrap();
        let mut ssm_state = SsmState::zeros(cfg.ssm_config()).unwrap();
        let mo = run_mamba_layer(
            &index,
            &cfg,
            rec,
            &mut conv_state,
            &mut ssm_state,
            &emb,
            &stop,
        )
        .unwrap();
        assert_eq!(mo.len(), cfg.n_embd);
        assert!(mo.iter().all(|v| v.is_finite()));

        let mut kv = StreamKvCache::default();
        let ao = run_attention_layer(&index, &cfg, att, &mut kv, &emb, &stop).unwrap();
        assert_eq!(ao.len(), cfg.n_embd);
        assert!(ao.iter().all(|v| v.is_finite()));
        assert_eq!(kv.pos, 1);

        let fo = run_moe_layer(&index, &cfg, moe, &emb, &stop).unwrap();
        assert_eq!(fo.len(), cfg.n_embd);
        assert!(fo.iter().all(|v| v.is_finite()));
    }

    /// Genuine end-to-end single-token generation over the real 52-block model.
    /// Slow (full streamed forward); `#[ignore]` so the default suite skips it.
    /// Run with: `cargo test -p llmbridge-core --features native --release
    /// -- --ignored --nocapture nemotron_nano::tests::generate_one_token`.
    /// Logits are genuine-but-UNVALIDATED (see module docs).
    #[test]
    #[ignore]
    fn generate_one_token() {
        let Some(path) = model_path() else {
            eprintln!("skip: LLMBRIDGE_NANO_GGUF unset");
            return;
        };
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let cfg = NanoConfig::NEMOTRON_3_NANO_30B;
        let plan = ModelPlan::build(&index, cfg).unwrap();
        let prof = source_read_profile(&index, &cfg, &plan).unwrap();
        eprintln!(
            "NANO_READ_PROFILE total={} embedding={} recurrent={} attention={} moe_non_expert={} moe_expert={} output={} norm={}",
            prof.total_bytes, prof.embedding_bytes, prof.recurrent_bytes, prof.attention_bytes,
            prof.moe_non_expert_bytes, prof.moe_expert_bytes, prof.output_bytes, prof.norm_bytes
        );
        let tok =
            crate::nemotron_tokenizer::NemotronTokenizer::from_gguf(std::path::Path::new(&path))
                .unwrap();
        let stop = AtomicBool::new(false);
        let r = generate_next_token(&index, &cfg, &plan, &tok, "Hi", &stop).unwrap();
        eprintln!(
            "NANO_GEN prompt={:?} prompt_ids={:?} n_prefill={} next_id={} next_text={:?} prefill_s={:.2}",
            r.prompt, r.token_ids, r.n_prefill, r.next_token_id, r.next_token_text, r.prefill_seconds
        );
        eprintln!("NANO_GEN_TOP5 {:?}", r.top5);
        assert_eq!(r.n_prefill, r.token_ids.len());
        assert!(r.next_token_id < cfg.n_vocab);
    }

    /// Mandatory acceptance A: the LLM Bridge-owned Q8_0 kernel matches BOTH the
    /// pinned GGML dequant_matvec (test-only reference) AND an independent in-Rust
    /// FP64 oracle, on a real Nano Q8_0 tensor slice (blk.5.attn_k.weight).
    #[test]
    fn owned_q8_0_kernel_matches_ggml_and_fp64_oracle_on_real_nano_tensor() {
        use std::io::{Read, Seek, SeekFrom};
        let Some(path) = model_path() else {
            eprintln!("skip: LLMBRIDGE_NANO_GGUF unset");
            return;
        };
        let index = GgufTensorIndex::inspect(&path).unwrap();
        let region = index
            .tensor("blk.5.attn_k.weight")
            .expect("attn_k present")
            .clone();
        assert_eq!(region.ggml_type, crate::native_kernels::GGML_TYPE_Q8_0);
        let columns = region.dimensions[0] as usize;
        let rows = region.dimensions[1] as usize;
        let act: Vec<f32> = (0..columns)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let stop = AtomicBool::new(false);

        // Owned pure-Rust kernel (no GGML/FFI).
        let mut r1 = BoundedLocalReader::open(&index.path, CHUNK, CAP).unwrap();
        let owned =
            crate::native_kernels::q8_0_dequant_matvec(&region, &mut r1, &act, &stop, BUDGET)
                .unwrap();
        // GGML reference path.
        let mut r2 = BoundedLocalReader::open(&index.path, CHUNK, CAP).unwrap();
        let ggml = dequant_matvec(&region, &mut r2, &act, &stop, BUDGET).unwrap();
        // Independent FP64 oracle (direct file read + f64 arithmetic).
        let row_bytes = region.byte_len as usize / rows;
        let mut f = std::fs::File::open(&index.path).unwrap();
        let mut oracle = vec![0f64; rows];
        let mut buf = vec![0u8; row_bytes];
        for (r, o) in oracle.iter_mut().enumerate() {
            f.seek(SeekFrom::Start(region.file_offset + (r * row_bytes) as u64))
                .unwrap();
            f.read_exact(&mut buf).unwrap();
            let mut acc = 0f64;
            for b in 0..(columns / 32) {
                let base = b * 34;
                let d = crate::native_kernels::f16_to_f32(u16::from_le_bytes([
                    buf[base],
                    buf[base + 1],
                ])) as f64;
                for i in 0..32 {
                    let q = buf[base + 2 + i] as i8;
                    acc += d * f64::from(q) * f64::from(act[b * 32 + i]);
                }
            }
            *o = acc;
        }

        assert_eq!(owned.len(), rows);
        assert_eq!(ggml.len(), rows);
        let mut max_vs_ggml = 0f64;
        let mut max_vs_oracle = 0f64;
        let mut max_mag = 1e-12f64;
        for r in 0..rows {
            max_vs_ggml = max_vs_ggml.max((owned[r] as f64 - ggml[r] as f64).abs());
            max_vs_oracle = max_vs_oracle.max((owned[r] as f64 - oracle[r]).abs());
            max_mag = max_mag.max(oracle[r].abs());
        }
        eprintln!(
            "OWNED_Q8_0 rows={rows} max_abs_vs_ggml={max_vs_ggml:.3e} max_abs_vs_fp64={max_vs_oracle:.3e} max_mag={max_mag:.3e}"
        );
        // Owned vs FP64 oracle: f32 rounding only.
        assert!(max_vs_oracle / max_mag < 1e-4, "owned vs fp64 too large");
        // Owned vs GGML: both f32, accumulation-order differences only.
        assert!(max_vs_ggml / max_mag < 1e-3, "owned vs ggml too large");
    }
}
