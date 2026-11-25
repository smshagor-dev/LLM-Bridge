//! Model-backed Nemotron (`nemotron_h_moe`) execution: the real 88-layer
//! tensor-binding plan and streamed, bounded, authenticated per-layer execution.
//!
//! This bridges the independently-validated math modules ([`crate::mamba2_conv`],
//! [`crate::mamba2_ssm`], [`crate::nemotron_attention`], [`crate::latent_moe_ffn`],
//! [`crate::nemotron_expert_stream`]) to the actual GGUF weights on disk. Every
//! projection is executed with [`crate::inference::dequant_matvec`] (one row
//! streamed at a time), packed experts via [`crate::nemotron_expert_stream`], and
//! only the small per-layer parameters (conv kernel/bias, A, D, dt bias, norms)
//! are materialized as f32. No large tensor is dequantized in full and the whole
//! model is never resident.
//!
//! Layer typing is read from the real tensor table (not assumed): a layer is
//! recurrent if it has `ssm_in`, attention if it has `attn_q` (and no FFN), and
//! MoE if it has `ffn_gate_inp`. Shapes are validated against the typed config,
//! which carries the values verified from the shard-1 header
//! (`attention.head_count=32`, `head_count_kv=2`, `key_length=128`,
//! `embedding_length=4096`, `expert_count=512`, `expert_used_count=22`,
//! `moe_latent_size=1024`, `expert_feed_forward_length=2688`,
//! `expert_shared_feed_forward_length=5376`, ssm conv/state/group/inner/dt from
//! the `ssm.*` keys). The config is confirmed against resolved tensor geometry
//! when the plan is built.

use std::sync::atomic::AtomicBool;

use crate::gguf_index::{SplitGgufIndex, TensorRegion};
use crate::inference::dequant_matvec;
use crate::latent_moe_routing::{route_sigmoid_topk, RoutingConfig};
use crate::mamba2_conv::{conv_forward, ConvState, ConvWeights};
use crate::mamba2_ssm::{ssm_scan, SsmConfig, SsmState};
use crate::native_dequant::dequantize_rows;
use crate::nemotron_expert_stream::PackedExperts;
use crate::storage::BoundedLocalReader;

/// Typed Nemotron 3 Super 120B-A12B configuration (verified from the shard-1
/// GGUF header; see module docs). Confirmed against resolved tensor shapes in
/// [`ModelPlan::build`].
#[derive(Debug, Clone, Copy)]
pub struct NemotronConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub head_dim: usize,
    // Mamba-2
    pub d_conv: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub ssm_n_head: usize,
    pub ssm_n_group: usize,
    pub ssm_norm_eps: f32,
    // MoE
    pub moe_latent: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub weights_scale: f32,
    pub weights_norm: bool,
    pub norm_eps: f32,
}

impl NemotronConfig {
    pub const NEMOTRON_3_SUPER_120B: NemotronConfig = NemotronConfig {
        n_layer: 88,
        n_embd: 4096,
        n_vocab: 131072,
        n_head: 32,
        head_dim: 128,
        d_conv: 4,
        d_inner: 8192,
        d_state: 128,
        ssm_n_head: 128,
        ssm_n_group: 8,
        ssm_norm_eps: 1e-5,
        moe_latent: 1024,
        n_expert: 512,
        n_expert_used: 22,
        n_ff_exp: 2688,
        n_ff_shexp: 5376,
        weights_scale: 5.0,
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

/// The resolved kind of one decoder layer, read from the real tensor table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    Recurrent,
    Attention,
    Moe,
}

/// Resolved, shape-validated tensor bindings for one layer.
#[derive(Debug, Clone)]
pub struct LayerPlan {
    pub index: usize,
    pub kind: LayerType,
    /// (tensor name -> ggml type) for every required tensor, for evidence.
    pub tensors: Vec<(String, u32)>,
}

/// A validated execution plan over the real 88-layer model.
#[derive(Debug, Clone)]
pub struct ModelPlan {
    pub cfg: NemotronConfig,
    pub layers: Vec<LayerPlan>,
    pub n_attention: usize,
    pub n_recurrent: usize,
    pub n_moe: usize,
}

fn tn(i: usize, suffix: &str) -> String {
    format!("blk.{i}.{suffix}")
}

/// Resolve a tensor or produce a descriptive error.
fn require<'a>(index: &'a SplitGgufIndex, name: &str) -> Result<&'a TensorRegion, String> {
    index
        .tensor(name)
        .map(|(_, r)| r)
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
    /// Build and validate the full layer plan from the real split index. Reads
    /// only the tensor table (no weight payload). Rejects any missing or
    /// shape-incompatible tensor before any execution.
    pub fn build(index: &SplitGgufIndex, cfg: NemotronConfig) -> Result<Self, String> {
        // Embedding + output head.
        let tok = require(index, "token_embd.weight")?;
        dims_eq(tok, &[cfg.n_embd as u64, cfg.n_vocab as u64], "token_embd")?;
        require(index, "output_norm.weight")?;
        if index.tensor("output.weight").is_none() && index.tensor("token_embd.weight").is_none() {
            return Err("missing output projection".into());
        }

        let (ne, hd) = (cfg.n_embd as u64, cfg.head_dim as u64);
        let conv_ch = cfg.conv_channels() as u64;
        let d_in_proj = cfg.d_in_proj() as u64;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        let (mut n_att, mut n_rec, mut n_moe) = (0usize, 0usize, 0usize);

        for i in 0..cfg.n_layer {
            require(index, &tn(i, "attn_norm.weight"))?;
            let mut tensors: Vec<(String, u32)> = Vec::new();
            let push = |index: &SplitGgufIndex, name: String, t: &mut Vec<(String, u32)>| {
                if let Some((_, r)) = index.tensor(&name) {
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
                // ---- attention ----
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
                // ---- MoE FFN ----
                let gate = require(index, &tn(i, "ffn_gate_inp.weight"))?;
                dims_eq(gate, &[ne, cfg.n_expert as u64], "ffn_gate_inp")?;
                require(index, &tn(i, "exp_probs_b.bias"))?;
                let ld = require(index, &tn(i, "ffn_latent_down.weight"))?;
                dims_eq(ld, &[ne, cfg.moe_latent as u64], "ffn_latent_down")?;
                let lu = require(index, &tn(i, "ffn_latent_up.weight"))?;
                dims_eq(lu, &[cfg.moe_latent as u64, ne], "ffn_latent_up")?;
                let up = require(index, &tn(i, "ffn_up_exps.weight"))?;
                dims_eq(
                    up,
                    &[
                        cfg.moe_latent as u64,
                        cfg.n_ff_exp as u64,
                        cfg.n_expert as u64,
                    ],
                    "ffn_up_exps",
                )?;
                let down = require(index, &tn(i, "ffn_down_exps.weight"))?;
                dims_eq(
                    down,
                    &[
                        cfg.n_ff_exp as u64,
                        cfg.moe_latent as u64,
                        cfg.n_expert as u64,
                    ],
                    "ffn_down_exps",
                )?;
                let ups = require(index, &tn(i, "ffn_up_shexp.weight"))?;
                dims_eq(ups, &[ne, cfg.n_ff_shexp as u64], "ffn_up_shexp")?;
                let downs = require(index, &tn(i, "ffn_down_shexp.weight"))?;
                dims_eq(downs, &[cfg.n_ff_shexp as u64, ne], "ffn_down_shexp")?;
                for s in [
                    "ffn_gate_inp.weight",
                    "exp_probs_b.bias",
                    "ffn_latent_down.weight",
                    "ffn_latent_up.weight",
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
// Streamed, bounded, authenticated execution helpers.
// ----------------------------------------------------------------------------

const BUDGET: usize = 8 * 1024 * 1024;
const CHUNK: usize = 1 << 16;
const CAP: usize = 1 << 17;

/// Stream a full (small) tensor and dequantize it row-major to f32. For small
/// per-layer parameters only (conv kernel/bias, A, D, dt bias, norms).
fn stream_small(index: &SplitGgufIndex, name: &str, stop: &AtomicBool) -> Result<Vec<f32>, String> {
    let (member, region) = index
        .tensor(name)
        .ok_or_else(|| format!("tensor missing: {name}"))?;
    let gi = member
        .index
        .as_ref()
        .ok_or("tensor owner shard has no index")?;
    if region.byte_len > 64 * 1024 * 1024 {
        return Err(format!("{name} too large to materialize as f32"));
    }
    let mut reader = BoundedLocalReader::open(&member.path, CHUNK, CAP)
        .map_err(|e| format!("open {}: {e}", member.path.display()))?;
    let mut encoded = Vec::with_capacity(region.byte_len as usize);
    gi.stream_tensor_cancellable(&mut reader, name, stop, |c| {
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
    // Float formats have no ggml `to_float` dequantizer (F32 is already float);
    // decode them directly. Quantized formats go through the pinned kernel.
    match region.ggml_type {
        0 => {
            // F32
            if encoded.len() < n_elems * 4 {
                return Err(format!("{name}: truncated F32 payload"));
            }
            let mut out = Vec::with_capacity(n_elems);
            for i in 0..n_elems {
                let b = [
                    encoded[i * 4],
                    encoded[i * 4 + 1],
                    encoded[i * 4 + 2],
                    encoded[i * 4 + 3],
                ];
                out.push(f32::from_le_bytes(b));
            }
            if !out.iter().all(|v| v.is_finite()) {
                return Err(format!("{name}: non-finite F32 value"));
            }
            Ok(out)
        }
        1 => {
            // F16
            if encoded.len() < n_elems * 2 {
                return Err(format!("{name}: truncated F16 payload"));
            }
            let mut out = Vec::with_capacity(n_elems);
            for i in 0..n_elems {
                let bits = u16::from_le_bytes([encoded[i * 2], encoded[i * 2 + 1]]);
                out.push(f16_to_f32(bits));
            }
            Ok(out)
        }
        30 => {
            // BF16
            if encoded.len() < n_elems * 2 {
                return Err(format!("{name}: truncated BF16 payload"));
            }
            let mut out = Vec::with_capacity(n_elems);
            for i in 0..n_elems {
                let bits = u16::from_le_bytes([encoded[i * 2], encoded[i * 2 + 1]]);
                out.push(f32::from_bits((bits as u32) << 16));
            }
            Ok(out)
        }
        _ => dequantize_rows(region.ggml_type, columns, rows, &encoded),
    }
}

/// IEEE-754 half-precision bit pattern to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let val = match exp {
        0 => {
            // subnormal / zero
            (mant as f32) * 2f32.powi(-24)
        }
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

/// Streamed rank-2 matrix-vector product `W . x` from the real GGUF tensor.
///
/// Quantized tensors use the bounded native kernel (one row streamed at a time).
/// Float formats (F32/F16/BF16) have no ggml dequantizer, so they are decoded by
/// [`stream_small`] and multiplied with a plain dense dot; the only float
/// projection in this model is the small F32 router (`ffn_gate_inp`, 4096x512).
fn matvec(
    index: &SplitGgufIndex,
    name: &str,
    x: &[f32],
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    let (member, region) = index
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
    if matches!(region.ggml_type, 0 | 1 | 30) {
        // Float: decode (row-major [rows][columns]) and dense-dot.
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
    let mut reader = BoundedLocalReader::open(&member.path, CHUNK, CAP)
        .map_err(|e| format!("open {}: {e}", member.path.display()))?;
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

/// Read one token's embedding row via a single bounded range read.
pub fn embed_token(
    index: &SplitGgufIndex,
    cfg: &NemotronConfig,
    token: u32,
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let (member, region) = index
        .tensor("token_embd.weight")
        .ok_or("token_embd missing")?;
    let id = token as u64;
    if id >= cfg.n_vocab as u64 {
        return Err("token id out of vocab range".into());
    }
    if stop.load(std::sync::atomic::Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let row_bytes = region.byte_len / cfg.n_vocab as u64;
    let mut f = std::fs::File::open(&member.path).map_err(|e| e.to_string())?;
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

/// Execute one real recurrent (Mamba-2) layer for a single token, streaming all
/// weights. `x` is the pre-norm input (length n_embd); returns the pre-residual
/// sub-layer output (length n_embd).
#[allow(clippy::needless_range_loop)] // ch = h*hd + i1 indexing reads clearest as-is
pub fn run_mamba_layer(
    index: &SplitGgufIndex,
    cfg: &NemotronConfig,
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

    let conv_flat = stream_small(index, &tn(il, "ssm_conv1d.weight"), stop)?; // [cc*d_conv]
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

/// Growing KV cache for streamed attention execution.
#[derive(Debug, Default, Clone)]
pub struct StreamKvCache {
    pub k: Vec<f32>, // [pos * kv_dim]
    pub v: Vec<f32>,
    pub pos: usize,
    pub kv_dim: usize,
}

/// Execute one real attention layer for a single token, streaming Q/K/V/O.
pub fn run_attention_layer(
    index: &SplitGgufIndex,
    cfg: &NemotronConfig,
    il: usize,
    kv: &mut StreamKvCache,
    x: &[f32],
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    let hd = cfg.head_dim;
    let q = matvec(index, &tn(il, "attn_q.weight"), x, stop)?; // [n_head*hd]
    let k = matvec(index, &tn(il, "attn_k.weight"), x, stop)?; // [n_head_kv*hd]
    let v = matvec(index, &tn(il, "attn_v.weight"), x, stop)?;
    let kv_dim = k.len();
    if v.len() != kv_dim || kv_dim % hd != 0 {
        return Err("attention K/V geometry invalid".into());
    }
    let n_head_kv = kv_dim / hd;
    if kv.pos == 0 {
        kv.kv_dim = kv_dim;
    } else if kv.kv_dim != kv_dim {
        return Err("KV cache dim changed between tokens".into());
    }
    if q.len() != cfg.n_head * hd {
        return Err("attention Q geometry invalid".into());
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

/// Execute one real LatentMoE FFN layer for a single token, streaming all
/// projections and only the selected packed experts.
pub fn run_moe_layer(
    index: &SplitGgufIndex,
    cfg: &NemotronConfig,
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

    // routed experts in latent space
    let latent = matvec(index, &tn(il, "ffn_latent_down.weight"), x, stop)?; // [moe_latent]

    let (up_member, up_region) = index
        .tensor(&tn(il, "ffn_up_exps.weight"))
        .ok_or("ffn_up_exps missing")?;
    let (down_member, down_region) = index
        .tensor(&tn(il, "ffn_down_exps.weight"))
        .ok_or("ffn_down_exps missing")?;
    let up = PackedExperts::new(
        up_region.file_offset,
        up_region.ggml_type,
        up_region.dimensions[0],
        up_region.dimensions[1],
        up_region.dimensions[2],
        up_region.byte_len,
    )?;
    let down = PackedExperts::new(
        down_region.file_offset,
        down_region.ggml_type,
        down_region.dimensions[0],
        down_region.dimensions[1],
        down_region.dimensions[2],
        down_region.byte_len,
    )?;
    let mut up_reader =
        BoundedLocalReader::open(&up_member.path, CHUNK, CAP).map_err(|e| e.to_string())?;
    let mut moe = vec![0.0f32; cfg.moe_latent];
    for (&e, &w) in routing.expert_ids.iter().zip(&routing.weights) {
        let mut u = up.project_expert(&mut up_reader, e, &latent, stop, BUDGET)?;
        relu_sqr_in_place(&mut u);
        let mut down_reader =
            BoundedLocalReader::open(&down_member.path, CHUNK, CAP).map_err(|e| e.to_string())?;
        let dvec = down.project_expert(&mut down_reader, e, &u, stop, BUDGET)?;
        for (m, dv) in moe.iter_mut().zip(&dvec) {
            *m += w * dv;
        }
    }
    let moe_up = matvec(index, &tn(il, "ffn_latent_up.weight"), &moe, stop)?; // [n_embd]
    let out: Vec<f32> = moe_up.iter().zip(&shexp).map(|(a, b)| a + b).collect();
    if !out.iter().all(|v| v.is_finite()) {
        return Err("non-finite moe output".into());
    }
    Ok(out)
}

/// Per-layer recurrent/KV state for a full-model session.
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
    pub fn new(cfg: &NemotronConfig, plan: &ModelPlan) -> Result<Self, String> {
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

/// Name of the output (vocabulary) projection: a dedicated `output.weight` when
/// present, otherwise the tied `token_embd.weight`.
fn output_tensor_name(index: &SplitGgufIndex) -> &'static str {
    if index.tensor("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    }
}

/// Run the full 88-layer streamed forward over `tokens` (fed one at a time,
/// advancing all per-layer state), then return the final-position logits over
/// the vocabulary. Every weight is read bounded and authenticated from the
/// shards; no full tensor is materialized and the model is never resident.
///
/// This produces genuine model-backed logits. It is NOT validated against a
/// full-model resident reference (none is runnable on this host); only the
/// per-layer kernels and the reduced-stack composition are reference-checked.
pub fn run_full_forward(
    index: &SplitGgufIndex,
    cfg: &NemotronConfig,
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
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return Err("full forward cancelled".into());
        }
        let mut h = embed_token(index, cfg, tok, stop)?;
        for (li, layer) in plan.layers.iter().enumerate() {
            let attn_norm = stream_small(index, &tn(layer.index, "attn_norm.weight"), stop)?;
            let hn = {
                let ms: f32 = h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32;
                let inv = 1.0 / (ms + cfg.norm_eps).sqrt();
                (0..h.len())
                    .map(|i| h[i] * inv * attn_norm[i])
                    .collect::<Vec<f32>>()
            };
            let out = match (&layer.kind, &mut session.states[li]) {
                (LayerType::Recurrent, FullLayerState::Recurrent(conv, ssm)) => {
                    run_mamba_layer(index, cfg, layer.index, conv, ssm, &hn, stop)?
                }
                (LayerType::Attention, FullLayerState::Attention(kv)) => {
                    run_attention_layer(index, cfg, layer.index, kv, &hn, stop)?
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
    // final norm + output projection on the last token
    let h = last_hidden.ok_or("no hidden state produced")?;
    let on = stream_small(index, "output_norm.weight", stop)?;
    let ms: f32 = h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32;
    let inv = 1.0 / (ms + cfg.norm_eps).sqrt();
    let hf: Vec<f32> = (0..h.len()).map(|i| h[i] * inv * on[i]).collect();
    let logits = matvec(index, output_tensor_name(index), &hf, stop)?;
    if logits.len() != cfg.n_vocab || !logits.iter().all(|v| v.is_finite()) {
        return Err("invalid full-forward logits".into());
    }
    Ok(logits)
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

/// Encode `prompt` with the Nemotron tokenizer, run the full streamed 88-layer
/// prefill over the resulting token ids (shared recurrent/KV state advanced one
/// token at a time), and greedily pick the next token from the final-position
/// logits. This is the genuine prompt-conditioned path; it reuses
/// [`run_full_forward`] and performs no weight residency.
///
/// Honest caveat: the tokenizer's non-Latin / edge-case ID parity is unverified
/// in this environment (see [`crate::nemotron_tokenizer`]), so for non-ASCII
/// prompts the token ids may differ from a corrected llama.cpp reference.
pub fn generate_next_token(
    index: &SplitGgufIndex,
    cfg: &NemotronConfig,
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
    // top-5 by descending logit (small bounded partial selection)
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
    let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, logits[i])).collect();
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

/// Greedy argmax token id over a logit vector.
pub fn greedy_token(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .fold(None, |best, (i, &v)| match best {
            Some((_, bv)) if bv >= v => best,
            _ => Some((i, v)),
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shards() -> Option<Vec<String>> {
        let dir = std::env::var("LLMBRIDGE_NEMOTRON_DIR").ok()?;
        Some(
            (1..=3)
                .map(|i| {
                    format!(
                        "{dir}/NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ2_XXS-0000{i}-of-00003.gguf"
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn config_derived_dimensions_are_consistent() {
        let c = NemotronConfig::NEMOTRON_3_SUPER_120B;
        assert_eq!(c.conv_channels(), 8192 + 2 * 8 * 128);
        assert_eq!(c.d_in_proj(), 2 * 8192 + 2 * 8 * 128 + 128);
        assert_eq!(c.ssm_head_dim(), 64);
        assert_eq!(c.n_head * c.head_dim, c.n_embd);
    }

    #[test]
    fn full_88_layer_plan_builds_and_validates() {
        let Some(sh) = shards() else {
            eprintln!("skip: LLMBRIDGE_NEMOTRON_DIR unset");
            return;
        };
        let index = SplitGgufIndex::inspect(&sh).unwrap();
        let cfg = NemotronConfig::NEMOTRON_3_SUPER_120B;
        let plan = ModelPlan::build(&index, cfg).expect("plan builds");
        assert_eq!(plan.layers.len(), 88);
        assert_eq!(plan.n_attention, 8, "expected 8 attention layers");
        assert!(plan.n_recurrent > 0 && plan.n_moe > 0);
        let att: Vec<usize> = plan
            .layers
            .iter()
            .filter(|l| l.kind == LayerType::Attention)
            .map(|l| l.index)
            .collect();
        assert_eq!(att, vec![7, 16, 25, 36, 47, 58, 69, 78]);
        eprintln!(
            "plan: {} layers = {} recurrent + {} attention + {} moe",
            plan.layers.len(),
            plan.n_recurrent,
            plan.n_attention,
            plan.n_moe
        );
    }

    #[test]
    fn model_backed_layers_execute_through_streamed_reader() {
        let Some(sh) = shards() else {
            eprintln!("skip: LLMBRIDGE_NEMOTRON_DIR unset");
            return;
        };
        let index = SplitGgufIndex::inspect(&sh).unwrap();
        let cfg = NemotronConfig::NEMOTRON_3_SUPER_120B;
        let plan = ModelPlan::build(&index, cfg).unwrap();
        let stop = AtomicBool::new(false);

        // Test A: real embedding of token 42.
        let emb = embed_token(&index, &cfg, 42, &stop).unwrap();
        assert_eq!(emb.len(), cfg.n_embd);
        assert!(emb.iter().all(|v| v.is_finite()));

        let first = |k: LayerType| plan.layers.iter().find(|l| l.kind == k).unwrap().index;
        let rec = first(LayerType::Recurrent);
        let att = first(LayerType::Attention);
        let moe = first(LayerType::Moe);

        // Test B: one real Mamba layer.
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

        // Test C: one real attention layer.
        let mut kv = StreamKvCache::default();
        let ao = run_attention_layer(&index, &cfg, att, &mut kv, &emb, &stop).unwrap();
        assert_eq!(ao.len(), cfg.n_embd);
        assert!(ao.iter().all(|v| v.is_finite()));
        assert_eq!(kv.pos, 1);

        // Test D: one real LatentMoE layer.
        let fo = run_moe_layer(&index, &cfg, moe, &emb, &stop).unwrap();
        assert_eq!(fo.len(), cfg.n_embd);
        assert!(fo.iter().all(|v| v.is_finite()));

        eprintln!(
            "model-backed layers ok: embed+mamba(l{rec})+attn(l{att})+moe(l{moe}), all finite"
        );
    }

    // Full 88-layer streamed forward on a single raw token. Heavy: reads most of
    // the non-expert weights (plus 22/512 experts per MoE layer) from the slow G:
    // drive, so it is gated on BOTH LLMBRIDGE_NEMOTRON_DIR and
    // LLMBRIDGE_NEMOTRON_FULL to keep it out of normal runs. Produces genuine
    // finite logits and a greedy token id; it is NOT a reference-parity check.
    #[test]
    fn full_88_layer_forward_produces_finite_logits_and_a_token() {
        let Some(sh) = shards() else {
            eprintln!("skip: LLMBRIDGE_NEMOTRON_DIR unset");
            return;
        };
        if std::env::var("LLMBRIDGE_NEMOTRON_FULL").is_err() {
            eprintln!("skip: LLMBRIDGE_NEMOTRON_FULL unset (heavy full-forward test)");
            return;
        }
        let index = SplitGgufIndex::inspect(&sh).unwrap();
        let cfg = NemotronConfig::NEMOTRON_3_SUPER_120B;
        let plan = ModelPlan::build(&index, cfg).unwrap();
        let mut session = FullSession::new(&cfg, &plan).unwrap();
        let stop = AtomicBool::new(false);
        // A single real token id (BOS = <s> = 1); a true prompt needs the
        // tokenizer, which is a separate outstanding deliverable.
        let t0 = std::time::Instant::now();
        let logits = run_full_forward(&index, &cfg, &plan, &mut session, &[1u32], &stop).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        assert_eq!(logits.len(), cfg.n_vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
        let tok = greedy_token(&logits).unwrap();
        assert_eq!(session.position(), 1);
        eprintln!(
            "FULL 88-layer forward ok in {secs:.1}s: finite logits[{}], greedy token id = {tok}",
            logits.len()
        );
    }

    // Genuine prompt-conditioned generation: encode a text prompt, run the full
    // 88-layer streamed prefill, and emit one real next token. Heavy (each prompt
    // token is a full streamed pass over the slow G: drive), so gated on
    // LLMBRIDGE_NEMOTRON_DIR + LLMBRIDGE_NEMOTRON_FULL.
    #[test]
    fn prompt_conditioned_generation_emits_one_real_token() {
        let Some(sh) = shards() else {
            eprintln!("skip: LLMBRIDGE_NEMOTRON_DIR unset");
            return;
        };
        if std::env::var("LLMBRIDGE_NEMOTRON_FULL").is_err() {
            eprintln!("skip: LLMBRIDGE_NEMOTRON_FULL unset (heavy prompt-generation test)");
            return;
        }
        let index = SplitGgufIndex::inspect(&sh).unwrap();
        let cfg = NemotronConfig::NEMOTRON_3_SUPER_120B;
        let plan = ModelPlan::build(&index, cfg).unwrap();
        let shard1 = std::path::PathBuf::from(&sh[0]);
        let tk = crate::nemotron_tokenizer::NemotronTokenizer::from_gguf(&shard1).unwrap();
        let stop = AtomicBool::new(false);
        let prompt = std::env::var("LLMBRIDGE_NEMOTRON_PROMPT").unwrap_or_else(|_| "Hi".into());
        let r = generate_next_token(&index, &cfg, &plan, &tk, &prompt, &stop).unwrap();
        assert!(r.next_token_id < cfg.n_vocab);
        assert_eq!(r.n_prefill, r.token_ids.len());
        eprintln!(
            "PROMPT GEN ok: prompt={:?} ids={:?} n_prefill={} next_id={} next_text={:?} top5={:?} prefill={:.1}s",
            r.prompt, r.token_ids, r.n_prefill, r.next_token_id, r.next_token_text, r.top5, r.prefill_seconds
        );
    }
}
