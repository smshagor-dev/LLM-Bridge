//! Reduced-size `nemotron_h_moe` hybrid decoder forward pass.
//!
//! Sequences the three validated sub-layers — the Mamba-2 block
//! ([`crate::mamba2_block`]), NoPE GQA attention ([`crate::nemotron_attention`]),
//! and the LatentMoE FFN ([`crate::latent_moe_ffn`]) — into the decoder exactly
//! as the pinned graph `llama_model_nemotron_h::graph`
//! (build/reference/src/models/nemotron-h.cpp:202, rev cd959ba) defines it:
//!
//! ```text
//!   inpL = tok_embd[token]
//!   for each layer il:
//!       inpSA = inpL
//!       cur   = rms_norm(inpL, attn_norm[il])      // every block pre-norms with attn_norm
//!       cur   = sublayer(cur)                      // Mamba OR attention OR MoE-FFN
//!       inpL  = cur + inpSA                         // residual
//!   cur    = rms_norm(inpL, output_norm)
//!   logits = output . cur
//! ```
//!
//! Layer typing follows the real model (per-layer metadata arrays): a layer is
//! recurrent (Mamba) iff `head_count_kv == 0 && feed_forward_length == 0`,
//! attention iff `head_count_kv != 0` (`feed_forward_length == 0`), and MoE-FFN
//! iff `feed_forward_length != 0`. For the 120B these are 88 layers with the
//! 8 attention layers at indices 7,16,25,36,47,58,69,78 and the rest alternating
//! recurrent / MoE. This module composes a REDUCED stack (small dims, a handful
//! of layers covering all three types) — it does not rewrite the sub-layers and
//! does not claim full-model parity.
//!
//! State is session-local and per layer: Mamba conv+SSM state and attention KV
//! cache. Running the whole stack is layer-major within a call; calling once per
//! token is token-major. Both yield identical results because every sub-layer is
//! causal/streaming (verified), so `forward` over a sequence equals sequential
//! single-token `forward` calls.

use crate::latent_moe_ffn::{LatentMoeWeights, LinearF32};
use crate::mamba2_block::{Mamba2BlockState, Mamba2BlockWeights};
use crate::nemotron_attention::{AttnState, AttnWeights};

/// One decoder sub-layer's weights (plus its pre-norm gain is stored separately).
pub enum LayerKind {
    Recurrent(Box<Mamba2BlockWeights>),
    Attention(Box<AttnWeights>),
    Moe(Box<LatentMoeWeights>),
}

/// One layer: the `attn_norm` pre-norm gain (`[n_embd]`) and its sub-layer.
pub struct HybridLayer {
    pub attn_norm: Vec<f32>,
    pub kind: LayerKind,
}

/// A reduced Nemotron hybrid decoder.
pub struct NemotronHybrid {
    n_embd: usize,
    n_vocab: usize,
    norm_eps: f32,
    tok_embd: Vec<f32>, // [n_vocab * n_embd], row per token id
    layers: Vec<HybridLayer>,
    output_norm: Vec<f32>, // [n_embd]
    output: LinearF32,     // [n_vocab, n_embd]
}

/// Per-layer recurrent/KV state for one session.
pub enum LayerState {
    Recurrent(Box<Mamba2BlockState>),
    Attention(Box<AttnState>),
    Moe, // stateless
}

/// Session-local state for the whole stack.
pub struct HybridState {
    layers: Vec<LayerState>,
    pos: usize,
}

impl HybridState {
    pub fn position(&self) -> usize {
        self.pos
    }
    pub fn reset(&mut self) {
        for s in &mut self.layers {
            match s {
                LayerState::Recurrent(st) => st.reset(),
                LayerState::Attention(st) => st.reset(),
                LayerState::Moe => {}
            }
        }
        self.pos = 0;
    }
}

/// RMS norm: `x[i] / sqrt(mean(x^2) + eps) * w[i]`, single precision.
fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ms = 0.0f32;
    for &v in x {
        ms += v * v;
    }
    ms /= n as f32;
    let inv = 1.0f32 / (ms + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w[i]).collect()
}

impl NemotronHybrid {
    pub fn new(
        n_embd: usize,
        n_vocab: usize,
        norm_eps: f32,
        tok_embd: Vec<f32>,
        layers: Vec<HybridLayer>,
        output_norm: Vec<f32>,
        output: LinearF32,
    ) -> Result<Self, String> {
        if n_embd == 0 || n_vocab == 0 {
            return Err("hybrid requires non-zero n_embd and n_vocab".into());
        }
        if !norm_eps.is_finite() || norm_eps < 0.0 {
            return Err("norm_eps must be finite and non-negative".into());
        }
        if tok_embd.len() != n_vocab * n_embd {
            return Err("tok_embd length must equal n_vocab*n_embd".into());
        }
        if !tok_embd.iter().all(|v| v.is_finite()) {
            return Err("non-finite token embedding rejected".into());
        }
        if output_norm.len() != n_embd {
            return Err("output_norm length must equal n_embd".into());
        }
        if output.out_dim() != n_vocab || output.in_dim() != n_embd {
            return Err("output projection shape must be [n_vocab, n_embd]".into());
        }
        if layers.is_empty() {
            return Err("hybrid requires at least one layer".into());
        }
        for (i, l) in layers.iter().enumerate() {
            if l.attn_norm.len() != n_embd {
                return Err(format!("layer {i} attn_norm length must equal n_embd"));
            }
            let ok = match &l.kind {
                LayerKind::Recurrent(m) => m.config().n_embd == n_embd,
                LayerKind::Attention(a) => a.config().n_embd == n_embd,
                LayerKind::Moe(f) => f.n_embd() == n_embd,
            };
            if !ok {
                return Err(format!("layer {i} sub-layer n_embd disagrees with model"));
            }
        }
        Ok(Self {
            n_embd,
            n_vocab,
            norm_eps,
            tok_embd,
            layers,
            output_norm,
            output,
        })
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Fresh zero state for one session.
    pub fn new_state(&self) -> Result<HybridState, String> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            layers.push(match &l.kind {
                LayerKind::Recurrent(m) => {
                    LayerState::Recurrent(Box::new(Mamba2BlockState::zeros(m.config())?))
                }
                LayerKind::Attention(a) => {
                    LayerState::Attention(Box::new(AttnState::zeros(a.config())?))
                }
                LayerKind::Moe(_) => LayerState::Moe,
            });
        }
        Ok(HybridState { layers, pos: 0 })
    }

    /// Run the decoder over `tokens`, advancing `state`, returning logits for
    /// every input token (`tokens.len() * n_vocab`, token-major).
    pub fn forward(&self, state: &mut HybridState, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("hybrid forward requires at least one token".into());
        }
        if state.layers.len() != self.layers.len() {
            return Err("state does not match model layer count".into());
        }
        let n_t = tokens.len();
        let n_embd = self.n_embd;

        // Embedding lookup.
        let mut h = vec![0.0f32; n_t * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let id = tok as usize;
            if id >= self.n_vocab {
                return Err(format!("token id {id} out of vocab range {}", self.n_vocab));
            }
            h[t * n_embd..(t + 1) * n_embd]
                .copy_from_slice(&self.tok_embd[id * n_embd..(id + 1) * n_embd]);
        }

        // Layers: pre-norm -> sub-layer -> residual.
        for (li, layer) in self.layers.iter().enumerate() {
            // pre-norm each token
            let mut hn = vec![0.0f32; n_t * n_embd];
            for t in 0..n_t {
                let normed = rms_norm(
                    &h[t * n_embd..(t + 1) * n_embd],
                    &layer.attn_norm,
                    self.norm_eps,
                );
                hn[t * n_embd..(t + 1) * n_embd].copy_from_slice(&normed);
            }
            // sub-layer
            let out = match (&layer.kind, &mut state.layers[li]) {
                (LayerKind::Recurrent(m), LayerState::Recurrent(st)) => m.forward(st, &hn)?,
                (LayerKind::Attention(a), LayerState::Attention(st)) => a.forward(st, &hn)?,
                (LayerKind::Moe(f), LayerState::Moe) => {
                    let mut o = vec![0.0f32; n_t * n_embd];
                    for t in 0..n_t {
                        let r = f.forward(&hn[t * n_embd..(t + 1) * n_embd])?;
                        o[t * n_embd..(t + 1) * n_embd].copy_from_slice(&r.ffn_out);
                    }
                    o
                }
                _ => return Err(format!("layer {li} state/kind mismatch")),
            };
            // residual
            for i in 0..n_t * n_embd {
                h[i] += out[i];
            }
        }

        // Final norm + output projection, per token.
        let mut logits = vec![0.0f32; n_t * self.n_vocab];
        for t in 0..n_t {
            let hf = rms_norm(
                &h[t * n_embd..(t + 1) * n_embd],
                &self.output_norm,
                self.norm_eps,
            );
            let l = self.output.matvec(&hf)?;
            for v in &l {
                if !v.is_finite() {
                    return Err("non-finite logit rejected".into());
                }
            }
            logits[t * self.n_vocab..(t + 1) * self.n_vocab].copy_from_slice(&l);
        }

        state.pos += n_t;
        Ok(logits)
    }

    /// Convenience: argmax (top-1 token id) of the last position's logits.
    pub fn argmax_last(&self, logits: &[f32]) -> Option<usize> {
        if logits.len() < self.n_vocab || logits.len() % self.n_vocab != 0 {
            return None;
        }
        let last = &logits[logits.len() - self.n_vocab..];
        last.iter()
            .enumerate()
            .fold(None, |best, (i, &v)| match best {
                Some((_, bv)) if bv >= v => best,
                _ => Some((i, v)),
            })
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::latent_moe_routing::RoutingConfig;
    use crate::mamba2_block::Mamba2BlockConfig;
    use crate::mamba2_conv::ConvWeights;
    use crate::nemotron_attention::AttnConfig;

    fn gen(tag: &str, n: usize, scale: f32, shift: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = tag
                    .bytes()
                    .fold(2166136261u32, |a, b| (a ^ b as u32).wrapping_mul(16777619));
                let k = h.wrapping_add((i as u32).wrapping_mul(2654435761));
                ((k % 1000) as f32 / 500.0 - 1.0) * scale + shift
            })
            .collect()
    }

    // Reduced but structurally faithful model: [recurrent, attention, moe].
    const NE: usize = 8;
    const NV: usize = 12;
    const EPS: f32 = 1e-5;

    fn mamba_cfg() -> Mamba2BlockConfig {
        Mamba2BlockConfig {
            n_embd: NE,
            d_inner: 8,
            d_conv: 4,
            d_state: 2,
            n_head: 4,
            n_group: 2,
            ssm_norm_eps: EPS,
        }
    }
    fn attn_cfg() -> AttnConfig {
        AttnConfig {
            n_embd: NE,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 4,
            capacity: 32,
        }
    }
    fn moe_cfg() -> RoutingConfig {
        RoutingConfig {
            n_expert: 6,
            n_expert_used: 2,
            weights_norm: true,
            weights_scale: 5.0,
        }
    }
    const NLAT: usize = 4;
    const NFE: usize = 5;
    const NFS: usize = 6;

    // Hold every raw weight so the independent oracle can recompute from scratch.
    struct Raw {
        tok_embd: Vec<f32>,
        output_norm: Vec<f32>,
        output: Vec<f32>,
        // per layer attn_norm
        an: [Vec<f32>; 3],
        // mamba (layer 0)
        m_in: Vec<f32>,
        m_conv: Vec<f32>,
        m_cb: Vec<f32>,
        m_dtb: Vec<f32>,
        m_a: Vec<f32>,
        m_d: Vec<f32>,
        m_norm: Vec<f32>,
        m_out: Vec<f32>,
        // attention (layer 1)
        a_wq: Vec<f32>,
        a_wk: Vec<f32>,
        a_wv: Vec<f32>,
        a_wo: Vec<f32>,
        // moe (layer 2)
        f_router: Vec<f32>,
        f_ups: Vec<f32>,
        f_downs: Vec<f32>,
        f_ld: Vec<f32>,
        f_lu: Vec<f32>,
        f_upe: Vec<Vec<f32>>,
        f_downe: Vec<Vec<f32>>,
    }

    fn build() -> (NemotronHybrid, Raw) {
        let mc = mamba_cfg();
        let ac = attn_cfg();
        let fc = moe_cfg();
        let cc = mc.conv_channels();
        let dip = mc.d_in_proj();

        let raw = Raw {
            tok_embd: gen("emb", NV * NE, 1.0, 0.0),
            output_norm: gen("on", NE, 0.2, 1.0),
            output: gen("out", NV * NE, 0.4, 0.0),
            an: [
                gen("an0", NE, 0.2, 1.0),
                gen("an1", NE, 0.2, 1.0),
                gen("an2", NE, 0.2, 1.0),
            ],
            m_in: gen("min", dip * NE, 0.3, 0.0),
            m_conv: gen("mcv", cc * mc.d_conv, 0.4, 0.0),
            m_cb: gen("mcb", cc, 0.1, 0.0),
            m_dtb: gen("mdt", mc.n_head, 0.2, -0.5),
            m_a: gen("ma", mc.n_head, 0.5, -1.5),
            m_d: gen("md", mc.n_head, 0.3, 0.0),
            m_norm: gen("mn", mc.d_inner, 0.2, 1.0),
            m_out: gen("mo", NE * mc.d_inner, 0.3, 0.0),
            a_wq: gen("wq", ac.q_dim() * NE, 0.4, 0.0),
            a_wk: gen("wk", ac.kv_dim() * NE, 0.4, 0.0),
            a_wv: gen("wv", ac.kv_dim() * NE, 0.4, 0.0),
            a_wo: gen("wo", NE * ac.q_dim(), 0.4, 0.0),
            f_router: gen("fr", fc.n_expert * NE, 0.4, 0.0),
            f_ups: gen("fus", NFS * NE, 0.3, 0.0),
            f_downs: gen("fds", NE * NFS, 0.3, 0.0),
            f_ld: gen("fld", NLAT * NE, 0.3, 0.0),
            f_lu: gen("flu", NE * NLAT, 0.3, 0.0),
            f_upe: (0..fc.n_expert)
                .map(|e| gen(&format!("fue{e}"), NFE * NLAT, 0.4, 0.0))
                .collect(),
            f_downe: (0..fc.n_expert)
                .map(|e| gen(&format!("fde{e}"), NLAT * NFE, 0.4, 0.0))
                .collect(),
        };

        let mamba = Mamba2BlockWeights::new(
            mc,
            LinearF32::new(dip, NE, raw.m_in.clone()).unwrap(),
            ConvWeights::new(cc, mc.d_conv, raw.m_conv.clone()).unwrap(),
            raw.m_cb.clone(),
            raw.m_dtb.clone(),
            raw.m_a.clone(),
            raw.m_d.clone(),
            Some(raw.m_norm.clone()),
            LinearF32::new(NE, mc.d_inner, raw.m_out.clone()).unwrap(),
        )
        .unwrap();

        let attn = AttnWeights::new(
            ac,
            LinearF32::new(ac.q_dim(), NE, raw.a_wq.clone()).unwrap(),
            LinearF32::new(ac.kv_dim(), NE, raw.a_wk.clone()).unwrap(),
            LinearF32::new(ac.kv_dim(), NE, raw.a_wv.clone()).unwrap(),
            LinearF32::new(NE, ac.q_dim(), raw.a_wo.clone()).unwrap(),
            None,
        )
        .unwrap();

        let moe = LatentMoeWeights::new(
            fc,
            LinearF32::new(fc.n_expert, NE, raw.f_router.clone()).unwrap(),
            None,
            LinearF32::new(NFS, NE, raw.f_ups.clone()).unwrap(),
            LinearF32::new(NE, NFS, raw.f_downs.clone()).unwrap(),
            LinearF32::new(NLAT, NE, raw.f_ld.clone()).unwrap(),
            LinearF32::new(NE, NLAT, raw.f_lu.clone()).unwrap(),
            raw.f_upe
                .iter()
                .map(|w| LinearF32::new(NFE, NLAT, w.clone()).unwrap())
                .collect(),
            raw.f_downe
                .iter()
                .map(|w| LinearF32::new(NLAT, NFE, w.clone()).unwrap())
                .collect(),
        )
        .unwrap();

        let layers = vec![
            HybridLayer {
                attn_norm: raw.an[0].clone(),
                kind: LayerKind::Recurrent(Box::new(mamba)),
            },
            HybridLayer {
                attn_norm: raw.an[1].clone(),
                kind: LayerKind::Attention(Box::new(attn)),
            },
            HybridLayer {
                attn_norm: raw.an[2].clone(),
                kind: LayerKind::Moe(Box::new(moe)),
            },
        ];

        let model = NemotronHybrid::new(
            NE,
            NV,
            EPS,
            raw.tok_embd.clone(),
            layers,
            raw.output_norm.clone(),
            LinearF32::new(NV, NE, raw.output.clone()).unwrap(),
        )
        .unwrap();
        (model, raw)
    }

    // ---------- Independent f64 oracle (from scratch, no production calls) -----

    fn mvf(w: &[f32], out: usize, inn: usize, x: &[f64]) -> Vec<f64> {
        (0..out)
            .map(|o| (0..inn).map(|i| w[o * inn + i] as f64 * x[i]).sum())
            .collect()
    }
    fn rmsf(x: &[f64], w: &[f32], eps: f32) -> Vec<f64> {
        let n = x.len();
        let ms: f64 = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        (0..n).map(|i| x[i] * inv * w[i] as f64).collect()
    }
    fn siluf(x: f64) -> f64 {
        x * (1.0 / (1.0 + (-x).exp()))
    }
    fn relu2f(v: &[f64]) -> Vec<f64> {
        v.iter()
            .map(|&x| if x > 0.0 { x * x } else { 0.0 })
            .collect()
    }

    // f64 Mamba block over a sequence, with its own persistent state.
    fn mamba_oracle(
        cfg: &Mamba2BlockConfig,
        r: &Raw,
        conv_state: &mut [f64],
        ssm_state: &mut [f64],
        hn: &[f64],
    ) -> Vec<f64> {
        let (ne, di, cc, k) = (cfg.n_embd, cfg.d_inner, cfg.conv_channels(), cfg.d_conv);
        let gds = cfg.n_group * cfg.d_state;
        let hd = cfg.head_dim();
        let dip = cfg.d_in_proj();
        let n_t = hn.len() / ne;
        let hpg = cfg.n_head / cfg.n_group;
        let mut out = vec![0.0f64; n_t * ne];
        for t in 0..n_t {
            let x: Vec<f64> = hn[t * ne..(t + 1) * ne].to_vec();
            let proj = mvf(&r.m_in, dip, ne, &x);
            let z = &proj[0..di];
            let xbc_in = &proj[di..di + cc];
            let dt: Vec<f64> = (0..cfg.n_head)
                .map(|h| proj[2 * di + gds + h] + r.m_dtb[h] as f64)
                .collect();
            // conv
            let mut xbc = vec![0.0f64; cc];
            for ch in 0..cc {
                let mut acc = 0.0f64;
                for j in 0..k {
                    let v = if j < k - 1 {
                        conv_state[ch * (k - 1) + j]
                    } else {
                        xbc_in[ch]
                    };
                    acc += v * r.m_conv[ch * k + j] as f64;
                }
                xbc[ch] = siluf(acc + r.m_cb[ch] as f64);
            }
            for ch in 0..cc {
                if k >= 2 {
                    for j in 0..(k - 2) {
                        conv_state[ch * (k - 1) + j] = conv_state[ch * (k - 1) + j + 1];
                    }
                    conv_state[ch * (k - 1) + (k - 2)] = xbc_in[ch];
                }
            }
            let xv = &xbc[0..di];
            let b = &xbc[di..di + gds];
            let cvec = &xbc[di + gds..di + 2 * gds];
            let mut y = vec![0.0f64; di];
            for h in 0..cfg.n_head {
                let dtsp = {
                    let zz = dt[h];
                    if zz > 20.0 {
                        zz
                    } else {
                        zz.exp().ln_1p()
                    }
                };
                let da = (dtsp * r.m_a[h] as f64).exp();
                let g = h / hpg;
                for i1 in 0..hd {
                    let ch = h * hd + i1;
                    let x_dt = xv[ch] * dtsp;
                    let mut sum = 0.0f64;
                    for i0 in 0..cfg.d_state {
                        let si = ch * cfg.d_state + i0;
                        let ig = g * cfg.d_state + i0;
                        let st = ssm_state[si] * da + b[ig] * x_dt;
                        sum += st * cvec[ig];
                        ssm_state[si] = st;
                    }
                    y[ch] = sum;
                }
            }
            for h in 0..cfg.n_head {
                for i1 in 0..hd {
                    let ch = h * hd + i1;
                    let yi = y[ch] + xv[ch] * r.m_d[h] as f64;
                    y[ch] = siluf(z[ch]) * yi;
                }
            }
            // grouped rms norm
            let gsize = di / cfg.n_group;
            for grp in 0..cfg.n_group {
                let lo = grp * gsize;
                let hi = lo + gsize;
                let ms: f64 = y[lo..hi].iter().map(|v| v * v).sum::<f64>() / gsize as f64;
                let inv = 1.0 / (ms + cfg.ssm_norm_eps as f64).sqrt();
                for ch in lo..hi {
                    y[ch] = y[ch] * inv * r.m_norm[ch] as f64;
                }
            }
            let o = mvf(&r.m_out, ne, di, &y);
            out[t * ne..(t + 1) * ne].copy_from_slice(&o);
        }
        out
    }

    // f64 attention over a sequence with its own KV cache (grows with pos).
    fn attn_oracle(
        cfg: &AttnConfig,
        r: &Raw,
        kc: &mut Vec<f64>,
        vc: &mut Vec<f64>,
        base: usize,
        hn: &[f64],
    ) -> Vec<f64> {
        let ne = cfg.n_embd;
        let hd = cfg.head_dim;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let hpk = cfg.n_head / cfg.n_head_kv;
        let scale = 1.0 / (hd as f64).sqrt();
        let n_t = hn.len() / ne;
        let mut qc = vec![0.0f64; n_t * qd];
        for t in 0..n_t {
            let x = &hn[t * ne..(t + 1) * ne];
            qc[t * qd..(t + 1) * qd].copy_from_slice(&mvf(&r.a_wq, qd, ne, x));
            let kk = mvf(&r.a_wk, kvd, ne, x);
            let vv = mvf(&r.a_wv, kvd, ne, x);
            kc.extend_from_slice(&kk);
            vc.extend_from_slice(&vv);
        }
        let mut out = vec![0.0f64; n_t * ne];
        for t in 0..n_t {
            let abs = base + t;
            let n_kv = abs + 1;
            let mut concat = vec![0.0f64; qd];
            for h in 0..cfg.n_head {
                let g = h / hpk;
                let qh = &qc[t * qd + h * hd..t * qd + (h + 1) * hd];
                let mut s = vec![0.0f64; n_kv];
                for (j, sj) in s.iter_mut().enumerate() {
                    let kh = &kc[j * kvd + g * hd..j * kvd + (g + 1) * hd];
                    *sj = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f64>() * scale;
                }
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for x in s.iter_mut() {
                    *x = (*x - m).exp();
                    sum += *x;
                }
                for x in s.iter_mut() {
                    *x /= sum;
                }
                for (j, &w) in s.iter().enumerate() {
                    let vh = &vc[j * kvd + g * hd..j * kvd + (g + 1) * hd];
                    for d in 0..hd {
                        concat[h * hd + d] += w * vh[d];
                    }
                }
            }
            out[t * ne..(t + 1) * ne].copy_from_slice(&mvf(&r.a_wo, ne, qd, &concat));
        }
        out
    }

    // f64 MoE FFN for one token.
    fn moe_oracle(fc: &RoutingConfig, r: &Raw, x: &[f64]) -> Vec<f64> {
        let logits = mvf(&r.f_router, fc.n_expert, NE, x);
        let probs: Vec<f64> = logits.iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect();
        let mut idx: Vec<usize> = (0..fc.n_expert).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(fc.n_expert_used);
        let mut w: Vec<f64> = idx.iter().map(|&i| probs[i]).collect();
        let sum: f64 = w.iter().sum::<f64>().max(6.103_515_625e-5);
        for v in &mut w {
            *v = *v / sum * fc.weights_scale as f64;
        }
        let shexp = mvf(&r.f_downs, NE, NFS, &relu2f(&mvf(&r.f_ups, NFS, NE, x)));
        let lat = mvf(&r.f_ld, NLAT, NE, x);
        let mut moe = vec![0.0f64; NLAT];
        for (&e, &we) in idx.iter().zip(&w) {
            let u = relu2f(&mvf(&r.f_upe[e], NFE, NLAT, &lat));
            let d = mvf(&r.f_downe[e], NLAT, NFE, &u);
            for (m, dv) in moe.iter_mut().zip(&d) {
                *m += we * dv;
            }
        }
        let moe_up = mvf(&r.f_lu, NE, NLAT, &moe);
        moe_up.iter().zip(&shexp).map(|(a, b)| a + b).collect()
    }

    // Full f64 hybrid oracle over a token sequence; its own per-layer state.
    fn hybrid_oracle(r: &Raw, tokens: &[u32]) -> Vec<f64> {
        let mc = mamba_cfg();
        let ac = attn_cfg();
        let fc = moe_cfg();
        let n_t = tokens.len();
        let mut h = vec![0.0f64; n_t * NE];
        for (t, &tok) in tokens.iter().enumerate() {
            for i in 0..NE {
                h[t * NE + i] = r.tok_embd[tok as usize * NE + i] as f64;
            }
        }
        // persistent oracle states
        let mut conv_state = vec![0.0f64; mc.conv_channels() * (mc.d_conv - 1)];
        let mut ssm_state = vec![0.0f64; mc.n_head * mc.head_dim() * mc.d_state];
        let mut kc: Vec<f64> = Vec::new();
        let mut vc: Vec<f64> = Vec::new();

        // layer 0: recurrent
        let mut hn = vec![0.0f64; n_t * NE];
        for t in 0..n_t {
            hn[t * NE..(t + 1) * NE].copy_from_slice(&rmsf(
                &h[t * NE..(t + 1) * NE],
                &r.an[0],
                EPS,
            ));
        }
        let o0 = mamba_oracle(&mc, r, &mut conv_state, &mut ssm_state, &hn);
        for i in 0..n_t * NE {
            h[i] += o0[i];
        }
        // layer 1: attention
        for t in 0..n_t {
            hn[t * NE..(t + 1) * NE].copy_from_slice(&rmsf(
                &h[t * NE..(t + 1) * NE],
                &r.an[1],
                EPS,
            ));
        }
        let o1 = attn_oracle(&ac, r, &mut kc, &mut vc, 0, &hn);
        for i in 0..n_t * NE {
            h[i] += o1[i];
        }
        // layer 2: moe
        for t in 0..n_t {
            hn[t * NE..(t + 1) * NE].copy_from_slice(&rmsf(
                &h[t * NE..(t + 1) * NE],
                &r.an[2],
                EPS,
            ));
            let o2 = moe_oracle(&fc, r, &hn[t * NE..(t + 1) * NE]);
            for i in 0..NE {
                h[t * NE + i] += o2[i];
            }
        }
        // final norm + output
        let mut logits = vec![0.0f64; n_t * NV];
        for t in 0..n_t {
            let hf = rmsf(&h[t * NE..(t + 1) * NE], &r.output_norm, EPS);
            logits[t * NV..(t + 1) * NV].copy_from_slice(&mvf(&r.output, NV, NE, &hf));
        }
        logits
    }

    fn max_abs(a: &[f32], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(g, r)| (*g as f64 - r).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn hybrid_prefill_matches_independent_f64_oracle_and_logits_finite() {
        let (model, raw) = build();
        let tokens: Vec<u32> = vec![3, 7, 1, 9, 0, 5];
        let mut st = model.new_state().unwrap();
        let logits = model.forward(&mut st, &tokens).unwrap();
        assert_eq!(logits.len(), tokens.len() * NV);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must be finite"
        );
        let want = hybrid_oracle(&raw, &tokens);
        let ma = max_abs(&logits, &want);
        assert!(ma < 2e-3, "hybrid logits max_abs {ma}");
        assert_eq!(st.position(), tokens.len());
        // top-1 of the last token must agree with the oracle.
        let got_top = model.argmax_last(&logits).unwrap();
        let last = &want[want.len() - NV..];
        let want_top = last
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 {
                    (i, v)
                } else {
                    b
                }
            })
            .0;
        assert_eq!(got_top, want_top, "top-1 token must match oracle");
    }

    #[test]
    fn hybrid_single_token_matches_oracle() {
        let (model, raw) = build();
        let tokens = vec![4u32];
        let mut st = model.new_state().unwrap();
        let logits = model.forward(&mut st, &tokens).unwrap();
        let want = hybrid_oracle(&raw, &tokens);
        assert!(max_abs(&logits, &want) < 2e-3);
    }

    #[test]
    fn hybrid_prefill_equals_sequential_decode() {
        let (model, _raw) = build();
        let tokens: Vec<u32> = vec![2, 8, 5, 1, 11, 6, 0];
        // one-shot prefill
        let mut s_pre = model.new_state().unwrap();
        let pre = model.forward(&mut s_pre, &tokens).unwrap();
        // sequential decode
        let mut s_dec = model.new_state().unwrap();
        let mut dec = Vec::new();
        for &tok in &tokens {
            dec.extend(model.forward(&mut s_dec, &[tok]).unwrap());
        }
        let ma = pre
            .iter()
            .zip(&dec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(ma < 1e-4, "prefill vs sequential decode max_abs {ma}");
        assert_eq!(s_pre.position(), s_dec.position());
    }

    #[test]
    fn different_single_tokens_produce_different_logits() {
        // Structural input-sensitivity proof (reference-free): two different input
        // tokens must yield different embeddings AND different final logits. This
        // shows the full hybrid pipeline (embed -> Mamba -> attention -> MoE ->
        // norm -> output) propagates the input, so an identical top-1 token for
        // different inputs on the real model is not, by itself, a wiring defect.
        let (model, raw) = build();
        let (mut sa, mut sb) = (model.new_state().unwrap(), model.new_state().unwrap());
        let la = model.forward(&mut sa, &[3u32]).unwrap();
        let lb = model.forward(&mut sb, &[9u32]).unwrap();
        // embeddings differ
        let emb_diff: f32 = (0..NE)
            .map(|i| (raw.tok_embd[3 * NE + i] - raw.tok_embd[9 * NE + i]).abs())
            .sum();
        assert!(
            emb_diff > 0.0,
            "distinct tokens must have distinct embeddings"
        );
        // final logits differ
        assert_ne!(la, lb, "distinct inputs must yield distinct logits");
        let l2: f64 = la
            .iter()
            .zip(&lb)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(l2 > 1e-4, "logit L2 distance must be nonzero (got {l2})");
    }

    #[test]
    fn reset_then_repeat_is_deterministic() {
        let (model, _raw) = build();
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let mut st = model.new_state().unwrap();
        let a = model.forward(&mut st, &tokens).unwrap();
        st.reset();
        assert_eq!(st.position(), 0);
        let b = model.forward(&mut st, &tokens).unwrap();
        assert_eq!(a, b, "reset + repeat must reproduce identical logits");
    }

    #[test]
    fn invalid_token_and_empty_and_state_mismatch_rejected() {
        let (model, _raw) = build();
        let mut st = model.new_state().unwrap();
        // out-of-vocab token
        assert!(model.forward(&mut st, &[NV as u32]).is_err());
        // empty input
        assert!(model.forward(&mut st, &[]).is_err());
    }
}
