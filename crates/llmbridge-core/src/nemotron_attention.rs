//! Nemotron (`nemotron_h_moe`) attention sub-layer for the hybrid decoder.
//!
//! Faithful CPU implementation of the pinned reference attention path
//! `llama_model_nemotron_h::graph::build_attention_layer`
//! (build/reference/src/models/nemotron-h.cpp:278) -> `build_qkv` ->
//! `build_attn` -> `build_attn_mha` (build/reference/src/llama-graph.cpp, rev
//! cd959ba). Verified against the real model metadata (shard 1 header):
//!
//!   * `attention.head_count = 32` query heads (scalar),
//!   * `attention.head_count_kv = 2` on the attention layers (GQA 16:1); the
//!     per-layer array is 0 on recurrent layers and 2 on the 8 attention layers
//!     (indices 7,16,25,36,47,58,69,78 for the 88-layer 120B),
//!   * `attention.key_length = value_length = 128` (head dim),
//!   * **NoPE**: the reference `build_attention_layer` applies NO RoPE and NO
//!     Q/K normalization — it is a plain projection -> attention. (The GGUF
//!     carries `rope.*` keys, but the pinned graph does not use them for this
//!     arch; this is the "do not assume generic Llama attention" case.)
//!   * no attention sinks, no sliding window (`swa_type == NONE` is asserted),
//!   * `f_attention_scale` is absent, so the scale is `1/sqrt(head_dim)`,
//!   * separate `wq`/`wk`/`wv`; output projection `wo` with optional bias `wo_b`.
//!
//! The pre-layer RMS norm and the residual add are the enclosing block's job
//! (`build_attention_layer` returns the attention output before the residual),
//! matching [`crate::mamba2_block`] and [`crate::latent_moe_ffn`].
//!
//! Memory: the KV cache is a fixed-capacity store of `capacity` positions; each
//! query is scored against only the currently-cached keys (an `O(n_kv)` score
//! buffer per query), so no full context×context matrix is ever materialized.
//! Scores/dots accumulate in `f32` to mirror the ggml CPU path.

use crate::latent_moe_ffn::LinearF32;

/// Static shape parameters of one attention sub-layer.
#[derive(Debug, Clone, Copy)]
pub struct AttnConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    /// Maximum number of cached positions (context capacity).
    pub capacity: usize,
}

impl AttnConfig {
    fn validate(&self) -> Result<(), String> {
        if self.n_embd == 0
            || self.n_head == 0
            || self.n_head_kv == 0
            || self.head_dim == 0
            || self.capacity == 0
        {
            return Err("attention dimensions must be non-zero".into());
        }
        if self.n_head % self.n_head_kv != 0 {
            return Err("n_head must be a multiple of n_head_kv (GQA)".into());
        }
        if self.capacity > (1 << 24) {
            return Err("attention capacity exceeds bounded limit".into());
        }
        Ok(())
    }

    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
    /// `1/sqrt(head_dim)` (the model sets no explicit attention scale).
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// Dense weights of one attention sub-layer.
#[derive(Debug, Clone)]
pub struct AttnWeights {
    cfg: AttnConfig,
    wq: LinearF32,          // [q_dim, n_embd]
    wk: LinearF32,          // [kv_dim, n_embd]
    wv: LinearF32,          // [kv_dim, n_embd]
    wo: LinearF32,          // [n_embd, q_dim]
    wo_b: Option<Vec<f32>>, // [n_embd]
}

/// Persistent, session-local KV cache for one attention sub-layer.
#[derive(Debug, Clone)]
pub struct AttnState {
    cfg: AttnConfig,
    k: Vec<f32>, // [capacity * kv_dim], k[pos*kv_dim + g*head_dim + d]
    v: Vec<f32>,
    pos: usize, // number of cached positions
}

impl AttnState {
    pub fn zeros(cfg: &AttnConfig) -> Result<Self, String> {
        cfg.validate()?;
        let n = cfg
            .capacity
            .checked_mul(cfg.kv_dim())
            .ok_or("KV cache size overflow")?;
        Ok(Self {
            cfg: *cfg,
            k: vec![0.0; n],
            v: vec![0.0; n],
            pos: 0,
        })
    }

    pub fn reset(&mut self) {
        self.k.iter_mut().for_each(|x| *x = 0.0);
        self.v.iter_mut().for_each(|x| *x = 0.0);
        self.pos = 0;
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bytes of KV cache held (k + v), for memory accounting.
    pub fn kv_bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * std::mem::size_of::<f32>()
    }
}

#[inline]
fn stable_softmax_inplace(s: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &x in s.iter() {
        if x > m {
            m = x;
        }
    }
    let mut sum = 0.0f64;
    for x in s.iter_mut() {
        let e = ((*x - m) as f64).exp();
        *x = e as f32;
        sum += e;
    }
    if sum > 0.0 {
        for x in s.iter_mut() {
            *x = (*x as f64 / sum) as f32;
        }
    }
}

impl AttnWeights {
    pub fn new(
        cfg: AttnConfig,
        wq: LinearF32,
        wk: LinearF32,
        wv: LinearF32,
        wo: LinearF32,
        wo_b: Option<Vec<f32>>,
    ) -> Result<Self, String> {
        cfg.validate()?;
        if wq.out_dim() != cfg.q_dim() || wq.in_dim() != cfg.n_embd {
            return Err("wq shape must be [n_head*head_dim, n_embd]".into());
        }
        if wk.out_dim() != cfg.kv_dim() || wk.in_dim() != cfg.n_embd {
            return Err("wk shape must be [n_head_kv*head_dim, n_embd]".into());
        }
        if wv.out_dim() != cfg.kv_dim() || wv.in_dim() != cfg.n_embd {
            return Err("wv shape must be [n_head_kv*head_dim, n_embd]".into());
        }
        if wo.out_dim() != cfg.n_embd || wo.in_dim() != cfg.q_dim() {
            return Err("wo shape must be [n_embd, n_head*head_dim]".into());
        }
        if let Some(b) = &wo_b {
            if b.len() != cfg.n_embd {
                return Err("wo_b length must equal n_embd".into());
            }
            if !b.iter().all(|x| x.is_finite()) {
                return Err("non-finite wo_b rejected".into());
            }
        }
        Ok(Self {
            cfg,
            wq,
            wk,
            wv,
            wo,
            wo_b,
        })
    }

    pub fn config(&self) -> &AttnConfig {
        &self.cfg
    }

    /// Run attention over `cur` (`n_t` tokens, `n_embd` each), appending the new
    /// keys/values to `state` and attending causally over all cached positions.
    /// Works identically for multi-token prefill and single-token decode.
    pub fn forward(&self, state: &mut AttnState, cur: &[f32]) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        c.validate()?;
        if state.cfg.n_embd != c.n_embd
            || state.cfg.n_head_kv != c.n_head_kv
            || state.cfg.head_dim != c.head_dim
            || state.cfg.capacity != c.capacity
        {
            return Err("KV cache config does not match weights".into());
        }
        if cur.is_empty() || cur.len() % c.n_embd != 0 {
            return Err("attention input length is not a positive multiple of n_embd".into());
        }
        if !cur.iter().all(|x| x.is_finite()) {
            return Err("non-finite attention input rejected".into());
        }
        let n_t = cur.len() / c.n_embd;
        let kv_dim = c.kv_dim();
        let head_dim = c.head_dim;
        let hpk = c.n_head / c.n_head_kv; // query heads per kv head
        let scale = c.scale();

        if state.pos.checked_add(n_t).ok_or("position overflow")? > c.capacity {
            return Err(format!(
                "attention context capacity {} exceeded (have {}, adding {})",
                c.capacity, state.pos, n_t
            ));
        }

        // 1) Project + append all new K/V to the cache first (so intra-batch
        //    causal attention sees the current token's own key).
        let base = state.pos;
        let mut q_all = vec![0.0f32; n_t * c.q_dim()];
        for t in 0..n_t {
            let x = &cur[t * c.n_embd..(t + 1) * c.n_embd];
            let q = self.wq.matvec(x)?;
            let k = self.wk.matvec(x)?;
            let v = self.wv.matvec(x)?;
            q_all[t * c.q_dim()..(t + 1) * c.q_dim()].copy_from_slice(&q);
            let slot = (base + t) * kv_dim;
            state.k[slot..slot + kv_dim].copy_from_slice(&k);
            state.v[slot..slot + kv_dim].copy_from_slice(&v);
        }

        // 2) For each query token, attend causally over cached keys [0 ..= abs_pos].
        let mut out = vec![0.0f32; n_t * c.n_embd];
        for t in 0..n_t {
            let abs = base + t; // absolute position of this query
            let n_kv = abs + 1; // causal: keys 0..=abs
            let mut attn_concat = vec![0.0f32; c.q_dim()];
            let mut scores = vec![0.0f32; n_kv];
            for h in 0..c.n_head {
                let g = h / hpk; // kv head for this query head
                let qh = &q_all[t * c.q_dim() + h * head_dim..t * c.q_dim() + (h + 1) * head_dim];
                // scores over cached keys
                for (j, sc) in scores.iter_mut().enumerate() {
                    let kslot = j * kv_dim + g * head_dim;
                    let kh = &state.k[kslot..kslot + head_dim];
                    let mut dot = 0.0f32;
                    for (qv, kv) in qh.iter().zip(kh) {
                        dot += qv * kv;
                    }
                    *sc = dot * scale;
                }
                stable_softmax_inplace(&mut scores);
                // weighted value aggregation
                let oh = &mut attn_concat[h * head_dim..(h + 1) * head_dim];
                for (j, &w) in scores.iter().enumerate() {
                    let vslot = j * kv_dim + g * head_dim;
                    let vh = &state.v[vslot..vslot + head_dim];
                    for (od, &vd) in oh.iter_mut().zip(vh) {
                        *od += w * vd;
                    }
                }
            }
            // output projection (+ optional bias)
            let mut o = self.wo.matvec(&attn_concat)?;
            if let Some(b) = &self.wo_b {
                for (ov, bv) in o.iter_mut().zip(b) {
                    *ov += bv;
                }
            }
            for v in &o {
                if !v.is_finite() {
                    return Err("non-finite attention output rejected".into());
                }
            }
            out[t * c.n_embd..(t + 1) * c.n_embd].copy_from_slice(&o);
        }

        state.pos += n_t;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent f64 attention oracle over a full sequence, from scratch (no
    /// shared code with the production path). Returns (outputs, k_cache, v_cache).
    #[allow(clippy::too_many_arguments)]
    fn oracle(
        cfg: &AttnConfig,
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        wo_b: Option<&[f32]>,
        cur: &[f32],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n_embd = cfg.n_embd;
        let head_dim = cfg.head_dim;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let hpk = cfg.n_head / cfg.n_head_kv;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let n_t = cur.len() / n_embd;
        let mv = |w: &[f32], out: usize, inn: usize, x: &[f64]| -> Vec<f64> {
            (0..out)
                .map(|o| (0..inn).map(|i| w[o * inn + i] as f64 * x[i]).sum())
                .collect()
        };
        // project all tokens, build caches
        let mut kc = vec![0.0f64; n_t * kvd];
        let mut vc = vec![0.0f64; n_t * kvd];
        let mut qc = vec![0.0f64; n_t * qd];
        for t in 0..n_t {
            let x: Vec<f64> = cur[t * n_embd..(t + 1) * n_embd]
                .iter()
                .map(|&v| v as f64)
                .collect();
            qc[t * qd..(t + 1) * qd].copy_from_slice(&mv(wq, qd, n_embd, &x));
            kc[t * kvd..(t + 1) * kvd].copy_from_slice(&mv(wk, kvd, n_embd, &x));
            vc[t * kvd..(t + 1) * kvd].copy_from_slice(&mv(wv, kvd, n_embd, &x));
        }
        let mut out = vec![0.0f64; n_t * n_embd];
        for t in 0..n_t {
            let n_kv = t + 1;
            let mut concat = vec![0.0f64; qd];
            for h in 0..cfg.n_head {
                let g = h / hpk;
                let qh = &qc[t * qd + h * head_dim..t * qd + (h + 1) * head_dim];
                let mut s = vec![0.0f64; n_kv];
                for (j, sj) in s.iter_mut().enumerate() {
                    let kh = &kc[j * kvd + g * head_dim..j * kvd + (g + 1) * head_dim];
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
                    let vh = &vc[j * kvd + g * head_dim..j * kvd + (g + 1) * head_dim];
                    for (d, &vd) in vh.iter().enumerate() {
                        concat[h * head_dim + d] += w * vd;
                    }
                }
            }
            let mut o = mv(wo, n_embd, qd, &concat);
            if let Some(b) = wo_b {
                for (ov, bv) in o.iter_mut().zip(b) {
                    *ov += *bv as f64;
                }
            }
            out[t * n_embd..(t + 1) * n_embd].copy_from_slice(&o);
        }
        (out, kc, vc)
    }

    fn gen(tag: &str, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = tag
                    .bytes()
                    .fold(2166136261u32, |a, b| (a ^ b as u32).wrapping_mul(16777619));
                let k = h.wrapping_add((i as u32).wrapping_mul(2654435761));
                ((k % 1000) as f32 / 500.0 - 1.0) * scale
            })
            .collect()
    }

    #[allow(clippy::type_complexity)]
    fn make(
        cfg: AttnConfig,
        bias: bool,
    ) -> (
        AttnWeights,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Option<Vec<f32>>,
    ) {
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let wq = gen("wq", qd * cfg.n_embd, 0.4);
        let wk = gen("wk", kvd * cfg.n_embd, 0.4);
        let wv = gen("wv", kvd * cfg.n_embd, 0.4);
        let wo = gen("wo", cfg.n_embd * qd, 0.4);
        let wo_b = if bias {
            Some(gen("wob", cfg.n_embd, 0.2))
        } else {
            None
        };
        let w = AttnWeights::new(
            cfg,
            LinearF32::new(qd, cfg.n_embd, wq.clone()).unwrap(),
            LinearF32::new(kvd, cfg.n_embd, wk.clone()).unwrap(),
            LinearF32::new(kvd, cfg.n_embd, wv.clone()).unwrap(),
            LinearF32::new(cfg.n_embd, qd, wo.clone()).unwrap(),
            wo_b.clone(),
        )
        .unwrap();
        (w, wq, wk, wv, wo, wo_b)
    }

    // Nemotron-shaped reduced config: GQA 4:2, head_dim 4 (full uses 32:2, 128).
    fn cfg_gqa() -> AttnConfig {
        AttnConfig {
            n_embd: 16,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 4,
            capacity: 32,
        }
    }

    fn max_abs(a: &[f32], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(g, r)| (*g as f64 - r).abs())
            .fold(0.0, f64::max)
    }
    fn rmse(a: &[f32], b: &[f64]) -> f64 {
        (a.iter()
            .zip(b)
            .map(|(g, r)| {
                let d = *g as f64 - r;
                d * d
            })
            .sum::<f64>()
            / a.len() as f64)
            .sqrt()
    }

    #[test]
    fn single_token_matches_oracle() {
        let cfg = cfg_gqa();
        let (w, wq, wk, wv, wo, wob) = make(cfg, true);
        let cur = gen("x1", cfg.n_embd, 1.0);
        let mut st = AttnState::zeros(&cfg).unwrap();
        let got = w.forward(&mut st, &cur).unwrap();
        let (want, _, _) = oracle(&cfg, &wq, &wk, &wv, &wo, wob.as_deref(), &cur);
        assert!(
            max_abs(&got, &want) < 1e-3,
            "max_abs {}",
            max_abs(&got, &want)
        );
        assert_eq!(st.position(), 1);
    }

    #[test]
    fn multitoken_prefill_matches_oracle_with_metrics_and_kv() {
        let cfg = cfg_gqa();
        let (w, wq, wk, wv, wo, wob) = make(cfg, true);
        let n_t = 6usize;
        let cur = gen("xp", n_t * cfg.n_embd, 1.0);
        let mut st = AttnState::zeros(&cfg).unwrap();
        let got = w.forward(&mut st, &cur).unwrap();
        let (want, kc, vc) = oracle(&cfg, &wq, &wk, &wv, &wo, wob.as_deref(), &cur);
        let ma = max_abs(&got, &want);
        let rm = rmse(&got, &want);
        assert!(ma < 1e-3, "output max_abs {ma}");
        assert!(rm < 1e-4, "output rmse {rm}");
        assert_eq!(st.position(), n_t);
        // KV cache contents must match the oracle for all cached positions.
        let kvd = cfg.kv_dim();
        let mut kmax = 0f64;
        for i in 0..n_t * kvd {
            kmax = kmax.max((st.k[i] as f64 - kc[i]).abs());
            kmax = kmax.max((st.v[i] as f64 - vc[i]).abs());
        }
        assert!(kmax < 1e-3, "kv cache max_abs {kmax}");
    }

    #[test]
    fn prefill_equals_incremental_decode() {
        let cfg = cfg_gqa();
        let (w, ..) = make(cfg, true);
        let n_t = 7usize;
        let cur = gen("xd", n_t * cfg.n_embd, 1.0);
        // one-shot prefill
        let mut s_pre = AttnState::zeros(&cfg).unwrap();
        let pre = w.forward(&mut s_pre, &cur).unwrap();
        // incremental decode, one token at a time
        let mut s_dec = AttnState::zeros(&cfg).unwrap();
        let mut dec = Vec::new();
        for t in 0..n_t {
            dec.extend(
                w.forward(&mut s_dec, &cur[t * cfg.n_embd..(t + 1) * cfg.n_embd])
                    .unwrap(),
            );
        }
        let ma = pre
            .iter()
            .zip(&dec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(ma < 1e-5, "prefill vs decode max_abs {ma}");
        assert_eq!(s_pre.k, s_dec.k, "KV cache (k) must match");
        assert_eq!(s_pre.v, s_dec.v, "KV cache (v) must match");
        assert_eq!(s_pre.position(), s_dec.position());
    }

    #[test]
    fn causality_future_tokens_do_not_affect_past_outputs() {
        let cfg = cfg_gqa();
        let (w, ..) = make(cfg, false);
        // Prefill 3 tokens, record outputs.
        let seq = gen("cz", 5 * cfg.n_embd, 1.0);
        let mut s3 = AttnState::zeros(&cfg).unwrap();
        let o3 = w.forward(&mut s3, &seq[..3 * cfg.n_embd]).unwrap();
        // Prefill 5 tokens; the first 3 outputs must be identical (causal).
        let mut s5 = AttnState::zeros(&cfg).unwrap();
        let o5 = w.forward(&mut s5, &seq).unwrap();
        let ma = o3
            .iter()
            .zip(&o5[..3 * cfg.n_embd])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            ma < 1e-6,
            "future tokens must not change past outputs: {ma}"
        );
    }

    #[test]
    fn decode_after_prefill_uses_prior_kv() {
        let cfg = cfg_gqa();
        let (w, wq, wk, wv, wo, wob) = make(cfg, true);
        let full = gen("pk", 5 * cfg.n_embd, 1.0);
        // split: prefill 3, then decode 2 more
        let mut st = AttnState::zeros(&cfg).unwrap();
        let mut got = w.forward(&mut st, &full[..3 * cfg.n_embd]).unwrap();
        got.extend(w.forward(&mut st, &full[3 * cfg.n_embd..]).unwrap());
        // oracle over the whole sequence
        let (want, _, _) = oracle(&cfg, &wq, &wk, &wv, &wo, wob.as_deref(), &full);
        assert!(
            max_abs(&got, &want) < 1e-3,
            "split prefill+decode vs oracle"
        );
        assert_eq!(st.position(), 5);
    }

    #[test]
    fn reset_and_capacity_and_invalid_geometry() {
        let cfg = cfg_gqa();
        let (w, ..) = make(cfg, false);
        let mut st = AttnState::zeros(&cfg).unwrap();
        let tok = gen("rk", cfg.n_embd, 1.0);
        let _ = w.forward(&mut st, &tok).unwrap();
        assert_eq!(st.position(), 1);
        st.reset();
        assert_eq!(st.position(), 0);
        assert!(st.k.iter().all(|&x| x == 0.0));

        // Capacity exceeded.
        let small = AttnConfig { capacity: 2, ..cfg };
        let (ws, ..) = make(small, false);
        let mut ss = AttnState::zeros(&small).unwrap();
        assert!(ws
            .forward(&mut ss, &gen("c3", 3 * small.n_embd, 1.0))
            .is_err());

        // Invalid geometry: n_head not divisible by n_head_kv.
        let bad = AttnConfig {
            n_embd: 12,
            n_head: 3,
            n_head_kv: 2,
            head_dim: 4,
            capacity: 8,
        };
        assert!(AttnState::zeros(&bad).is_err());

        // Non-finite input.
        let mut bad_in = gen("bi", cfg.n_embd, 1.0);
        bad_in[0] = f32::NAN;
        let mut st2 = AttnState::zeros(&cfg).unwrap();
        assert!(w.forward(&mut st2, &bad_in).is_err());

        // Wrong input length.
        assert!(w.forward(&mut st2, &[0.0; 3]).is_err());
    }

    #[test]
    fn mha_non_gqa_also_matches_oracle() {
        // n_head == n_head_kv (plain MHA).
        let cfg = AttnConfig {
            n_embd: 12,
            n_head: 3,
            n_head_kv: 3,
            head_dim: 4,
            capacity: 16,
        };
        let (w, wq, wk, wv, wo, wob) = make(cfg, false);
        let cur = gen("mha", 4 * cfg.n_embd, 1.0);
        let mut st = AttnState::zeros(&cfg).unwrap();
        let got = w.forward(&mut st, &cur).unwrap();
        let (want, _, _) = oracle(&cfg, &wq, &wk, &wv, &wo, wob.as_deref(), &cur);
        assert!(max_abs(&got, &want) < 1e-3);
    }
}
