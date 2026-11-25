//! Complete executable Mamba-2 block for the `nemotron_h_moe` hybrid layers.
//!
//! This assembles the two independently-validated cores — the causal depthwise
//! convolution ([`crate::mamba2_conv`]) and the selective SSM scan
//! ([`crate::mamba2_ssm`]) — into the full block, exactly as the pinned reference
//! graph `llm_build_mamba_base::build_mamba2_layer`
//! (build/reference/src/models/mamba-base.cpp:153, revision cd959ba) defines it.
//! Neither core is reimplemented here; this module is the surrounding glue.
//!
//! ## Exact composition (from the pinned graph, not inferred)
//!
//! With `silu(v) = v * sigmoid(v)`, per token (`cur` has dimension `n_embd`):
//!
//! ```text
//!   zxBCdt = ssm_in · cur                 // [d_in_proj]; in-projection
//!   split zxBCdt:
//!       z   = zxBCdt[0 .. d_inner]                               // gate
//!       xBC = zxBCdt[d_inner .. d_inner + d_conv_ch]             // conv input
//!       dt  = zxBCdt[2*d_inner + 2*n_group*d_state .. + n_head]  // raw dt
//!   where d_conv_ch = d_inner + 2*n_group*d_state.
//!
//!   xBC = ssm_conv(conv_state, xBC)       // causal depthwise conv over d_conv_ch
//!   xBC = silu(xBC + conv_bias)
//!   split xBC:
//!       x = xBC[0 .. d_inner]                        // [head_dim, n_head]
//!       B = xBC[d_inner .. + n_group*d_state]        // [d_state, n_group]
//!       C = xBC[d_inner + n_group*d_state .. + n_group*d_state]
//!
//!   dt = dt + dt_bias
//!   y  = ssm_scan(ssm_state, x, dt, A, B, C)         // [head_dim, n_head]
//!   y  = y + x * D                                   // per-head D skip term
//!   y  = silu(z) * y                                 // gated (swiglu_split(z, y))
//!   y  = grouped_rms_norm(y) * ssm_norm              // n_group groups, if present
//!   cur = ssm_out · y                                // [n_embd]; out-projection
//! ```
//!
//! The pre-layer RMS norm and the residual add are performed by the enclosing
//! decoder block, not here (matching the reference, where `build_mamba2_layer`
//! returns `mamba_out` before the residual). Convolution and SSM state are
//! persistent and session-local. Matvecs accumulate in `f32` to mirror the ggml
//! CPU path; the dt bias and gating are each applied exactly once.

use crate::latent_moe_ffn::LinearF32;
use crate::mamba2_conv::{conv_forward, ConvState, ConvWeights};
use crate::mamba2_ssm::{ssm_scan, SsmConfig, SsmState};

/// Static shape parameters of one Mamba-2 block.
#[derive(Debug, Clone, Copy)]
pub struct Mamba2BlockConfig {
    pub n_embd: usize,
    pub d_inner: usize,
    pub d_conv: usize,
    pub d_state: usize,
    pub n_head: usize,
    pub n_group: usize,
    pub ssm_norm_eps: f32,
}

impl Mamba2BlockConfig {
    pub fn head_dim(&self) -> usize {
        self.d_inner / self.n_head
    }
    /// Channels carried through the convolution: `x`, `B`, and `C` stacked.
    pub fn conv_channels(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }
    /// Width of the input projection output.
    pub fn d_in_proj(&self) -> usize {
        2 * self.d_inner + 2 * self.n_group * self.d_state + self.n_head
    }

    fn validate(&self) -> Result<(), String> {
        if self.n_embd == 0
            || self.d_inner == 0
            || self.d_conv == 0
            || self.d_state == 0
            || self.n_head == 0
            || self.n_group == 0
        {
            return Err("Mamba2 block dimensions must be non-zero".into());
        }
        if self.d_inner % self.n_head != 0 {
            return Err("d_inner must be a multiple of n_head".into());
        }
        if self.d_inner % self.n_group != 0 {
            return Err("d_inner must be a multiple of n_group".into());
        }
        if !self.ssm_norm_eps.is_finite() || self.ssm_norm_eps < 0.0 {
            return Err("ssm_norm_eps must be finite and non-negative".into());
        }
        Ok(())
    }

    fn ssm_config(&self) -> SsmConfig {
        SsmConfig {
            n_head: self.n_head,
            dim: self.head_dim(),
            d_state: self.d_state,
            n_group: self.n_group,
        }
    }
}

/// Dense weights of one Mamba-2 block.
#[derive(Debug, Clone)]
pub struct Mamba2BlockWeights {
    cfg: Mamba2BlockConfig,
    ssm_in: LinearF32,          // [d_in_proj, n_embd]
    conv: ConvWeights,          // channels = conv_channels, kernel = d_conv
    conv_bias: Vec<f32>,        // [conv_channels]
    dt_bias: Vec<f32>,          // [n_head]
    a: Vec<f32>,                // [n_head] scalar-A (negative)
    d: Vec<f32>,                // [n_head] skip term
    ssm_norm: Option<Vec<f32>>, // [d_inner], grouped RMS gain
    ssm_out: LinearF32,         // [n_embd, d_inner]
}

/// Persistent, session-local recurrent state for one Mamba-2 block.
#[derive(Debug, Clone)]
pub struct Mamba2BlockState {
    conv: ConvState,
    ssm: SsmState,
}

impl Mamba2BlockState {
    pub fn zeros(cfg: &Mamba2BlockConfig) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            conv: ConvState::zeros(cfg.conv_channels(), cfg.d_conv)?,
            ssm: SsmState::zeros(cfg.ssm_config())?,
        })
    }

    pub fn reset(&mut self) {
        self.conv.reset();
        self.ssm.reset();
    }

    /// Total recurrent state bytes held (conv + SSM), for memory accounting.
    pub fn state_bytes(&self) -> usize {
        self.conv.state_bytes() + self.ssm.state_bytes()
    }

    /// Read-only view of the SSM state (for tests / inspection).
    pub fn ssm_state(&self) -> &[f32] {
        self.ssm.raw()
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    // x * sigmoid(x), numerically stable for large |x|.
    let s = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    };
    x * s
}

impl Mamba2BlockWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: Mamba2BlockConfig,
        ssm_in: LinearF32,
        conv: ConvWeights,
        conv_bias: Vec<f32>,
        dt_bias: Vec<f32>,
        a: Vec<f32>,
        d: Vec<f32>,
        ssm_norm: Option<Vec<f32>>,
        ssm_out: LinearF32,
    ) -> Result<Self, String> {
        cfg.validate()?;
        if ssm_in.out_dim() != cfg.d_in_proj() || ssm_in.in_dim() != cfg.n_embd {
            return Err("ssm_in shape must be [d_in_proj, n_embd]".into());
        }
        if conv.channels() != cfg.conv_channels() || conv.kernel_width() != cfg.d_conv {
            return Err("conv weights shape must be [conv_channels, d_conv]".into());
        }
        if conv_bias.len() != cfg.conv_channels() {
            return Err("conv_bias length must equal conv_channels".into());
        }
        if dt_bias.len() != cfg.n_head {
            return Err("dt_bias length must equal n_head".into());
        }
        if a.len() != cfg.n_head {
            return Err("A length must equal n_head".into());
        }
        if d.len() != cfg.n_head {
            return Err("D length must equal n_head".into());
        }
        if let Some(n) = &ssm_norm {
            if n.len() != cfg.d_inner {
                return Err("ssm_norm length must equal d_inner".into());
            }
            if !n.iter().all(|v| v.is_finite()) {
                return Err("non-finite ssm_norm rejected".into());
            }
        }
        if ssm_out.out_dim() != cfg.n_embd || ssm_out.in_dim() != cfg.d_inner {
            return Err("ssm_out shape must be [n_embd, d_inner]".into());
        }
        for (slice, name) in [
            (&conv_bias, "conv_bias"),
            (&dt_bias, "dt_bias"),
            (&a, "A"),
            (&d, "D"),
        ] {
            if !slice.iter().all(|v| v.is_finite()) {
                return Err(format!("non-finite {name} rejected"));
            }
        }
        Ok(Self {
            cfg,
            ssm_in,
            conv,
            conv_bias,
            dt_bias,
            a,
            d,
            ssm_norm,
            ssm_out,
        })
    }

    pub fn config(&self) -> &Mamba2BlockConfig {
        &self.cfg
    }

    /// Run the block over `cur` (`n_t` tokens, `n_embd` each, token-outermost),
    /// advancing `state`. Returns the `n_t * n_embd` pre-residual outputs.
    /// Handles both multi-token prefill and single-token decode identically.
    pub fn forward(&self, state: &mut Mamba2BlockState, cur: &[f32]) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        c.validate()?;
        if cur.is_empty() || cur.len() % c.n_embd != 0 {
            return Err("Mamba2 input length is not a positive multiple of n_embd".into());
        }
        if !cur.iter().all(|v| v.is_finite()) {
            return Err("non-finite Mamba2 input rejected".into());
        }
        let n_t = cur.len() / c.n_embd;
        let d_inner = c.d_inner;
        let cc = c.conv_channels();
        let gds = c.n_group * c.d_state;
        let head_dim = c.head_dim();

        // 1) In-projection + split into z, xBC(conv input), dt (per token).
        let mut z_all = vec![0.0f32; n_t * d_inner];
        let mut conv_in = vec![0.0f32; n_t * cc];
        let mut dt_all = vec![0.0f32; n_t * c.n_head];
        let dt_off = 2 * d_inner + gds;
        for t in 0..n_t {
            let proj = self.ssm_in.matvec(&cur[t * c.n_embd..(t + 1) * c.n_embd])?;
            z_all[t * d_inner..(t + 1) * d_inner].copy_from_slice(&proj[0..d_inner]);
            conv_in[t * cc..(t + 1) * cc].copy_from_slice(&proj[d_inner..d_inner + cc]);
            for h in 0..c.n_head {
                dt_all[t * c.n_head + h] = proj[dt_off + h] + self.dt_bias[h];
            }
        }

        // 2) Causal depthwise conv over all conv channels, then bias + SiLU.
        let mut xbc = conv_forward(&self.conv, &mut state.conv, &conv_in)?;
        for t in 0..n_t {
            for ch in 0..cc {
                let idx = t * cc + ch;
                let v = xbc[idx] + self.conv_bias[ch];
                xbc[idx] = silu(v);
            }
        }

        // 3) Split conv output into x, B, C and run the SSM scan.
        let mut x_all = vec![0.0f32; n_t * d_inner];
        let mut b_all = vec![0.0f32; n_t * gds];
        let mut c_all = vec![0.0f32; n_t * gds];
        for t in 0..n_t {
            let base = t * cc;
            x_all[t * d_inner..(t + 1) * d_inner].copy_from_slice(&xbc[base..base + d_inner]);
            b_all[t * gds..(t + 1) * gds]
                .copy_from_slice(&xbc[base + d_inner..base + d_inner + gds]);
            c_all[t * gds..(t + 1) * gds]
                .copy_from_slice(&xbc[base + d_inner + gds..base + d_inner + 2 * gds]);
        }
        let y_ssm = ssm_scan(
            &c.ssm_config(),
            &self.a,
            &mut state.ssm,
            &x_all,
            &dt_all,
            &b_all,
            &c_all,
        )?;

        // 4) D skip, SiLU gate, grouped RMS norm, out-projection — per token.
        let gsize = d_inner / c.n_group;
        let mut out = vec![0.0f32; n_t * c.n_embd];
        for t in 0..n_t {
            let mut y = vec![0.0f32; d_inner];
            for h in 0..c.n_head {
                for i1 in 0..head_dim {
                    let ch = h * head_dim + i1;
                    let xi = x_all[t * d_inner + ch];
                    // D skip term (per-head D broadcast over head_dim).
                    let yi = y_ssm[t * d_inner + ch] + xi * self.d[h];
                    // SiLU gate from z.
                    y[ch] = silu(z_all[t * d_inner + ch]) * yi;
                }
            }
            // Grouped RMS norm over consecutive groups of gsize channels.
            if let Some(norm) = &self.ssm_norm {
                for g in 0..c.n_group {
                    let lo = g * gsize;
                    let hi = lo + gsize;
                    let mut ms = 0.0f32;
                    for &v in &y[lo..hi] {
                        ms += v * v;
                    }
                    ms /= gsize as f32;
                    let inv = 1.0f32 / (ms + c.ssm_norm_eps).sqrt();
                    for ch in lo..hi {
                        y[ch] = y[ch] * inv * norm[ch];
                    }
                }
            }
            for (o, v) in out[t * c.n_embd..(t + 1) * c.n_embd]
                .iter_mut()
                .zip(self.ssm_out.matvec(&y)?.into_iter())
            {
                if !v.is_finite() {
                    return Err("non-finite Mamba2 output rejected".into());
                }
                *o = v;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Independent f64 oracle: a wholly separate re-derivation of the block,
    // including its own conv and scan, in f64. It shares no code with the
    // production f32 path (which reuses mamba2_conv / mamba2_ssm). ----

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn oracle(
        cfg: &Mamba2BlockConfig,
        ssm_in: &[f32], // [d_in_proj * n_embd]
        conv_w: &[f32], // [conv_channels * d_conv], c-major (w[c*k+j])
        conv_b: &[f32],
        dt_b: &[f32],
        a: &[f32],
        d: &[f32],
        norm: Option<&[f32]>,
        ssm_out: &[f32], // [n_embd * d_inner]
        cur: &[f32],
    ) -> (Vec<f64>, Vec<f64>) {
        let n_embd = cfg.n_embd;
        let d_inner = cfg.d_inner;
        let cc = cfg.conv_channels();
        let gds = cfg.n_group * cfg.d_state;
        let head_dim = cfg.head_dim();
        let k = cfg.d_conv;
        let n_t = cur.len() / n_embd;
        let dip = cfg.d_in_proj();

        let silu = |x: f64| x * (1.0 / (1.0 + (-x).exp()));
        let mv = |w: &[f32], out: usize, inn: usize, x: &[f64]| -> Vec<f64> {
            (0..out)
                .map(|o| (0..inn).map(|i| w[o * inn + i] as f64 * x[i]).sum())
                .collect()
        };

        // Persistent f64 states.
        let mut conv_state = vec![0.0f64; cc * (k - 1)]; // s[c*(k-1)+j], oldest first
        let mut ssm_state = vec![0.0f64; cfg.n_head * head_dim * cfg.d_state];
        let hpg = cfg.n_head / cfg.n_group;

        let mut out = vec![0.0f64; n_t * n_embd];
        for t in 0..n_t {
            let x_in: Vec<f64> = cur[t * n_embd..(t + 1) * n_embd]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let proj = mv(ssm_in, dip, n_embd, &x_in);
            let z = &proj[0..d_inner];
            let xbc_in = &proj[d_inner..d_inner + cc];
            let dt: Vec<f64> = (0..cfg.n_head)
                .map(|h| proj[2 * d_inner + gds + h] + dt_b[h] as f64)
                .collect();

            // conv per channel: window = [state(k-1) .. current], then bias + silu.
            let mut xbc = vec![0.0f64; cc];
            for ch in 0..cc {
                let mut acc = 0.0f64;
                for j in 0..k {
                    let v = if j < k - 1 {
                        conv_state[ch * (k - 1) + j]
                    } else {
                        xbc_in[ch]
                    };
                    acc += v * conv_w[ch * k + j] as f64;
                }
                xbc[ch] = silu(acc + conv_b[ch] as f64);
            }
            // advance conv state: last k-1 of (state ++ current)
            for ch in 0..cc {
                if k >= 2 {
                    for j in 0..(k - 2) {
                        conv_state[ch * (k - 1) + j] = conv_state[ch * (k - 1) + j + 1];
                    }
                    conv_state[ch * (k - 1) + (k - 2)] = xbc_in[ch];
                }
            }

            // split conv output
            let x = &xbc[0..d_inner];
            let b = &xbc[d_inner..d_inner + gds];
            let cc_ = &xbc[d_inner + gds..d_inner + 2 * gds];

            // SSM scan (scalar-A), advancing state; produce y.
            let mut y = vec![0.0f64; d_inner];
            for h in 0..cfg.n_head {
                let dtsp = {
                    let zz = dt[h];
                    if zz > 20.0 {
                        zz
                    } else {
                        zz.exp().ln_1p()
                    }
                };
                let da = (dtsp * a[h] as f64).exp();
                let g = h / hpg;
                for i1 in 0..head_dim {
                    let ch = h * head_dim + i1;
                    let x_dt = x[ch] * dtsp;
                    let mut sum = 0.0f64;
                    for i0 in 0..cfg.d_state {
                        let si = ch * cfg.d_state + i0;
                        let ig = g * cfg.d_state + i0;
                        let st = ssm_state[si] * da + b[ig] * x_dt;
                        sum += st * cc_[ig];
                        ssm_state[si] = st;
                    }
                    y[ch] = sum;
                }
            }
            // D skip + silu gate
            for h in 0..cfg.n_head {
                for i1 in 0..head_dim {
                    let ch = h * head_dim + i1;
                    let yi = y[ch] + x[ch] * d[h] as f64;
                    y[ch] = silu(z[ch]) * yi;
                }
            }
            // grouped RMS norm
            if let Some(norm) = norm {
                let gsize = d_inner / cfg.n_group;
                for grp in 0..cfg.n_group {
                    let lo = grp * gsize;
                    let hi = lo + gsize;
                    let ms: f64 = y[lo..hi].iter().map(|v| v * v).sum::<f64>() / gsize as f64;
                    let inv = 1.0 / (ms + cfg.ssm_norm_eps as f64).sqrt();
                    for ch in lo..hi {
                        y[ch] = y[ch] * inv * norm[ch] as f64;
                    }
                }
            }
            let o = mv(ssm_out, n_embd, d_inner, &y);
            out[t * n_embd..(t + 1) * n_embd].copy_from_slice(&o);
        }
        (out, ssm_state)
    }

    // Deterministic small-magnitude weight generator.
    fn gen(tag: &str, n: usize, scale: f32, shift: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = tag.bytes().fold(2166136261u32, |acc, byte| {
                    (acc ^ byte as u32).wrapping_mul(16777619)
                });
                let k = h.wrapping_add((i as u32).wrapping_mul(2654435761));
                ((k % 1000) as f32 / 500.0 - 1.0) * scale + shift
            })
            .collect()
    }

    #[allow(clippy::type_complexity)]
    fn make(
        cfg: Mamba2BlockConfig,
        with_norm: bool,
    ) -> (
        Mamba2BlockWeights,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Option<Vec<f32>>,
        Vec<f32>,
    ) {
        let dip = cfg.d_in_proj();
        let cc = cfg.conv_channels();
        let ssm_in_w = gen("in", dip * cfg.n_embd, 0.3, 0.0);
        let conv_w = gen("conv", cc * cfg.d_conv, 0.4, 0.0);
        let conv_b = gen("cb", cc, 0.1, 0.0);
        let dt_b = gen("dtb", cfg.n_head, 0.2, -0.5);
        // A must be negative for a stable decay.
        let a = gen("a", cfg.n_head, 0.5, -1.5);
        let d = gen("d", cfg.n_head, 0.3, 0.0);
        let norm = if with_norm {
            Some(gen("norm", cfg.d_inner, 0.2, 1.0))
        } else {
            None
        };
        let ssm_out_w = gen("out", cfg.n_embd * cfg.d_inner, 0.3, 0.0);
        let w = Mamba2BlockWeights::new(
            cfg,
            LinearF32::new(dip, cfg.n_embd, ssm_in_w.clone()).unwrap(),
            ConvWeights::new(cc, cfg.d_conv, conv_w.clone()).unwrap(),
            conv_b.clone(),
            dt_b.clone(),
            a.clone(),
            d.clone(),
            norm.clone(),
            LinearF32::new(cfg.n_embd, cfg.d_inner, ssm_out_w.clone()).unwrap(),
        )
        .unwrap();
        (w, ssm_in_w, conv_w, conv_b, dt_b, a, d, norm, ssm_out_w)
    }

    fn test_cfg() -> Mamba2BlockConfig {
        // Small but structurally faithful: n_head and n_group both divide d_inner,
        // and n_group != n_head so the grouped norm is non-trivial.
        Mamba2BlockConfig {
            n_embd: 6,
            d_inner: 8,
            d_conv: 4,
            d_state: 3,
            n_head: 4,
            n_group: 2,
            ssm_norm_eps: 1e-5,
        }
    }

    #[test]
    fn block_matches_independent_f64_reference() {
        let cfg = test_cfg();
        let (w, inw, cw, cb, dtb, a, d, norm, outw) = make(cfg, true);
        let n_t = 5usize;
        let cur = gen("x", n_t * cfg.n_embd, 1.0, 0.0);

        let mut st = Mamba2BlockState::zeros(&cfg).unwrap();
        let got = w.forward(&mut st, &cur).unwrap();

        let (want, want_state) = oracle(
            &cfg,
            &inw,
            &cw,
            &cb,
            &dtb,
            &a,
            &d,
            norm.as_deref(),
            &outw,
            &cur,
        );

        let mut max_abs = 0f64;
        let mut sse = 0f64;
        for (g, r) in got.iter().zip(&want) {
            let e = (*g as f64 - r).abs();
            max_abs = max_abs.max(e);
            sse += e * e;
        }
        let rmse = (sse / want.len() as f64).sqrt();
        assert!(max_abs < 1e-3, "output max_abs {max_abs}");
        assert!(rmse < 1e-4, "output rmse {rmse}");
        // Final recurrent SSM state must also match.
        let mut st_max = 0f64;
        for (g, r) in st.ssm_state().iter().zip(&want_state) {
            st_max = st_max.max((*g as f64 - r).abs());
        }
        assert!(st_max < 1e-3, "ssm state max_abs {st_max}");
    }

    #[test]
    fn works_without_grouped_norm() {
        let cfg = test_cfg();
        let (w, inw, cw, cb, dtb, a, d, _norm, outw) = make(cfg, false);
        let n_t = 3usize;
        let cur = gen("xn", n_t * cfg.n_embd, 1.0, 0.0);
        let mut st = Mamba2BlockState::zeros(&cfg).unwrap();
        let got = w.forward(&mut st, &cur).unwrap();
        let (want, _) = oracle(&cfg, &inw, &cw, &cb, &dtb, &a, &d, None, &outw, &cur);
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(g, r)| (*g as f64 - r).abs())
            .fold(0.0, f64::max);
        assert!(max_abs < 1e-3, "no-norm max_abs {max_abs}");
    }

    #[test]
    fn prefill_equals_sequential_single_token_decode() {
        let cfg = test_cfg();
        let (w, ..) = make(cfg, true);
        let n_t = 6usize;
        let cur = gen("seq", n_t * cfg.n_embd, 1.0, 0.0);

        let mut s_pre = Mamba2BlockState::zeros(&cfg).unwrap();
        let prefill = w.forward(&mut s_pre, &cur).unwrap();

        let mut s_dec = Mamba2BlockState::zeros(&cfg).unwrap();
        let mut decode = Vec::new();
        for t in 0..n_t {
            let tok = &cur[t * cfg.n_embd..(t + 1) * cfg.n_embd];
            decode.extend(w.forward(&mut s_dec, tok).unwrap());
        }
        // The prefill batches conv/scan but the arithmetic is identical, so the
        // results match to f32 rounding.
        let max_abs = prefill
            .iter()
            .zip(&decode)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 1e-5, "prefill vs decode max_abs {max_abs}");
        assert_eq!(
            s_pre.ssm_state(),
            s_dec.ssm_state(),
            "final state must match"
        );
    }

    #[test]
    fn reset_and_session_isolation() {
        let cfg = test_cfg();
        let (w, ..) = make(cfg, true);
        let cur = gen("iso", cfg.n_embd, 1.0, 0.0);
        let mut a = Mamba2BlockState::zeros(&cfg).unwrap();
        let b = Mamba2BlockState::zeros(&cfg).unwrap();
        let _ = w.forward(&mut a, &cur).unwrap();
        assert!(
            a.ssm_state().iter().any(|&v| v != 0.0),
            "state should advance"
        );
        assert!(
            b.ssm_state().iter().all(|&v| v == 0.0),
            "independent session must be untouched"
        );
        a.reset();
        assert!(
            a.ssm_state().iter().all(|&v| v == 0.0),
            "reset clears state"
        );
    }

    #[test]
    fn invalid_inputs_and_shapes_are_rejected() {
        let cfg = test_cfg();
        let (w, ..) = make(cfg, true);
        let mut st = Mamba2BlockState::zeros(&cfg).unwrap();
        // Wrong input length (not a multiple of n_embd).
        assert!(w.forward(&mut st, &[0.0; 5]).is_err());
        // Empty input.
        assert!(w.forward(&mut st, &[]).is_err());
        // Non-finite input.
        let mut bad = vec![0.0f32; cfg.n_embd];
        bad[0] = f32::NAN;
        assert!(w.forward(&mut st, &bad).is_err());
        // Malformed construction: wrong ssm_out shape.
        let dip = cfg.d_in_proj();
        let cc = cfg.conv_channels();
        let bad_build = Mamba2BlockWeights::new(
            cfg,
            LinearF32::new(dip, cfg.n_embd, vec![0.0; dip * cfg.n_embd]).unwrap(),
            ConvWeights::new(cc, cfg.d_conv, vec![0.0; cc * cfg.d_conv]).unwrap(),
            vec![0.0; cc],
            vec![0.0; cfg.n_head],
            vec![-1.0; cfg.n_head],
            vec![0.0; cfg.n_head],
            None,
            // wrong: out_dim should be n_embd, not d_inner
            LinearF32::new(
                cfg.d_inner,
                cfg.d_inner,
                vec![0.0; cfg.d_inner * cfg.d_inner],
            )
            .unwrap(),
        );
        assert!(bad_build.is_err());
    }
}
