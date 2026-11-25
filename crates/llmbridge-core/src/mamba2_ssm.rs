//! Mamba-2 selective state-space (SSM) scan for the `nemotron_h_moe` hybrid
//! layers.
//!
//! Faithful CPU port of the pinned ggml scalar kernel
//! `ggml_compute_forward_ssm_scan_f32`, Mamba-2 scalar-`A` branch
//! (build/reference/ggml/src/ggml-cpu/ops.cpp:10155, revision cd959ba):
//!
//! ```text
//!   dt_sp = softplus(dt[h])                     // softplus(x)= x>20 ? x : log1p(exp(x))
//!   dA    = exp(dt_sp * A[h])                   // A is scalar per head (Mamba-2)
//!   x_dt  = x[h,i1] * dt_sp
//!   for i0 in 0..d_state:                       // per head h, inner dim i1
//!     state[h,i1,i0] = state[h,i1,i0]*dA + B[g,i0]*x_dt   // g = h / (n_head/n_group)
//!     y[h,i1]       += state[h,i1,i0]*C[g,i0]
//! ```
//!
//! Accumulation is single precision to match the reference kernel. This is the
//! recurrent scan only: the preceding conv + SiLU (see `mamba2_conv`), the `dt`
//! bias/`D` skip term, gating and the in/out projections are applied elsewhere.
//!
//! Layouts (row-major): state `s[(h*dim + i1)*d_state + i0]`; per-token inputs
//! `x[h*dim + i1]`, `dt[h]`, `B[g*d_state + i0]`, `C[g*d_state + i0]`; per-head
//! decay `A[h]`. The multi-token `ssm_scan` lays tokens outermost.

#[derive(Debug, Clone, Copy)]
pub struct SsmConfig {
    pub n_head: usize,
    pub dim: usize, // inner dimension per head (nr)
    pub d_state: usize,
    pub n_group: usize,
}

impl SsmConfig {
    fn validate(&self) -> Result<(), String> {
        let Self {
            n_head,
            dim,
            d_state,
            n_group,
        } = *self;
        if n_head == 0 || dim == 0 || d_state == 0 || n_group == 0 {
            return Err("SSM config dimensions must be non-zero".into());
        }
        if n_group > n_head || n_head % n_group != 0 {
            return Err("n_head must be a positive multiple of n_group".into());
        }
        if n_head > (1 << 20) || dim > (1 << 16) || d_state > (1 << 16) {
            return Err("SSM config dimensions exceed bounded limits".into());
        }
        self.state_len().map(|_| ())
    }

    fn state_len(&self) -> Result<usize, String> {
        self.n_head
            .checked_mul(self.dim)
            .and_then(|v| v.checked_mul(self.d_state))
            .ok_or_else(|| "SSM state size overflow".into())
    }
}

/// Persistent recurrent SSM state, session-local.
#[derive(Debug, Clone)]
pub struct SsmState {
    cfg: SsmConfig,
    s: Vec<f32>,
}

impl SsmState {
    pub fn zeros(cfg: SsmConfig) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            s: vec![0.0; cfg.state_len()?],
        })
    }

    pub fn reset(&mut self) {
        self.s.iter_mut().for_each(|v| *v = 0.0);
    }

    pub fn state_bytes(&self) -> usize {
        self.s.len() * std::mem::size_of::<f32>()
    }

    /// Read-only view of the raw state (for tests / inspection).
    pub fn raw(&self) -> &[f32] {
        &self.s
    }
}

#[inline]
fn softplus(x: f32) -> f32 {
    // Matches ggml op_softplus: avoid overflow of exp for large x.
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// One recurrent step over a single token, advancing `state` and returning
/// `y` of length `n_head * dim`.
///
/// `x`: `n_head*dim`, `dt`: `n_head`, `b`/`c`: `n_group*d_state`.
pub fn ssm_step(
    cfg: &SsmConfig,
    a: &[f32],
    state: &mut SsmState,
    x: &[f32],
    dt: &[f32],
    b: &[f32],
    c: &[f32],
) -> Result<Vec<f32>, String> {
    cfg.validate()?;
    if state.cfg.n_head != cfg.n_head
        || state.cfg.dim != cfg.dim
        || state.cfg.d_state != cfg.d_state
        || state.cfg.n_group != cfg.n_group
    {
        return Err("SSM state shape does not match config".into());
    }
    let (nh, dim, nc, ng) = (cfg.n_head, cfg.dim, cfg.d_state, cfg.n_group);
    if a.len() != nh
        || x.len() != nh * dim
        || dt.len() != nh
        || b.len() != ng * nc
        || c.len() != ng * nc
    {
        return Err("SSM input slice length disagrees with config".into());
    }
    for (slice, name) in [(a, "A"), (x, "x"), (dt, "dt"), (b, "B"), (c, "C")] {
        if !slice.iter().all(|v| v.is_finite()) {
            return Err(format!("non-finite SSM input ({name}) rejected"));
        }
    }
    let heads_per_group = nh / ng;
    let mut y = vec![0.0f32; nh * dim];
    for h in 0..nh {
        let dt_sp = softplus(dt[h]);
        let d_a = (dt_sp * a[h]).exp();
        if !d_a.is_finite() {
            return Err("non-finite SSM decay factor".into());
        }
        let g = h / heads_per_group;
        for i1 in 0..dim {
            let ii = h * dim + i1;
            let x_dt = x[ii] * dt_sp;
            let mut sumf = 0.0f32; // single precision, matching the ggml kernel
            for i0 in 0..nc {
                let si = ii * nc + i0;
                let ig = g * nc + i0;
                let st = state.s[si] * d_a + b[ig] * x_dt;
                sumf += st * c[ig];
                state.s[si] = st;
            }
            if !sumf.is_finite() {
                return Err("non-finite SSM output rejected".into());
            }
            y[ii] = sumf;
        }
    }
    Ok(y)
}

/// Multi-token prefill: run `n_t` tokens through the scan, advancing `state`.
/// `x`: `n_t*n_head*dim`, `dt`: `n_t*n_head`, `b`/`c`: `n_t*n_group*d_state`.
/// Returns `y` of length `n_t*n_head*dim`. Equivalent to calling `ssm_step`
/// once per token in order.
pub fn ssm_scan(
    cfg: &SsmConfig,
    a: &[f32],
    state: &mut SsmState,
    x: &[f32],
    dt: &[f32],
    b: &[f32],
    c: &[f32],
) -> Result<Vec<f32>, String> {
    cfg.validate()?;
    let (nh, dim, nc, ng) = (cfg.n_head, cfg.dim, cfg.d_state, cfg.n_group);
    let xstep = nh * dim;
    let dstep = nh;
    let bstep = ng * nc;
    if xstep == 0 || x.len() % xstep != 0 {
        return Err("SSM x length is not a multiple of n_head*dim".into());
    }
    let n_t = x.len() / xstep;
    if n_t == 0 || dt.len() != n_t * dstep || b.len() != n_t * bstep || c.len() != n_t * bstep {
        return Err("SSM per-token input lengths disagree with token count".into());
    }
    let mut y = Vec::with_capacity(x.len());
    for t in 0..n_t {
        let yt = ssm_step(
            cfg,
            a,
            state,
            &x[t * xstep..(t + 1) * xstep],
            &dt[t * dstep..(t + 1) * dstep],
            &b[t * bstep..(t + 1) * bstep],
            &c[t * bstep..(t + 1) * bstep],
        )?;
        y.extend_from_slice(&yt);
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent f64 reference for the Mamba-2 scan over a sequence, returning
    /// (outputs, final_state). Separate code path and precision from the kernel.
    fn reference(
        cfg: &SsmConfig,
        a: &[f32],
        s0: &[f32],
        x: &[f32],
        dt: &[f32],
        b: &[f32],
        c: &[f32],
    ) -> (Vec<f64>, Vec<f64>) {
        let (nh, dim, nc, ng) = (cfg.n_head, cfg.dim, cfg.d_state, cfg.n_group);
        let hpg = nh / ng;
        let xstep = nh * dim;
        let n_t = x.len() / xstep;
        let mut state: Vec<f64> = s0.iter().map(|&v| v as f64).collect();
        let mut out = Vec::new();
        for t in 0..n_t {
            let mut y = vec![0.0f64; nh * dim];
            for h in 0..nh {
                let dtsp = {
                    let z = dt[t * nh + h] as f64;
                    if z > 20.0 {
                        z
                    } else {
                        z.exp().ln_1p()
                    }
                };
                let da = (dtsp * a[h] as f64).exp();
                let g = h / hpg;
                for i1 in 0..dim {
                    let ii = h * dim + i1;
                    let x_dt = x[t * xstep + ii] as f64 * dtsp;
                    let mut sum = 0.0f64;
                    for i0 in 0..nc {
                        let si = ii * nc + i0;
                        let ig = (t * ng + g) * nc + i0;
                        let st = state[si] * da + b[ig] as f64 * x_dt;
                        sum += st * c[ig] as f64;
                        state[si] = st;
                    }
                    y[ii] = sum;
                }
            }
            out.extend(y);
        }
        (out, state)
    }

    #[test]
    fn single_step_matches_closed_form() {
        // nh=1, dim=1, d_state=1, ng=1. A=-1, zero state, x=2, dt=0, B=1, C=1.
        let cfg = SsmConfig {
            n_head: 1,
            dim: 1,
            d_state: 1,
            n_group: 1,
        };
        let mut st = SsmState::zeros(cfg).unwrap();
        let y = ssm_step(&cfg, &[-1.0], &mut st, &[2.0], &[0.0], &[1.0], &[1.0]).unwrap();
        // dt_sp = softplus(0) = ln(2); dA = exp(ln2 * -1) = 0.5; x_dt = 2*ln2.
        // state = 0*0.5 + 1*(2*ln2) = 2 ln2; y = state*1 = 2 ln2.
        let want = 2.0f32 * std::f32::consts::LN_2;
        assert!((y[0] - want).abs() < 1e-5, "y={} want={}", y[0], want);
        assert!((st.raw()[0] - want).abs() < 1e-5);
    }

    #[test]
    fn multihead_multigroup_matches_independent_f64_reference() {
        let cfg = SsmConfig {
            n_head: 4,
            dim: 2,
            d_state: 3,
            n_group: 2,
        };
        let a: Vec<f32> = vec![-0.5, -1.0, -0.25, -2.0];
        let n_t = 5usize;
        let xstep = cfg.n_head * cfg.dim;
        let bstep = cfg.n_group * cfg.d_state;
        let x: Vec<f32> = (0..n_t * xstep)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
            .collect();
        let dt: Vec<f32> = (0..n_t * cfg.n_head)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.3)
            .collect();
        let b: Vec<f32> = (0..n_t * bstep)
            .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.2)
            .collect();
        let c: Vec<f32> = (0..n_t * bstep)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.15)
            .collect();

        let s0 = vec![0.0f32; cfg.state_len().unwrap()];
        let (ref_y, ref_s) = reference(&cfg, &a, &s0, &x, &dt, &b, &c);

        let mut st = SsmState::zeros(cfg).unwrap();
        let y = ssm_scan(&cfg, &a, &mut st, &x, &dt, &b, &c).unwrap();

        let mut max_abs = 0f64;
        let mut sse = 0f64;
        for (g, r) in y.iter().zip(&ref_y) {
            let d = (*g as f64 - r).abs();
            max_abs = max_abs.max(d);
            sse += d * d;
        }
        let rmse = (sse / ref_y.len() as f64).sqrt();
        assert!(max_abs < 1e-4, "output max abs {max_abs}");
        assert!(rmse < 1e-5, "output rmse {rmse}");
        // Final recurrent state must also match the reference.
        let mut state_max = 0f64;
        for (g, r) in st.raw().iter().zip(&ref_s) {
            state_max = state_max.max((*g as f64 - r).abs());
        }
        assert!(state_max < 1e-4, "state max abs {state_max}");
    }

    #[test]
    fn prefill_equals_sequential_single_token_decode() {
        let cfg = SsmConfig {
            n_head: 2,
            dim: 2,
            d_state: 2,
            n_group: 1,
        };
        let a = vec![-0.7f32, -1.3];
        let n_t = 4usize;
        let xstep = cfg.n_head * cfg.dim;
        let bstep = cfg.n_group * cfg.d_state;
        let x: Vec<f32> = (0..n_t * xstep).map(|i| (i as f32 % 5.0) - 2.0).collect();
        let dt: Vec<f32> = (0..n_t * cfg.n_head)
            .map(|i| (i as f32 % 3.0) - 1.0)
            .collect();
        let b: Vec<f32> = (0..n_t * bstep).map(|i| (i as f32 % 4.0) - 1.5).collect();
        let c: Vec<f32> = (0..n_t * bstep).map(|i| (i as f32 % 6.0) - 2.5).collect();

        let mut s_pre = SsmState::zeros(cfg).unwrap();
        let pre = ssm_scan(&cfg, &a, &mut s_pre, &x, &dt, &b, &c).unwrap();

        let mut s_dec = SsmState::zeros(cfg).unwrap();
        let mut dec = Vec::new();
        for t in 0..n_t {
            let yt = ssm_step(
                &cfg,
                &a,
                &mut s_dec,
                &x[t * xstep..(t + 1) * xstep],
                &dt[t * cfg.n_head..(t + 1) * cfg.n_head],
                &b[t * bstep..(t + 1) * bstep],
                &c[t * bstep..(t + 1) * bstep],
            )
            .unwrap();
            dec.extend(yt);
        }
        assert_eq!(pre, dec, "prefill must equal sequential decode");
        assert_eq!(s_pre.raw(), s_dec.raw(), "final state must match");
    }

    #[test]
    fn reset_and_session_isolation() {
        let cfg = SsmConfig {
            n_head: 1,
            dim: 1,
            d_state: 1,
            n_group: 1,
        };
        let mut a_state = SsmState::zeros(cfg).unwrap();
        let b_state = SsmState::zeros(cfg).unwrap();
        let _ = ssm_step(&cfg, &[-1.0], &mut a_state, &[1.0], &[0.0], &[1.0], &[1.0]).unwrap();
        assert!(a_state.raw()[0] != 0.0);
        assert_eq!(b_state.raw()[0], 0.0, "independent session unaffected");
        a_state.reset();
        assert_eq!(a_state.raw()[0], 0.0);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let cfg = SsmConfig {
            n_head: 2,
            dim: 2,
            d_state: 2,
            n_group: 1,
        };
        let mut st = SsmState::zeros(cfg).unwrap();
        let ok = |st: &mut SsmState| {
            ssm_step(
                &cfg,
                &[-1.0, -1.0],
                st,
                &[1.0; 4],
                &[0.0, 0.0],
                &[1.0, 1.0],
                &[1.0, 1.0],
            )
        };
        assert!(ok(&mut st).is_ok());
        // Wrong x length.
        assert!(ssm_step(
            &cfg,
            &[-1.0, -1.0],
            &mut st,
            &[1.0; 3],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 1.0]
        )
        .is_err());
        // Non-finite dt.
        assert!(ssm_step(
            &cfg,
            &[-1.0, -1.0],
            &mut st,
            &[1.0; 4],
            &[f32::NAN, 0.0],
            &[1.0, 1.0],
            &[1.0, 1.0]
        )
        .is_err());
        // Invalid config (n_group not dividing n_head).
        let bad = SsmConfig {
            n_head: 3,
            dim: 1,
            d_state: 1,
            n_group: 2,
        };
        assert!(SsmState::zeros(bad).is_err());
    }
}
