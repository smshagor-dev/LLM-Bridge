//! Executable LatentMoE feed-forward network for the `nemotron_h_moe`
//! architecture.
//!
//! This composes the independently-validated router ([`crate::latent_moe_routing`])
//! with the real expert arithmetic, exactly as the pinned reference graph defines
//! it. The authoritative sources (revision cd959ba) are:
//!
//!   * `build/reference/src/models/nemotron-h-moe.cpp` — the FFN sub-layer wiring
//!     (shared expert + latent projections + MoE), and
//!   * `build/reference/src/llama-graph.cpp` `build_moe_ffn` — the per-expert math.
//!
//! ## Exact composition (derived from the pinned graph, not inferred)
//!
//! Let `x` be the FFN sub-layer input (dimension `n_embd`, e.g. 4096). With
//! `relu2(v) = relu(v)^2` applied elementwise:
//!
//! ```text
//!   router_logits = ffn_gate_inp · x                      // [n_expert], on full x
//!   routing       = route_sigmoid_topk(router_logits,     // sigmoid gating,
//!                       cfg, ffn_exp_probs_b)              //   bias for selection only
//!
//!   shexp = ffn_down_shexp · relu2(ffn_up_shexp · x)      // shared expert, on full x
//!
//!   lat   = ffn_latent_down · x                           // [n_latent], e.g. 1024
//!   moe   = 0                                             // [n_latent]
//!   for (e, w) in routing:                                // SELECTED experts only
//!       u   = relu2(ffn_up_exps[e]   · lat)               // [n_ff_exp]
//!       d   = ffn_down_exps[e] · u                        // [n_latent]
//!       moe += w * d                                      // router-weighted sum
//!   moe   = ffn_latent_up · moe                           // [n_embd]
//!
//!   ffn_out = moe + shexp                                 // residual added by caller
//! ```
//!
//! Key facts enforced here, each taken from the pinned graph:
//!   * gating is **sigmoid**, never softmax (`LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`);
//!   * experts have **no gate** and use **squared-ReLU** (`LLM_FFN_RELU_SQR`);
//!   * the router and shared expert run on the **full** `n_embd` input, while the
//!     routed experts run in the **latent** space between `latent_down`/`latent_up`;
//!   * the selection bias (`ffn_exp_probs_b`) affects ranking only — expert weights
//!     come from the unbiased sigmoid probabilities (handled in the router);
//!   * `ffn_out = moe + shexp`; the block adds the residual afterwards.
//!
//! ## Scope and precision
//!
//! Matrix-vector products accumulate in `f32`, mirroring the ggml CPU path, and
//! only the **selected** experts are evaluated (sparse compute). The dense `f32`
//! weights here are the validation substrate: the native streamed path reuses the
//! same composition but sources each selected expert row through
//! [`crate::inference::dequant_matvec`] + `SplitGgufIndex`, so no expert matrix is
//! materialized in full and no unselected expert is read. This module does not run
//! attention, the Mamba block, the tokenizer, or establish full-model parity.

use crate::latent_moe_routing::{route_sigmoid_topk, Routing, RoutingConfig};

/// A dense row-major linear map `y = W·x`, with `W` laid out as `out_dim` rows of
/// `in_dim` contiguous weights (`w[o*in_dim + i]`), matching ggml `mul_mat`
/// semantics (`W->ne = [in_dim, out_dim]`, result length `out_dim`).
#[derive(Debug, Clone)]
pub struct LinearF32 {
    out_dim: usize,
    in_dim: usize,
    w: Vec<f32>,
}

impl LinearF32 {
    pub fn new(out_dim: usize, in_dim: usize, w: Vec<f32>) -> Result<Self, String> {
        if out_dim == 0 || in_dim == 0 {
            return Err("linear map requires non-zero dimensions".into());
        }
        let expected = out_dim
            .checked_mul(in_dim)
            .ok_or("linear weight size overflow")?;
        if w.len() != expected {
            return Err(format!(
                "linear weight length {} != out_dim*in_dim ({})",
                w.len(),
                expected
            ));
        }
        if !w.iter().all(|v| v.is_finite()) {
            return Err("non-finite linear weight rejected".into());
        }
        Ok(Self { out_dim, in_dim, w })
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// `y = W·x`, single-precision accumulation (matching the ggml CPU kernel).
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        if x.len() != self.in_dim {
            return Err(format!(
                "matvec input length {} != in_dim {}",
                x.len(),
                self.in_dim
            ));
        }
        let mut y = vec![0.0f32; self.out_dim];
        for (o, yo) in y.iter_mut().enumerate() {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            let mut acc = 0.0f32;
            for (wv, xv) in row.iter().zip(x) {
                acc += wv * xv;
            }
            if !acc.is_finite() {
                return Err("non-finite matvec output rejected".into());
            }
            *yo = acc;
        }
        Ok(y)
    }
}

/// Elementwise squared ReLU, `relu(v)^2`, in place. This is the nemotron expert
/// and shared-expert activation (`LLM_FFN_RELU_SQR`).
#[inline]
fn relu_sqr_in_place(v: &mut [f32]) {
    for x in v.iter_mut() {
        let r = if *x > 0.0 { *x } else { 0.0 };
        *x = r * r;
    }
}

/// Dense weights for one Nemotron LatentMoE FFN sub-layer.
///
/// All experts are held here for validation; the native path streams each
/// selected expert instead (see the module docs). Dimensions:
///   * router:        `[n_expert, n_embd]`
///   * shared up:     `[n_ff_shexp, n_embd]`, shared down: `[n_embd, n_ff_shexp]`
///   * latent_down:   `[n_latent, n_embd]`,   latent_up:   `[n_embd, n_latent]`
///   * each up_exp:   `[n_ff_exp, n_latent]`,  each down_exp: `[n_latent, n_ff_exp]`
#[derive(Debug, Clone)]
pub struct LatentMoeWeights {
    cfg: RoutingConfig,
    router: LinearF32,
    exp_probs_b: Option<Vec<f32>>,
    up_shexp: LinearF32,
    down_shexp: LinearF32,
    latent_down: LinearF32,
    latent_up: LinearF32,
    up_exps: Vec<LinearF32>,
    down_exps: Vec<LinearF32>,
}

/// The result of one FFN forward: the pre-residual output plus the routing taken
/// (recorded so evidence fixtures can assert the exact experts and weights).
#[derive(Debug, Clone)]
pub struct LatentMoeOutput {
    /// `moe + shexp`; the caller adds the residual.
    pub ffn_out: Vec<f32>,
    /// Selected expert ids, descending by gated selection score.
    pub expert_ids: Vec<usize>,
    /// Aligned router weights (unbiased sigmoid probs, normalized, scaled).
    pub weights: Vec<f32>,
}

impl LatentMoeWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: RoutingConfig,
        router: LinearF32,
        exp_probs_b: Option<Vec<f32>>,
        up_shexp: LinearF32,
        down_shexp: LinearF32,
        latent_down: LinearF32,
        latent_up: LinearF32,
        up_exps: Vec<LinearF32>,
        down_exps: Vec<LinearF32>,
    ) -> Result<Self, String> {
        if cfg.n_expert == 0 || cfg.n_expert_used == 0 {
            return Err("LatentMoE requires non-zero expert counts".into());
        }
        if cfg.n_expert_used > cfg.n_expert {
            return Err("n_expert_used exceeds n_expert".into());
        }
        let n_embd = latent_down.in_dim();
        let n_latent = latent_down.out_dim();

        // Router maps the full input to one logit per expert.
        if router.out_dim() != cfg.n_expert || router.in_dim() != n_embd {
            return Err(format!(
                "router shape [{}, {}] disagrees with [n_expert={}, n_embd={}]",
                router.out_dim(),
                router.in_dim(),
                cfg.n_expert,
                n_embd
            ));
        }
        if let Some(b) = &exp_probs_b {
            if b.len() != cfg.n_expert {
                return Err("selection bias length disagrees with n_expert".into());
            }
            if !b.iter().all(|v| v.is_finite()) {
                return Err("non-finite selection bias rejected".into());
            }
        }
        // Shared expert runs on the full n_embd input and returns to n_embd.
        if up_shexp.in_dim() != n_embd {
            return Err("shared-expert up input dim must equal n_embd".into());
        }
        if down_shexp.in_dim() != up_shexp.out_dim() {
            return Err("shared-expert down input dim must equal shared-up output".into());
        }
        if down_shexp.out_dim() != n_embd {
            return Err("shared-expert down output dim must equal n_embd".into());
        }
        // latent_up is the inverse projection latent -> n_embd.
        if latent_up.in_dim() != n_latent || latent_up.out_dim() != n_embd {
            return Err("latent_up shape must be [n_embd, n_latent]".into());
        }
        // Routed experts: there must be exactly n_expert of each, all in latent space.
        if up_exps.len() != cfg.n_expert || down_exps.len() != cfg.n_expert {
            return Err(format!(
                "expected {} up/down experts, got {}/{}",
                cfg.n_expert,
                up_exps.len(),
                down_exps.len()
            ));
        }
        let n_ff_exp = up_exps[0].out_dim();
        for (e, (u, d)) in up_exps.iter().zip(&down_exps).enumerate() {
            if u.in_dim() != n_latent {
                return Err(format!("expert {e} up input dim must equal n_latent"));
            }
            if u.out_dim() != n_ff_exp {
                return Err(format!("expert {e} up output dim is not uniform"));
            }
            if d.in_dim() != n_ff_exp {
                return Err(format!("expert {e} down input dim must equal expert n_ff"));
            }
            if d.out_dim() != n_latent {
                return Err(format!("expert {e} down output dim must equal n_latent"));
            }
        }
        Ok(Self {
            cfg,
            router,
            exp_probs_b,
            up_shexp,
            down_shexp,
            latent_down,
            latent_up,
            up_exps,
            down_exps,
        })
    }

    pub fn n_embd(&self) -> usize {
        self.latent_down.in_dim()
    }
    pub fn n_latent(&self) -> usize {
        self.latent_down.out_dim()
    }

    /// One shared-expert path: `down_shexp · relu2(up_shexp · x)`.
    fn shared_expert(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        let mut h = self.up_shexp.matvec(x)?;
        relu_sqr_in_place(&mut h);
        self.down_shexp.matvec(&h)
    }

    /// Full FFN forward over a single token `x` (`n_embd`). Only the selected
    /// experts are evaluated.
    pub fn forward(&self, x: &[f32]) -> Result<LatentMoeOutput, String> {
        if x.len() != self.n_embd() {
            return Err(format!(
                "FFN input length {} != n_embd {}",
                x.len(),
                self.n_embd()
            ));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err("non-finite FFN input rejected".into());
        }

        // Router logits on the full input, then sigmoid top-k selection.
        let logits = self.router.matvec(x)?;
        let Routing {
            expert_ids,
            weights,
        } = route_sigmoid_topk(&logits, &self.cfg, self.exp_probs_b.as_deref())?;

        // Shared expert on the full input.
        let shexp = self.shared_expert(x)?;

        // Routed experts in the latent space.
        let lat = self.latent_down.matvec(x)?;
        let n_latent = self.n_latent();
        let mut moe = vec![0.0f32; n_latent];
        for (&e, &w) in expert_ids.iter().zip(&weights) {
            if e >= self.up_exps.len() {
                return Err(format!("selected expert id {e} out of range"));
            }
            let mut u = self.up_exps[e].matvec(&lat)?;
            relu_sqr_in_place(&mut u);
            let d = self.down_exps[e].matvec(&u)?;
            for (m, dv) in moe.iter_mut().zip(&d) {
                *m += w * dv;
            }
        }
        let moe_up = self.latent_up.matvec(&moe)?;

        // FFN output = routed MoE (projected back) + shared expert.
        let mut ffn_out = moe_up;
        for (o, s) in ffn_out.iter_mut().zip(&shexp) {
            *o += s;
            if !o.is_finite() {
                return Err("non-finite FFN output rejected".into());
            }
        }
        Ok(LatentMoeOutput {
            ffn_out,
            expert_ids,
            weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Independent f64 oracle (separate code path and precision) ----

    /// Deterministic weight generator: `(tag, out_idx, in_idx, extra) -> value`.
    type SeedFn = dyn Fn(&str, usize, usize, usize) -> f32;

    struct Oracle {
        cfg: RoutingConfig,
        router: Vec<Vec<f64>>,     // [n_expert][n_embd]
        bias: Option<Vec<f64>>,    // [n_expert]
        up_shexp: Vec<Vec<f64>>,   // [n_ff_shexp][n_embd]
        down_shexp: Vec<Vec<f64>>, // [n_embd][n_ff_shexp]
        latent_down: Vec<Vec<f64>>,
        latent_up: Vec<Vec<f64>>,
        up_exps: Vec<Vec<Vec<f64>>>,
        down_exps: Vec<Vec<Vec<f64>>>,
    }

    fn mv(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        w.iter()
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }
    fn relu2(v: &[f64]) -> Vec<f64> {
        v.iter()
            .map(|&x| if x > 0.0 { x * x } else { 0.0 })
            .collect()
    }
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// `gating`: "sigmoid" (production) or "softmax" (the historical defect).
    fn oracle_forward(o: &Oracle, x: &[f64], gating: &str) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
        let logits = mv(&o.router, x);
        let probs: Vec<f64> = match gating {
            "sigmoid" => logits.iter().map(|&l| sigmoid(l)).collect(),
            "softmax" => {
                let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let ex: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
                let s: f64 = ex.iter().sum();
                ex.iter().map(|&e| e / s).collect()
            }
            _ => unreachable!(),
        };
        let selection: Vec<f64> = match &o.bias {
            None => probs.clone(),
            Some(b) => probs.iter().zip(b).map(|(p, bb)| p + bb).collect(),
        };
        // stable descending top-k by selection score, ties -> lower index
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| {
            selection[b]
                .partial_cmp(&selection[a])
                .unwrap()
                .then(a.cmp(&b))
        });
        idx.truncate(o.cfg.n_expert_used);
        let mut w: Vec<f64> = idx.iter().map(|&i| probs[i]).collect();
        if o.cfg.weights_norm {
            let sum: f64 = w.iter().sum::<f64>().max(6.103_515_625e-5);
            for v in &mut w {
                *v /= sum;
            }
        }
        for v in &mut w {
            *v *= o.cfg.weights_scale as f64;
        }
        // shared expert on full x
        let shexp = mv(&o.down_shexp, &relu2(&mv(&o.up_shexp, x)));
        // routed experts in latent space
        let lat = mv(&o.latent_down, x);
        let n_latent = o.latent_down.len();
        let mut moe = vec![0.0f64; n_latent];
        for (&e, &we) in idx.iter().zip(&w) {
            let u = relu2(&mv(&o.up_exps[e], &lat));
            let d = mv(&o.down_exps[e], &u);
            for (m, dv) in moe.iter_mut().zip(&d) {
                *m += we * dv;
            }
        }
        let moe_up = mv(&o.latent_up, &moe);
        let out: Vec<f64> = moe_up.iter().zip(&shexp).map(|(a, b)| a + b).collect();
        (out, idx, w)
    }

    // Build matching dense weights + oracle from a deterministic generator.
    #[allow(clippy::type_complexity)]
    fn build(
        cfg: RoutingConfig,
        n_embd: usize,
        n_latent: usize,
        n_ff_exp: usize,
        n_ff_shexp: usize,
        bias: Option<Vec<f32>>,
        seedf: &SeedFn,
    ) -> (LatentMoeWeights, Oracle) {
        let gen = |tag: &str, out: usize, inn: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(out * inn);
            for o in 0..out {
                for i in 0..inn {
                    v.push(seedf(tag, o, i, 0));
                }
            }
            v
        };
        let to_rows = |w: &[f32], out: usize, inn: usize| -> Vec<Vec<f64>> {
            (0..out)
                .map(|o| (0..inn).map(|i| w[o * inn + i] as f64).collect())
                .collect()
        };

        let router_w = gen("router", cfg.n_expert, n_embd);
        let ups_w = gen("upsh", n_ff_shexp, n_embd);
        let downs_w = gen("downsh", n_embd, n_ff_shexp);
        let ld_w = gen("ld", n_latent, n_embd);
        let lu_w = gen("lu", n_embd, n_latent);

        let mut up_exps = Vec::new();
        let mut down_exps = Vec::new();
        let mut o_up = Vec::new();
        let mut o_down = Vec::new();
        for e in 0..cfg.n_expert {
            let uw: Vec<f32> = (0..n_ff_exp * n_latent)
                .map(|k| seedf("ue", e, k / n_latent, k % n_latent))
                .collect();
            let dw: Vec<f32> = (0..n_latent * n_ff_exp)
                .map(|k| seedf("de", e, k / n_ff_exp, k % n_ff_exp))
                .collect();
            o_up.push(to_rows(&uw, n_ff_exp, n_latent));
            o_down.push(to_rows(&dw, n_latent, n_ff_exp));
            up_exps.push(LinearF32::new(n_ff_exp, n_latent, uw).unwrap());
            down_exps.push(LinearF32::new(n_latent, n_ff_exp, dw).unwrap());
        }

        let oracle = Oracle {
            cfg,
            router: to_rows(&router_w, cfg.n_expert, n_embd),
            bias: bias.as_ref().map(|b| b.iter().map(|&v| v as f64).collect()),
            up_shexp: to_rows(&ups_w, n_ff_shexp, n_embd),
            down_shexp: to_rows(&downs_w, n_embd, n_ff_shexp),
            latent_down: to_rows(&ld_w, n_latent, n_embd),
            latent_up: to_rows(&lu_w, n_embd, n_latent),
            up_exps: o_up,
            down_exps: o_down,
        };
        let weights = LatentMoeWeights::new(
            cfg,
            LinearF32::new(cfg.n_expert, n_embd, router_w).unwrap(),
            bias,
            LinearF32::new(n_ff_shexp, n_embd, ups_w).unwrap(),
            LinearF32::new(n_embd, n_ff_shexp, downs_w).unwrap(),
            LinearF32::new(n_latent, n_embd, ld_w).unwrap(),
            LinearF32::new(n_embd, n_latent, lu_w).unwrap(),
            up_exps,
            down_exps,
        )
        .unwrap();
        (weights, oracle)
    }

    fn small_seed() -> Box<SeedFn> {
        // Deterministic, small-magnitude, sign-varying weights (produce negative
        // intermediates so squared-ReLU's clamping is genuinely exercised).
        Box::new(|tag, a, b, c| {
            let h = tag.bytes().fold(2166136261u32, |acc, byte| {
                (acc ^ byte as u32).wrapping_mul(16777619)
            });
            let k = h
                .wrapping_add((a as u32).wrapping_mul(2654435761))
                .wrapping_add((b as u32).wrapping_mul(40503))
                .wrapping_add((c as u32).wrapping_mul(97));
            ((k % 1000) as f32 / 500.0) - 1.0 // in [-1, 1)
        })
    }

    fn max_abs(a: &[f32], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(g, r)| (*g as f64 - r).abs())
            .fold(0.0, f64::max)
    }
    fn rmse(a: &[f32], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        (a.iter()
            .zip(b)
            .map(|(g, r)| {
                let d = *g as f64 - r;
                d * d
            })
            .sum::<f64>()
            / n)
            .sqrt()
    }

    #[test]
    fn single_selected_expert_matches_oracle() {
        let cfg = RoutingConfig {
            n_expert: 6,
            n_expert_used: 1,
            weights_norm: true,
            weights_scale: 5.0,
        };
        let seed = small_seed();
        let (w, o) = build(cfg, 8, 4, 5, 7, None, &seed);
        let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 - 0.4).collect();
        let got = w.forward(&x).unwrap();
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let (want, ids, wts) = oracle_forward(&o, &xf, "sigmoid");
        assert_eq!(got.expert_ids, ids, "selected expert");
        assert_eq!(got.expert_ids.len(), 1);
        // With n_expert_used=1 and norm, the single weight must be the scale.
        assert!((got.weights[0] - cfg.weights_scale).abs() < 1e-5);
        assert!((wts[0] - cfg.weights_scale as f64).abs() < 1e-5);
        let ma = max_abs(&got.ffn_out, &want);
        assert!(ma < 1e-3, "max_abs {ma}");
    }

    #[test]
    fn multiple_experts_match_oracle_with_metrics() {
        let cfg = RoutingConfig {
            n_expert: 16,
            n_expert_used: 5,
            weights_norm: true,
            weights_scale: 5.0,
        };
        let seed = small_seed();
        let n_embd = 12usize;
        let (base, mut o) = build(cfg, n_embd, 6, 9, 10, None, &seed);
        // Separated router: row e = (e+1)*0.05 over all inputs. With an all-positive
        // x the logits strictly increase in e, so selection is unambiguous and the
        // f32 production path and f64 oracle must agree exactly on the experts.
        let mut router_w = vec![0.0f32; cfg.n_expert * n_embd];
        for e in 0..cfg.n_expert {
            for i in 0..n_embd {
                router_w[e * n_embd + i] = (e as f32 + 1.0) * 0.05;
            }
        }
        o.router = (0..cfg.n_expert)
            .map(|e| {
                (0..n_embd)
                    .map(|i| router_w[e * n_embd + i] as f64)
                    .collect()
            })
            .collect();
        let w = LatentMoeWeights::new(
            cfg,
            LinearF32::new(cfg.n_expert, n_embd, router_w).unwrap(),
            None,
            base.up_shexp.clone(),
            base.down_shexp.clone(),
            base.latent_down.clone(),
            base.latent_up.clone(),
            base.up_exps.clone(),
            base.down_exps.clone(),
        )
        .unwrap();
        let x: Vec<f32> = (0..n_embd).map(|i| 0.2 + (i as f32) * 0.05).collect();
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let got = w.forward(&x).unwrap();
        let (want, ids, _) = oracle_forward(&o, &xf, "sigmoid");
        assert_eq!(got.expert_ids, ids);
        // Strictly increasing logits -> the 5 highest-indexed experts.
        assert_eq!(got.expert_ids, vec![15, 14, 13, 12, 11]);
        let ma = max_abs(&got.ffn_out, &want);
        let rm = rmse(&got.ffn_out, &want);
        assert!(ma < 1e-3, "max_abs {ma}");
        assert!(rm < 1e-4, "rmse {rm}");
        // normalized+scaled weights sum to the scale
        let sum: f32 = got.weights.iter().sum();
        assert!((sum - cfg.weights_scale).abs() < 1e-4, "weight sum {sum}");
    }

    #[test]
    fn different_tokens_select_different_experts() {
        let cfg = RoutingConfig {
            n_expert: 12,
            n_expert_used: 3,
            weights_norm: true,
            weights_scale: 5.0,
        };
        let seed = small_seed();
        let (w, o) = build(cfg, 10, 5, 8, 6, None, &seed);
        let x1: Vec<f32> = (0..10).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let x2: Vec<f32> = (0..10).map(|i| ((9 - i) as f32) * 0.2 - 1.0).collect();
        let g1 = w.forward(&x1).unwrap();
        let g2 = w.forward(&x2).unwrap();
        assert_ne!(
            g1.expert_ids, g2.expert_ids,
            "distinct tokens should route differently"
        );
        for (x, g) in [(&x1, &g1), (&x2, &g2)] {
            let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let (want, ids, _) = oracle_forward(&o, &xf, "sigmoid");
            assert_eq!(g.expert_ids, ids);
            assert!(max_abs(&g.ffn_out, &want) < 1e-3);
        }
    }

    #[test]
    fn selection_bias_changes_rank_but_not_weight_value() {
        let cfg = RoutingConfig {
            n_expert: 6,
            n_expert_used: 2,
            weights_norm: false,
            weights_scale: 1.0,
        };
        let seed = small_seed();
        // No-bias baseline.
        let (w0, _) = build(cfg, 6, 3, 4, 4, None, &seed);
        let x: Vec<f32> = vec![0.3, -0.2, 0.5, -0.1, 0.4, -0.3];
        let base = w0.forward(&x).unwrap();
        // Force expert 0 into the selection via a large bias on index 0.
        let mut bias = vec![0.0f32; cfg.n_expert];
        bias[0] = 10.0;
        let (wb, _) = build(cfg, 6, 3, 4, 4, Some(bias), &seed);
        let biased = wb.forward(&x).unwrap();
        assert_eq!(
            biased.expert_ids[0], 0,
            "bias must pull expert 0 to the top"
        );
        assert!(
            !base.expert_ids.contains(&0),
            "baseline should not pick expert 0"
        );
        // The weight attached to expert 0 is its UNBIASED sigmoid prob (<=1 here,
        // never the +10 biased score). weights_norm is off and scale is 1.
        assert!(
            biased.weights[0] > 0.0 && biased.weights[0] <= 1.0,
            "weight {} must be the unbiased prob, not the biased score",
            biased.weights[0]
        );
    }

    #[test]
    fn equal_scores_prefer_lower_index() {
        let cfg = RoutingConfig {
            n_expert: 5,
            n_expert_used: 2,
            weights_norm: true,
            weights_scale: 5.0,
        };
        let seed = small_seed();
        let n_embd = 4usize;
        // A zero router row makes every logit exactly equal -> ties.
        let router =
            LinearF32::new(cfg.n_expert, n_embd, vec![0.0; cfg.n_expert * n_embd]).unwrap();
        let base = build(cfg, n_embd, 3, 4, 4, None, &seed).0;
        let w = LatentMoeWeights::new(
            cfg,
            router,
            None,
            base.up_shexp.clone(),
            base.down_shexp.clone(),
            base.latent_down.clone(),
            base.latent_up.clone(),
            base.up_exps.clone(),
            base.down_exps.clone(),
        )
        .unwrap();
        let x: Vec<f32> = vec![0.5, -0.5, 0.25, -0.25];
        let got = w.forward(&x).unwrap();
        assert_eq!(got.expert_ids, vec![0, 1], "equal scores -> lowest indices");
    }

    #[test]
    fn squared_relu_zeroes_negative_preactivations() {
        // If the up pre-activation is negative everywhere, squared ReLU zeroes the
        // routed contribution, leaving only the shared expert in the output.
        let cfg = RoutingConfig {
            n_expert: 3,
            n_expert_used: 1,
            weights_norm: true,
            weights_scale: 5.0,
        };
        let n_embd = 3usize;
        let n_latent = 2usize;
        let n_ff = 2usize;
        let n_shexp = 2usize;
        let router =
            LinearF32::new(cfg.n_expert, n_embd, vec![0.1; cfg.n_expert * n_embd]).unwrap();
        let up_shexp = LinearF32::new(n_shexp, n_embd, vec![0.2; n_shexp * n_embd]).unwrap();
        let down_shexp = LinearF32::new(n_embd, n_shexp, vec![0.3; n_embd * n_shexp]).unwrap();
        let latent_down = LinearF32::new(n_latent, n_embd, vec![0.5; n_latent * n_embd]).unwrap();
        let latent_up = LinearF32::new(n_embd, n_latent, vec![0.5; n_embd * n_latent]).unwrap();
        // up_exps all negative, latent positive -> up pre-activation negative.
        let up_exps: Vec<_> = (0..cfg.n_expert)
            .map(|_| LinearF32::new(n_ff, n_latent, vec![-1.0; n_ff * n_latent]).unwrap())
            .collect();
        let down_exps: Vec<_> = (0..cfg.n_expert)
            .map(|_| LinearF32::new(n_latent, n_ff, vec![1.0; n_latent * n_ff]).unwrap())
            .collect();
        let w = LatentMoeWeights::new(
            cfg,
            router,
            None,
            up_shexp.clone(),
            down_shexp.clone(),
            latent_down,
            latent_up,
            up_exps,
            down_exps,
        )
        .unwrap();
        let x = vec![1.0f32, 1.0, 1.0];
        let got = w.forward(&x).unwrap();
        // Routed contribution is zero; output equals the shared expert alone.
        let shexp = down_shexp
            .matvec(&{
                let mut h = up_shexp.matvec(&x).unwrap();
                relu_sqr_in_place(&mut h);
                h
            })
            .unwrap();
        for (o, s) in got.ffn_out.iter().zip(&shexp) {
            assert!(
                (o - s).abs() < 1e-6,
                "routed path should vanish: {o} vs {s}"
            );
        }
    }

    #[test]
    fn regression_sigmoid_differs_from_softmax_defect() {
        // The historical defect used softmax gating. On this fixture the two
        // gatings produce DIFFERENT outputs; the production path must match the
        // sigmoid oracle and NOT the softmax one.
        let cfg = RoutingConfig {
            n_expert: 10,
            n_expert_used: 3,
            weights_norm: true,
            weights_scale: 5.0,
        };
        let seed = small_seed();
        let (w, o) = build(cfg, 8, 4, 6, 5, None, &seed);
        let x: Vec<f32> = (0..8).map(|i| ((i * 3 % 5) as f32 - 2.0) * 0.6).collect();
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let got = w.forward(&x).unwrap();
        let (sig, sig_ids, sig_w) = oracle_forward(&o, &xf, "sigmoid");
        let (soft, _soft_ids, soft_w) = oracle_forward(&o, &xf, "softmax");

        // Production matches the sigmoid oracle.
        assert_eq!(got.expert_ids, sig_ids);
        assert!(
            max_abs(&got.ffn_out, &sig) < 1e-3,
            "must match sigmoid oracle"
        );
        // Sigmoid and softmax weights differ (softmax normalizes over ALL experts,
        // sigmoid is per-expert) -> the outputs differ materially.
        let diff = sig
            .iter()
            .zip(&soft)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(diff > 1e-3, "sigmoid vs softmax must differ (got {diff})");
        assert!(sig_w[0] != soft_w[0], "gating must yield different weights");
        // And production is NOT the softmax result.
        let to_soft = max_abs(&got.ffn_out, &soft);
        assert!(
            to_soft > 1e-3,
            "production must not equal the softmax defect"
        );
    }

    #[test]
    fn invalid_shapes_and_non_finite_are_rejected() {
        let cfg = RoutingConfig {
            n_expert: 4,
            n_expert_used: 2,
            weights_norm: true,
            weights_scale: 5.0,
        };
        let seed = small_seed();
        let (w, _) = build(cfg, 5, 3, 4, 4, None, &seed);
        // Wrong input length.
        assert!(w.forward(&[0.0; 4]).is_err());
        // Non-finite input.
        let mut bad = vec![0.0f32; 5];
        bad[2] = f32::INFINITY;
        assert!(w.forward(&bad).is_err());
        // Malformed construction: expert count mismatch (1 provided, need 4).
        let router = LinearF32::new(cfg.n_expert, 5, vec![0.0; cfg.n_expert * 5]).unwrap();
        let bad_build = LatentMoeWeights::new(
            cfg,
            router,
            None,
            LinearF32::new(4, 5, vec![0.0; 20]).unwrap(),
            LinearF32::new(5, 4, vec![0.0; 20]).unwrap(),
            LinearF32::new(3, 5, vec![0.0; 15]).unwrap(),
            LinearF32::new(5, 3, vec![0.0; 15]).unwrap(),
            vec![LinearF32::new(4, 3, vec![0.0; 12]).unwrap()],
            vec![LinearF32::new(3, 4, vec![0.0; 12]).unwrap()],
        );
        assert!(bad_build.is_err());
        // Non-finite weight rejected at the linear-map level.
        assert!(LinearF32::new(2, 2, vec![0.0, f32::NAN, 1.0, 2.0]).is_err());
    }

    #[test]
    fn real_nemotron_config_shapes_route_22_of_512() {
        // Real routing config with tiny projection dims so it stays bounded;
        // verifies 22-of-512 selection flows through the full FFN.
        let cfg = RoutingConfig::NEMOTRON_3_SUPER_120B;
        let seed = small_seed();
        let (w, o) = build(cfg, 4, 3, 2, 2, None, &seed);
        let x: Vec<f32> = vec![0.7, -0.4, 0.2, -0.6];
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let got = w.forward(&x).unwrap();
        assert_eq!(got.expert_ids.len(), 22);
        assert_eq!(got.weights.len(), 22);
        let (want, ids, _) = oracle_forward(&o, &xf, "sigmoid");
        assert_eq!(got.expert_ids, ids);
        assert!(max_abs(&got.ffn_out, &want) < 1e-3);
        let sum: f32 = got.weights.iter().sum();
        assert!((sum - cfg.weights_scale).abs() < 1e-4);
    }
}
