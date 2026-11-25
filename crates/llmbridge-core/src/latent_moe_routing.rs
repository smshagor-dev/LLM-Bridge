//! LatentMoE expert routing for the `nemotron_h_moe` architecture.
//!
//! This module implements ONLY the router selection/weighting arithmetic, which
//! is well-defined and independently verifiable without loading the model.
//!
//! IMPORTANT — gating function. The authoritative pinned graph
//! `build/reference/src/models/nemotron-h-moe.cpp` calls `build_moe_ffn` with
//! `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`, i.e. the real model uses **sigmoid**
//! gating, NOT softmax. `build_moe_ffn` (llama-graph.cpp:2275+) does:
//!   1. `probs = sigmoid(logits)`                    (per-expert, independent),
//!   2. `selection_probs = probs + exp_probs_b`      (DeepSeek-V3 selection bias;
//!      used ONLY to choose experts, not to weight them),
//!   3. `selected = top_k(selection_probs, n_used)`  (argsort top-k),
//!   4. `weights = probs[selected]`                  (UNBIASED sigmoid probs),
//!   5. if `norm_w`: `weights /= sum(weights)`        (sum clamped to F16 min),
//!   6. `weights *= w_scale`.
//!
//! (`expert_group_count = 1` for this model, so the group-masking path is skipped.)
//!
//! `route_sigmoid_topk` implements that exact recipe. `route_softmax_topk` and
//! `softmax` are retained for softmax-gated architectures and as a reference;
//! they are NOT the nemotron_h_moe recipe. An earlier revision (cfe61ad) used
//! softmax for nemotron, which this module corrects.
//!
//! Config comes from the real model metadata (`expert_count = 512`,
//! `expert_used_count = 22`, `expert_weights_norm = true`,
//! `expert_weights_scale = 5.0`, `expert_group_count = 1`).
//!
//! Scope limit: this is the routing stage only. It does NOT perform the latent
//! projection, expert matrix products, shared-expert path, aggregation, or any
//! Mamba-2/attention computation, and it does not establish full-model parity.

/// Router configuration derived from model metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingConfig {
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub weights_norm: bool,
    pub weights_scale: f32,
}

impl RoutingConfig {
    /// The configuration read from the real Nemotron 3 Super 120B-A12B metadata.
    pub const NEMOTRON_3_SUPER_120B: RoutingConfig = RoutingConfig {
        n_expert: 512,
        n_expert_used: 22,
        weights_norm: true,
        weights_scale: 5.0,
    };
}

/// A routing decision for one token: the selected expert ids (descending by
/// gated probability, ties broken by lower index) and their aligned weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Numerically stable softmax over `logits` (max-subtracted), in f32. Returns an
/// error if any logit is non-finite or the result is degenerate.
pub fn softmax(logits: &[f32]) -> Result<Vec<f32>, String> {
    if logits.is_empty() {
        return Err("softmax requires at least one logit".into());
    }
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if !l.is_finite() {
            return Err("non-finite router logit rejected".into());
        }
        if l > max {
            max = l;
        }
    }
    let mut sum = 0.0f64;
    let mut probs = Vec::with_capacity(logits.len());
    for &l in logits {
        let e = ((l - max) as f64).exp();
        probs.push(e);
        sum += e;
    }
    if !(sum.is_finite() && sum > 0.0) {
        return Err("degenerate softmax normalization".into());
    }
    Ok(probs.into_iter().map(|e| (e / sum) as f32).collect())
}

/// Compute the Nemotron LatentMoE routing for one token's router logits.
///
/// Selection uses repeated maximum extraction (O(n * k)) rather than a full
/// sort, so the tests can cross-check it against an independent full-sort
/// reference. Tie-break: when probabilities are equal, the lower expert index is
/// preferred (matching a stable descending argsort).
pub fn route_softmax_topk(logits: &[f32], cfg: &RoutingConfig) -> Result<Routing, String> {
    if cfg.n_expert == 0 || cfg.n_expert_used == 0 {
        return Err("routing requires non-zero expert counts".into());
    }
    if cfg.n_expert_used > cfg.n_expert {
        return Err("n_expert_used exceeds n_expert".into());
    }
    if logits.len() != cfg.n_expert {
        return Err(format!(
            "router logit count {} disagrees with n_expert {}",
            logits.len(),
            cfg.n_expert
        ));
    }
    if !cfg.weights_scale.is_finite() {
        return Err("non-finite expert weights scale".into());
    }
    let probs = softmax(logits)?;

    // Repeated-argmax top-k with stable lower-index tie-break.
    let mut taken = vec![false; cfg.n_expert];
    let mut expert_ids = Vec::with_capacity(cfg.n_expert_used);
    let mut weights = Vec::with_capacity(cfg.n_expert_used);
    for _ in 0..cfg.n_expert_used {
        let mut best_idx = usize::MAX;
        let mut best_prob = f32::NEG_INFINITY;
        for (idx, &p) in probs.iter().enumerate() {
            if taken[idx] {
                continue;
            }
            // Strictly greater keeps the first (lowest) index on ties.
            if p > best_prob {
                best_prob = p;
                best_idx = idx;
            }
        }
        taken[best_idx] = true;
        expert_ids.push(best_idx);
        weights.push(best_prob);
    }

    if cfg.weights_norm {
        let sum: f64 = weights.iter().map(|&w| w as f64).sum();
        if !(sum.is_finite() && sum > 0.0) {
            return Err("degenerate selected-weight normalization".into());
        }
        for w in &mut weights {
            *w = (*w as f64 / sum) as f32;
        }
    }
    for w in &mut weights {
        *w *= cfg.weights_scale;
    }
    Ok(Routing {
        expert_ids,
        weights,
    })
}

/// Numerically stable logistic sigmoid over `logits` (per-expert, independent),
/// rejecting non-finite input. This is the nemotron_h_moe gating function.
pub fn sigmoid_probs(logits: &[f32]) -> Result<Vec<f32>, String> {
    if logits.is_empty() {
        return Err("sigmoid requires at least one logit".into());
    }
    let mut out = Vec::with_capacity(logits.len());
    for &l in logits {
        if !l.is_finite() {
            return Err("non-finite router logit rejected".into());
        }
        let x = l as f64;
        // Branch to avoid overflow of exp for large-magnitude logits.
        let p = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        };
        out.push(p as f32);
    }
    Ok(out)
}

/// The exact `nemotron_h_moe` routing (see module docs / `build_moe_ffn`):
/// sigmoid gating, top-k selection on `sigmoid(logits) + selection_bias`,
/// weights taken from the UNBIASED sigmoid probabilities, optional
/// renormalization (sum clamped to the F16 minimum, matching the reference),
/// then `weights_scale`. `selection_bias` (`exp_probs_b`) is optional and, when
/// present, must have length `n_expert`; it affects selection only, not weights.
pub fn route_sigmoid_topk(
    logits: &[f32],
    cfg: &RoutingConfig,
    selection_bias: Option<&[f32]>,
) -> Result<Routing, String> {
    if cfg.n_expert == 0 || cfg.n_expert_used == 0 {
        return Err("routing requires non-zero expert counts".into());
    }
    if cfg.n_expert_used > cfg.n_expert {
        return Err("n_expert_used exceeds n_expert".into());
    }
    if logits.len() != cfg.n_expert {
        return Err(format!(
            "router logit count {} disagrees with n_expert {}",
            logits.len(),
            cfg.n_expert
        ));
    }
    if !cfg.weights_scale.is_finite() {
        return Err("non-finite expert weights scale".into());
    }
    let probs = sigmoid_probs(logits)?;
    let selection: Vec<f32> = match selection_bias {
        None => probs.clone(),
        Some(bias) => {
            if bias.len() != cfg.n_expert {
                return Err("selection bias length disagrees with n_expert".into());
            }
            if !bias.iter().all(|v| v.is_finite()) {
                return Err("non-finite selection bias rejected".into());
            }
            probs.iter().zip(bias).map(|(p, b)| p + b).collect()
        }
    };
    // top-k on selection scores; repeated-argmax, stable lower-index tie-break.
    let mut taken = vec![false; cfg.n_expert];
    let mut expert_ids = Vec::with_capacity(cfg.n_expert_used);
    let mut weights = Vec::with_capacity(cfg.n_expert_used);
    for _ in 0..cfg.n_expert_used {
        let mut best_idx = usize::MAX;
        let mut best = f32::NEG_INFINITY;
        for (idx, &s) in selection.iter().enumerate() {
            if taken[idx] {
                continue;
            }
            if s > best {
                best = s;
                best_idx = idx;
            }
        }
        taken[best_idx] = true;
        expert_ids.push(best_idx);
        weights.push(probs[best_idx]); // UNBIASED probability
    }
    if cfg.weights_norm {
        let sum: f64 = weights.iter().map(|&w| w as f64).sum();
        // build_moe_ffn clamps the sum to the smallest F16 before dividing.
        let denom = sum.max(6.103_515_625e-5);
        for w in &mut weights {
            *w = (*w as f64 / denom) as f32;
        }
    }
    for w in &mut weights {
        *w *= cfg.weights_scale;
    }
    Ok(Routing {
        expert_ids,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent softmax reference in f64.
    fn ref_softmax(logits: &[f32]) -> Vec<f64> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = logits.iter().map(|&l| ((l as f64) - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    }

    /// Independent top-k via a full stable descending sort (different algorithm
    /// from the production repeated-argmax selection).
    fn ref_topk(probs: &[f64], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| {
            probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)) // ties: lower index first
        });
        idx.truncate(k);
        idx
    }

    #[test]
    fn softmax_matches_independent_reference_and_normalizes() {
        let logits = [2.0f32, -1.0, 0.5, 3.5, 3.5, -4.0];
        let got = softmax(&logits).unwrap();
        let want = ref_softmax(&logits);
        let sum: f32 = got.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");
        let mut max_abs = 0f64;
        for (g, w) in got.iter().zip(&want) {
            max_abs = max_abs.max((*g as f64 - w).abs());
        }
        assert!(max_abs < 1e-6, "softmax vs reference max abs = {max_abs}");
    }

    #[test]
    fn topk_selection_matches_full_sort_reference() {
        let cfg = RoutingConfig {
            n_expert: 8,
            n_expert_used: 3,
            weights_norm: true,
            weights_scale: 5.0,
        };
        let logits = [0.1f32, 2.0, 2.0, -1.0, 4.0, 3.0, 2.0, 0.0];
        let routing = route_softmax_topk(&logits, &cfg).unwrap();
        let probs = ref_softmax(&logits);
        let expected_ids = ref_topk(&probs, cfg.n_expert_used);
        assert_eq!(routing.expert_ids, expected_ids, "top-k ids");
        // Expected: idx 4 (4.0), idx 5 (3.0), then idx 1 (2.0, lowest-index tie).
        assert_eq!(routing.expert_ids, vec![4, 5, 1]);
    }

    #[test]
    fn normalized_weights_sum_to_scale_and_scale_is_applied() {
        let cfg = RoutingConfig::NEMOTRON_3_SUPER_120B;
        // 512 logits, a clear ordering so selection is deterministic.
        let logits: Vec<f32> = (0..cfg.n_expert).map(|i| (i as f32) * 0.01).collect();
        let routing = route_softmax_topk(&logits, &cfg).unwrap();
        assert_eq!(routing.expert_ids.len(), 22);
        assert_eq!(routing.weights.len(), 22);
        // Highest logits are the largest indices; top-22 = 511..=490 descending.
        assert_eq!(routing.expert_ids[0], 511);
        assert_eq!(routing.expert_ids[21], 490);
        // With norm, selected weights sum to 1 BEFORE scale, so AFTER scale they
        // sum to weights_scale.
        let sum: f32 = routing.weights.iter().sum();
        assert!(
            (sum - cfg.weights_scale).abs() < 1e-4,
            "weight sum {sum} != scale {}",
            cfg.weights_scale
        );
        // Weights are descending (highest-prob expert first).
        for pair in routing.weights.windows(2) {
            assert!(pair[0] >= pair[1]);
        }
    }

    #[test]
    fn unnormalized_weights_are_raw_softmax_times_scale() {
        let cfg = RoutingConfig {
            n_expert: 4,
            n_expert_used: 2,
            weights_norm: false,
            weights_scale: 2.0,
        };
        let logits = [1.0f32, 3.0, 2.0, 0.0];
        let routing = route_softmax_topk(&logits, &cfg).unwrap();
        let probs = ref_softmax(&logits);
        // Selected: idx 1 (3.0), idx 2 (2.0).
        assert_eq!(routing.expert_ids, vec![1, 2]);
        let want0 = (probs[1] * 2.0) as f32;
        let want1 = (probs[2] * 2.0) as f32;
        assert!((routing.weights[0] - want0).abs() < 1e-6);
        assert!((routing.weights[1] - want1).abs() < 1e-6);
    }

    #[test]
    fn ties_prefer_lower_expert_index() {
        let cfg = RoutingConfig {
            n_expert: 5,
            n_expert_used: 2,
            weights_norm: false,
            weights_scale: 1.0,
        };
        // All equal logits: the two lowest indices must be chosen, in order.
        let routing = route_softmax_topk(&[1.0; 5], &cfg).unwrap();
        assert_eq!(routing.expert_ids, vec![0, 1]);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let cfg = RoutingConfig {
            n_expert: 4,
            n_expert_used: 2,
            weights_norm: true,
            weights_scale: 1.0,
        };
        // Wrong logit count.
        assert!(route_softmax_topk(&[1.0, 2.0, 3.0], &cfg).is_err());
        // Non-finite logit.
        assert!(route_softmax_topk(&[1.0, f32::NAN, 2.0, 3.0], &cfg).is_err());
        // n_expert_used > n_expert.
        let bad = RoutingConfig {
            n_expert: 2,
            n_expert_used: 3,
            weights_norm: true,
            weights_scale: 1.0,
        };
        assert!(route_softmax_topk(&[1.0, 2.0], &bad).is_err());
    }

    fn ref_sigmoid(x: f32) -> f64 {
        1.0 / (1.0 + (-(x as f64)).exp())
    }

    #[test]
    fn sigmoid_matches_independent_reference() {
        let logits = [0.0f32, 1.0, -2.0, 5.0, -8.0];
        let got = sigmoid_probs(&logits).unwrap();
        let mut max_abs = 0f64;
        for (g, &l) in got.iter().zip(&logits) {
            max_abs = max_abs.max((*g as f64 - ref_sigmoid(l)).abs());
        }
        assert!(max_abs < 1e-6, "sigmoid vs reference max abs = {max_abs}");
    }

    #[test]
    fn sigmoid_routing_selects_by_sigmoid_and_weights_are_unbiased() {
        let cfg = RoutingConfig {
            n_expert: 4,
            n_expert_used: 2,
            weights_norm: false,
            weights_scale: 1.0,
        };
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        // No bias: top-2 by sigmoid are experts 3 and 2.
        let r = route_sigmoid_topk(&logits, &cfg, None).unwrap();
        assert_eq!(r.expert_ids, vec![3, 2]);
        assert!((r.weights[0] as f64 - ref_sigmoid(3.0)).abs() < 1e-6);
        assert!((r.weights[1] as f64 - ref_sigmoid(2.0)).abs() < 1e-6);

        // Bias steers SELECTION to expert 0, but its WEIGHT stays the unbiased
        // sigmoid(0.0) = 0.5 (bias must not leak into the weight).
        let bias = [10.0f32, 0.0, 0.0, 0.0];
        let rb = route_sigmoid_topk(&logits, &cfg, Some(&bias)).unwrap();
        assert_eq!(rb.expert_ids, vec![0, 3]);
        assert!(
            (rb.weights[0] as f64 - 0.5).abs() < 1e-6,
            "weight must be unbiased"
        );
        assert!((rb.weights[1] as f64 - ref_sigmoid(3.0)).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_routing_normalizes_then_scales() {
        let cfg = RoutingConfig::NEMOTRON_3_SUPER_120B; // sigmoid, norm, x5.0
        let logits: Vec<f32> = (0..cfg.n_expert).map(|i| (i as f32) * 0.01 - 2.0).collect();
        let r = route_sigmoid_topk(&logits, &cfg, None).unwrap();
        assert_eq!(r.expert_ids.len(), 22);
        assert_eq!(r.expert_ids[0], 511); // largest logit -> largest sigmoid
        let sum: f32 = r.weights.iter().sum();
        assert!(
            (sum - cfg.weights_scale).abs() < 1e-4,
            "normalized+scaled weights sum {sum} != scale {}",
            cfg.weights_scale
        );
    }

    #[test]
    fn sigmoid_routing_rejects_bad_bias_and_inputs() {
        let cfg = RoutingConfig {
            n_expert: 3,
            n_expert_used: 2,
            weights_norm: true,
            weights_scale: 5.0,
        };
        assert!(route_sigmoid_topk(&[0.0, 1.0, 2.0], &cfg, Some(&[0.0, 1.0])).is_err()); // bias len
        assert!(route_sigmoid_topk(&[0.0, f32::NAN, 2.0], &cfg, None).is_err()); // non-finite
        assert!(route_sigmoid_topk(&[0.0, 1.0], &cfg, None).is_err()); // logit count
    }
}
