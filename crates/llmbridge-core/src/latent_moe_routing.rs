//! LatentMoE expert routing for the `nemotron_h_moe` architecture.
//!
//! This module implements ONLY the router selection/weighting arithmetic, which
//! is well-defined and independently verifiable without loading the model:
//!
//!   1. gating: softmax over all `n_expert` router logits,
//!   2. select the `n_expert_used` experts with the largest gated probability
//!      (stable tie-break: lower expert index wins),
//!   3. if `weights_norm`, renormalize the selected weights to sum to 1,
//!   4. multiply the selected weights by `weights_scale`.
//!
//! Config comes from the real model metadata (`expert_count = 512`,
//! `expert_used_count = 22`, `expert_weights_norm = true`,
//! `expert_weights_scale = 5.0`, `expert_group_count = 1`). The gating function
//! is softmax: this model carries no `expert_gating_func` key, so llama.cpp's
//! default applies. Softmax is monotonic, so the top-k set is identical whether
//! ranked by logit or by probability; the weights are taken from the softmax
//! probabilities.
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
}
