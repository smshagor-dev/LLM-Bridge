//! Opt-in numerical diagnostics comparing two logit vectors captured at the
//! same inference position (the streamed decoder vs the resident llama.cpp
//! reference). The metric computation is pure and unit-tested without the
//! native backend; the capture wrappers are gated behind the `native` feature.
//!
//! This never changes normal inference: it is a separate diagnostic path.

/// Numerical comparison of two equal-length logit vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitDiff {
    pub vocab: usize,
    /// Non-finite (NaN/inf) counts are reported, never hidden.
    pub streamed_nonfinite: usize,
    pub resident_nonfinite: usize,
    /// Element-wise error statistics over positions finite in both vectors.
    pub compared: usize,
    pub max_abs_err: f64,
    pub mean_abs_err: f64,
    pub rmse: f64,
    pub max_rel_err: f64,
    pub rel_denominator_floor: f64,
    pub streamed_top1: usize,
    pub resident_top1: usize,
    pub top1_agree: bool,
    pub top_k: usize,
    pub topk_overlap: usize,
    /// Rank (0 = best) of one backend's selected token within the other's logits.
    pub resident_top1_rank_in_streamed: usize,
    pub streamed_top1_rank_in_resident: usize,
}

impl LogitDiff {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"vocab\":{},\"compared\":{},\"streamed_nonfinite\":{},\"resident_nonfinite\":{},\
\"max_abs_err\":{},\"mean_abs_err\":{},\"rmse\":{},\"max_rel_err\":{},\"rel_denominator_floor\":{},\
\"streamed_top1\":{},\"resident_top1\":{},\"top1_agree\":{},\"top_k\":{},\"topk_overlap\":{},\
\"resident_top1_rank_in_streamed\":{},\"streamed_top1_rank_in_resident\":{}}}",
            self.vocab,
            self.compared,
            self.streamed_nonfinite,
            self.resident_nonfinite,
            json_f64(self.max_abs_err),
            json_f64(self.mean_abs_err),
            json_f64(self.rmse),
            json_f64(self.max_rel_err),
            json_f64(self.rel_denominator_floor),
            self.streamed_top1,
            self.resident_top1,
            self.top1_agree,
            self.top_k,
            self.topk_overlap,
            self.resident_top1_rank_in_streamed,
            self.streamed_top1_rank_in_resident,
        )
    }
}

fn json_f64(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.9}")
    } else {
        "null".to_string()
    }
}

/// Index of the maximum finite value; NaN/inf are never selected. Ties resolve
/// to the lowest index. Returns 0 for an all-non-finite vector (reported via
/// the non-finite counts).
fn argmax_finite(values: &[f32]) -> usize {
    let mut best = usize::MAX;
    let mut best_value = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v.is_finite() && v > best_value {
            best_value = v;
            best = i;
        }
    }
    if best == usize::MAX {
        0
    } else {
        best
    }
}

/// Rank (0 = strictly best) of `token` within `values`: the number of entries
/// strictly greater than `values[token]`. Non-finite entries never count as
/// greater (NaN comparisons are false), matching `argmax_finite`.
fn rank_of(values: &[f32], token: usize) -> usize {
    if token >= values.len() {
        return values.len();
    }
    let reference = values[token];
    values.iter().filter(|&&v| v > reference).count()
}

/// Index set of the `k` largest finite values (descending), lowest index first
/// on ties. `k` is clamped to the vector length.
fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        let va = if values[a].is_finite() {
            values[a]
        } else {
            f32::NEG_INFINITY
        };
        let vb = if values[b].is_finite() {
            values[b]
        } else {
            f32::NEG_INFINITY
        };
        vb.partial_cmp(&va).unwrap().then(a.cmp(&b))
    });
    order.truncate(k.min(values.len()));
    order
}

/// Compare two logit vectors position-for-position. Both must be the same
/// length (same vocabulary and inference position); a mismatch is an error, not
/// a silently truncated comparison.
pub fn compute(streamed: &[f32], resident: &[f32], top_k: usize) -> Result<LogitDiff, String> {
    if streamed.len() != resident.len() {
        return Err(format!(
            "logit length mismatch: streamed {} vs resident {}",
            streamed.len(),
            resident.len()
        ));
    }
    if streamed.is_empty() {
        return Err("empty logit vectors".into());
    }
    let rel_denominator_floor = 1e-6_f64;
    let mut streamed_nonfinite = 0usize;
    let mut resident_nonfinite = 0usize;
    let mut compared = 0usize;
    let mut max_abs_err = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_rel_err = 0.0f64;
    for (&a, &b) in streamed.iter().zip(resident.iter()) {
        if !a.is_finite() {
            streamed_nonfinite += 1;
        }
        if !b.is_finite() {
            resident_nonfinite += 1;
        }
        if a.is_finite() && b.is_finite() {
            let diff = (f64::from(a) - f64::from(b)).abs();
            max_abs_err = max_abs_err.max(diff);
            sum_abs += diff;
            sum_sq += diff * diff;
            let denom = f64::from(b).abs().max(rel_denominator_floor);
            max_rel_err = max_rel_err.max(diff / denom);
            compared += 1;
        }
    }
    let mean_abs_err = if compared > 0 {
        sum_abs / compared as f64
    } else {
        f64::NAN
    };
    let rmse = if compared > 0 {
        (sum_sq / compared as f64).sqrt()
    } else {
        f64::NAN
    };
    let streamed_top1 = argmax_finite(streamed);
    let resident_top1 = argmax_finite(resident);
    let k = top_k.clamp(1, streamed.len());
    let s_top = top_k_indices(streamed, k);
    let r_top = top_k_indices(resident, k);
    let topk_overlap = s_top.iter().filter(|i| r_top.contains(i)).count();
    Ok(LogitDiff {
        vocab: streamed.len(),
        streamed_nonfinite,
        resident_nonfinite,
        compared,
        max_abs_err,
        mean_abs_err,
        rmse,
        max_rel_err,
        rel_denominator_floor,
        streamed_top1,
        resident_top1,
        top1_agree: streamed_top1 == resident_top1,
        top_k: k,
        topk_overlap,
        resident_top1_rank_in_streamed: rank_of(streamed, resident_top1),
        streamed_top1_rank_in_resident: rank_of(resident, streamed_top1),
    })
}

/// Per-position comparison across a decoded sequence. `prefix_identical_through`
/// is the number of leading positions at which both backends' greedy argmax
/// agree; a positional comparison beyond that point compares logits produced
/// from DIFFERENT fed prefixes and is flagged as such.
#[derive(Debug, Clone)]
pub struct MultiLogitDiff {
    pub positions: Vec<LogitDiff>,
    pub prefix_identical_through: usize,
}

impl MultiLogitDiff {
    pub fn to_json(&self) -> String {
        let per: Vec<String> = self
            .positions
            .iter()
            .enumerate()
            .map(|(i, d)| format!("{{\"position\":{},\"metrics\":{}}}", i, d.to_json()))
            .collect();
        format!(
            "{{\"prefix_identical_through\":{},\"position_count\":{},\"positions\":[{}]}}",
            self.prefix_identical_through,
            self.positions.len(),
            per.join(",")
        )
    }
}

/// Compare two per-position logit sequences (streamed vs resident), one
/// `LogitDiff` per position over the common length. The sequences must align
/// position-for-position (identical prompt, deterministic greedy decoding);
/// `prefix_identical_through` bounds where that alignment is guaranteed.
pub fn compute_multi(
    streamed: &[Vec<f32>],
    resident: &[Vec<f32>],
    top_k: usize,
) -> Result<MultiLogitDiff, String> {
    if streamed.is_empty() || resident.is_empty() {
        return Err("no logit positions captured".into());
    }
    let n = streamed.len().min(resident.len());
    let mut positions = Vec::with_capacity(n);
    let mut prefix_identical_through = 0usize;
    let mut still_identical = true;
    for i in 0..n {
        let diff = compute(&streamed[i], &resident[i], top_k)?;
        if still_identical && diff.streamed_top1 == diff.resident_top1 {
            prefix_identical_through += 1;
        } else {
            still_identical = false;
        }
        positions.push(diff);
    }
    Ok(MultiLogitDiff {
        positions,
        prefix_identical_through,
    })
}

#[cfg(feature = "native")]
mod capture {
    use crate::streamed::{generate_streamed, StreamedConfig};
    use std::ffi::{c_char, c_int, c_void, CStr, CString};
    use std::path::Path;
    use std::slice;
    use std::sync::atomic::AtomicBool;

    type LogitsCallback = extern "C" fn(*const f32, usize, *mut c_void) -> i32;

    extern "C" {
        fn llmbridge_resident_prompt_logits_v1(
            path: *const c_char,
            prompt: *const c_char,
            context_tokens: u32,
            threads: c_int,
            emit_logits: LogitsCallback,
            userdata: *mut c_void,
            error: *mut c_char,
            capacity: usize,
        ) -> i32;
        fn llmbridge_resident_decode_logits_v1(
            path: *const c_char,
            prompt: *const c_char,
            context_tokens: u32,
            max_tokens: u32,
            threads: c_int,
            emit_logits: LogitsCallback,
            userdata: *mut c_void,
            error: *mut c_char,
            capacity: usize,
        ) -> i32;
    }

    struct MultiCapture {
        steps: Vec<Vec<f32>>,
    }

    extern "C" fn multi_sink(data: *const f32, count: usize, userdata: *mut c_void) -> i32 {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if userdata.is_null() || data.is_null() || count == 0 || count > 1 << 22 {
                return false;
            }
            let state = unsafe { &mut *(userdata.cast::<MultiCapture>()) };
            let slice = unsafe { slice::from_raw_parts(data, count) };
            state.steps.push(slice.to_vec());
            true
        }));
        i32::from(matches!(result, Ok(true)))
    }

    /// Capture the resident llama.cpp logit vector at each of the first
    /// `max_tokens` greedy decode positions.
    pub fn resident_decode_logits(
        model: &Path,
        prompt: &str,
        context_tokens: u32,
        max_tokens: u32,
        threads: i32,
    ) -> Result<Vec<Vec<f32>>, String> {
        let path = CString::new(model.to_str().ok_or("non-UTF8 model path")?)
            .map_err(|_| "invalid model path")?;
        let prompt = CString::new(prompt).map_err(|_| "prompt contains NUL")?;
        crate::native_cpu::ensure_compatible()?;
        let mut state = MultiCapture { steps: Vec::new() };
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            llmbridge_resident_decode_logits_v1(
                path.as_ptr(),
                prompt.as_ptr(),
                context_tokens,
                max_tokens,
                threads,
                multi_sink,
                (&mut state as *mut MultiCapture).cast(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            let diagnostic = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            return Err(format!("resident decode-logits failed: {diagnostic}"));
        }
        if state.steps.is_empty() {
            return Err("resident backend produced no logits".into());
        }
        Ok(state.steps)
    }

    /// Capture the streamed decoder's logit vector at each of the first
    /// `max_tokens` greedy decode positions.
    pub fn streamed_decode_logits(
        model: &Path,
        prompt: &str,
        context_tokens: u32,
        max_tokens: u32,
        workspace_bytes: u64,
    ) -> Result<Vec<Vec<f32>>, String> {
        let config = StreamedConfig {
            context_tokens,
            max_tokens,
            workspace_bytes,
            ..Default::default()
        };
        let cancelled = AtomicBool::new(false);
        let mut steps: Vec<Vec<f32>> = Vec::new();
        let emit = |_: &[u8]| true;
        let on_logits = |values: &[f32]| {
            steps.push(values.to_vec());
            true
        };
        generate_streamed(model, prompt, config, &cancelled, emit, on_logits)?;
        if steps.is_empty() {
            return Err("streamed backend produced no logits".into());
        }
        Ok(steps)
    }

    /// Capture the streamed decoder's logits under teacher forcing: the final
    /// prompt-position logits, then one vector after each `forced_tokens` token
    /// is fed (`forced_tokens.len() + 1` vectors). Feeding a fixed sequence keeps
    /// every position comparable to an independently decoded reference prefix.
    pub fn streamed_forced_logits(
        model: &Path,
        prompt: &str,
        context_tokens: u32,
        forced_tokens: &[i32],
        workspace_bytes: u64,
    ) -> Result<Vec<Vec<f32>>, String> {
        let config = StreamedConfig {
            context_tokens,
            max_tokens: 1, // unused by the forced path; satisfies config validation
            workspace_bytes,
            ..Default::default()
        };
        let cancelled = AtomicBool::new(false);
        let mut steps: Vec<Vec<f32>> = Vec::new();
        let on_logits = |values: &[f32]| {
            steps.push(values.to_vec());
            true
        };
        crate::streamed::generate_streamed_forced_logits(
            model,
            prompt,
            config,
            &cancelled,
            forced_tokens,
            on_logits,
        )?;
        if steps.is_empty() {
            return Err("streamed forced backend produced no logits".into());
        }
        Ok(steps)
    }

    struct Capture {
        logits: Option<Vec<f32>>,
    }

    extern "C" fn sink(data: *const f32, count: usize, userdata: *mut c_void) -> i32 {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if userdata.is_null() || data.is_null() || count == 0 || count > 1 << 22 {
                return false;
            }
            let state = unsafe { &mut *(userdata.cast::<Capture>()) };
            if state.logits.is_none() {
                let slice = unsafe { slice::from_raw_parts(data, count) };
                state.logits = Some(slice.to_vec());
            }
            true
        }));
        i32::from(matches!(result, Ok(true)))
    }

    /// Capture the resident llama.cpp logit vector at the final prompt position.
    pub fn resident_prompt_logits(
        model: &Path,
        prompt: &str,
        context_tokens: u32,
        threads: i32,
    ) -> Result<Vec<f32>, String> {
        let path = CString::new(model.to_str().ok_or("non-UTF8 model path")?)
            .map_err(|_| "invalid model path")?;
        let prompt = CString::new(prompt).map_err(|_| "prompt contains NUL")?;
        crate::native_cpu::ensure_compatible()?;
        let mut state = Capture { logits: None };
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            llmbridge_resident_prompt_logits_v1(
                path.as_ptr(),
                prompt.as_ptr(),
                context_tokens,
                threads,
                sink,
                (&mut state as *mut Capture).cast(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            let diagnostic = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            return Err(format!("resident logit capture failed: {diagnostic}"));
        }
        state
            .logits
            .ok_or_else(|| "resident backend produced no logits".to_string())
    }

    /// Capture the streamed decoder's step-0 logit vector (the logits whose
    /// argmax is the first generated token), using a single-token generation.
    pub fn streamed_prompt_logits(
        model: &Path,
        prompt: &str,
        context_tokens: u32,
        workspace_bytes: u64,
    ) -> Result<Vec<f32>, String> {
        let config = StreamedConfig {
            context_tokens,
            max_tokens: 1,
            workspace_bytes,
            ..Default::default()
        };
        let cancelled = AtomicBool::new(false);
        let mut first: Option<Vec<f32>> = None;
        let emit = |_: &[u8]| true;
        let on_logits = |values: &[f32]| {
            if first.is_none() {
                first = Some(values.to_vec());
            }
            true
        };
        generate_streamed(model, prompt, config, &cancelled, emit, on_logits)?;
        first.ok_or_else(|| "streamed backend produced no logits".to_string())
    }
}

#[cfg(feature = "native")]
pub use capture::{
    resident_decode_logits, resident_prompt_logits, streamed_decode_logits, streamed_forced_logits,
    streamed_prompt_logits,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_report_zero_error_and_full_agreement() {
        let a = [1.0f32, 2.0, 3.0, 0.5];
        let d = compute(&a, &a, 2).unwrap();
        assert_eq!(d.vocab, 4);
        assert_eq!(d.compared, 4);
        assert_eq!(d.max_abs_err, 0.0);
        assert_eq!(d.mean_abs_err, 0.0);
        assert_eq!(d.rmse, 0.0);
        assert_eq!(d.max_rel_err, 0.0);
        assert_eq!(d.streamed_top1, 2);
        assert_eq!(d.resident_top1, 2);
        assert!(d.top1_agree);
        assert_eq!(d.topk_overlap, 2);
        assert_eq!(d.resident_top1_rank_in_streamed, 0);
        assert_eq!(d.streamed_top1_rank_in_resident, 0);
    }

    #[test]
    fn length_mismatch_is_rejected() {
        assert!(compute(&[1.0, 2.0], &[1.0], 1).is_err());
        assert!(compute(&[], &[], 1).is_err());
    }

    #[test]
    fn disagreement_quantifies_error_and_rank() {
        // streamed prefers index 0, resident prefers index 2.
        let streamed = [5.0f32, 1.0, 4.0];
        let resident = [4.0f32, 1.0, 5.0];
        let d = compute(&streamed, &resident, 1).unwrap();
        assert_eq!(d.streamed_top1, 0);
        assert_eq!(d.resident_top1, 2);
        assert!(!d.top1_agree);
        assert!((d.max_abs_err - 1.0).abs() < 1e-9);
        // resident top1 (index 2) has streamed value 4.0, beaten only by index 0 -> rank 1.
        assert_eq!(d.resident_top1_rank_in_streamed, 1);
        // streamed top1 (index 0) has resident value 4.0, beaten only by index 2 -> rank 1.
        assert_eq!(d.streamed_top1_rank_in_resident, 1);
        // top-1 sets disjoint.
        assert_eq!(d.topk_overlap, 0);
    }

    #[test]
    fn nonfinite_values_are_counted_not_hidden() {
        let streamed = [1.0f32, f32::NAN, 3.0];
        let resident = [1.0f32, 2.0, f32::INFINITY];
        let d = compute(&streamed, &resident, 2).unwrap();
        assert_eq!(d.streamed_nonfinite, 1);
        assert_eq!(d.resident_nonfinite, 1);
        // Only index 0 is finite in both.
        assert_eq!(d.compared, 1);
        assert_eq!(d.max_abs_err, 0.0);
        // NaN/inf never win argmax.
        assert_eq!(d.streamed_top1, 2);
        assert_eq!(d.resident_top1, 1);
    }

    #[test]
    fn multi_position_tracks_identical_prefix_length() {
        // Positions 0 and 1 agree on argmax; position 2 diverges (streamed->0,
        // resident->1), so the guaranteed-aligned prefix is 2 positions.
        let streamed = vec![
            vec![3.0f32, 1.0, 0.0],
            vec![0.0f32, 5.0, 1.0],
            vec![9.0f32, 0.0, 1.0],
        ];
        let resident = vec![
            vec![2.5f32, 1.0, 0.0],
            vec![0.0f32, 4.0, 1.0],
            vec![0.0f32, 7.0, 1.0],
        ];
        let m = compute_multi(&streamed, &resident, 2).unwrap();
        assert_eq!(m.positions.len(), 3);
        assert_eq!(m.prefix_identical_through, 2);
        assert!(m.positions[0].top1_agree);
        assert!(m.positions[1].top1_agree);
        assert!(!m.positions[2].top1_agree);
    }

    #[test]
    fn multi_position_requires_nonempty_sequences() {
        assert!(compute_multi(&[], &[vec![1.0]], 1).is_err());
    }
}
