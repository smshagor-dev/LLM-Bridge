//! Mamba-2 causal depthwise convolution for the `nemotron_h_moe` hybrid layers.
//!
//! This is a faithful CPU implementation of the pinned ggml operation
//! `ggml_compute_forward_ssm_conv_f32`
//! (build/reference/ggml/src/ggml-cpu/ops.cpp:9943, revision cd959ba), whose
//! math is:
//!
//! ```text
//!   out[t, c] = sum_{k=0}^{K-1} conv_x[t + k, c] * weight[k, c]
//! ```
//!
//! where `conv_x` is `(K-1)` timesteps of left-padding state followed by the
//! current token inputs (a causal, depthwise, per-channel 1-D convolution of
//! kernel width `K = d_conv`). ggml accumulates each output in single precision;
//! this module matches that to stay bit-comparable with the reference kernel.
//!
//! Layout (row-major, channel-fastest, chosen for a clean Rust API):
//!   - weights `w[c*K + k]`      (per channel `c`, kernel tap `k`),
//!   - token inputs `u[t*C + c]` (token `t`, channel `c`),
//!   - outputs    `y[t*C + c]`,
//!   - conv state `s[c*(K-1) + j]` holds the last `K-1` inputs per channel,
//!     oldest first.
//!
//! The convolution is the first stage of the Mamba-2 block; this module does NOT
//! apply the post-convolution activation (SiLU), the SSM scan, gating, or the
//! projections. Those remain to be implemented.

/// Per-channel convolution kernel weights.
#[derive(Debug, Clone)]
pub struct ConvWeights {
    channels: usize,
    k: usize,
    /// `w[c * k + j]`.
    w: Vec<f32>,
}

impl ConvWeights {
    pub fn new(channels: usize, k: usize, w: Vec<f32>) -> Result<Self, String> {
        if channels == 0 || k == 0 {
            return Err("conv weights require non-zero channels and kernel width".into());
        }
        if channels > (1 << 20) || k > 1024 {
            return Err("conv weight dimensions exceed bounded limits".into());
        }
        if w.len() != channels.checked_mul(k).ok_or("conv weight size overflow")? {
            return Err(format!(
                "conv weight length {} != channels*k ({})",
                w.len(),
                channels * k
            ));
        }
        if !w.iter().all(|v| v.is_finite()) {
            return Err("non-finite conv weight rejected".into());
        }
        Ok(Self { channels, k, w })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn kernel_width(&self) -> usize {
        self.k
    }
}

/// Persistent per-channel convolution state: the last `K-1` inputs per channel.
/// State is session-local; construct a fresh one per independent inference.
#[derive(Debug, Clone)]
pub struct ConvState {
    channels: usize,
    pad: usize, // K - 1
    /// `s[c * pad + j]`, oldest first.
    s: Vec<f32>,
}

impl ConvState {
    /// Zero-initialized state for a stream of `channels` channels, kernel `k`.
    pub fn zeros(channels: usize, k: usize) -> Result<Self, String> {
        if channels == 0 || k == 0 {
            return Err("conv state requires non-zero channels and kernel width".into());
        }
        let pad = k - 1;
        let len = channels
            .checked_mul(pad)
            .ok_or("conv state size overflow")?;
        Ok(Self {
            channels,
            pad,
            s: vec![0.0; len],
        })
    }

    /// Reset all state to zero, reusing the allocation.
    pub fn reset(&mut self) {
        self.s.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Bytes of recurrent convolution state held (for memory accounting).
    pub fn state_bytes(&self) -> usize {
        self.s.len() * std::mem::size_of::<f32>()
    }
}

/// Run the causal convolution over `input` (`n_t` tokens, channel-fastest),
/// advancing `state` by the sequence. Handles both multi-token prefill
/// (`n_t > 1`) and single-token incremental decode (`n_t == 1`) identically.
/// Returns the `n_t * channels` outputs.
pub fn conv_forward(
    weights: &ConvWeights,
    state: &mut ConvState,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let c = weights.channels;
    let k = weights.k;
    let pad = k - 1;
    if state.channels != c || state.pad != pad {
        return Err("conv state shape does not match weights".into());
    }
    if input.len() % c != 0 {
        return Err("conv input length is not a multiple of channels".into());
    }
    let n_t = input.len() / c;
    if n_t == 0 {
        return Err("conv input must contain at least one token".into());
    }
    if !input.iter().all(|v| v.is_finite()) {
        return Err("non-finite conv input rejected".into());
    }

    // conv_x position p in 0..pad+n_t: p < pad -> state[c*pad + p]; else input token (p-pad).
    let conv_x = |ch: usize, p: usize| -> f32 {
        if p < pad {
            state.s[ch * pad + p]
        } else {
            input[(p - pad) * c + ch]
        }
    };

    let mut out = vec![0.0f32; n_t * c];
    for ch in 0..c {
        for t in 0..n_t {
            let mut sumf = 0.0f32; // single precision, matching the ggml kernel
            for j in 0..k {
                sumf += conv_x(ch, t + j) * weights.w[ch * k + j];
            }
            if !sumf.is_finite() {
                return Err("non-finite convolution output rejected".into());
            }
            out[t * c + ch] = sumf;
        }
    }

    // Advance state to the last `pad` inputs of (state ++ input).
    if pad > 0 {
        let mut next = vec![0.0f32; c * pad];
        for ch in 0..c {
            for j in 0..pad {
                next[ch * pad + j] = conv_x(ch, n_t + j);
            }
        }
        state.s = next;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_hand_computed_causal_convolution() {
        // C=2 channels, K=3. ch0 weights [1,2,3]; ch1 weights [0.5,-1,2].
        let weights = ConvWeights::new(2, 3, vec![1.0, 2.0, 3.0, 0.5, -1.0, 2.0]).unwrap();
        let mut state = ConvState::zeros(2, 3).unwrap();
        // 3 tokens, channel-fastest: t0=[1,10], t1=[2,20], t2=[3,30].
        let input = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let out = conv_forward(&weights, &mut state, &input).unwrap();
        // Hand-computed (zero initial state):
        //   ch0: 3, 8, 14 ; ch1: 20, 30, 45
        let expected = vec![3.0, 20.0, 8.0, 30.0, 14.0, 45.0];
        for (g, e) in out.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "got {g} expected {e}");
        }
        // Final state = last K-1=2 inputs per channel: ch0 [2,3], ch1 [20,30].
        assert_eq!(state.s, vec![2.0, 3.0, 20.0, 30.0]);
    }

    #[test]
    fn prefill_equals_sequential_single_token_decode() {
        let weights =
            ConvWeights::new(3, 4, (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect()).unwrap();
        let tokens: Vec<f32> = (0..3 * 6).map(|i| ((i * 7 % 13) as f32) - 6.0).collect();

        // Prefill all 6 tokens at once.
        let mut s_prefill = ConvState::zeros(3, 4).unwrap();
        let prefill = conv_forward(&weights, &mut s_prefill, &tokens).unwrap();

        // Feed one token at a time, concatenating outputs.
        let mut s_decode = ConvState::zeros(3, 4).unwrap();
        let mut decode = Vec::new();
        for t in 0..6 {
            let tok = &tokens[t * 3..(t + 1) * 3];
            decode.extend(conv_forward(&weights, &mut s_decode, tok).unwrap());
        }
        assert_eq!(prefill, decode, "prefill must equal sequential decode");
        assert_eq!(s_prefill.s, s_decode.s, "final state must match");
    }

    #[test]
    fn nonzero_initial_state_is_used() {
        let weights = ConvWeights::new(1, 3, vec![1.0, 1.0, 1.0]).unwrap();
        let mut state = ConvState::zeros(1, 3).unwrap();
        state.s = vec![5.0, 7.0]; // last two prior inputs
                                  // one token = 2.0; window = [5,7,2] -> sum = 14.
        let out = conv_forward(&weights, &mut state, &[2.0]).unwrap();
        assert_eq!(out, vec![14.0]);
        // new state = [7, 2].
        assert_eq!(state.s, vec![7.0, 2.0]);
    }

    #[test]
    fn reset_clears_state_and_sessions_are_isolated() {
        let weights = ConvWeights::new(1, 2, vec![1.0, 1.0]).unwrap();
        let mut a = ConvState::zeros(1, 2).unwrap();
        let b = ConvState::zeros(1, 2).unwrap();
        // Advance only `a`.
        let _ = conv_forward(&weights, &mut a, &[3.0]).unwrap();
        assert_eq!(a.s, vec![3.0]);
        assert_eq!(b.s, vec![0.0], "independent session state must not change");
        a.reset();
        assert_eq!(a.s, vec![0.0]);
    }

    #[test]
    fn invalid_dimensions_and_non_finite_are_rejected() {
        assert!(ConvWeights::new(2, 3, vec![1.0, 2.0]).is_err()); // wrong length
        assert!(ConvWeights::new(1, 2, vec![1.0, f32::NAN]).is_err()); // non-finite
        let weights = ConvWeights::new(2, 3, vec![1.0; 6]).unwrap();
        let mut state = ConvState::zeros(2, 3).unwrap();
        assert!(conv_forward(&weights, &mut state, &[1.0, 2.0, 3.0]).is_err()); // not multiple of C
        assert!(conv_forward(&weights, &mut state, &[1.0, f32::INFINITY]).is_err()); // non-finite
        let mut wrong = ConvState::zeros(2, 2).unwrap();
        assert!(conv_forward(&weights, &mut wrong, &[1.0, 2.0]).is_err()); // state/weight mismatch
    }

    #[test]
    fn independent_full_window_reference_agrees() {
        // Independent formulation: build the explicit conv_x column per channel
        // and dot with the kernel, versus the streaming conv_forward.
        let c = 4usize;
        let k = 4usize;
        let weights = ConvWeights::new(
            c,
            k,
            (0..c * k).map(|i| ((i * 5 % 9) as f32) - 4.0).collect(),
        )
        .unwrap();
        let n_t = 5usize;
        let input: Vec<f32> = (0..n_t * c).map(|i| ((i * 3 % 11) as f32) - 5.0).collect();

        let mut state = ConvState::zeros(c, k).unwrap();
        let got = conv_forward(&weights, &mut state, &input).unwrap();

        // Reference with zero initial state: conv_x[p] = p<K-1 ? 0 : input[p-(K-1)].
        let pad = k - 1;
        let mut reference = vec![0.0f32; n_t * c];
        for ch in 0..c {
            for t in 0..n_t {
                let mut sum = 0.0f64;
                for j in 0..k {
                    let p = t + j;
                    let v = if p < pad {
                        0.0
                    } else {
                        input[(p - pad) * c + ch]
                    };
                    sum += v as f64 * weights.w[ch * k + j] as f64;
                }
                reference[t * c + ch] = sum as f32;
            }
        }
        let mut max_abs = 0f64;
        for (g, r) in got.iter().zip(&reference) {
            max_abs = max_abs.max((*g as f64 - *r as f64).abs());
        }
        assert!(
            max_abs < 1e-3,
            "streaming vs full-window reference max abs = {max_abs}"
        );
    }
}
