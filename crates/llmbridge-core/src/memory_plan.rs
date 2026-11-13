//! Conservative planning for the existing resident-weight CPU backend.
//! This is admission math, NOT out-of-core execution or an RSS enforcement mechanism.
use crate::hardware::Hardware;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanRequest {
    pub model_file_bytes: u64,
    pub context_tokens: u64,
    pub kv_bytes_per_token: u64,
    pub scratch_bytes: u64,
    pub os_reserve_bytes: u64,
    pub safety_margin_percent: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPlan {
    pub estimated_peak_bytes: u64,
    pub available_after_reserve_bytes: u64,
    pub logical_threads: usize,
}

/// Fail closed when the available-memory measurement is missing or arithmetic overflows.
/// File size is deliberately counted as resident weights even when the backend uses mmap.
pub fn plan_cpu(hardware: &Hardware, request: PlanRequest) -> Result<MemoryPlan, String> {
    if request.model_file_bytes == 0 || request.context_tokens == 0 {
        return Err("model size and context length must be nonzero".into());
    }
    if request.safety_margin_percent > 90 {
        return Err("safety margin must be between 0 and 90 percent".into());
    }
    let available = hardware
        .available_ram_bytes
        .ok_or("available RAM is unknown; refusing admission")?;
    let kv = request
        .context_tokens
        .checked_mul(request.kv_bytes_per_token)
        .ok_or("KV cache size overflow")?;
    let base = request
        .model_file_bytes
        .checked_add(kv)
        .and_then(|n| n.checked_add(request.scratch_bytes))
        .ok_or("memory estimate overflow")?;
    let multiplier = 100u64 + u64::from(request.safety_margin_percent);
    // Widen the intermediate multiplication before rounding, then reject
    // estimates that cannot be represented as u64.
    let rounded = (u128::from(base) * u128::from(multiplier) + 99) / 100;
    let estimated_peak_bytes =
        u64::try_from(rounded).map_err(|_| "memory safety margin overflow")?;
    let headroom = available.saturating_sub(request.os_reserve_bytes);
    if estimated_peak_bytes > headroom {
        return Err(format!(
            "CPU inference denied: estimated {estimated_peak_bytes} bytes exceeds available headroom {headroom} bytes"
        ));
    }
    Ok(MemoryPlan {
        estimated_peak_bytes,
        available_after_reserve_bytes: headroom,
        logical_threads: hardware.logical_cpus.max(1),
    })
}

/// Choose the largest admissible context within caller-specified bounds for the
/// current resident-weight CPU backend. Uses the same conservative estimate as
/// plan_cpu; does not make tensor weights out-of-core or enforce process RSS.
/// A minimum context that does not fit is rejected rather than silently reduced.
pub fn plan_cpu_adaptive_context(
    hardware: &Hardware,
    request: PlanRequest,
    min_context_tokens: u64,
) -> Result<(MemoryPlan, u64), String> {
    if min_context_tokens == 0 || min_context_tokens > request.context_tokens {
        return Err("minimum context must be nonzero and within requested context".into());
    }
    let minimum = PlanRequest {
        context_tokens: min_context_tokens,
        ..request
    };
    let minimum_plan = plan_cpu(hardware, minimum)?;
    if let Ok(full_plan) = plan_cpu(hardware, request) {
        return Ok((full_plan, request.context_tokens));
    }
    // Validate non-memory request constraints up front; a lower context must
    // never turn a malformed model or unknown hardware into an admission.
    let mut low = min_context_tokens;
    let mut high = request.context_tokens;
    let mut best = minimum_plan;
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        match plan_cpu(
            hardware,
            PlanRequest {
                context_tokens: mid,
                ..request
            },
        ) {
            Ok(plan) => {
                low = mid;
                best = plan;
            }
            Err(_) => high = mid,
        }
    }
    Ok((best, low))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hw(available: Option<u64>) -> Hardware {
        Hardware {
            logical_cpus: 4,
            total_ram_bytes: Some(4096),
            available_ram_bytes: available,
        }
    }
    fn req() -> PlanRequest {
        PlanRequest {
            model_file_bytes: 1000,
            context_tokens: 10,
            kv_bytes_per_token: 20,
            scratch_bytes: 300,
            os_reserve_bytes: 500,
            safety_margin_percent: 20,
        }
    }
    #[test]
    fn adaptive_context_selects_largest_admissible_token_count() {
        let requested = PlanRequest {
            context_tokens: 30,
            ..req()
        };
        let (plan, tokens) = plan_cpu_adaptive_context(&hw(Some(2500)), requested, 1).unwrap();
        assert_eq!(tokens, 18);
        assert_eq!(plan.estimated_peak_bytes, 1992);
        assert_eq!(plan.available_after_reserve_bytes, 2000);
        assert!(plan_cpu(
            &hw(Some(2500)),
            PlanRequest {
                context_tokens: 19,
                ..requested
            }
        )
        .is_err());
        assert!(tokens <= requested.context_tokens);
    }

    #[test]
    fn adaptive_context_never_exceeds_requested_context_when_memory_is_abundant() {
        let request = req();
        let (plan, tokens) = plan_cpu_adaptive_context(&hw(Some(2500)), request, 1).unwrap();
        assert_eq!(tokens, request.context_tokens);
        assert_eq!(plan.estimated_peak_bytes, 1800);
    }

    #[test]
    fn adaptive_context_rejects_minimum_above_memory_limit() {
        let request = PlanRequest {
            context_tokens: 30,
            ..req()
        };
        assert!(plan_cpu_adaptive_context(&hw(Some(2500)), request, 19).is_err());
    }

    #[test]
    fn adaptive_context_preserves_full_request_when_it_fits() {
        let (plan, tokens) = plan_cpu_adaptive_context(&hw(Some(4096)), req(), 2).unwrap();
        assert_eq!(tokens, 10);
        assert_eq!(plan.estimated_peak_bytes, 1800);
    }

    #[test]
    fn adaptive_context_rejects_unknown_ram_and_unreachable_minimum() {
        assert!(plan_cpu_adaptive_context(&hw(None), req(), 1).is_err());
        assert!(plan_cpu_adaptive_context(&hw(Some(1400)), req(), 1).is_err());
        assert!(plan_cpu_adaptive_context(&hw(Some(4096)), req(), 0).is_err());
        assert!(plan_cpu_adaptive_context(&hw(Some(4096)), req(), 11).is_err());
    }

    #[test]
    fn adaptive_context_handles_large_bounds_without_midpoint_overflow() {
        let request = PlanRequest {
            model_file_bytes: 1,
            context_tokens: u64::MAX,
            kv_bytes_per_token: 1,
            scratch_bytes: 0,
            os_reserve_bytes: 0,
            safety_margin_percent: 0,
        };
        let hardware = hw(Some(100));
        let (plan, tokens) = plan_cpu_adaptive_context(&hardware, request, 1).unwrap();
        assert_eq!(tokens, 99);
        assert_eq!(plan.estimated_peak_bytes, 100);
    }

    #[test]
    fn accounts_for_weights_kv_scratch_margin_and_reserve() {
        let plan = plan_cpu(&hw(Some(2500)), req()).unwrap();
        assert_eq!(plan.estimated_peak_bytes, 1800);
        assert_eq!(plan.available_after_reserve_bytes, 2000);
        assert_eq!(plan.logical_threads, 4);
        assert!(plan_cpu(&hw(Some(2200)), req()).is_err());
    }
    #[test]
    fn unknown_memory_and_invalid_inputs_fail_closed() {
        assert!(plan_cpu(&hw(None), req()).is_err());
        let mut request = req();
        request.context_tokens = 0;
        assert!(plan_cpu(&hw(Some(4096)), request).is_err());
        request.context_tokens = u64::MAX;
        assert!(plan_cpu(&hw(Some(4096)), request).is_err());
        request.context_tokens = 10;
        request.safety_margin_percent = 91;
        assert!(plan_cpu(&hw(Some(4096)), request).is_err());
    }
    #[test]
    fn large_exact_estimates_do_not_overflow_intermediate_math() {
        let mut request = req();
        request.model_file_bytes = u64::MAX - 1500;
        request.context_tokens = 1;
        request.kv_bytes_per_token = 0;
        request.scratch_bytes = 0;
        request.os_reserve_bytes = 0;
        request.safety_margin_percent = 0;
        let plan = plan_cpu(&hw(Some(u64::MAX)), request).unwrap();
        assert_eq!(plan.estimated_peak_bytes, u64::MAX - 1500);
        request.safety_margin_percent = 1;
        assert!(plan_cpu(&hw(Some(u64::MAX)), request).is_err());
    }

    #[test]
    fn percentage_rounds_up_without_spurious_overflow() {
        let mut request = req();
        request.model_file_bytes = u64::MAX / 2;
        request.context_tokens = 1;
        request.kv_bytes_per_token = 0;
        request.scratch_bytes = 0;
        request.os_reserve_bytes = 0;
        request.safety_margin_percent = 20;
        let expected = ((u128::from(request.model_file_bytes) * 120 + 99) / 100) as u64;
        assert_eq!(
            plan_cpu(&hw(Some(u64::MAX)), request)
                .unwrap()
                .estimated_peak_bytes,
            expected
        );
    }

    #[test]
    fn overflow_and_zero_headroom_never_admit() {
        let mut request = req();
        request.model_file_bytes = u64::MAX;
        assert!(plan_cpu(&hw(Some(u64::MAX)), request).is_err());
        request = req();
        request.os_reserve_bytes = 5000;
        assert!(plan_cpu(&hw(Some(4096)), request).is_err());
    }
}
