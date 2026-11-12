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
    let available = hardware.available_ram_bytes
        .ok_or("available RAM is unknown; refusing admission")?;
    let kv = request.context_tokens.checked_mul(request.kv_bytes_per_token)
        .ok_or("KV cache size overflow")?;
    let base = request.model_file_bytes.checked_add(kv)
        .and_then(|n| n.checked_add(request.scratch_bytes))
        .ok_or("memory estimate overflow")?;
    let multiplier = 100u64 + u64::from(request.safety_margin_percent);
    // Use a wider intermediate so a valid u64 estimate is not rejected
    // merely because multiplying by the percentage overflows u64.
    let rounded = (u128::from(base) * u128::from(multiplier) + 99) / 100;
    let estimated_peak_bytes = u64::try_from(rounded)
        .map_err(|_| "memory safety margin overflow")?;
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

#[cfg(test)]
mod tests {
    use super::*;
    fn hw(available: Option<u64>) -> Hardware {
        Hardware { logical_cpus: 4, total_ram_bytes: Some(4096), available_ram_bytes: available }
    }
    fn req() -> PlanRequest {
        PlanRequest {
            model_file_bytes: 1000, context_tokens: 10,
            kv_bytes_per_token: 20, scratch_bytes: 300,
            os_reserve_bytes: 500, safety_margin_percent: 20,
        }
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
        assert_eq!(plan_cpu(&hw(Some(u64::MAX)), request).unwrap().estimated_peak_bytes, expected);
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
