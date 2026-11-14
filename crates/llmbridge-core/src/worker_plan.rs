//! Conservative per-worker scratch admission for future parallel CPU stages.
//! This planner never starts workers or promises that the allocator returns RSS.
use crate::hardware::Hardware;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerRequest {
    /// Bytes retained once for the whole inference session.
    pub shared_bytes: u64,
    /// Independent scratch charged for every simultaneously active worker.
    pub scratch_per_worker_bytes: u64,
    /// Other in-flight reservations not included in shared_bytes.
    pub already_reserved_bytes: u64,
    /// Keep this much OS/container headroom uncommitted.
    pub reserve_bytes: u64,
    /// Upper bound requested by the caller.
    pub requested_workers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerPlan {
    pub workers: usize,
    pub charged_bytes: u64,
    pub remaining_bytes: u64,
}

/// Fail closed on unknown memory, invalid worker counts and arithmetic overflow.
/// This is admission arithmetic only: caller must still enforce shared reservations
/// atomically, recheck pressure, and charge real allocator/graph overhead.
pub fn plan_workers(hardware: &Hardware, request: WorkerRequest) -> Result<WorkerPlan, String> {
    if request.requested_workers == 0 {
        return Err("worker count must be nonzero".into());
    }
    let available = hardware
        .available_ram_bytes
        .ok_or("available memory unknown; refusing worker admission")?;
    let fixed = request
        .shared_bytes
        .checked_add(request.already_reserved_bytes)
        .and_then(|n| n.checked_add(request.reserve_bytes))
        .ok_or("fixed memory charge overflow")?;
    let budget = available
        .checked_sub(fixed)
        .ok_or("fixed memory charge exceeds available memory")?;
    let cap = hardware.logical_cpus.max(1).min(request.requested_workers);
    let workers = if request.scratch_per_worker_bytes == 0 {
        cap
    } else {
        let possible = budget / request.scratch_per_worker_bytes;
        let possible = usize::try_from(possible).unwrap_or(usize::MAX);
        cap.min(possible)
    };
    if workers == 0 {
        return Err("insufficient memory for one worker".into());
    }
    let worker_bytes = u64::try_from(workers)
        .ok()
        .and_then(|n| n.checked_mul(request.scratch_per_worker_bytes))
        .ok_or("worker scratch charge overflow")?;
    let charged_bytes = fixed
        .checked_add(worker_bytes)
        .ok_or("total worker charge overflow")?;
    let remaining_bytes = available
        .checked_sub(charged_bytes)
        .ok_or("worker charge exceeds available memory")?;
    Ok(WorkerPlan {
        workers,
        charged_bytes,
        remaining_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hardware(cpus: usize, available: Option<u64>) -> Hardware {
        Hardware {
            logical_cpus: cpus,
            total_ram_bytes: None,
            available_ram_bytes: available,
        }
    }
    fn request() -> WorkerRequest {
        WorkerRequest {
            shared_bytes: 100,
            scratch_per_worker_bytes: 30,
            already_reserved_bytes: 20,
            reserve_bytes: 10,
            requested_workers: 8,
        }
    }
    #[test]
    fn chooses_largest_admissible_worker_count() {
        let plan = plan_workers(&hardware(16, Some(230)), request()).unwrap();
        assert_eq!(plan.workers, 3);
        assert_eq!(plan.charged_bytes, 220);
        assert_eq!(plan.remaining_bytes, 10);
    }
    #[test]
    fn respects_cpu_limit_and_explicit_request() {
        assert_eq!(
            plan_workers(&hardware(2, Some(1000)), request())
                .unwrap()
                .workers,
            2
        );
        let req = WorkerRequest {
            requested_workers: 1,
            ..request()
        };
        assert_eq!(
            plan_workers(&hardware(16, Some(1000)), req)
                .unwrap()
                .workers,
            1
        );
    }
    #[test]
    fn denies_unknown_memory_and_fixed_overcommit() {
        assert!(plan_workers(&hardware(4, None), request()).is_err());
        assert!(plan_workers(&hardware(4, Some(129)), request()).is_err());
        assert!(plan_workers(&hardware(4, Some(159)), request()).is_err());
        assert_eq!(
            plan_workers(&hardware(4, Some(160)), request())
                .unwrap()
                .workers,
            1
        );
    }
    #[test]
    fn zero_scratch_is_still_bounded_by_cpu_and_request() {
        let req = WorkerRequest {
            scratch_per_worker_bytes: 0,
            ..request()
        };
        assert_eq!(
            plan_workers(&hardware(3, Some(130)), req).unwrap().workers,
            3
        );
        assert!(plan_workers(&hardware(3, Some(129)), req).is_err());
    }
    #[test]
    fn rejects_zero_workers_and_overflow() {
        let req = WorkerRequest {
            requested_workers: 0,
            ..request()
        };
        assert!(plan_workers(&hardware(4, Some(1000)), req).is_err());
        let req = WorkerRequest {
            shared_bytes: u64::MAX,
            ..request()
        };
        assert!(plan_workers(&hardware(4, Some(u64::MAX)), req).is_err());
        let req = WorkerRequest {
            scratch_per_worker_bytes: u64::MAX,
            ..request()
        };
        assert!(plan_workers(&hardware(4, Some(u64::MAX)), req).is_err());
    }
    #[test]
    fn clamps_zero_reported_cpu_count_conservatively() {
        assert_eq!(
            plan_workers(&hardware(0, Some(1000)), request())
                .unwrap()
                .workers,
            1
        );
    }
    #[test]
    fn does_not_overflow_on_large_memory_and_zero_scratch() {
        let req = WorkerRequest {
            shared_bytes: 0,
            scratch_per_worker_bytes: 0,
            already_reserved_bytes: 0,
            reserve_bytes: 0,
            requested_workers: usize::MAX,
        };
        let result = plan_workers(&hardware(2, Some(u64::MAX)), req).unwrap();
        assert_eq!(result.workers, 2);
        assert_eq!(result.remaining_bytes, u64::MAX);
    }
}
