//! Runtime allocation admission based on observed memory headroom.
//!
//! This is a point-in-time guard for *additional* allocations, not an RSS cap:
//! concurrent allocations and OS memory pressure can change after admission.
use crate::hardware::{Hardware, ProcessMemory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationAdmission {
    pub requested_bytes: u64,
    pub headroom_after_reserve_bytes: u64,
    pub remaining_after_request_bytes: u64,
}

/// Reject allocations when available memory cannot be measured reliably.
/// The hardware profile already incorporates cgroup headroom on Linux. When
/// a process snapshot also supplies a cgroup limit/current pair, intersect
/// both measurements rather than adding them or counting current RSS twice.
pub fn admit_additional_allocation(
    hardware: &Hardware,
    process: &ProcessMemory,
    requested_bytes: u64,
    reserve_bytes: u64,
) -> Result<AllocationAdmission, String> {
    let available = hardware
        .available_ram_bytes
        .ok_or("available memory unknown; refusing allocation")?;
    let effective = match (process.cgroup_limit_bytes, process.cgroup_current_bytes) {
        (Some(limit), Some(current)) => available.min(limit.saturating_sub(current)),
        (Some(_), None) | (None, Some(_)) => {
            return Err("incomplete cgroup memory measurement".into());
        }
        (None, None) => available,
    };
    let headroom = effective.saturating_sub(reserve_bytes);
    if requested_bytes > headroom {
        return Err(format!(
            "allocation denied: need {requested_bytes} bytes, headroom {headroom} bytes"
        ));
    }
    Ok(AllocationAdmission {
        requested_bytes,
        headroom_after_reserve_bytes: headroom,
        remaining_after_request_bytes: headroom - requested_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hardware(available: Option<u64>) -> Hardware {
        Hardware {
            logical_cpus: 2,
            total_ram_bytes: Some(1000),
            available_ram_bytes: available,
        }
    }

    #[test]
    fn intersects_cgroup_headroom_without_double_counting_rss() {
        let process = ProcessMemory {
            rss_bytes: Some(300),
            cgroup_limit_bytes: Some(800),
            cgroup_current_bytes: Some(650),
            ..ProcessMemory::default()
        };
        let plan = admit_additional_allocation(&hardware(Some(500)), &process, 100, 40)
            .unwrap();
        assert_eq!(plan.headroom_after_reserve_bytes, 110);
        assert_eq!(plan.remaining_after_request_bytes, 10);
        assert!(admit_additional_allocation(&hardware(Some(500)), &process, 111, 40)
            .is_err());
    }

    #[test]
    fn fails_closed_on_unknown_or_incomplete_measurements() {
        let process = ProcessMemory::default();
        assert!(admit_additional_allocation(&hardware(None), &process, 1, 0).is_err());
        let incomplete = ProcessMemory {
            cgroup_limit_bytes: Some(100),
            ..ProcessMemory::default()
        };
        assert!(admit_additional_allocation(&hardware(Some(100)), &incomplete, 1, 0)
            .is_err());
    }

    #[test]
    fn handles_exhausted_cgroup_and_reserve_without_underflow() {
        let process = ProcessMemory {
            cgroup_limit_bytes: Some(100),
            cgroup_current_bytes: Some(200),
            ..ProcessMemory::default()
        };
        assert!(admit_additional_allocation(&hardware(Some(500)), &process, 1, 0).is_err());
        let plan = admit_additional_allocation(&hardware(Some(10)), &ProcessMemory::default(), 0, 100)
            .unwrap();
        assert_eq!(plan.remaining_after_request_bytes, 0);
    }
}
