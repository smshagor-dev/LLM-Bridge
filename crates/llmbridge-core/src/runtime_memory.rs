//! Cooperative runtime memory guard for native inference boundaries.
//!
//! This is an observation/admission mechanism, NOT a hard RSS cap: native
//! allocations may occur between samples, and OS/cgroup accounting is delayed.
use crate::hardware::{process_memory, ProcessMemory};
use std::io;

/// Independent ceilings for process RSS and cgroup-wide memory consumption.
/// A caller must sample at each native execution boundary; unknown metrics fail closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeMemoryLimits {
    pub max_process_rss_bytes: u64,
    pub max_cgroup_current_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryViolation {
    InvalidLimit,
    UnknownProcessRss,
    ProcessRssExceeded { observed: u64, limit: u64 },
    UnknownCgroupUsage,
    CgroupUsageExceeded { observed: u64, limit: u64 },
}

impl RuntimeMemoryLimits {
    /// Pure admission check; intended for use before and after native operations.
    /// No arithmetic on untrusted counters and no silent fallback for unknown RSS.
    pub fn check(self, sample: &ProcessMemory) -> Result<(), MemoryViolation> {
        if self.max_process_rss_bytes == 0 || self.max_cgroup_current_bytes == Some(0) {
            return Err(MemoryViolation::InvalidLimit);
        }
        let rss = sample.rss_bytes.ok_or(MemoryViolation::UnknownProcessRss)?;
        if rss > self.max_process_rss_bytes {
            return Err(MemoryViolation::ProcessRssExceeded {
                observed: rss,
                limit: self.max_process_rss_bytes,
            });
        }
        if let Some(limit) = self.max_cgroup_current_bytes {
            let used = sample.cgroup_current_bytes.ok_or(MemoryViolation::UnknownCgroupUsage)?;
            if used > limit {
                return Err(MemoryViolation::CgroupUsageExceeded {
                    observed: used,
                    limit,
                });
            }
        }
        Ok(())
    }

    /// Admit a prospective native allocation using checked arithmetic and explicit
    /// headroom. This is a cooperative preflight, not a reservation or hard cap.
    /// Both process RSS and cgroup usage must fit after the requested growth.
    pub fn check_growth(
        self,
        sample: &ProcessMemory,
        growth_bytes: u64,
        reserve_bytes: u64,
    ) -> Result<(), MemoryViolation> {
        if self.max_process_rss_bytes == 0 || self.max_cgroup_current_bytes == Some(0) {
            return Err(MemoryViolation::InvalidLimit);
        }
        let rss = sample.rss_bytes.ok_or(MemoryViolation::UnknownProcessRss)?;
        let projected = rss.checked_add(growth_bytes)
            .and_then(|value| value.checked_add(reserve_bytes))
            .ok_or(MemoryViolation::ProcessRssExceeded {
                observed: u64::MAX,
                limit: self.max_process_rss_bytes,
            })?;
        if projected > self.max_process_rss_bytes {
            return Err(MemoryViolation::ProcessRssExceeded {
                observed: projected,
                limit: self.max_process_rss_bytes,
            });
        }
        if let Some(limit) = self.max_cgroup_current_bytes {
            let used = sample.cgroup_current_bytes.ok_or(MemoryViolation::UnknownCgroupUsage)?;
            let projected = used.checked_add(growth_bytes)
                .and_then(|value| value.checked_add(reserve_bytes))
                .ok_or(MemoryViolation::CgroupUsageExceeded {
                    observed: u64::MAX,
                    limit,
                })?;
            if projected > limit {
                return Err(MemoryViolation::CgroupUsageExceeded {
                    observed: projected,
                    limit,
                });
            }
        }
        Ok(())
    }

    /// Read current process/cgroup counters and reject on any limit violation.
    /// On platforms without RSS measurement this deliberately rejects admission.
    pub fn check_current(self) -> Result<ProcessMemory, MemoryViolation> {
        let sample = process_memory();
        self.check(&sample)?;
        Ok(sample)
    }

    /// Sample live counters before admission and after the operation. A successful
    /// preflight does not reserve physical pages or guarantee a hard RSS ceiling.
    pub fn around_growth<T>(
        self,
        growth_bytes: u64,
        reserve_bytes: u64,
        operation: impl FnOnce() -> io::Result<T>,
    ) -> io::Result<T> {
        self.around_growth_with_sampler(growth_bytes, reserve_bytes, process_memory, operation)
    }

    /// Injectable sampler allows deterministic verification of both boundaries.
    /// The postflight check uses actual observed usage, not predicted growth.
    pub fn around_growth_with_sampler<T>(
        self,
        growth_bytes: u64,
        reserve_bytes: u64,
        mut sample: impl FnMut() -> ProcessMemory,
        operation: impl FnOnce() -> io::Result<T>,
    ) -> io::Result<T> {
        self.check_growth(&sample(), growth_bytes, reserve_bytes)
            .map_err(violation_io)?;
        let result = operation();
        self.check(&sample()).map_err(violation_io)?;
        result
    }

    /// Execute a fallible native boundary only when both surrounding samples pass.
    /// The closure is never invoked on preflight failure. Its side effects cannot
    /// be rolled back if the postflight sample exceeds the limit.
    pub fn around<T>(
        self,
        operation: impl FnOnce() -> io::Result<T>,
    ) -> io::Result<T> {
        self.check_current().map_err(violation_io)?;
        let result = operation();
        // Check even if the native call failed; do not hide a memory breach.
        let postflight = self.check_current().map_err(violation_io);
        postflight?;
        result
    }
}

fn violation_io(violation: MemoryViolation) -> io::Error {
    io::Error::new(io::ErrorKind::OutOfMemory, format!("runtime memory admission: {violation:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(rss: Option<u64>, cgroup: Option<u64>) -> ProcessMemory {
        ProcessMemory {
            rss_bytes: rss,
            cgroup_current_bytes: cgroup,
            ..ProcessMemory::default()
        }
    }
    #[test]
    fn rejects_unknown_and_zero_limits() {
        let limits = RuntimeMemoryLimits {
            max_process_rss_bytes: 100,
            max_cgroup_current_bytes: None,
        };
        assert_eq!(limits.check(&sample(None, None)), Err(MemoryViolation::UnknownProcessRss));
        assert_eq!(RuntimeMemoryLimits { max_process_rss_bytes: 0, ..limits }
            .check(&sample(Some(0), None)), Err(MemoryViolation::InvalidLimit));
        assert_eq!(RuntimeMemoryLimits { max_cgroup_current_bytes: Some(0), ..limits }
            .check(&sample(Some(0), Some(0))), Err(MemoryViolation::InvalidLimit));
    }
    #[test]
    fn enforces_both_independent_counters_inclusively() {
        let limits = RuntimeMemoryLimits {
            max_process_rss_bytes: 100,
            max_cgroup_current_bytes: Some(200),
        };
        assert!(limits.check(&sample(Some(100), Some(200))).is_ok());
        assert_eq!(limits.check(&sample(Some(101), Some(1))),
            Err(MemoryViolation::ProcessRssExceeded { observed: 101, limit: 100 }));
        assert_eq!(limits.check(&sample(Some(1), None)), Err(MemoryViolation::UnknownCgroupUsage));
        assert_eq!(limits.check(&sample(Some(1), Some(201))),
            Err(MemoryViolation::CgroupUsageExceeded { observed: 201, limit: 200 }));
    }
    #[test]
    fn unknown_cgroup_is_allowed_only_when_no_cgroup_ceiling_requested() {
        let limits = RuntimeMemoryLimits {
            max_process_rss_bytes: u64::MAX,
            max_cgroup_current_bytes: None,
        };
        assert!(limits.check(&sample(Some(u64::MAX), None)).is_ok());
    }
    #[test]
    fn prospective_growth_requires_headroom_in_both_domains() {
        let limits = RuntimeMemoryLimits {
            max_process_rss_bytes: 100,
            max_cgroup_current_bytes: Some(120),
        };
        assert!(limits.check_growth(&sample(Some(50), Some(70)), 30, 20).is_ok());
        assert_eq!(limits.check_growth(&sample(Some(50), Some(70)), 31, 20),
            Err(MemoryViolation::ProcessRssExceeded { observed: 101, limit: 100 }));
        assert_eq!(limits.check_growth(&sample(Some(10), Some(90)), 20, 11),
            Err(MemoryViolation::CgroupUsageExceeded { observed: 121, limit: 120 }));
        assert_eq!(limits.check_growth(&sample(Some(u64::MAX), Some(0)), 1, 0),
            Err(MemoryViolation::ProcessRssExceeded { observed: u64::MAX, limit: 100 }));
        assert_eq!(limits.check_growth(&sample(Some(1), None), 1, 0),
            Err(MemoryViolation::UnknownCgroupUsage));
    }

    #[test]
    fn growth_boundary_rejects_before_execution_and_checks_after_failure() {
        use std::cell::Cell;
        let limits = RuntimeMemoryLimits {
            max_process_rss_bytes: 100,
            max_cgroup_current_bytes: None,
        };
        let invoked = Cell::new(false);
        let rejected = limits.around_growth_with_sampler(30, 10,
            || sample(Some(70), None),
            || { invoked.set(true); Ok(()) });
        assert_eq!(rejected.unwrap_err().kind(), io::ErrorKind::OutOfMemory);
        assert!(!invoked.get());

        let samples = Cell::new(0usize);
        let result: io::Result<()> = limits.around_growth_with_sampler(20, 10,
            || {
                let n = samples.get();
                samples.set(n + 1);
                sample(Some(if n == 0 { 50 } else { 101 }), None)
            },
            || { invoked.set(true); Err(io::Error::new(io::ErrorKind::Other, "native failed")) });
        assert!(invoked.get());
        assert_eq!(samples.get(), 2);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::OutOfMemory);
    }

    #[test]
    fn growth_boundary_preserves_operation_error_when_postflight_is_safe() {
        let limits = RuntimeMemoryLimits {
            max_process_rss_bytes: 100,
            max_cgroup_current_bytes: None,
        };
        let result: io::Result<()> = limits.around_growth_with_sampler(10, 5,
            || sample(Some(40), None),
            || Err(io::Error::new(io::ErrorKind::InvalidData, "native failed")));
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

}
