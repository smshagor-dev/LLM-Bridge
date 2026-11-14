#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hardware {
    pub logical_cpus: usize,
    pub total_ram_bytes: Option<u64>,
    /// OS available RAM intersected with every observable cgroup-v2 ancestor.
    /// Unknown visible constraints fail closed; hidden hierarchies cannot be measured.
    pub available_ram_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcessMemory {
    /// Current virtual address size (Linux VmSize), distinct from RSS.
    pub virtual_bytes: Option<u64>,
    /// Sampled current VmRSS; taken after the inference call.
    pub rss_bytes: Option<u64>,
    /// Linux VmHWM high-water RSS from process lifetime.
    pub peak_rss_bytes: Option<u64>,
    /// Linux smaps_rollup Pss sampled after the inference call, not peak PSS.
    pub pss_bytes: Option<u64>,
    /// Leaf cgroup usage, including descendants; not ancestor usage.
    pub cgroup_current_bytes: Option<u64>,
    /// Leaf hard limit; None also represents the kernel unlimited value.
    pub cgroup_limit_bytes: Option<u64>,
}

pub fn profile() -> Hardware {
    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    #[cfg(target_os = "linux")]
    {
        let content = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let total = parse_meminfo(&content, "MemTotal:");
        let available = parse_meminfo(&content, "MemAvailable:");
        let cgroup = crate::cgroup_memory::read();
        let cg_limit = cgroup.as_ref().ok().and_then(|s| s.min_limit);
        let effective_total = match (total, cg_limit) {
            (Some(physical), Some(limit)) => Some(physical.min(limit)),
            (None, limit) => limit,
            (total, None) => total,
        };
        let effective_available = cgroup.ok().and_then(|s| s.available(available));
        Hardware {
            logical_cpus,
            total_ram_bytes: effective_total,
            available_ram_bytes: effective_available,
        }
    }
    #[cfg(not(target_os = "linux"))]
    Hardware {
        logical_cpus,
        total_ram_bytes: None,
        available_ram_bytes: None,
    }
}

/// Never treats container memory as an unlimited fraction of host MemAvailable.
pub fn admit(available: Option<u64>, reserved: u64, estimated_required: u64) -> Result<(), String> {
    let available = available.ok_or("Available RAM unknown; refusing unverified admission")?;
    let remaining = available.saturating_sub(reserved);
    if estimated_required > remaining {
        return Err(format!(
            "Insufficient RAM: need {estimated_required} bytes, available after reserve {remaining} bytes"
        ));
    }
    Ok(())
}

pub fn process_memory() -> ProcessMemory {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let rollup = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
        let cgroup = crate::cgroup_memory::read().ok();
        let cg_limit = cgroup.as_ref().and_then(|s| s.leaf_limit);
        let cg_current = cgroup.as_ref().and_then(|s| s.leaf_current);
        ProcessMemory {
            virtual_bytes: parse_meminfo(&status, "VmSize:"),
            rss_bytes: parse_meminfo(&status, "VmRSS:"),
            peak_rss_bytes: parse_meminfo(&status, "VmHWM:"),
            pss_bytes: parse_meminfo(&rollup, "Pss:"),
            cgroup_current_bytes: cg_current,
            cgroup_limit_bytes: cg_limit,
        }
    }
    #[cfg(not(target_os = "linux"))]
    ProcessMemory::default()
}

#[cfg(target_os = "linux")]
fn parse_meminfo(contents: &str, key: &str) -> Option<u64> {
    contents.lines().find_map(|line| {
        let mut parts = line.strip_prefix(key)?.split_whitespace();
        let amount: u64 = parts.next()?.parse().ok()?;
        if parts.next()? != "kB" {
            return None;
        }
        amount.checked_mul(1024)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_respects_os_reserve_and_unknown_ram() {
        assert!(admit(Some(2_000), 1_500, 501).is_err());
        assert!(admit(Some(2_000), 1_500, 500).is_ok());
        assert!(admit(None, 0, 1).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn reject_bad_meminfo_units_or_overflow() {
        assert_eq!(
            parse_meminfo("MemAvailable:  1024 kB\n", "MemAvailable:"),
            Some(1_048_576)
        );
        assert_eq!(
            parse_meminfo("MemAvailable:  999 MB\n", "MemAvailable:"),
            None
        );
        assert_eq!(
            parse_meminfo("MemAvailable:  18446744073709551615 kB\n", "MemAvailable:"),
            None
        );
        assert_eq!(
            parse_meminfo("VmHWM: 1337 kB\nVmRSS: 1200 kB\n", "VmHWM:"),
            Some(1_369_088)
        );
    }
}
