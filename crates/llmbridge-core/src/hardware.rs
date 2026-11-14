#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hardware {
    pub logical_cpus: usize,
    pub total_ram_bytes: Option<u64>,
    /// OS available RAM or remaining cgroup memory, whichever is smaller on Linux.
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
    pub cgroup_current_bytes: Option<u64>,
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
        let (cg_limit, cg_current) = read_cgroup_v2();
        let effective_total = match (total, cg_limit) {
            (Some(physical), Some(limit)) => Some(physical.min(limit)),
            (None, limit) => limit,
            (total, None) => total,
        };
        let effective_available = limit_by_cgroup(available, cg_limit, cg_current);
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
        let (cg_limit, cg_current) = read_cgroup_v2();
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

#[cfg(target_os = "linux")]
fn limit_by_cgroup(
    physical_available: Option<u64>,
    cgroup_limit: Option<u64>,
    cgroup_current: Option<u64>,
) -> Option<u64> {
    if cgroup_limit.is_some() && cgroup_current.is_none() {
        return None;
    }
    let headroom = cgroup_limit
        .zip(cgroup_current)
        .map(|(limit, used)| limit.saturating_sub(used));
    match (physical_available, headroom) {
        (Some(physical), Some(cgroup)) => Some(physical.min(cgroup)),
        (Some(physical), None) => Some(physical),
        (None, Some(cgroup)) => Some(cgroup),
        (None, None) => None,
    }
}

#[cfg(target_os = "linux")]
fn parse_cgroup_number(content: &str) -> Option<u64> {
    let value = content.trim();
    if value == "max" {
        None
    } else {
        value.parse::<u64>().ok()
    }
}

#[cfg(target_os = "linux")]
fn cgroup_directory() -> Option<std::path::PathBuf> {
    use std::path::{Component, Path, PathBuf};
    let membership = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let hierarchy = membership.lines().find_map(|line| {
        let mut parts = line.splitn(3, ':');
        if parts.next()? == "0" && parts.next()?.is_empty() {
            parts.next()
        } else {
            None
        }
    })?;
    let mut directory = PathBuf::from("/sys/fs/cgroup");
    for component in Path::new(hierarchy).components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(path) => directory.push(path),
            _ => return None,
        }
    }
    Some(directory)
}

#[cfg(target_os = "linux")]
fn read_cgroup_v2() -> (Option<u64>, Option<u64>) {
    let Some(directory) = cgroup_directory() else {
        return (None, None);
    };
    let read = |name: &str| {
        std::fs::read_to_string(directory.join(name))
            .ok()
            .and_then(|value| parse_cgroup_number(&value))
    };
    (read("memory.max"), read("memory.current"))
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
    fn known_cgroup_limit_without_usage_never_falls_back_to_host_ram() {
        assert_eq!(limit_by_cgroup(Some(1000), Some(100), None), None);
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

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_headroom_never_exceeds_hardware_or_underflows() {
        assert_eq!(
            limit_by_cgroup(Some(2_000), Some(1_000), Some(700)),
            Some(300)
        );
        assert_eq!(
            limit_by_cgroup(Some(2_000), Some(1_000), Some(1_100)),
            Some(0)
        );
        assert_eq!(
            limit_by_cgroup(Some(300), Some(1_000), Some(500)),
            Some(300)
        );
        assert_eq!(limit_by_cgroup(None, None, None), None);
        assert_eq!(parse_cgroup_number("max\n"), None);
        assert_eq!(parse_cgroup_number("1048576\n"), Some(1_048_576));
    }
}
