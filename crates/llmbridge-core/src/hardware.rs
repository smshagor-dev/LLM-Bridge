#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hardware {
    pub logical_cpus: usize,
    pub total_ram_bytes: Option<u64>,
    pub available_ram_bytes: Option<u64>,
}
pub fn profile() -> Hardware {
    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    #[cfg(target_os = "linux")]
    let (total_ram_bytes, available_ram_bytes) = {
        let content = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        (
            parse_meminfo(&content, "MemTotal:"),
            parse_meminfo(&content, "MemAvailable:"),
        )
    };
    #[cfg(not(target_os = "linux"))]
    let (total_ram_bytes, available_ram_bytes) = (None, None);
    Hardware {
        logical_cpus,
        total_ram_bytes,
        available_ram_bytes,
    }
}
pub fn admit(available: Option<u64>, reserved: u64, estimated_required: u64) -> Result<(), String> {
    let available = available.ok_or("Available RAM unknown; refusing unverified admission")?;
    let remaining = available.saturating_sub(reserved);
    if estimated_required > remaining {
        return Err(format!("Insufficient RAM: need {estimated_required} bytes, available after reserve {remaining} bytes"));
    }
    Ok(())
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
    }
}
