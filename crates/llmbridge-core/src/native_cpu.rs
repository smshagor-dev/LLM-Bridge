//! Baseline Rust checks before entering the explicitly targeted native library.
//! Direct C ABI consumers must perform their own compatible CPU/OS admission.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuCapabilities {
    pub sse2: bool,
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub bmi2: bool,
}
impl CpuCapabilities {
    fn requirements(self) -> [(&'static str, bool); 7] {
        [
            ("sse2", self.sse2),
            ("sse4.2", self.sse42),
            ("avx", self.avx),
            ("avx2", self.avx2),
            ("fma", self.fma),
            ("f16c", self.f16c),
            ("bmi2", self.bmi2),
        ]
    }
}
#[derive(Debug, Clone, Copy)]
pub struct NativeCpuCompatibility {
    pub profile: &'static str,
    pub architecture: &'static str,
    pub capabilities: CpuCapabilities,
}
impl NativeCpuCompatibility {
    pub fn compatible(&self) -> bool {
        match self.profile {
            "x86-baseline" => self.architecture == "x86_64" && self.capabilities.sse2,
            "x86-avx2" => {
                matches!(self.architecture, "x86" | "x86_64")
                    && self
                        .capabilities
                        .requirements()
                        .iter()
                        .all(|(_, present)| *present)
            }
            _ => false,
        }
    }
    fn required_features(&self) -> impl Iterator<Item = (&'static str, bool)> {
        let baseline = self.profile == "x86-baseline";
        self.capabilities
            .requirements()
            .into_iter()
            .filter(move |(name, _)| !baseline || *name == "sse2")
    }
    pub fn ensure(&self) -> Result<(), String> {
        if self.compatible() {
            return Ok(());
        }
        let missing = self
            .required_features()
            .filter_map(|(name, present)| (!present).then_some(name))
            .collect::<Vec<_>>()
            .join(", ");
        Err(format!("native CPU incompatible: build profile {}; architecture {}; missing CPU/OS capabilities [{}]. Select a compatible explicit build profile; automatic ISA fallback is disabled",
                    self.profile, self.architecture, missing))
    }
    pub fn to_json(&self) -> String {
        let c = self.capabilities;
        let missing = self
            .required_features()
            .filter_map(|(name, present)| (!present).then(|| format!("\"{name}\"")))
            .collect::<Vec<_>>()
            .join(",");
        format!(concat!("{{\"native_profile\":\"{}\",\"architecture\":\"{}\",\"native_compatible\":{},",
            "\"capabilities\":{{\"sse2\":{},\"sse42\":{},\"avx\":{},\"avx2\":{},\"fma\":{},\"f16c\":{},\"bmi2\":{}}},",
            "\"missing_cpu_os_capabilities\":[{}],\"scalar_fallback_validated\":false}}"),
            self.profile, self.architecture, self.compatible(), c.sse2, c.sse42, c.avx, c.avx2, c.fma, c.f16c, c.bmi2, missing)
    }
}

/// Rust's AVX feature checks also require OS extended-state support. CPUID's
/// hardware bit alone is insufficient. No native function is called here.
pub fn compatibility() -> NativeCpuCompatibility {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let capabilities = CpuCapabilities {
        sse2: std::is_x86_feature_detected!("sse2"),
        sse42: std::is_x86_feature_detected!("sse4.2"),
        avx: std::is_x86_feature_detected!("avx"),
        avx2: std::is_x86_feature_detected!("avx2"),
        fma: std::is_x86_feature_detected!("fma"),
        f16c: std::is_x86_feature_detected!("f16c"),
        bmi2: std::is_x86_feature_detected!("bmi2"),
    };
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let capabilities = CpuCapabilities::default();
    NativeCpuCompatibility {
        profile: option_env!("LLMBRIDGE_NATIVE_CPU_PROFILE").unwrap_or("disabled"),
        architecture: std::env::consts::ARCH,
        capabilities,
    }
}
pub fn ensure_compatible() -> Result<(), String> {
    compatibility().ensure()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn capable() -> NativeCpuCompatibility {
        NativeCpuCompatibility {
            profile: "x86-avx2",
            architecture: "x86_64",
            capabilities: CpuCapabilities {
                sse2: true,
                sse42: true,
                avx: true,
                avx2: true,
                fma: true,
                f16c: true,
                bmi2: true,
            },
        }
    }
    #[test]
    fn every_required_cpu_os_feature_must_be_present() {
        assert!(capable().ensure().is_ok());
        for index in 0..7 {
            let mut cpu = capable();
            let flags = [
                &mut cpu.capabilities.sse2,
                &mut cpu.capabilities.sse42,
                &mut cpu.capabilities.avx,
                &mut cpu.capabilities.avx2,
                &mut cpu.capabilities.fma,
                &mut cpu.capabilities.f16c,
                &mut cpu.capabilities.bmi2,
            ];
            *flags.into_iter().nth(index).unwrap() = false;
            let name = cpu.capabilities.requirements()[index].0;
            assert!(cpu.ensure().unwrap_err().contains(name));
            assert!(!cpu.compatible());
        }
    }
    #[test]
    fn hardware_avx2_without_os_avx_state_cannot_admit() {
        let mut cpu = capable();
        cpu.capabilities.avx = false;
        assert!(cpu.ensure().is_err());
    }
    #[test]
    fn disabled_unknown_profiles_and_other_architectures_fail_closed() {
        for profile in ["disabled", "unvalidated", "future-profile"] {
            assert!(NativeCpuCompatibility {
                profile,
                ..capable()
            }
            .ensure()
            .is_err());
        }
        assert!(NativeCpuCompatibility {
            architecture: "aarch64",
            ..capable()
        }
        .ensure()
        .is_err());
    }
    #[test]
    fn diagnostic_reports_missing_features_without_fallback_claim() {
        let mut cpu = capable();
        cpu.capabilities.f16c = false;
        let json = cpu.to_json();
        assert!(json.contains("\"native_compatible\":false"));
        assert!(json.contains("\"missing_cpu_os_capabilities\":[\"f16c\"]"));
        assert!(json.contains("\"scalar_fallback_validated\":false"));
    }
    #[test]
    fn baseline_admits_sse2_without_avx_but_avx2_build_does_not() {
        let cpu = NativeCpuCompatibility {
            profile: "x86-baseline",
            architecture: "x86_64",
            capabilities: CpuCapabilities {
                sse2: true,
                ..Default::default()
            },
        };
        assert!(cpu.ensure().is_ok());
        assert!(cpu.to_json().contains("\"missing_cpu_os_capabilities\":[]"));
        assert!(NativeCpuCompatibility {
            profile: "x86-avx2",
            ..cpu
        }
        .ensure()
        .is_err());
    }
    #[test]
    fn baseline_requires_sse2_and_a_validated_architecture() {
        let cpu = NativeCpuCompatibility {
            profile: "x86-baseline",
            ..capable()
        };
        assert!(NativeCpuCompatibility {
            capabilities: CpuCapabilities {
                sse2: false,
                ..cpu.capabilities
            },
            ..cpu
        }
        .ensure()
        .is_err());
        for architecture in ["x86", "aarch64", "riscv64"] {
            assert!(NativeCpuCompatibility {
                architecture,
                ..cpu
            }
            .ensure()
            .is_err());
        }
    }
}
