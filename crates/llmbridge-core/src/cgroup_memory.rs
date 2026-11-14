//! Linux cgroup-v2 memory observations. Only visible ancestors are measurable.
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Memory {
    pub leaf_limit: Option<u64>,
    pub leaf_current: Option<u64>,
    pub min_limit: Option<u64>,
    pub headroom: Option<u64>,
}
impl Memory {
    pub fn available(&self, physical: Option<u64>) -> Option<u64> {
        match (physical, self.headroom) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
}
fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "incomplete cgroup-v2 memory observation",
    )
}
fn read_text(path: &Path) -> io::Result<String> {
    // mountinfo may contain many mounts; individual interfaces need far less.
    let bound = if path == Path::new("/proc/self/mountinfo") {
        1024 * 1024
    } else {
        64 * 1024
    };
    let mut value = String::new();
    std::fs::File::open(path)?
        .take(bound + 1)
        .read_to_string(&mut value)?;
    if value.len() as u64 > bound {
        return Err(invalid());
    }
    Ok(value)
}
pub(crate) fn read() -> io::Result<Memory> {
    let membership = read_text(Path::new("/proc/self/cgroup"))?;
    let mounts = read_text(Path::new("/proc/self/mountinfo"))?;
    let observation = observe(&membership, &mounts, read_text)?;
    if read_text(Path::new("/proc/self/cgroup"))? != membership {
        return Err(invalid());
    }
    Ok(observation)
}
fn absolute(value: &str) -> io::Result<PathBuf> {
    let path = Path::new(value);
    if !path.is_absolute()
        || value.contains('\0')
        || value.ends_with(" (deleted)")
        || value.split('/').any(|part| part == "." || part == "..")
        || path
            .components()
            .any(|c| !matches!(c, Component::RootDir | Component::Normal(_)))
    {
        return Err(invalid());
    }
    Ok(path.to_path_buf())
}
fn mount_path(value: &str) -> io::Result<PathBuf> {
    let mut decoded = String::new();
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            decoded.push(c);
            continue;
        }
        let code: String = chars.by_ref().take(3).collect();
        decoded.push(match code.as_str() {
            "040" => ' ',
            "011" => '\t',
            "012" => '\n',
            "134" => '\\',
            _ => return Err(invalid()),
        });
    }
    absolute(&decoded)
}
fn number(value: &str) -> io::Result<u64> {
    let value = value.trim();
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid());
    }
    value.parse().map_err(|_| invalid())
}
fn optional(value: io::Result<String>) -> io::Result<Option<String>> {
    match value {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}
fn observe(
    membership: &str,
    mounts: &str,
    mut read: impl FnMut(&Path) -> io::Result<String>,
) -> io::Result<Memory> {
    let mut unified = membership
        .lines()
        .filter_map(|line| line.strip_prefix("0::"));
    let Some(path) = unified.next() else {
        return Ok(Memory::default());
    };
    if unified.next().is_some() {
        return Err(invalid());
    }
    let membership = absolute(path)?;
    // Prefer the widest visible hierarchy, so a narrower bind mount cannot hide
    // an observable parent. Resolve mount roots by path components, not strings.
    let mut selected: Option<(PathBuf, PathBuf)> = None;
    let mut has_v2 = false;
    for line in mounts.lines() {
        let Some((left, right)) = line.split_once(" - ") else {
            continue;
        };
        if right.split_whitespace().next() != Some("cgroup2") {
            continue;
        }
        has_v2 = true;
        let fields: Vec<_> = left.split_whitespace().take(6).collect();
        if fields.len() < 6 {
            return Err(invalid());
        }
        let root = mount_path(fields[3])?;
        let mount = mount_path(fields[4])?;
        if membership.strip_prefix(&root).is_err() {
            continue;
        }
        if selected
            .as_ref()
            .is_none_or(|(prior, _)| root.components().count() < prior.components().count())
        {
            selected = Some((root, mount));
        }
    }
    let Some((root, mount)) = selected else {
        // Some restricted environments expose no cgroup filesystem at all.
        // There is no visible constraint to report; this is not unlimited RAM.
        return if has_v2 {
            Err(invalid())
        } else {
            Ok(Memory::default())
        };
    };
    let relative = membership.strip_prefix(&root).map_err(|_| invalid())?;
    let mut directory = mount.join(relative);
    let mut snapshot = Memory::default();
    for depth in 0..128 {
        let max = optional(read(&directory.join("memory.max")))?;
        let current = optional(read(&directory.join("memory.current")))?;
        match (max, current) {
            (Some(max), Some(current)) => {
                let limit = if max.trim() == "max" {
                    None
                } else {
                    Some(number(&max)?)
                };
                let current = number(&current)?;
                if depth == 0 {
                    snapshot.leaf_limit = limit;
                    snapshot.leaf_current = Some(current);
                }
                if let Some(limit) = limit {
                    snapshot.min_limit = Some(snapshot.min_limit.map_or(limit, |n| n.min(limit)));
                    let remaining = limit.saturating_sub(current);
                    snapshot.headroom =
                        Some(snapshot.headroom.map_or(remaining, |n| n.min(remaining)));
                }
            }
            (None, None) => {
                // The true hierarchy root has no memory.max/current. Disabled
                // controllers may omit them below root, but the node must exist.
                let controllers = read(&directory.join("cgroup.controllers"))?;
                let true_root = directory == mount && root == Path::new("/");
                if !true_root && controllers.split_whitespace().any(|c| c == "memory") {
                    return Err(invalid());
                }
            }
            _ => return Err(invalid()),
        }
        if directory == mount {
            return Ok(snapshot);
        }
        if !directory.pop() || !directory.starts_with(&mount) {
            return Err(invalid());
        }
    }
    Err(invalid())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    struct Fixture {
        files: HashMap<PathBuf, String>,
    }
    impl Fixture {
        fn new() -> Self {
            Self {
                files: HashMap::new(),
            }
        }
        fn node(&mut self, directory: &str, max: &str, current: &str) {
            self.files
                .insert(Path::new(directory).join("memory.max"), max.into());
            self.files
                .insert(Path::new(directory).join("memory.current"), current.into());
        }
        fn root(&mut self, directory: &str) {
            self.files.insert(
                Path::new(directory).join("cgroup.controllers"),
                "cpu memory io".into(),
            );
        }
        fn sample(&self, member: &str, mounts: &str) -> io::Result<Memory> {
            observe(member, mounts, |p| {
                self.files
                    .get(p)
                    .cloned()
                    .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))
            })
        }
    }
    const MOUNT: &str = "20 1 0:20 / /cg rw - cgroup2 cgroup rw\n";
    #[test]
    fn ancestor_usage_includes_siblings_and_is_not_subtracted_from_leaf_limit() {
        let mut f = Fixture::new();
        f.root("/cg");
        f.node("/cg/team", "2000", "1950");
        f.node("/cg/team/job", "1000", "100");
        let s = f.sample("0::/team/job\n", MOUNT).unwrap();
        assert_eq!((s.leaf_limit, s.leaf_current), (Some(1000), Some(100)));
        assert_eq!((s.min_limit, s.headroom), (Some(1000), Some(50)));
        assert_eq!(s.available(Some(9000)), Some(50));
        assert_eq!(s.available(Some(30)), Some(30));
        assert_eq!(s.available(None), Some(50));
    }
    #[test]
    fn unlimited_leaf_still_observes_parent_and_exhaustion_saturates() {
        let mut f = Fixture::new();
        f.root("/cg");
        f.node("/cg/p", "100", "101");
        f.node("/cg/p/job", "max\n", "7");
        let s = f.sample("0::/p/job", MOUNT).unwrap();
        assert_eq!(s.headroom, Some(0));
        assert_eq!(s.leaf_limit, None);
        assert_eq!(s.leaf_current, Some(7));
    }
    #[test]
    fn namespaced_root_and_escaped_mount_are_resolved_without_walking_above_mount() {
        let mut f = Fixture::new();
        f.node("/cg space", "500", "300");
        f.node("/cg space/job", "max", "10");
        let s = f
            .sample(
                "0::/tenant/job",
                "1 0 0:1 /tenant /cg\\040space rw - cgroup2 cgroup rw",
            )
            .unwrap();
        assert_eq!(s.headroom, Some(200));
        assert_eq!(s.min_limit, Some(500));
    }
    #[test]
    fn widest_mount_exposes_more_ancestors_and_prefixes_are_component_aware() {
        let mut f = Fixture::new();
        f.root("/cg");
        f.node("/cg/p", "100", "99");
        f.node("/cg/p/job", "max", "0");
        f.node("/bind/job", "1000", "0");
        let mounts = format!("1 0 0:1 /p /bind rw - cgroup2 cgroup rw\n{MOUNT}");
        assert_eq!(f.sample("0::/p/job", &mounts).unwrap().headroom, Some(1));
        assert!(f
            .sample(
                "0::/prefix/job",
                "1 0 0:1 /pre /bind rw - cgroup2 cgroup rw"
            )
            .is_err());
    }
    #[test]
    fn unreadable_malformed_or_disappeared_finite_ancestor_fails_closed() {
        let mut f = Fixture::new();
        f.root("/cg");
        f.node("/cg/p", "100", "50");
        f.node("/cg/p/job", "max", "0");
        for bad in ["", "max", "-1", "+1", "18446744073709551616", "1 2"] {
            f.files.insert("/cg/p/memory.current".into(), bad.into());
            assert!(f.sample("0::/p/job", MOUNT).is_err());
        }
        f.files.remove(Path::new("/cg/p/memory.current"));
        assert!(f.sample("0::/p/job", MOUNT).is_err());
        assert!(observe("0::/", MOUNT, |_| Err(io::Error::from(
            io::ErrorKind::PermissionDenied
        )))
        .is_err());
    }
    #[test]
    fn malformed_limit_is_not_confused_with_unlimited() {
        let mut f = Fixture::new();
        f.root("/cg");
        for bad in ["", "MAX", "-1", "+1", "18446744073709551616", "max 1"] {
            f.node("/cg/job", bad, "0");
            assert!(f.sample("0::/job", MOUNT).is_err());
        }
        f.node("/cg/job", "0", "0");
        assert_eq!(f.sample("0::/job", MOUNT).unwrap().headroom, Some(0));
    }
    #[test]
    fn true_root_and_disabled_controller_have_no_fabricated_limit() {
        let mut f = Fixture::new();
        f.root("/cg");
        assert_eq!(f.sample("0::/", MOUNT).unwrap(), Memory::default());
        f.files
            .insert("/cg/job/cgroup.controllers".into(), "cpu io".into());
        assert_eq!(f.sample("0::/job", MOUNT).unwrap(), Memory::default());
        f.files
            .insert("/cg/job/cgroup.controllers".into(), "memory".into());
        assert!(f.sample("0::/job", MOUNT).is_err());
    }
    #[test]
    fn rejects_unsafe_deleted_duplicate_or_unresolvable_membership() {
        let f = Fixture::new();
        for member in [
            "0::relative",
            "0::/../escape",
            "0::/a/./b",
            "0::/a (deleted)",
            "0::/a\n0::/b",
            "0::/a\0b",
        ] {
            assert!(f.sample(member, MOUNT).is_err());
        }
        assert!(f
            .sample("0::/", "1 0 0:1 / /cg\\999 rw - cgroup2 cgroup rw")
            .is_err());
    }
    #[test]
    fn no_visible_v2_retains_only_host_observation() {
        let f = Fixture::new();
        assert_eq!(
            f.sample("0::/", "").unwrap().available(Some(100)),
            Some(100)
        );
        assert_eq!(f.sample("1:memory:/job", MOUNT).unwrap(), Memory::default());
    }
    #[test]
    fn ancestor_observation_drives_context_and_allocation_admission() {
        use crate::allocation_admission::admit_additional_allocation;
        use crate::hardware::{Hardware, ProcessMemory};
        use crate::streamed_plan::{plan_streamed_shape, StreamedPlanRequest, StreamedShape};
        let mut f = Fixture::new();
        f.root("/cg");
        f.node("/cg/team", "2000", "1150");
        f.node("/cg/team/job", "10000", "100");
        let sample = f.sample("0::/team/job", MOUNT).unwrap();
        let hardware = Hardware {
            logical_cpus: 2,
            total_ram_bytes: sample.min_limit,
            available_ram_bytes: sample.available(Some(9000)),
        };
        let shape = StreamedShape {
            workspace_fixed_bytes: 100,
            workspace_per_token_bytes: 10,
            kv_bytes_per_token: 2,
            training_context: 512,
            file_bytes: 10000,
            metadata_prefix_bytes: 16,
            chunk_bytes: 8,
            ..Default::default()
        };
        let request = StreamedPlanRequest {
            workspace_budget_bytes: 2000,
            runtime_reserve_bytes: 0,
            os_reserve_bytes: 0,
            allow_context_reduction: true,
            ..Default::default()
        };
        let plan = plan_streamed_shape(shape, &hardware, request).unwrap();
        assert_eq!(plan.selected_context, 71); // (850 - metadata32 - chunk8 - fixed100)/10.
        assert!(plan_streamed_shape(
            shape,
            &hardware,
            StreamedPlanRequest {
                allow_context_reduction: false,
                ..request
            }
        )
        .is_err());
        let process = ProcessMemory {
            cgroup_limit_bytes: sample.leaf_limit,
            cgroup_current_bytes: sample.leaf_current,
            ..Default::default()
        };
        assert!(admit_additional_allocation(&hardware, &process, 851, 0).is_err());
        assert!(admit_additional_allocation(&hardware, &process, 850, 0).is_ok());
        f.node("/cg/team", "2000", "1999");
        let hardware = Hardware {
            available_ram_bytes: f
                .sample("0::/team/job", MOUNT)
                .unwrap()
                .available(Some(9000)),
            ..hardware
        };
        assert!(plan_streamed_shape(shape, &hardware, request).is_err());
    }

    #[test]
    fn real_files_follow_parent_updates_and_reject_disappearance_and_oversize() {
        let root =
            std::env::temp_dir().join(format!("llmbridge-cgroup-tree-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(root.clone());
        let parent = root.join("team");
        let leaf = parent.join("job");
        std::fs::create_dir_all(&leaf).unwrap();
        std::fs::write(root.join("cgroup.controllers"), "memory").unwrap();
        std::fs::write(parent.join("memory.max"), "2000").unwrap();
        std::fs::write(parent.join("memory.current"), "1900").unwrap();
        std::fs::write(leaf.join("memory.max"), "max").unwrap();
        std::fs::write(leaf.join("memory.current"), "20").unwrap();
        let mount = root
            .to_str()
            .unwrap()
            .replace('\\', "\\134")
            .replace(' ', "\\040");
        let mounts = format!("1 0 0:1 / {mount} rw - cgroup2 cgroup rw");
        assert_eq!(
            observe("0::/team/job", &mounts, read_text)
                .unwrap()
                .headroom,
            Some(100)
        );
        std::fs::write(parent.join("memory.current"), "1999").unwrap();
        assert_eq!(
            observe("0::/team/job", &mounts, read_text)
                .unwrap()
                .headroom,
            Some(1)
        );
        std::fs::write(parent.join("memory.current"), vec![b'1'; 65537]).unwrap();
        assert!(observe("0::/team/job", &mounts, read_text).is_err());
        std::fs::remove_file(parent.join("memory.current")).unwrap();
        assert!(observe("0::/team/job", &mounts, read_text).is_err());
    }

    #[test]
    fn hierarchy_walk_is_bounded() {
        let mut f = Fixture::new();
        let mut path = PathBuf::from("/cg");
        f.root("/cg");
        for _ in 0..129 {
            path.push("a");
            f.node(path.to_str().unwrap(), "max", "0");
        }
        let member = format!("0::{}", path.to_str().unwrap().strip_prefix("/cg").unwrap());
        assert!(f.sample(&member, MOUNT).is_err());
    }
}
