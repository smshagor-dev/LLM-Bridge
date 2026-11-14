# Visible cgroup-v2 memory admission

Linux hardware profiling intersects host `MemAvailable` with `memory.max - memory.current` at each visible ancestor of the process cgroup. Subtraction saturates at zero. The minimum remaining headroom wins; the minimum hard limit separately caps reported total RAM. Ancestor usage includes sibling processes, so subtracting only leaf usage from a parent's limit would overstate availability.

For example, a leaf with limit 1000 and usage 100 has 900 bytes of local headroom. If its parent has limit 2000 and usage 1950, effective cgroup headroom is 50, even though the parent's absolute limit is larger. The same hardware profile feeds streamed context/cache admission and the optional cache-pressure monitor.

Discovery uses `/proc/self/cgroup` and cgroup2 entries in `/proc/self/mountinfo`, including mount roots, bind-mounted subtrees and standard escaped mount paths. Among compatible mounts, the widest visible root is selected. Ancestor traversal stops at that mount root and is limited to 128 nodes. File reads are bounded (1 MiB mountinfo; 64 KiB other interfaces). Membership is checked again after observation; a detected migration invalidates the sample.

Malformed counters, finite limits without usage, permission failures, disappeared nodes, unmappable visible mounts and excessive depth invalidate availability. They cannot silently become an unlimited constraint. The true hierarchy root and confirmed nodes without an enabled memory controller may omit memory interfaces; visible parents are still examined. `memory.max=max` with numeric usage is a valid unlimited **leaf**, so additional-allocation admission intersects the already constrained hardware availability instead of rejecting that pair as incomplete.

`ProcessMemory.cgroup_limit_bytes` and `cgroup_current_bytes` remain **leaf** observations for callers with explicit leaf-usage ceilings. Hardware available/total values incorporate visible ancestors; an ancestor's usage is never mislabeled as leaf usage.

## Validation and limits

Regression fixtures cover sibling usage, exhausted ancestors, unlimited leafs, distinct minimum-limit/headroom constraints, namespaced/bind roots, escaped mounts, component-aware prefixes, unavailable/malformed interfaces, disabled controllers, hierarchy depth and unsafe/deleted membership. A real temporary directory exercises production file reads as parent usage changes and then disappears. A planner regression demonstrates context reduction and exact additional-allocation admission from the ancestor observation.

These are parser/filesystem/policy regressions, not privileged kernel cgroup enforcement or physical low-RAM experiments. Hidden ancestors above a namespace/bind root cannot be observed. With no visible cgroup2 mount (or only v1 membership), only host RAM is reported; that is not proof of an unlimited container. Cgroup v1, memory.high throttling, swap policy, reclaimability and atomic controller snapshots are outside this change. Concurrent usage/limit/mount changes can still race sampling; inference uses separate explicit workspace/Linux address-space guards where requested. Profiling performs extra bounded proc/interface reads, with no throughput improvement claimed.

Kernel interface reference: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html (hierarchical controllers, memory.current and memory.max).
