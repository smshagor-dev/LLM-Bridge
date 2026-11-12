# Memory metrics and constrained host admission

M2 instrumentation on Linux uses only OS-provided observations, never estimated benchmark results:
- \`/proc/self/status\`: current \`VmRSS\` and process-lifetime \`VmHWM\`, both in KiB converted with checked arithmetic.
- \`/proc/self/smaps_rollup\`: current proportional set size (\`Pss\`) when permission allows.
- cgroup v2 \`memory.max\` and \`memory.current\`: a bounded container's memory headroom. Effective available RAM is the smaller of host \`MemAvailable\` and cgroup headroom.
- Flags: \`llmbridge chat /path/model.gguf --prompt "..." --metrics\`.

\`LLMBRIDGE_MEMORY {json}\` is written to stderr after native generation returns. \`benchmarks/run_cpu_baseline.py\` records the raw values and three-run wall times in JSON.

**Important sampling semantics**: only \`VmHWM\` is a process-lifetime peak RSS. Current RSS/PSS and cgroup usage are sampled **after** the inference call, which may already have freed weights. They are *not* peak PSS or total OS page-cache measurements. A cgroup admission estimate is **not** an enforceable hard limit. For a genuine 2GB-machine claim, run the workload under a real cgroup/Job Object hard limit with system survival metrics, record failures and benchmark against independently verified golden model output.
