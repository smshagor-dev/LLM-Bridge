# Conservative parallel-worker memory admission

`worker_plan::plan_workers` chooses the largest CPU worker count that fits a
caller-supplied fixed session charge, existing reservations, OS reserve, and
**per-worker** scratch estimate. Unknown memory and arithmetic overflow fail
closed. No workers are spawned by the planner.

For example, 230 bytes available, 100 shared, 20 already reserved, 10 OS
reserve and 30 per worker admits three workers and leaves 10 bytes. This is
synthetic arithmetic, not a physical hardware measurement.

The caller must account for native allocations, scheduler queue storage and
the actual shared graph; it must atomically reserve memory before dispatch,
recheck changing cgroup pressure, and honor cancellation. This helper does not
enforce RSS, reclaim allocator pages, or demonstrate inference parity, older
CPU support, 2 GB machines or large-model feasibility.

Validation pending on the new branch: `cargo test -p llmbridge-core worker_plan`,
`cargo fmt --all -- --check`, and workspace Clippy. Seven deterministic
unit tests cover limits, zero scratch, unknown RAM and overflows.
