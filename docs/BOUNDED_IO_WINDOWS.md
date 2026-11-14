# Bounded adaptive I/O windows (experimental M3 primitive)

`plan_io_window` computes a conservative, block-aligned staging window for one indexed tensor range. It caps each read by the source reader chunk, requires a complete tensor block, charges every concurrent staging buffer, and refuses impossible budgets rather than silently falling back to an unbounded allocation.

The caller must first verify an immutable authenticated model session, hold an atomic memory reservation covering `staging_charge_bytes`, and poll cancellation around every bounded read. This planner does **not** start workers, reserve memory, perform reads, guarantee RSS reclaim, or establish physical low-RAM performance. It is intentionally separate from concurrent worker and FIFO reservation PRs.

Validation: six deterministic Rust unit tests are included. CI/native execution on this new branch is pending. No new real-hardware performance or output-parity claim is made.
