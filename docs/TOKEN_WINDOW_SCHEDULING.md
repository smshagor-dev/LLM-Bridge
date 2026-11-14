# Bounded CPU token-window scheduler (experimental M3 primitive)

`TokenWindows::plan` divides a requested token range into contiguous, non-overlapping windows. Each window charges fixed scratch plus per-token scratch against the caller-supplied budget **after** outstanding reservations. It rejects zero work, zero per-token estimates, underfunded fixed scratch, and overflow. `next_checked` polls a cooperative cancellation flag before exposing work and never advances on cancellation.

The caller must reserve the returned memory **atomically** with other concurrent requests and retain that reservation while the native work executes. This is a deterministic scheduling primitive, not native graph integration, a measured throughput gain, RSS enforcement, or proof of running large models on low-RAM machines.

Run `cargo test -p llmbridge-core token_window` and `cargo fmt --all -- --check` in a Rust-equipped checkout before merging. Tests cover exact memory boundaries, full coverage, cancellation, reservations, malformed requests and u64 extremes.
