
# LLM Bridge implementation status (2026-11-12)

## M0 — verified in CI
- Empty repo inspected, main initialized, feature branch and draft PR #1.
- Workspace, GGUF header inspector, CPU hardware profile and admission, tests, benchmark process harness, CI.
- GitHub Actions native compile and 10 initial tests passed on run 43113641423.
- Foundation checks (fmt, no-default-features tests and clippy) passed on run 43114276598.
- Local sandbox has no rustc/cargo or outbound github.com DNS; CI executes these checks.

## M1 — native built, end-to-end GGUF parity pending
- C++20 pinned llama.cpp v0.9.0 CPU kernels and Rust FFI; streaming greedy decode, cancellation, scoped cleanup, strict argument checking and model admission.
- CLI doctor, hardware, models inspect, chat.
- Native ABI and invalid-model unit tests passed. Real model parity job added and running independently.
- No validated chat-template mode, numerical reference parity, time-to-first-token, RSS/PSS, or two-gigabyte hardware proof yet.

## Early M2 slice (separate stacked PR branch)
- Bounded local file range reader, real SHA-256 verification and CLI integration. Tests pending.
- **Not** wired to inference graph; no out-of-core model inference or strict process memory limit proven.

## M2–M11
Full out-of-core, hard RSS budget, remote connectors, GPU, MoE, API, desktop and huge-model research remain unimplemented. No fabricated performance claims.
