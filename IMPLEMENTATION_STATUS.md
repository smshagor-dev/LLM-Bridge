
# LLM Bridge implementation status — 2026-11-12

## M0 — implemented and initial CI validated
- Empty repository inspected, main initialized, isolated feature branch and PR #1.
- Rust workspace, bounded GGUF header inspection, Linux RAM profiler and conservative admission, tests, benchmarks harness, CI, ADR and license/security notices.
- GitHub Actions run 43114276598: foundation rustfmt, parser tests, clippy passed; native CPU tests passed (10 tests on initial native run 43113641423).
- Local execution container lacks rustc/cargo and GitHub DNS; tests ran in remote GitHub Actions.

## M1 — CPU functional verification on one tiny GGUF
- C++20 native ggml/llama.cpp v0.9.0 pinned at cd959bae99ff4bed22bef2604f45241c09bbb8c8, Rust C ABI, CPU greedy decode, stream callback and Ctrl+C request.
- Verified **real model** in GitHub Actions run 43114276598: ggml-org/tiny-llamas stories15M Q8 GGUF; model SHA-256 2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6 verified before run. Independent pinned upstream `llama-simple` produced exactly matching generated output bytes for 'The sky is' at 16 token max with greedy sampling.
- Additional real-model cancellation, oversized context rejection and 3 repeat trials are newly added; CI confirmation pending.
- **Not established**: broader GGUF model families, stochastic sampling parity, chat-template support, reproducible TTFT/tokens/s, RSS/PSS under actual 2GB, OS safety on constrained hosts.

## M2 (stacked draft PR #2) — early reader slice
- Rust reusable-budget local range reader, checked ranges, changed-file detection and actual SHA-256 file verification, CLI `models verify`.
- Not integrated into native transformer execution; no out-of-core model inference or hard RSS cap claim.

## M3–M11 — not implemented
- Remote provider connectors, GPU/MoE, API, desktop, universal architecture adapters and very-large-model research await verified M2 groundwork.
