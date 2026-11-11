# LLM Bridge implementation status (2026-11-12)

## M0 — code committed; CI pending
- Repository inspected: empty, main initialized, feature branch created.
- Rust workspace, bounded GGUF header inspection, Linux memory availability + conservative admission, tests, benchmark process harness, GitHub Actions, ADR.
- Local sandbox: no rustc/cargo, and no direct GitHub DNS. CI must provide compilation evidence.

## M1 — real code implemented, verification pending
- C++20 llama.cpp v0.9.0 pinned at cd959bae99ff4bed22bef2604f45241c09bbb8c8; CMake CPU-only build.
- Stable versioned C ABI with bounded args, native exception containment, single-request lock, real llama.cpp tokenizer, prefill/decode, greedy sampler, raw token byte callback, cooperative callback cancellation and RAII cleanup.
- Rust FFI with panic-catching callback, streaming CLI, SIGINT cancellation and honest GGUF header inspection.
- CLI: doctor, hardware --json, models inspect, chat <local-GGUF> --prompt ... --stream.
- Not yet verified: native compile, real model output, reference parity, real peak RSS, cross-platform behavior.
- No chat-template behavior or stochastic decoding validated. Native generation is **raw prompt completion**, not fully featured chat.
- No unattended model download; see docs/M1_TEST_PROTOCOL.md.

## M2–M11 — pending
- **No genuine tensor block streaming**, hard per-process cap, out-of-core proof, 2GB huge-model validation, remote storage, GPU, MoE, API or desktop.
- mmap alone is not M2. No performance or output metrics fabricated.

## Environment constraints
- Container has CMake and C++ compilers but no Rust toolchain, no outbound github.com DNS.
- Remote GitHub edits are through the GitHub connector; GitHub Actions may validate builds independently.
