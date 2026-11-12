# LLM Bridge implementation status — 2026-11-12

## M0 — validated
- Repository inspected from empty state. Rust workspace, constrained GGUF header inspector, conservative Linux RAM admission, CI, documentation and third-party notices.
- GitHub Actions 43114276598 and 43115094993: foundation fmt/no-default-features tests/clippy and native compilation passed.

## M1 — tested CPU baseline, additional validation pending
- Pinned llama.cpp v0.9.0 on CPU, Rust C ABI, raw prompt greedy decoding and streaming; no mandatory GPU/cloud.
- Real 15M Q8 GGUF checksum and byte-for-byte independent upstream greedy reference parity passed on run 43114276598.
- Follow-up live cancel/long prompt/3-run baseline validation added to PR #1; latest CI has an outstanding rustfmt issue in the new example (not a native inference failure).

## M2 — native-independent tensor I/O advancement; not complete
- Bounded local read and actual streamed whole-file SHA-256 verified with unit tests on PR #2. Existing CI on f2ef7a38 passed.
- New GGUF v2/v3 tensor index with limited verified ggml type layouts, strict metadata count/length caps, alignment checks, tensor names, shapes and exact byte offsets.
- New tensor range streaming, per-tensor SHA-256, and CLI commands \`models tensor-index\` and \`models tensor-sha256\` (optional expected digest). Synthetic fixture tests cover exact offsets, corrupt/misaligned/overlapping/truncated inputs.
- Independent Python reference parser in CI now compares **actual** 15M GGUF tensor offsets and bounded stream hash; this new test has not yet passed.
- CRITICAL: Rust index reader is **not connected to native llama.cpp model execution**. No full M2 out-of-core inference, enforced process RSS bound, 2GB host survival test or large-model proof.

## Next
- Complete real-checkpoint tensor parity CI, address bugs and add robust source pinning.
- Integrate verified storage callbacks with native graph tensor acquisition and execute a checkpoint larger than its configured RAM budget under enforceable host controls. Measure RSS/PSS and compare logits/outputs.
- Continue M3 and beyond only after sound M2 evidence. No fabricated throughput, universality or 550B claims.
