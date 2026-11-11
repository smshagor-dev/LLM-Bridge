# M1 CPU model validation protocol

Native build and invalid-model tests have passed in GitHub Actions (initial run 43113641423: 10 tests).
This is not yet a pass on genuine model output or a bound on resident memory.

- Model: ggml-org/tiny-llamas/stories15M-q8_0.gguf.
- Model SHA-256: 2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6.
- Native GGUF tokenizer/compute backend: llama.cpp v0.9.0 pinned commit cd959bae99ff4bed22bef2604f45241c09bbb8c8.
- GitHub CI model-parity job downloads (never bundles) the checkpoint, checks hash, builds native CLI and the **independent pinned upstream llama-simple binary**, runs both greedy with identical prompt/quantization/thread count, and checks generated raw bytes.
- The upstream simple example prints prompt then generation; Bridge prints only generation. Parity harness verifies that upstream stdout ends in the exact Bridge bytes.
- Both outputs and JSON test result are uploaded to the workflow as reproducible evidence, including git SHA.
- Run scripts/m1_smoke.sh with an authorized local GGUF for local validation.
- For speed metrics use benchmarks/run_cpu_baseline.py with >=3 repeats, record file SHA, hardware, cache state, RSS/PSS and process load independently.

M1 is a standard CPU mmap baseline, **not an M2 out-of-core streaming proof**. The original specification requires numeric parity, cancellation under active execution, TTFT, memory validation and broader models before declaring full M1 verified. Results must not be invented.
