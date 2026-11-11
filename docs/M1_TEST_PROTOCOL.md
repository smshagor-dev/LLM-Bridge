# M1 CPU model validation protocol

1. Use an explicitly authorized GGUF checkpoint. Suggested tiny functional test: ggml-org/tiny-llamas stories15M-q8_0.gguf, SHA-256 2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6.
2. Build CPU only: cargo build --release -p llmbridge. Native backend is pinned to llama.cpp commit cd959bae99ff4bed22bef2604f45241c09bbb8c8.
3. Check SHA-256 of the downloaded model and run scripts/m1_smoke.sh MODEL_PATH.
4. Run the upstream pinned llama.cpp simple example with the **same** GGUF, prompt, greedy sampling, ctx and maximum output tokens and compare raw token pieces/decoded output, not just the presence of nonempty text.
5. Run benchmarks/run_cpu_baseline.py for at least 3 runs and retain raw observations, OS and exact binary SHA.
6. Test SIGINT during generation, bad magic, unsupported model, prompt > context, and unavailable RAM rejection.
7. Publish measured TTFT, RSS/PSS and process/system memory separately. Until these exist, do not mark M1 verified or claim M2 memory bounds.

M1 uses upstream mmap/page cache. It does not actively page a full transformer through a bounded tensor arena.
