# Explicit x86-64 baseline build

`LLMBRIDGE_CPU_PROFILE=baseline` builds a separate SSE2 x86-64 native artifact without requiring SSE4.2, AVX, AVX2, FMA, F16C or BMI2. The default remains `avx2`. This is a **compile-time choice**, not automatic runtime dispatch or a silent fallback inside the AVX2 binary.

```sh
LLMBRIDGE_CPU_PROFILE=baseline cargo build --release --locked --target-dir target/baseline
target/baseline/release/llmbridge doctor --json
```

Use separate target directories to retain both binaries. Cargo tracks profile environment changes; wrapper and ggml flags change together. Advanced AVX512/AMX flags are explicitly disabled in both profiles to prevent stale CMake options from raising the ISA. Baseline also disables CPU weight repacking. Unknown profile values fail the build. The binary's embedded profile is authoritative: setting this environment variable only at runtime cannot turn an AVX2 binary into a baseline binary.

| Profile | Minimum admitted CPU/OS | Numerical configuration |
| --- | --- | --- |
| `avx2` (default) | SSE2/SSE4.2/AVX/AVX2/FMA/F16C/BMI2 with OS AVX state support | Existing explicit AVX2 wrapper/ggml kernels; Q4 reference repacking caveat still applies |
| `baseline` | x86-64 with SSE2 | Wrapper/ggml compiler baseline, optional higher ISA flags off, CPU repacking off |

`doctor` reports actual detected hardware features. `native_compatible` and missing features are evaluated against the **embedded profile**, so a baseline build can be compatible even when AVX2 is false. Baseline is SSE2 code, not a validated automatic scalar fallback. Other architectures remain unvalidated.

## Validation

Build the pinned independent reference with `GGML_NATIVE=OFF`, `GGML_SSE42=OFF`, `GGML_AVX=OFF`, `GGML_AVX2=OFF`, `GGML_FMA=OFF`, `GGML_F16C=OFF`, `GGML_BMI2=OFF`, `GGML_CPU_REPACK=OFF` and `GGML_OPENMP=OFF`. Retain the existing reference instrumentation for FP32 KV, disabled flash attention and explicit extra-buffer refusal.

The golden runner accepts `--emulator /path/to/qemu-x86_64 --guest-cpu core2duo --max-tokens 2`. Both the real baseline binary and the independent baseline reference run with that guest CPU model. Full output and all logits are compared; the report records the actual build/CPU profile, emulator version/hash, maximum token count and binary hashes. Three eight-token native-host trials remain separate from three two-token emulated trials for each pinned Q8/Q4 fixture. The existing comparison tolerance and checkpoint hash gates are unchanged.

QEMU user-mode memory observations can include emulator resources, and its CPU model does not constrain physical core count or host RAM. The runner refuses to combine emulation with the process-cap proof. The Q8 24 MiB process-limit experiment runs **natively** on the current host with the baseline artifact; Q4 is too small for that particular model-larger-than-cap proof and remains uncapped.

This establishes only the tested tiny-checkpoint execution on the current host and emulated SSE2 CPU. It is not a physical Core 2 Duo/2GB-machine test, a speed guarantee, broad model support, dynamic SIMD selection or larger-model feasibility result. Requantized Q4 quality limits and AVX2 default-repacking mismatch remain documented in [Q4 validation](STREAMED_Q4_VALIDATION.md).
