# Experimental streamed Q4_0 execution

The dense Llama streamed decoder accepts F32, Q8_0 and Q4_0 tensors. Q4_0 uses the pinned ggml packed-row geometry (18 bytes per 32 weights), its dequantizer for embedding rows and its CPU dot/activation-quantization traits for projections. Weights continue through bounded Rust range reads; no resident weight loader or full dequantized matrix is introduced. Q4_1 and K-quant formats are still rejected by this decoder.

RMS normalization follows the pinned upstream CPU arithmetic: square each activation in FP32, then accumulate those rounded squares in double precision. Squaring directly in double is a different rounding path and can cross later activation-quantization thresholds. The wrapper uses the same explicit AVX2/FMA profile as the pinned CPU kernels. Normal RoPE uses incremental FP32 angles; FP32 attention uses pinned dot/softmax primitives with padded score/value scratch; SwiGLU uses the pinned CPU primitive. Both Q8_0 and Q4_0 need reference regression coverage.

## Reproducible checkpoint

The test fixture is **requantized from an already lossy Q8_0 checkpoint**. It is not a new trained model, an original-F32 quantization or a language-quality evaluation. No weights are committed.

| Item | Pinned value |
| --- | --- |
| Upstream source/reference/quantizer revision | `cd959bae99ff4bed22bef2604f45241c09bbb8c8` |
| Source stories15M Q8_0 SHA-256 | `2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6` |
| Derived pure Q4_0 SHA-256 | `b5cd77bb0d1a785a82d7a8a2bcb4170dfe6e43e881b9cbb0d1c8184a6dd99839` |
| Source file size | 26,671,328 bytes |
| Derived file size | 14,469,344 bytes |
| Tensor inventory | 44 Q4_0 matrices and 13 F32 tensors |

Using the pinned independent reference build:

```sh
cmake --build reference-build --target llama-quantize --parallel 2
reference-build/bin/llama-quantize --allow-requantize --pure \
  models/stories15M-q8_0.gguf models/stories15M-derived-q4_0.gguf Q4_0 1
```

CI checks the exact derived hash. A separate `--pure Q4_1` conversion supplies a real unsupported-format fixture; Rust indexing must accept it, then native planning and generation must refuse its tensor type before token/logit output.

## Reference protocol

```sh
python3 scripts/check_streamed_parity.py --binary target/release/llmbridge \
  --reference reference-build/bin/llama-simple \
  --model models/stories15M-derived-q4_0.gguf \
  --checkpoint stories15M-derived-q4_0 --reference-no-repack --verify \
  --output /tmp/q4-uncached
```

Repeat with `--cache-mb 1`, then use `scripts/compare_cache_results.py` to require identical cached/uncached output and raw logits plus measured application range-read reduction. The separate real-model integrity contract checks cancellation, insufficient verification budget and same-identity late weight mutation with the pinned Q4_0 digest.

The reference instrumentation only exports logits and supplies explicit runtime configuration through stock upstream APIs. Both paths use FP32 KV and disabled flash attention. `--reference-no-repack` explicitly sets `llama_model_params.use_extra_bufts = false`; it does not alter weights, replace expected output or relax the existing `0.01 + 0.001*abs(reference)` tolerance. Reports distinguish the default upstream repacking configuration and the explicitly disabled configuration. The golden runner clears an inherited repacking override unless requested, and fails if repacking is observed despite the flag. The default repacked Q4 reference did **not** meet the same comparison gate on this fixture; successful agreement is restricted to the explicitly unpacked reference configuration. That negative result is retained in the evidence rather than weakening the tolerance. This is not proof of agreement with every optimized upstream backend.

The smaller Q4_0 file is **not** used to claim the existing 24 MiB out-of-core proof: it is smaller than that ceiling. Q4_0 numerical trials use no process cap. The original Q8_0 model-larger-than-cap gate and 24 MiB trials remain unchanged. File-size reduction is not a measured process-RAM reduction. Physical 2GB hardware, larger models, other Q4 checkpoints and language-quality evaluation remain pending.

The padded attention score/value scratch is explicitly accounted: native per-context-token workspace is KV bytes plus 8 bytes, and fixed workspace includes both 255-float padding tails. This replaces the earlier single score-array coefficient (KV plus 4). KV storage itself remains FP32 and the shape-probe ABI still returns actual fixed/per-token coefficients. Exact-budget and one-byte-below contracts remain required; no hidden allocation or changed process ceiling is used to obtain parity.
