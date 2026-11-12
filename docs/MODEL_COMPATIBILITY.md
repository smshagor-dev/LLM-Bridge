
# LLM Bridge verified execution support (2026-11-12)

| Model/format | Bounded header inspection | CPU generation | Trusted greedy parity |
|---|---|---|---|
| ggml-org/tiny-llamas stories15M-q8_0.gguf | Yes | **Verified in GitHub Actions** | **Pass** vs pinned upstream llama-simple, byte-for-byte generated suffix |
| GGUF v2/v3 other architectures | Header only | Available through native upstream adapter, unverified per-family | Not tested |
| SafeTensors, ONNX | Not yet | No | No |
| MoE, multimodal, recurrent | Not adapter-verified | Not advertised | No |
| GPU, hard-RSS-bounded out-of-core | N/A | No | No |

**Evidence**: GitHub Actions run 43114276598, `model-parity` job, verified model SHA-256 `2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6`, pinned upstream llama.cpp v0.9.0. The model output matched the independent upstream example exactly in the tested greedy case. Artifact contains source outputs and JSON result.

Scope is **raw prompt continuation**, not a chat-template-validated conversational model. No claim is made that every llama.cpp-supported quantization or family is verified. No memory-bounded tensor streaming proof is established.
