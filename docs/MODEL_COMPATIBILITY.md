# Verified execution support — 2026-11-13

| Model/format/path | Inspection | Generation | Validation |
| --- | --- | --- | --- |
| stories15M-q8_0, resident CPU | GGUF index | Yes | Complete generated-byte reference comparison |
| stories15M-q8_0, streamed CPU | GGUF index | Yes, experimental full decoder | Three prompts/eight tokens; complete output and FP32-KV logits; Linux 24 MiB address-space cap below model file size |
| Other dense Llama F32/Q8_0, standard full-head RoPE | GGUF index | Streamed path accepts implemented metadata/shapes | No independent per-model verification |
| Other GGUF families/quantizations | Bounded header/index for known layouts | Resident upstream adapter may support them | Not adapter-verified; streamed path rejects |
| SafeTensors/ONNX, MoE/multimodal/recurrent | Not implemented/adapter-verified | Not advertised | None |
| GPU, remote storage | Not implemented | No | None |

Pinned checkpoint SHA-256: `2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6`. Reference: llama.cpp `cd959bae99ff4bed22bef2604f45241c09bbb8c8` (v0.9.0). See [streamed decoder protocol](STREAMED_DECODER.md) and checked-in JSON evidence.

Raw prompt continuation only. Linux RLIMIT_AS validation is a constrained process test, not physical 2GB hardware, a general RSS API or evidence for large models. Unknown Llama metadata options and extra tensors fail closed in the streamed path.
