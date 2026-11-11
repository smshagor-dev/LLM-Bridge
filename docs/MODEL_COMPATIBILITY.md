# M1 compatibility matrix (2026-11-12)

| Format/model family | Header inspection | Native execution | Verified output/reference |
|---|---|---|---|
| GGUF v2/v3 (bounded header only) | Implemented; CI pending | Not implied by header | No |
| llama.cpp compatible decoder GGUF on CPU | Yes | Implemented via upstream v0.9.0 C++ API; CI/model test pending | No |
| SafeTensors / ONNX | No | No | No |
| MoE / multimodal / recurrent adapters | No model-specific validation | Not advertised | No |
| GPU, out-of-core execution | N/A | No | No |

The native upstream library may support additional architectures but **LLM Bridge does not mark them verified** until family-specific controlled experiments exist. Raw text completion does not yet apply the model's chat template; M1 is not a fully validated chat model adapter.
