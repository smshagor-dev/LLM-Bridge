# LLM Bridge

**Independent, vendor-agnostic AI inference engine** in Rust and C++20. CPU-only native GGUF inference is the initial execution path. No cloud account or OpenMindAI integration is required.

## Build

Requirements: Rust >=1.82, CMake >=3.22, C++20 compiler, Git, network during first build (FetchContent pulls pinned llama.cpp v0.9.0). Build with:

```sh
cargo build --release -p llmbridge
cargo run --release -p llmbridge -- doctor
cargo run --release -p llmbridge -- hardware --json
cargo run --release -p llmbridge -- models inspect /models/model.gguf
cargo run --release -p llmbridge -- chat /models/model.gguf --prompt "The sky is" --max-tokens 16 --ctx 128 --threads 2 --stream
```

This early CLI performs **raw prompt completion**, using the model's real tokenizer, attention, native kernels and greedy sampling. It does not fabricate text or silently run remote compute. A terminal Ctrl+C requests cooperative cancellation between token pieces.

No models are bundled. To test a small Q8 GGUF, use the explicit model URL and SHA-256 in [M1 test protocol](docs/M1_TEST_PROTOCOL.md). Check licenses before redistribution.

## Scope and limitations

The C++ runtime is pinned to [llama.cpp v0.9.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.9.0), commit cd959bae99ff4bed22bef2604f45241c09bbb8c8; Rust talks to C++ through a versioned C ABI. This **is not** a new tensor streaming engine yet. Default llama.cpp memory mapping does not prove bounded RSS or 2GB suitability.

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md), [support matrix](docs/MODEL_COMPATIBILITY.md), and [ADR](docs/adr/0001-baseline.md). Only M0/M1 have implementation code; tests and model validation are separately tracked.
