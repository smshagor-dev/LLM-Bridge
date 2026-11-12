# M2 native GGUF callback integration (experimental, not out-of-core)

The first Rust-to-C++ storage integration is implemented in:
- \`native/include/llmbridge.h\`: a narrow synchronous source callback API.
- \`native/src/bridge.cpp\`: call \`gguf_init_from_callback\` with \`no_alloc=true\`, a fixed maximum chunk request, and a caller-declared file size; return actual native tensor count and tensor-data offset.
- \`crates/llmbridge-core/src/inference.rs\`: Rust callback that reads through \`BoundedLocalReader\`, copies only requested bytes into the native temporary metadata buffer, catches Rust callback panics and forwards read errors without unwinding across the FFI.
- \`crates/llmbridge-cli/src/main.rs\`: \`llmbridge models native-index-probe /path/checkpoint.gguf\`.

The probe validates that an actual pinned ggml/llama.cpp parser receives its GGUF metadata through Rust-managed bounded range reads. It must agree with the Rust parser's tensor count and data offset. CI tests this on the real stories15M Q8 checkpoint, with the actual results exported as a workflow artifact.

**Important**: This runs the native *metadata* parser but does **not** make the inference graph load layer weights on demand. The existing M1 llama.cpp generator still uses its normal model-load pathway. The native callback probe is necessary adapter groundwork, not an M2 correctness or RSS proof. The GGUF parser is not yet isolated in a sandbox; malformed untrusted weights remain risky native inputs.
