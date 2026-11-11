
# M2 pre-integration: bounded local byte range reader

The first M2 source slice is executable Rust in \`llmbridge-core/src/storage.rs\`. It uses a **reusable preallocated bounded buffer** for local exact reads, checks file length / modified timestamp before and after each range, prevents overflow and over-budget requests, rejects symlinks, and supports whole-file SHA-256 using chunks. Tests cover actual file bytes and digest.

CLI: \`llmbridge models verify /path/model.gguf <sha256-hex>\` verifies exact file bytes without reading entire file into memory. It reports disk bytes read and allocated reader chunk bytes.

**Not a completed M2 milestone**: no GGUF tensor inventory/offset mapping, scheduled layer execution, cache/prefetch, actual model-weight streaming into llama.cpp graph, strict **process RSS** admission or 2GB test. This slice bounds its own explicit allocation, **not** the OS page cache or model execution. The verified file bytes are not a substitute for a signed manifest or compatible model validation.
