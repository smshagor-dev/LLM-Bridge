# M2 tensor index and bounded streaming (experimental slice)

This is **real source code**, not proof of bounded model inference.

Implemented in \`crates/llmbridge-core/src/gguf_index.rs\` and \`src/storage.rs\`:
- GGUF v2/v3 metadata scanner that never allocates the model's tensor weights.
- Parsing limits for metadata bytes, key/value arrays, index tensor count, tensor rank, alignment, unknown quantization types, and tensor dimensions.
- Exact tensor data offsets (including GGUF general.alignment and padding), checked lengths and overlap rejection for known ggml layouts.
- Deterministic local file ranged reads with an explicit reusable 64KiB chunk buffer.
- SHA-256 over just the tensor's declared byte region, plus fail-closed expected-digest validation.
- Fixture tests: Q8_0 block sizes, F16 offsets, output reconstruction, wrong hash, malformed type, misaligned/overlapping/out-of-file ranges.

CLI:
\`\`\`sh
llmbridge models tensor-index /models/checkpoint.gguf
llmbridge models tensor-sha256 /models/checkpoint.gguf token_embd.weight
llmbridge models tensor-sha256 /models/checkpoint.gguf token_embd.weight <known-64-hex-digest>
\`\`\`

Independent reference check: \`scripts/check_tensor_regions.py\` parses the real model in Python, compares **every** declared tensor's offset/dimensions/size with Rust output, and compares a streamed Rust tensor SHA-256 with Python's direct range hash. GitHub Actions runs this alongside the previously verified 15M Q8 CPU inference baseline. The Python tool is a validation utility only.

**Open risks:** The metadata source and streaming reader are opened separately; the current metadata/mtime checks do not guarantee immutable identity under adversarial file replacement. Parsed metadata remains resident, bounded by static admission limits rather than a user-selected per-model metadata budget. Full-model signature/provenance and real selective ggml tensor loading have not been integrated. The Rust range reader does not alter llama.cpp's existing mmap graph execution and is not M2 acceptance.
