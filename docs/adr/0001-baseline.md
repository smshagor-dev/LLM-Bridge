# ADR 0001 — M1 CPU baseline (2026-11-12)

Rust owns the CLI, hardware/resource admission, model inspection and cancellation. Pinned llama.cpp v0.9.0 (commit cd959bae99ff4bed22bef2604f45241c09bbb8c8) supplies actual C++ CPU kernels and tokenization behind a narrow, versioned C ABI. Non-cooperative load/compute cancellation and out-of-core streaming remain separate problems. Standard llama.cpp mmap does not satisfy the M2 out-of-core requirement. Reject unsupported requests and never secretly substitute a model.
