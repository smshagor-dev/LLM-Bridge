# LLM Bridge engineering rules

- Standalone, vendor-agnostic inference. Follow the user-supplied master specification.
- CPU only first; no GPU or unverified formats marketed as supported.
- Only mark M1 verified after actual GGUF model inference and trusted comparison.
- mmap is not proof of out-of-core tensor streaming or bounded resident memory.
- Every FFI entry point must be versioned and catch C++ exceptions.
- Don't commit weights, prompts, keys, tokens, or fabricated measurements.
- Keep status honest and work on isolated branches with reviewable PRs.
