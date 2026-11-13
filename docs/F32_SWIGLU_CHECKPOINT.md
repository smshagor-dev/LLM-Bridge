# Bounded F32 SwiGLU and residual checkpoint (experimental)

The following library APIs operate on three rank-2 F32 GGUF tensors
`gate`, `up`, and `down`, using the indexed local range provider:

- `f32_swiglu_cancellable`: `down(SiLU(gate(input)) * up(input))`
- `f32_swiglu_residual_cancellable`: `residual + down(SiLU(gate(input)) * up(input))`

These are genuinely executed three-tensor F32 computations, but are **not**
connected to the full native llama.cpp transformer execution graph.

## Allocation and failure contract

All projection weights are read in bounded ranges with one reusable file
buffer. Input, gate, up, final scratch, residual (if supplied), and output
are caller-owned. The requested activation budget charges every provided
slice; output is only copied after all three projections, gating, residual
addition and cancellation checks pass. A failed operation may mutate the
intermediate or scratch slices, but does not mutate caller-visible output.

Preflight validates the three indexed tensor shapes, input/residual values,
GGUF reader identity, and declared budgets before performing projection reads.
No whole-weight matrix is allocated by these functions. They do **not**
reserve system RAM, bound unrelated allocations or OS page cache, enforce
process RSS, or provide exact ggml SIMD numerical parity.

## Validation commands

```sh
cargo fmt --all -- --check
cargo test -p llmbridge-core f32_swiglu
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

The four initial core SwiGLU tests passed in native CI job
[132064978097](https://github.com/smshagor-dev/LLM-Bridge/actions/runs/43477477928/job/132064978097).
The overall job **failed** because four pre-existing tests in
`f32_stream` and `tensor_batch` failed (102 passed, 4 failed). The fixture
and formatting repairs are covered separately by draft PR #62.
That CI run predates the residual implementation, which must be retested
before this draft PR is ready for review.

New residual-specific tests exercise independent scalar numerical values,
overflow after all projections with unchanged final output, and invalid
residual/budget admission before file reads.

## Missing M2 acceptance evidence

Whole-model streamed weights connected to attention and decoder graph,
native reference logit parity, enforceable memory ceiling, cancellation
during complete generation, and reproducible constrained-hardware
measurements remain **unverified**. No model-size or low-RAM support claims
follow from this checkpoint.
