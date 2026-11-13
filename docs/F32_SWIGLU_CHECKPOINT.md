# Bounded F32 SwiGLU checkpoint (experimental)

This branch introduces `f32_swiglu_cancellable`, a caller-buffered feed-forward
composition using three rank-2 F32 matrices from one indexed GGUF source:

`down(SiLU(gate(input)) * up(input))`

Each projection reads tensor rows through `BoundedLocalReader` and reuses one
bounded file-chunk buffer. Gate, up, final scratch and output activation slices
are supplied by the caller. The function validates all tensor shapes and
requested staging/activation budgets before tensor I/O, rejects non-finite
arithmetic, checks cancellation between operations, and only commits the
caller-visible output after all stages succeed.

## Validation protocol

```sh
cargo fmt --all -- --check
cargo test -p llmbridge-core f32_swiglu
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The synthetic GGUF regression fixtures cover reference scalar SwiGLU values,
exact tensor-read accounting, invalid metadata and budgets before I/O, failure
of the final projection with unchanged output, and cancellation before I/O.

**Status at proposal:** These commands have **not** been executed in the
current connector environment (Rust toolchain unavailable). PR remains draft.

**Not proven:** The current full-model generation path still uses standard
llama.cpp weight loading. This F32 operator is not integrated into the native
transformer graph, is not an authenticated whole-model benchmark, is not
bitwise ggml/logit parity, and does not enforce a process RSS ceiling. Caller
activation-budget accounting does not reserve RAM, and the OS page cache is
outside this per-operation staging budget.

## Residual composition extension

`f32_swiglu_residual_cancellable` computes
`residual + down(SiLU(gate(input)) * up(input))` with an extra
caller-provided output-sized `projection` buffer. It preflights residual
values and total activation bytes, computes the three tensor projections,
checks all residual additions for overflow, then commits output in one pass.
Tests cover scalar-reference results, pre-I/O budget and NaN rejection, and
late residual overflow with unchanged output. The residual and projection
buffers are included in activation accounting; neither is silently allocated.

**All newly added tests are unexecuted until CI or a local Rust toolchain
runs them.** A successful CI result must be inspected before merging.
