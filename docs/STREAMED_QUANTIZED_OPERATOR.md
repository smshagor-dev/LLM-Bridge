# Bounded Q8_0 CPU tensor operator

The Rust q8_0_matvec function uses the GGUF tensor index to identify a genuine
rank-two Q8_0 matrix, then supplies its rows to a native C++ operator through
the existing bounded, synchronous Rust range callback.

- Explicit RAM allocations: a Rust byte buffer capped by the caller, a native
  single-row Q8_0 buffer, a native float32 dequantized row, the user-supplied
  activation vector, and the float32 output vector. Weight matrix rows are
  not retained between iterations.
- Numeric work: actual ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float
  runs on raw Q8_0 bytes; native CPU then performs a dot product per row.
  Input is a float32 vector (the CLI chooses all ones for repeatability).
- Rejections: non-Q8_0 or non-rank-two tensor, invalid block-alignment,
  mismatched activation width, non-finite input, oversized shape, insufficient
  chunk budget, source-identity changes, short reads and cancellations.
- Failure is atomic at the Rust API boundary: native output may be partly
  written, but Rust never returns it on any failure.
- No cache prediction, mmap-backed pretend streaming or network backend
  is involved in this small operator.
- Validation: synthetic Q8_0 fixture with two rows and known F16 scales,
  native ABI tests, plus independent Python parity on one genuine checkpoint
  tensor in scripts/check_tensor_matvec.py. Numeric tolerance 1e-5 relative
  and 1e-4 absolute accounts for F32 output rounding.

Build and run the native CLI, then run:

    llmbridge models tensor-matvec /path/model.gguf blk.0.attn_q.weight

CRITICAL LIMITATION: The full llama.cpp model graph still loads weights
through its preexisting model loader; this operator does not yet replace
attention, MLP, embedding, KV cache, or full token decode. The per-tensor
working-set bound is not a process-wide resident memory bound. Completion
of the out-of-core milestone requires integrating these mechanisms into actual
layer execution, full-model correctness validation and measured RSS.
