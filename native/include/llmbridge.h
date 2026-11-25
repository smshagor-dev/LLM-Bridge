#ifndef LLMBRIDGE_NATIVE_H
#define LLMBRIDGE_NATIVE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define LLMBRIDGE_NATIVE_ABI_VERSION 1

/* Must return 1 to continue or 0 to cancel. token bytes need not be valid UTF-8 individually. */
typedef int32_t (*llmbridge_token_callback)(const char *bytes, size_t size, void *userdata);
typedef int32_t (*llmbridge_cancel_poll)(void *userdata);
uint32_t llmbridge_abi_version(void);
/* Additive ABI v1 entry point: poll returns 1 to continue, 0 to cancel.
 * Poll is synchronous and invoked before loading, before prefill and before
 * each decode step. It cannot interrupt a single blocking native operation. */
int32_t llmbridge_generate_cpu_polled(const char *model_path, const char *prompt,
    uint32_t context_tokens, uint32_t max_tokens, int32_t threads,
    llmbridge_token_callback callback, llmbridge_cancel_poll poll,
    void *userdata, char *error, size_t error_capacity);
/* Returns 0 on successful completion, 2 on cancellation and -1 on error.
 * No exceptions cross this ABI. All inputs are borrowed for call duration.
 * Error buffer is NUL-terminated if error_capacity > 0. */
int32_t llmbridge_generate_cpu(const char *model_path,
                               const char *prompt,
                               uint32_t context_tokens,
                               uint32_t max_tokens,
                               int32_t threads,
                               llmbridge_token_callback callback,
                               void *userdata,
                               char *error,
                               size_t error_capacity);

/* Bounded synchronous byte-range reader. A short read indicates failure.
 * Native GGUF metadata parser does not own userdata or output after callback returns. */
typedef size_t (*llmbridge_range_callback)(void *userdata, void *output,
                                            uint64_t file_offset, size_t length);
/* v3 opt-in lookahead hint: native supplies the exact next tensor row batch
 * and enclosing tensor range. Return 1 when admitted, 0 for synchronous fallback. */
typedef int32_t (*llmbridge_lookahead_schedule_callback)(void *userdata,
    uint64_t file_offset, size_t length, uint64_t tensor_offset, uint64_t tensor_bytes);

/* Experimental full dense decoder for the llama and qwen2 architectures (qwen2
 * adds NeoX RoPE and Q/K/V projection biases; llama supports the optional
 * rope_freqs.weight per-pair frequency factors used by llama-3). Weights use bounded range
 * reads; tokenizer-only loading uses path. Workspace budget excludes tokenizer
 * metadata, parser, allocator and OS pages: it is not a hard RSS cap. Admits F32,
 * Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 and the K-quants Q4_K/Q5_K/Q6_K.
 * Logits callback receives the complete vocabulary before greedy sampling. */
typedef int32_t (*llmbridge_logits_callback)(const float *values, size_t count, void *userdata);
int32_t llmbridge_generate_streamed_cpu(const char *path, const char *prompt,
    uint32_t context, uint32_t max_tokens, uint64_t file_bytes, size_t max_chunk,
    uint64_t workspace_budget, llmbridge_range_callback reader,
    llmbridge_token_callback emit, llmbridge_logits_callback logits,
    llmbridge_cancel_poll poll, void *userdata, char *error, size_t capacity);
/* Additive entry: nonzero tokenizer prefix must match authenticated GGUF metadata.
 * Caller owns sealed vocabulary file and verified weight reader for the call. */
int32_t llmbridge_generate_streamed_cpu_v2(const char *path, const char *prompt,
    uint32_t context, uint32_t max_tokens, uint64_t file_bytes, size_t max_chunk,
    uint64_t workspace_budget, uint64_t tokenizer_prefix_bytes, llmbridge_range_callback reader,
    llmbridge_token_callback emit, llmbridge_logits_callback logits,
    llmbridge_cancel_poll poll, void *userdata, char *error, size_t capacity);
/* Additive opt-in ABI. Existing v1/v2 callers retain their synchronous behavior. */
int32_t llmbridge_generate_streamed_cpu_v3(const char *path, const char *prompt,
    uint32_t context, uint32_t max_tokens, uint64_t file_bytes, size_t max_chunk,
    uint64_t workspace_budget, uint64_t tokenizer_prefix_bytes, llmbridge_range_callback reader,
    llmbridge_lookahead_schedule_callback schedule, llmbridge_token_callback emit,
    llmbridge_logits_callback logits, llmbridge_cancel_poll poll, void *userdata,
    char *error, size_t capacity);

/* Opt-in diagnostic (additive; ABI version unchanged): decode the prompt with
 * the resident llama.cpp backend and report the full logit vector at the final
 * prompt position via emit_logits, called exactly once. Samples/generates
 * nothing. Returns 0 on success, 2 on cancel (unused here), -1 on error.
 * Intended as an independent reference for streamed step-0 logits. */
int32_t llmbridge_resident_prompt_logits_v1(const char *path, const char *prompt,
    uint32_t context_tokens, int32_t threads, llmbridge_logits_callback emit_logits,
    void *userdata, char *error, size_t capacity);

/* Opt-in multi-position diagnostic (additive; ABI version unchanged): greedy-
 * decode up to max_tokens and call emit_logits once per step with the full logit
 * vector BEFORE sampling (same timing as the streamed per-step logits callback).
 * The chosen token is the argmax of each emitted vector. Returns 0 on success,
 * 2 if emit_logits requests stop, -1 on error. */
int32_t llmbridge_resident_decode_logits_v1(const char *path, const char *prompt,
    uint32_t context_tokens, uint32_t max_tokens, int32_t threads,
    llmbridge_logits_callback emit_logits, void *userdata, char *error, size_t capacity);

/* Opt-in teacher-forced streamed diagnostic (additive; ABI version unchanged):
 * prefill the prompt, emit the final-prompt-position logits, then feed each
 * caller-supplied continuation token in order and emit the logits after each.
 * Emits forced_count + 1 logit vectors. Unlike the generate path, it samples
 * nothing: the fed tokens are fixed, so every emitted position shares a prefix
 * with an independently decoded reference at the same index (valid multi-
 * position comparison even after greedy sequences would diverge).
 * tokenizer_prefix behaves as in _v2. Returns 0, 2 on cancel/stop, -1 on error. */
int32_t llmbridge_streamed_forced_logits_v1(const char *path, const char *prompt,
    const int32_t *forced_tokens, uint32_t forced_count, uint32_t context,
    uint64_t file_bytes, size_t max_chunk, uint64_t workspace_budget,
    uint64_t tokenizer_prefix_bytes, llmbridge_range_callback reader,
    llmbridge_logits_callback emit_logits, llmbridge_cancel_poll poll,
    void *userdata, char *error, size_t capacity);

/* Reuses decoder validation without allocating activations, KV or weights.
 * Workspace is the native admission requirement, not process RSS. */
typedef struct llmbridge_streamed_shape_v1 {
    uint64_t workspace_fixed_bytes, workspace_per_token_bytes, kv_bytes_per_token;
    uint64_t file_bytes, tensor_payload_bytes, metadata_prefix_bytes, training_context;
    uint64_t embedding, heads, kv_heads, layers, feed_forward, vocabulary, chunk_bytes;
} llmbridge_streamed_shape_v1;
int32_t llmbridge_inspect_streamed_plan_v1(uint64_t file_bytes, size_t chunk,
    llmbridge_range_callback reader, llmbridge_cancel_poll poll, void *userdata,
    llmbridge_streamed_shape_v1 *out, char *error, size_t capacity);

/* Parses GGUF metadata via caller-supplied range reads, no tensor weight allocation.
 * NOT an inference graph adapter and NOT a hard process RSS limit.
 * Returns 0 for success, -1 for failure. */
int32_t llmbridge_probe_gguf_source(uint64_t file_bytes,
                                    size_t max_chunk_bytes,
                                    llmbridge_range_callback reader,
                                    void *userdata,
                                    uint64_t *tensor_count,
                                    uint64_t *tensor_data_offset,
                                    char *error,
                                    size_t error_capacity);

/* Run actual native GGML Q8_0 dequantization + a matrix-vector product, one
 * row of a named GGUF tensor at a time. Reader is called synchronously and
 * only for bytes inside [tensor_offset, tensor_offset + tensor_bytes).
 * Shape must be [columns, rows], with columns a multiple of 32.
 * Native scratch memory is limited to O(columns); it never loads the matrix.
 * Returns 0 on success, -1 on read/validation error. A failed call's output
 * must be discarded. This is an operator, not full model execution. */
int32_t llmbridge_q8_0_matvec(uint64_t tensor_offset,
                               uint64_t tensor_bytes,
                               uint32_t columns,
                               uint32_t rows,
                               size_t max_chunk_bytes,
                               llmbridge_range_callback reader,
                               void *userdata,
                               const float *input,
                               float *output,
                               char *error,
                               size_t error_capacity);

/* Dequantize `rows` rows of `columns` elements each, of the given ggml tensor
 * type, from the in-memory encoded buffer `src` (src_len bytes) into `dst`
 * (dst_count floats = rows*columns). Uses the pinned ggml type-trait to_float
 * kernel, not a reimplementation. `columns` must be a whole number of the type's
 * blocks; `src_len` must equal the exact packed byte length. Output is rejected
 * if any value is non-finite. Returns 0 on success, -1 on validation/decode
 * error. This decodes weights to FP32; it is not model execution. */
int32_t llmbridge_dequantize_rows_v1(uint32_t ggml_type,
                                     uint32_t columns,
                                     uint32_t rows,
                                     const uint8_t *src,
                                     uint64_t src_len,
                                     float *dst,
                                     uint64_t dst_count,
                                     char *error,
                                     size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
