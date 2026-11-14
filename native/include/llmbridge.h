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

/* Experimental full dense Llama decoder. All weights use bounded range reads;
 * tokenizer-only loading uses path. Workspace budget excludes tokenizer metadata,
 * parser, allocator and OS pages: it is not a hard RSS cap. F32/Q8_0 only.
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

#ifdef __cplusplus
}
#endif
#endif
