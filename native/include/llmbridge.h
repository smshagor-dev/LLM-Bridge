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
uint32_t llmbridge_abi_version(void);
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
#ifdef __cplusplus
}
#endif
#endif
