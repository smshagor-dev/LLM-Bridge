#include "llmbridge.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::mutex engine_mutex;
std::once_flag init_once;

void write_error(char *dst, size_t capacity, const char *message) noexcept {
    if (dst != nullptr && capacity != 0) {
        std::snprintf(dst, capacity, "%s", message);
    }
}

void set_tokens(llama_batch_ext *batch, const llama_token *tokens, int32_t count, int32_t position) {
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < count; ++i) {
        const auto index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        if (index < 0) {
            throw std::runtime_error("llama batch token admission failed");
        }
        const llama_pos p = position + i;
        llama_batch_ext_set_pos(batch, index, &p);
    }
    llama_batch_ext_set_output_logits(batch, count - 1, true);
}

int32_t generate(const char *path, const char *prompt, uint32_t context_tokens,
                 uint32_t max_tokens, int32_t threads, llmbridge_token_callback callback,
                 void *userdata, llmbridge_cancel_poll poll = nullptr) {
    if (!path || !prompt || !callback || !path[0] ||
        context_tokens < 64 || context_tokens > 8192 ||
        max_tokens == 0 || max_tokens > 2048 ||
        threads <= 0 || threads > 64 ||
        std::strlen(prompt) > 65536) {
        throw std::invalid_argument("invalid inference arguments or unsupported context/thread limits");
    }
    if (poll && poll(userdata) == 0) return 2;
    // Single model execution at a time while the baseline does not have a worker scheduler.
    std::unique_lock<std::mutex> lock(engine_mutex);
    if (poll && poll(userdata) == 0) return 2;
    std::call_once(init_once, [] { llama_backend_init(); });

    if (poll && poll(userdata) == 0) return 2;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path, mp), llama_model_free);
    if (!model) {
        // Return the process-wide inference gate before reporting the load error.
        // Keep later requests from waiting indefinitely if llama.cpp's failed
        // loader path leaves native state that prevents immediate reuse.
        lock.unlock();
        throw std::runtime_error("cannot load GGUF model; check format, architecture and available memory");
    }
    if (llama_model_has_encoder(model.get())) {
        throw std::runtime_error("M1 currently supports decoder-only models");
    }
    const auto *vocab = llama_model_get_vocab(model.get());
    const size_t prompt_bytes = std::strlen(prompt);
    if (prompt_bytes == 0 || prompt_bytes > static_cast<size_t>(INT32_MAX)) {
        throw std::invalid_argument("empty or oversized prompt");
    }
    int32_t token_count = llama_tokenize(vocab, prompt,
        static_cast<int32_t>(prompt_bytes), nullptr, 0, true, true);
    if (token_count >= 0 || token_count == INT32_MIN) {
        throw std::runtime_error("failed to determine prompt token count");
    }
    token_count = -token_count;
    if (token_count < 1 || static_cast<uint64_t>(token_count) + max_tokens > context_tokens) {
        throw std::invalid_argument("prompt and response exceed configured context; no silent truncation");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(token_count));
    if (llama_tokenize(vocab, prompt, static_cast<int32_t>(prompt_bytes),
                       tokens.data(), token_count, true, true) != token_count) {
        throw std::runtime_error("tokenization failed");
    }

    if (poll && poll(userdata) == 0) return 2;
    auto cp = llama_context_default_params();
    cp.n_ctx = context_tokens;
    cp.n_batch = static_cast<uint32_t>(token_count);
    cp.n_ubatch = std::min<uint32_t>(static_cast<uint32_t>(token_count), 512);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) {
        throw std::runtime_error("unable to initialize CPU context, possibly out of memory");
    }
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    if (!batch) {
        throw std::runtime_error("unable to initialize model batch");
    }
    auto chain_params = llama_sampler_chain_default_params();
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(chain_params), llama_sampler_free);
    if (!sampler) {
        throw std::runtime_error("unable to initialize greedy sampler");
    }
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

    int32_t position = 0;
    int32_t current_count = token_count;
    for (uint32_t generated = 0; generated < max_tokens; ++generated) {
        if (poll && poll(userdata) == 0) return 2;
        set_tokens(batch.get(), tokens.data(), current_count, position);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("CPU model decode failed");
        }
        position += current_count;
        const llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        if (token < 0) {
            throw std::runtime_error("sampler returned an invalid token");
        }
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        char small[256];
        int32_t length = llama_token_to_piece(vocab, token, small, sizeof(small), 0, true);
        std::vector<char> large;
        const char *piece = small;
        if (length < 0) {
            const auto needed = static_cast<size_t>(-static_cast<int64_t>(length));
            if (needed > 1024 * 1024) {
                throw std::runtime_error("token piece exceeds safety limit");
            }
            large.resize(needed);
            length = llama_token_to_piece(vocab, token, large.data(),
                                         static_cast<int32_t>(large.size()), 0, true);
            piece = large.data();
        }
        if (length < 0) {
            throw std::runtime_error("failed converting token to bytes");
        }
        if (callback(piece, static_cast<size_t>(length), userdata) == 0) {
            return 2;
        }
        tokens.assign(1, token);
        current_count = 1;
    }
    return 0;
}

// Opt-in diagnostic: decode the prompt and surface the logit vector at the
// final prompt position (the position whose argmax is the first generated
// token). Mirrors generate()'s load/tokenize/prefill exactly but samples
// nothing, so it is a faithful resident reference for streamed step-0 logits.
int32_t prompt_logits(const char *path, const char *prompt, uint32_t context_tokens,
                      int32_t threads, llmbridge_logits_callback emit_logits, void *userdata) {
    if (!path || !prompt || !emit_logits || !path[0] ||
        context_tokens < 64 || context_tokens > 8192 ||
        threads <= 0 || threads > 64 || std::strlen(prompt) > 65536) {
        throw std::invalid_argument("invalid resident logit-capture arguments");
    }
    std::unique_lock<std::mutex> lock(engine_mutex);
    std::call_once(init_once, [] { llama_backend_init(); });
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path, mp), llama_model_free);
    if (!model) {
        lock.unlock();
        throw std::runtime_error("cannot load GGUF model for logit capture");
    }
    if (llama_model_has_encoder(model.get())) {
        throw std::runtime_error("logit capture supports decoder-only models");
    }
    const auto *vocab = llama_model_get_vocab(model.get());
    const size_t prompt_bytes = std::strlen(prompt);
    if (prompt_bytes == 0) {
        throw std::invalid_argument("empty prompt");
    }
    int32_t token_count = llama_tokenize(vocab, prompt,
        static_cast<int32_t>(prompt_bytes), nullptr, 0, true, true);
    if (token_count >= 0 || token_count == INT32_MIN) {
        throw std::runtime_error("failed to determine prompt token count");
    }
    token_count = -token_count;
    if (token_count < 1 || static_cast<uint64_t>(token_count) >= context_tokens) {
        throw std::invalid_argument("prompt exceeds configured context");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(token_count));
    if (llama_tokenize(vocab, prompt, static_cast<int32_t>(prompt_bytes),
                       tokens.data(), token_count, true, true) != token_count) {
        throw std::runtime_error("tokenization failed");
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = context_tokens;
    cp.n_batch = static_cast<uint32_t>(token_count);
    cp.n_ubatch = std::min<uint32_t>(static_cast<uint32_t>(token_count), 512);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) {
        throw std::runtime_error("unable to initialize CPU context for logit capture");
    }
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    if (!batch) {
        throw std::runtime_error("unable to initialize model batch");
    }
    set_tokens(batch.get(), tokens.data(), token_count, 0);
    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        throw std::runtime_error("CPU model decode failed");
    }
    const float *logits = llama_get_logits_ith(ctx.get(), -1);
    if (!logits) {
        throw std::runtime_error("logits unavailable at final prompt position");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        throw std::runtime_error("invalid vocabulary size");
    }
    // Raw logits are reported unmodified; non-finite values are the caller's
    // diagnostic signal and must not be hidden here.
    (void)emit_logits(logits, static_cast<size_t>(n_vocab), userdata);
    return 0;
}

// Opt-in multi-position diagnostic: greedy-decode up to max_tokens and emit the
// full logit vector at EACH step before sampling (same timing as the streamed
// decoder's per-step logits callback). The chosen token is the argmax of the
// emitted vector, so the caller can reconstruct the generated sequence and
// confirm an identical prefix before comparing a given position.
int32_t decode_logits(const char *path, const char *prompt, uint32_t context_tokens,
                      uint32_t max_tokens, int32_t threads,
                      llmbridge_logits_callback emit_logits, void *userdata) {
    if (!path || !prompt || !emit_logits || !path[0] ||
        context_tokens < 64 || context_tokens > 8192 ||
        max_tokens == 0 || max_tokens > 2048 ||
        threads <= 0 || threads > 64 || std::strlen(prompt) > 65536) {
        throw std::invalid_argument("invalid resident decode-logits arguments");
    }
    std::unique_lock<std::mutex> lock(engine_mutex);
    std::call_once(init_once, [] { llama_backend_init(); });
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path, mp), llama_model_free);
    if (!model) {
        lock.unlock();
        throw std::runtime_error("cannot load GGUF model for decode-logits");
    }
    if (llama_model_has_encoder(model.get())) {
        throw std::runtime_error("decode-logits supports decoder-only models");
    }
    const auto *vocab = llama_model_get_vocab(model.get());
    const size_t prompt_bytes = std::strlen(prompt);
    if (prompt_bytes == 0) {
        throw std::invalid_argument("empty prompt");
    }
    int32_t token_count = llama_tokenize(vocab, prompt,
        static_cast<int32_t>(prompt_bytes), nullptr, 0, true, true);
    if (token_count >= 0 || token_count == INT32_MIN) {
        throw std::runtime_error("failed to determine prompt token count");
    }
    token_count = -token_count;
    if (token_count < 1 ||
        static_cast<uint64_t>(token_count) + max_tokens > context_tokens) {
        throw std::invalid_argument("prompt and response exceed configured context");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(token_count));
    if (llama_tokenize(vocab, prompt, static_cast<int32_t>(prompt_bytes),
                       tokens.data(), token_count, true, true) != token_count) {
        throw std::runtime_error("tokenization failed");
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = context_tokens;
    cp.n_batch = static_cast<uint32_t>(token_count);
    cp.n_ubatch = std::min<uint32_t>(static_cast<uint32_t>(token_count), 512);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) {
        throw std::runtime_error("unable to initialize CPU context for decode-logits");
    }
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    if (!batch) {
        throw std::runtime_error("unable to initialize model batch");
    }
    auto chain_params = llama_sampler_chain_default_params();
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(chain_params), llama_sampler_free);
    if (!sampler) {
        throw std::runtime_error("unable to initialize greedy sampler");
    }
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        throw std::runtime_error("invalid vocabulary size");
    }
    int32_t position = 0;
    int32_t current_count = token_count;
    for (uint32_t generated = 0; generated < max_tokens; ++generated) {
        set_tokens(batch.get(), tokens.data(), current_count, position);
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("CPU model decode failed");
        }
        position += current_count;
        const float *logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits) {
            throw std::runtime_error("logits unavailable during decode");
        }
        if (!emit_logits(logits, static_cast<size_t>(n_vocab), userdata)) {
            return 2;
        }
        const llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        if (token < 0) {
            throw std::runtime_error("sampler returned an invalid token");
        }
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        tokens.assign(1, token);
        current_count = 1;
    }
    return 0;
}
} // namespace

extern "C" uint32_t llmbridge_abi_version(void) {
    return LLMBRIDGE_NATIVE_ABI_VERSION;
}

extern "C" int32_t llmbridge_generate_cpu(
    const char *model_path, const char *prompt, uint32_t context_tokens,
    uint32_t max_tokens, int32_t threads, llmbridge_token_callback callback,
    void *userdata, char *error, size_t error_capacity) {
    write_error(error, error_capacity, "");
    try {
        return generate(model_path, prompt, context_tokens, max_tokens,
                        threads, callback, userdata);
    } catch (const std::exception &e) {
        write_error(error, error_capacity, e.what());
        return -1;
    } catch (...) {
        write_error(error, error_capacity, "unknown native exception");
        return -1;
    }
}


extern "C" int32_t llmbridge_resident_prompt_logits_v1(
    const char *path, const char *prompt, uint32_t context_tokens, int32_t threads,
    llmbridge_logits_callback emit_logits, void *userdata, char *error, size_t capacity) {
    write_error(error, capacity, "");
    try {
        return prompt_logits(path, prompt, context_tokens, threads, emit_logits, userdata);
    } catch (const std::exception &e) {
        write_error(error, capacity, e.what());
        return -1;
    } catch (...) {
        write_error(error, capacity, "unknown resident logit-capture exception");
        return -1;
    }
}

extern "C" int32_t llmbridge_resident_decode_logits_v1(
    const char *path, const char *prompt, uint32_t context_tokens, uint32_t max_tokens,
    int32_t threads, llmbridge_logits_callback emit_logits, void *userdata,
    char *error, size_t capacity) {
    write_error(error, capacity, "");
    try {
        return decode_logits(path, prompt, context_tokens, max_tokens, threads,
                             emit_logits, userdata);
    } catch (const std::exception &e) {
        write_error(error, capacity, e.what());
        return -1;
    } catch (...) {
        write_error(error, capacity, "unknown resident decode-logits exception");
        return -1;
    }
}

extern "C" int32_t llmbridge_probe_gguf_source(
    uint64_t file_bytes, size_t max_chunk_bytes,
    llmbridge_range_callback reader, void *userdata,
    uint64_t *tensor_count, uint64_t *tensor_data_offset,
    char *error, size_t error_capacity) {
    write_error(error, error_capacity, "");
    if (tensor_count != nullptr) {
        *tensor_count = 0;
    }
    if (tensor_data_offset != nullptr) {
        *tensor_data_offset = 0;
    }
    try {
        if (file_bytes < 24 || max_chunk_bytes == 0 || max_chunk_bytes > 1024 * 1024 ||
            !reader || !userdata || !tensor_count || !tensor_data_offset) {
            throw std::invalid_argument("invalid bounded GGUF source probe arguments");
        }
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = nullptr;
        // The underlying GGUF loader performs bounded calls to the supplied
        // reader, and no model weight tensors are allocated by this probe.
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_callback(reader, userdata, max_chunk_bytes, file_bytes, params),
            gguf_free);
        if (!metadata) {
            throw std::runtime_error("native GGUF metadata parse failed");
        }
        const int64_t count = gguf_get_n_tensors(metadata.get());
        if (count < 0) {
            throw std::runtime_error("native GGUF tensor count invalid");
        }
        *tensor_count = static_cast<uint64_t>(count);
        *tensor_data_offset = static_cast<uint64_t>(gguf_get_data_offset(metadata.get()));
        return 0;
    } catch (const std::exception &e) {
        write_error(error, error_capacity, e.what());
        return -1;
    } catch (...) {
        write_error(error, error_capacity, "unknown native GGUF probe exception");
        return -1;
    }
}

/* The input and output buffers are borrowed for the lifetime of this call.
 * Weights are read one Q8_0 row at a time, dequantized with the pinned ggml
 * type trait, and multiplied against the provided activation vector.
 * Failure has no committed output: the Rust wrapper discards it. */
extern "C" int32_t llmbridge_q8_0_matvec(
    uint64_t tensor_offset, uint64_t tensor_bytes,
    uint32_t columns, uint32_t rows, size_t max_chunk_bytes,
    llmbridge_range_callback reader, void *userdata,
    const float *input, float *output,
    char *error, size_t error_capacity) {
    write_error(error, error_capacity, "");
    try {
        if (!reader || !userdata || !input || !output ||
            columns == 0 || columns > 16384 || (columns % 32) != 0 ||
            rows == 0 || rows > 65536 ||
            max_chunk_bytes == 0 || max_chunk_bytes > 1024 * 1024) {
            throw std::invalid_argument("invalid Q8_0 streamed matvec arguments");
        }
        const size_t row_bytes = static_cast<size_t>(columns / 32) * 34;
        if (row_bytes > max_chunk_bytes ||
            tensor_bytes != static_cast<uint64_t>(row_bytes) * rows ||
            tensor_offset > UINT64_MAX - tensor_bytes) {
            throw std::invalid_argument("GGUF Q8_0 shape, byte range or chunk budget mismatch");
        }
        for (uint32_t col = 0; col < columns; ++col) {
            if (!std::isfinite(input[col])) {
                throw std::invalid_argument("non-finite activation rejected");
            }
        }
        const auto *traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        if (!traits || !traits->to_float) {
            throw std::runtime_error("pinned ggml Q8_0 dequantizer unavailable");
        }
        std::vector<uint8_t> packed(row_bytes);
        std::vector<float> decoded(columns);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint64_t offset = tensor_offset + static_cast<uint64_t>(row) * row_bytes;
            if (reader(userdata, packed.data(), offset, row_bytes) != row_bytes) {
                throw std::runtime_error("bounded GGUF range read failed or was cancelled");
            }
            traits->to_float(packed.data(), decoded.data(), columns);
            double sum = 0.0;
            for (uint32_t col = 0; col < columns; ++col) {
                sum += static_cast<double>(decoded[col]) * input[col];
            }
            if (!std::isfinite(sum) || std::abs(sum) > std::numeric_limits<float>::max()) {
                throw std::runtime_error("non-finite Q8_0 output rejected");
            }
            output[row] = static_cast<float>(sum);
        }
        return 0;
    } catch (const std::exception &e) {
        write_error(error, error_capacity, e.what());
        return -1;
    } catch (...) {
        write_error(error, error_capacity, "unknown native tensor operator exception");
        return -1;
    }
}

/* Dequantize whole rows of an arbitrary ggml tensor type with the pinned ggml
 * to_float trait (the same reference kernel llama.cpp uses). The caller owns the
 * encoded `src` bytes and the `dst` float buffer; both are borrowed only for the
 * call. Geometry is validated against the pinned ggml block size / row size, so a
 * wrong length or a type without a dequantizer is rejected rather than guessed.
 * No weights are retained and nothing beyond `src` is read. */
extern "C" int32_t llmbridge_dequantize_rows_v1(
    uint32_t ggml_type, uint32_t columns, uint32_t rows,
    const uint8_t *src, uint64_t src_len,
    float *dst, uint64_t dst_count,
    char *error, size_t error_capacity) {
    write_error(error, error_capacity, "");
    try {
        if (!src || !dst || columns == 0 || columns > (1u << 20) ||
            rows == 0 || rows > (1u << 20)) {
            throw std::invalid_argument("invalid dequantize_rows arguments");
        }
        // ggml_get_type_traits asserts (aborts) on an out-of-range id, so the
        // enum bound is checked here before any ggml call can be reached.
        if (ggml_type >= static_cast<uint32_t>(GGML_TYPE_COUNT)) {
            throw std::invalid_argument("ggml tensor type id out of range");
        }
        const enum ggml_type type = static_cast<enum ggml_type>(ggml_type);
        const auto *traits = ggml_get_type_traits(type);
        if (!traits || !traits->to_float) {
            throw std::runtime_error("pinned ggml type has no dequantizer");
        }
        const int64_t block = ggml_blck_size(type);
        if (block <= 0 || (columns % static_cast<uint32_t>(block)) != 0) {
            throw std::invalid_argument("columns is not a whole number of ggml blocks");
        }
        const size_t row_bytes = static_cast<size_t>(ggml_row_size(type, columns));
        // Exact packed length and output count, with overflow-checked products.
        if (row_bytes == 0 ||
            src_len != static_cast<uint64_t>(row_bytes) * rows ||
            dst_count != static_cast<uint64_t>(columns) * rows) {
            throw std::invalid_argument("src_len or dst_count disagrees with ggml geometry");
        }
        for (uint32_t row = 0; row < rows; ++row) {
            const uint8_t *row_src = src + static_cast<size_t>(row) * row_bytes;
            float *row_dst = dst + static_cast<size_t>(row) * columns;
            traits->to_float(row_src, row_dst, static_cast<int64_t>(columns));
            for (uint32_t col = 0; col < columns; ++col) {
                if (!std::isfinite(row_dst[col])) {
                    throw std::runtime_error("non-finite dequantized value rejected");
                }
            }
        }
        return 0;
    } catch (const std::exception &e) {
        write_error(error, error_capacity, e.what());
        return -1;
    } catch (...) {
        write_error(error, error_capacity, "unknown native dequantize exception");
        return -1;
    }
}

/* Additive entry point preserves the existing ABI and legacy callers. */
extern "C" int32_t llmbridge_generate_cpu_polled(
    const char *model_path, const char *prompt, uint32_t context_tokens,
    uint32_t max_tokens, int32_t threads, llmbridge_token_callback callback,
    llmbridge_cancel_poll poll, void *userdata, char *error, size_t error_capacity) {
    write_error(error, error_capacity, "");
    try {
        if (!poll) throw std::invalid_argument("cancellation poll callback required");
        return generate(model_path, prompt, context_tokens, max_tokens,
                        threads, callback, userdata, poll);
    } catch (const std::exception &e) {
        write_error(error, error_capacity, e.what());
        return -1;
    } catch (...) {
        write_error(error, error_capacity, "unknown native exception");
        return -1;
    }
}
