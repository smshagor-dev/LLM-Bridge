#include "llmbridge.h"
#include "llama.h"

#include <algorithm>
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
                 void *userdata) {
    if (!path || !prompt || !callback || !path[0] ||
        context_tokens < 64 || context_tokens > 8192 ||
        max_tokens == 0 || max_tokens > 2048 ||
        threads <= 0 || threads > 64 ||
        std::strlen(prompt) > 65536) {
        throw std::invalid_argument("invalid inference arguments or unsupported context/thread limits");
    }
    // Single model execution at a time while the baseline does not have a worker scheduler.
    std::lock_guard<std::mutex> lock(engine_mutex);
    std::call_once(init_once, [] { llama_backend_init(); });

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path, mp), llama_model_free);
    if (!model) {
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
