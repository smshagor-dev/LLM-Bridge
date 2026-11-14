#include "llmbridge.h"
#include "llama.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Cancelled {};
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
size_t checked_mul(size_t a, size_t b) {
    require(b == 0 || a <= SIZE_MAX / b, "streamed workspace arithmetic overflow");
    return a * b;
}
uint32_t number(const gguf_context *g, const char *key) {
    auto i = gguf_find_key(g, key);
    require(i >= 0 && gguf_get_kv_type(g, i) == GGUF_TYPE_UINT32, "missing/invalid streamed model integer metadata");
    return gguf_get_val_u32(g, i);
}
float real(const gguf_context *g, const char *key, float fallback) {
    auto i = gguf_find_key(g, key);
    if (i < 0) return fallback;
    require(gguf_get_kv_type(g, i) == GGUF_TYPE_FLOAT32, "invalid streamed model float metadata");
    auto v = gguf_get_val_f32(g, i);
    require(std::isfinite(v) && v > 0, "invalid streamed model float value");
    return v;
}
struct Tensor { uint64_t offset; size_t columns, rows, row_bytes; ggml_type type; };
struct Layer { Tensor an, q, k, v, o, fn, gate, up, down; };

// The vocabulary-only upstream loader allocates tokenizer metadata, never weights.
// Every weight row used by this decoder arrives through the caller's range reader.
struct Decoder {
    const gguf_context *g;
    uint64_t file_bytes;
    size_t chunk;
    llmbridge_range_callback reader;
    llmbridge_cancel_poll poll;
    void *userdata;
    size_t dim, hidden, heads, kvheads, head_dim, kvdim, layers, context;
    float epsilon, rope_base;
    Tensor embedding, final_norm, output;
    std::vector<Layer> blocks;
    std::vector<uint8_t> packed;
    std::vector<float> row, quantized, x, norm, q, k, v, att, projected, gate, up, logits, keys, values, scores;

    void alive() { if (poll(userdata) == 0) throw Cancelled{}; }
    Tensor tensor(const std::string &name, size_t columns, size_t rows) {
        auto id = gguf_find_tensor(g, name.c_str());
        require(id >= 0, "required streamed tensor missing");
        auto ne = gguf_get_tensor_ne(g, id);
        auto type = gguf_get_tensor_type(g, id);
        require((type == GGML_TYPE_F32 || type == GGML_TYPE_Q8_0) &&
                ne[0] == static_cast<int64_t>(columns) && ne[1] == static_cast<int64_t>(rows) &&
                ne[2] == 1 && ne[3] == 1, "unsupported streamed tensor type/shape");
        require(type != GGML_TYPE_Q8_0 || columns % 32 == 0, "invalid Q8_0 block geometry");
        size_t rb = type == GGML_TYPE_F32 ? checked_mul(columns, 4) : checked_mul(columns / 32, 34);
        uint64_t base = gguf_get_data_offset(g), relative = gguf_get_tensor_offset(g, id);
        require(relative <= UINT64_MAX - base, "tensor offset overflow");
        uint64_t offset = base + relative;
        auto bytes = checked_mul(rb, rows);
        require(gguf_get_tensor_size(g, id) == bytes && offset <= file_bytes && bytes <= file_bytes - offset,
                "streamed tensor outside model file");
        require(rb <= chunk, "one tensor row exceeds bounded reader chunk");
        return {offset, columns, rows, rb, type};
    }
    Decoder(const gguf_context *metadata, uint64_t size, size_t max_chunk,
            llmbridge_range_callback source, llmbridge_cancel_poll cancel, void *state,
            size_t ctx, size_t vocab, uint64_t budget)
        : g(metadata), file_bytes(size), chunk(max_chunk), reader(source), poll(cancel), userdata(state), context(ctx) {
        auto architecture = gguf_find_key(g, "general.architecture");
        require(architecture >= 0 && gguf_get_kv_type(g, architecture) == GGUF_TYPE_STRING &&
                std::string(gguf_get_val_str(g, architecture)) == "llama", "streamed backend supports llama only");
        // Reject unimplemented architecture options rather than silently approximate them.
        const std::set<std::string> known = {"llama.context_length", "llama.embedding_length", "llama.block_count",
            "llama.feed_forward_length", "llama.rope.dimension_count", "llama.rope.freq_base",
            "llama.attention.head_count", "llama.attention.head_count_kv", "llama.attention.layer_norm_rms_epsilon"};
        for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
            std::string key = gguf_get_key(g, i);
            require(key.rfind("llama.", 0) != 0 || known.count(key), "unsupported llama metadata option in streamed backend");
        }
        dim = number(g, "llama.embedding_length"); hidden = number(g, "llama.feed_forward_length");
        heads = number(g, "llama.attention.head_count"); layers = number(g, "llama.block_count");
        auto kh = gguf_find_key(g, "llama.attention.head_count_kv");
        kvheads = kh < 0 ? heads : number(g, "llama.attention.head_count_kv");
        require(dim && dim <= 8192 && hidden && hidden <= 32768 && heads && kvheads && heads % kvheads == 0 &&
                dim % heads == 0 && layers && layers <= 128 && vocab && vocab <= 262144 &&
                context <= number(g, "llama.context_length"), "streamed model configuration exceeds supported limits");
        head_dim = dim / heads; kvdim = head_dim * kvheads;
        require(head_dim % 2 == 0 && number(g, "llama.rope.dimension_count") == head_dim,
                "streamed backend requires full-head normal RoPE");
        epsilon = real(g, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        rope_base = real(g, "llama.rope.freq_base", 10000.f);
        // Fixed workspace: O(layers * context * KV width), no weight-matrix allocation.
        auto kvcount = checked_mul(checked_mul(layers, context), kvdim);
        auto floats = checked_mul(kvcount, 2) + checked_mul(dim, 7) + checked_mul(kvdim, 2) +
                      checked_mul(hidden, 2) + vocab + context + 2 * std::max(dim, hidden);
        auto bytes = checked_mul(floats, sizeof(float));
        require(bytes <= budget && chunk <= budget - bytes && checked_mul(layers, sizeof(Layer)) <= budget - bytes - chunk,
                "streamed KV/activation workspace exceeds budget");
        embedding = tensor("token_embd.weight", dim, vocab);
        final_norm = tensor("output_norm.weight", dim, 1);
        auto out_id = gguf_find_tensor(g, "output.weight");
        output = out_id < 0 ? embedding : tensor("output.weight", dim, vocab);
        for (size_t l = 0; l < layers; ++l) {
            std::string p = "blk." + std::to_string(l) + ".";
            blocks.push_back({tensor(p+"attn_norm.weight", dim, 1), tensor(p+"attn_q.weight", dim, dim),
                tensor(p+"attn_k.weight", dim, kvdim), tensor(p+"attn_v.weight", dim, kvdim),
                tensor(p+"attn_output.weight", dim, dim), tensor(p+"ffn_norm.weight", dim, 1),
                tensor(p+"ffn_gate.weight", dim, hidden), tensor(p+"ffn_up.weight", dim, hidden),
                tensor(p+"ffn_down.weight", hidden, dim)});
        }
        // Refuse unrecognized tensors (biases, special RoPE, adapters, etc.).
        require(gguf_get_n_tensors(g) == static_cast<int64_t>(layers * 9 + (out_id < 0 ? 2 : 3)),
                "unsupported extra tensors in streamed model");
        packed.resize(chunk); row.resize(std::max(dim, hidden)); quantized.resize(std::max(dim, hidden));
        x.resize(dim); norm.resize(dim); q.resize(dim); k.resize(kvdim); v.resize(kvdim);
        att.resize(dim); projected.resize(dim); gate.resize(hidden); up.resize(hidden);
        logits.resize(vocab); keys.resize(kvcount); values.resize(kvcount); scores.resize(context);
    }
    void read_row(const Tensor &t, size_t index, float *destination) {
        alive();
        require(index < t.rows, "streamed row index outside tensor");
        require(reader(userdata, packed.data(), t.offset + index * t.row_bytes, t.row_bytes) == t.row_bytes,
                "streamed tensor read failed");
        alive();
        if (t.type == GGML_TYPE_F32) std::memcpy(destination, packed.data(), t.row_bytes);
        else ggml_get_type_traits(t.type)->to_float(packed.data(), destination, t.columns);
        for (size_t i = 0; i < t.columns; ++i) require(std::isfinite(destination[i]), "nonfinite streamed weight");
    }
    void matvec(const Tensor &t, const std::vector<float> &input, std::vector<float> &result) {
        require(input.size() == t.columns && result.size() == t.rows, "internal streamed activation shape mismatch");
        for (auto value : input) require(std::isfinite(value), "nonfinite streamed activation");
        auto traits = ggml_get_type_traits_cpu(t.type);
        require(traits && traits->vec_dot, "streamed CPU dot kernel unavailable");
        const void * activation = input.data();
        if (t.type != GGML_TYPE_F32) {
            auto activation_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
            require(activation_traits && activation_traits->from_float, "streamed activation quantizer unavailable");
            require(ggml_row_size(traits->vec_dot_type, t.columns) <= quantized.size() * sizeof(float),
                    "quantized activation exceeds scratch");
            activation_traits->from_float(input.data(), quantized.data(), t.columns);
            activation = quantized.data();
        }
        for (size_t r = 0; r < t.rows; ++r) {
            alive();
            require(reader(userdata, packed.data(), t.offset + r * t.row_bytes, t.row_bytes) == t.row_bytes,
                    "streamed matrix row read failed");
            alive();
            float sum = 0;
            traits->vec_dot(static_cast<int>(t.columns), &sum, 0, packed.data(), 0, activation, 0, 1);
            require(std::isfinite(sum), "streamed matvec overflow/nonfinite weight");
            result[r] = sum;
        }
    }

    void rms(const Tensor &weight) {
        read_row(weight, 0, row.data());
        double sum = 0;
        for (auto z : x) { require(std::isfinite(z), "nonfinite RMS input"); sum += static_cast<double>(z) * z; }
        float scale = 1.f / std::sqrt(static_cast<float>(sum / dim) + epsilon);
        for (size_t i = 0; i < dim; ++i) { norm[i] = x[i] * scale * row[i]; require(std::isfinite(norm[i]), "RMS overflow"); }
    }
    void rope(std::vector<float> &vector, size_t count, size_t pos) {
        for (size_t h = 0; h < count; ++h) for (size_t i = 0; i < head_dim; i += 2) {
            float angle = static_cast<float>(pos) / std::pow(rope_base, static_cast<float>(i) / head_dim);
            float c = std::cos(angle), s = std::sin(angle);
            size_t a = h * head_dim + i;
            float first = vector[a], second = vector[a+1];
            vector[a] = first*c - second*s; vector[a+1] = first*s + second*c;
        }
    }
    void forward(llama_token token, size_t pos, bool need_logits) {
        require(pos < context && token >= 0 && static_cast<size_t>(token) < embedding.rows, "streamed token/context out of range");
        read_row(embedding, token, x.data());
        for (size_t l = 0; l < layers; ++l) {
            auto &b = blocks[l];
            rms(b.an); matvec(b.q, norm, q); matvec(b.k, norm, k); matvec(b.v, norm, v);
            rope(q, heads, pos); rope(k, kvheads, pos);
            auto base = (l * context + pos) * kvdim;
            std::copy(k.begin(), k.end(), keys.begin() + base);
            std::copy(v.begin(), v.end(), values.begin() + base);
            for (size_t h = 0; h < heads; ++h) {
                auto kh = h / (heads / kvheads);
                float maximum = -std::numeric_limits<float>::infinity();
                for (size_t t = 0; t <= pos; ++t) {
                    double sum = 0;
                    auto ki = (l * context + t) * kvdim + kh * head_dim;
                    for (size_t i = 0; i < head_dim; ++i) sum += static_cast<double>(q[h*head_dim+i]) * keys[ki+i];
                    scores[t] = static_cast<float>(sum / std::sqrt(static_cast<double>(head_dim)));
                    require(std::isfinite(scores[t]), "attention score overflow");
                    maximum = std::max(maximum, scores[t]);
                }
                double denominator = 0;
                for (size_t t = 0; t <= pos; ++t) { scores[t] = std::exp(scores[t] - maximum); denominator += scores[t]; }
                require(std::isfinite(denominator) && denominator > 0, "streamed attention softmax overflow");
                for (size_t i = 0; i < head_dim; ++i) {
                    double sum = 0;
                    for (size_t t = 0; t <= pos; ++t) {
                        auto vi = (l * context + t) * kvdim + kh * head_dim + i;
                        sum += (scores[t] / denominator) * values[vi];
                    }
                    att[h*head_dim+i] = static_cast<float>(sum);
                }
            }
            matvec(b.o, att, projected);
            for (size_t i = 0; i < dim; ++i) x[i] += projected[i];
            rms(b.fn); matvec(b.gate, norm, gate); matvec(b.up, norm, up);
            for (size_t i = 0; i < hidden; ++i) gate[i] = (gate[i] / (1.f + std::exp(-gate[i]))) * up[i];
            matvec(b.down, gate, projected);
            for (size_t i = 0; i < dim; ++i) { x[i] += projected[i]; require(std::isfinite(x[i]), "streamed residual overflow"); }
        }
        if (need_logits) { rms(final_norm); matvec(output, norm, logits); }
    }
};
} // namespace

extern "C" int32_t llmbridge_generate_streamed_cpu_v2(
    const char *path, const char *prompt, uint32_t context, uint32_t max_tokens,
    uint64_t file_bytes, size_t max_chunk, uint64_t workspace_budget, uint64_t tokenizer_prefix,
    llmbridge_range_callback reader, llmbridge_token_callback emit,
    llmbridge_logits_callback emit_logits, llmbridge_cancel_poll poll, void *userdata,
    char *error, size_t capacity) {
    if (error && capacity) error[0] = '\0';
    try {
        require(std::endian::native == std::endian::little, "streamed backend requires little endian");
        require(path && prompt && reader && emit && poll && userdata && context >= 64 && context <= 8192 &&
                max_tokens && max_tokens <= 2048 && max_chunk >= 34 && max_chunk <= 1024*1024 &&
                workspace_budget && workspace_budget <= 1024ull*1024*1024 && std::strlen(prompt) <= 65536,
                "invalid streamed generation arguments");
        if (!poll(userdata)) return 2;
        gguf_init_params gp{}; gp.no_alloc = true; gp.ctx = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_callback(reader, userdata, max_chunk, file_bytes, gp), gguf_free);
        require(static_cast<bool>(metadata), "streamed GGUF parse failed");
        if (tokenizer_prefix) require(gguf_get_data_offset(metadata.get()) == tokenizer_prefix, "authenticated tokenizer metadata prefix mismatch");
        if (!poll(userdata)) return 2;
        auto mp = llama_model_default_params(); mp.vocab_only = true; mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> vocabulary(llama_model_load_from_file(path, mp), llama_model_free);
        require(static_cast<bool>(vocabulary), "streamed tokenizer load failed");
        auto vocab = llama_model_get_vocab(vocabulary.get());
        int count = llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)), nullptr, 0, true, true);
        require(count < 0 && count != INT32_MIN, "streamed prompt tokenization failed"); count = -count;
        require(static_cast<uint64_t>(count) + max_tokens <= context, "streamed prompt and response exceed context");
        std::vector<llama_token> tokens(count);
        require(llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)), tokens.data(), count, true, true) == count,
                "streamed prompt tokenization changed");
        Decoder decoder(metadata.get(), file_bytes, max_chunk, reader, poll, userdata, context,
                        llama_vocab_n_tokens(vocab), workspace_budget);
        for (size_t i = 0; i < tokens.size(); ++i) decoder.forward(tokens[i], i, i+1 == tokens.size());
        size_t position = tokens.size();
        for (uint32_t step = 0; step < max_tokens; ++step) {
            decoder.alive();
            if (emit_logits && !emit_logits(decoder.logits.data(), decoder.logits.size(), userdata)) return 2;
            auto token = static_cast<llama_token>(std::max_element(decoder.logits.begin(), decoder.logits.end()) - decoder.logits.begin());
            if (llama_vocab_is_eog(vocab, token)) break;
            char piece[4096]; int length = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
            require(length >= 0, "streamed token piece exceeds supported limit");
            decoder.alive();
            if (!emit(piece, static_cast<size_t>(length), userdata)) return 2;
            if (step + 1 < max_tokens) decoder.forward(token, position++, true);
        }
        return 0;
    } catch (const Cancelled &) { return 2;
    } catch (const std::exception &e) { if (error && capacity) std::snprintf(error, capacity, "%s", e.what()); return -1;
    } catch (...) { if (error && capacity) std::snprintf(error, capacity, "unknown streamed native exception"); return -1; }
}

// Preserve additive ABI-v1 callers; v2 optionally binds a sealed vocabulary prefix.
extern "C" int32_t llmbridge_generate_streamed_cpu(
    const char *path, const char *prompt, uint32_t context, uint32_t max_tokens,
    uint64_t file_bytes, size_t max_chunk, uint64_t workspace_budget,
    llmbridge_range_callback reader, llmbridge_token_callback emit,
    llmbridge_logits_callback logits, llmbridge_cancel_poll poll, void *userdata,
    char *error, size_t capacity) {
    return llmbridge_generate_streamed_cpu_v2(path, prompt, context, max_tokens, file_bytes,
        max_chunk, workspace_budget, 0, reader, emit, logits, poll, userdata, error, capacity);
}
