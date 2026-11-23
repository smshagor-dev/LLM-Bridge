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

// Pinned ggml CPU primitive; keep this signature tied to the locked revision.
extern "C" void ggml_vec_dot_f32(int n, float *out, size_t bs, const float *x, size_t bx, const float *y, size_t by, int nrc);
extern "C" double ggml_vec_soft_max_f32(int n, float *out, const float *x, float maximum);
extern "C" void ggml_vec_swiglu_f32(int n, float *y, const float *x, const float *gate);

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
// qb/kb/vb are the optional Q/K/V projection biases (qwen2); they stay
// default-initialized and unread for architectures without QKV bias.
struct Layer { Tensor an, q, k, v, o, fn, gate, up, down; Tensor qb{}, kb{}, vb{}; };

// Tensor element types the streamed decoder admits. Dequantization (to_float)
// and the dot kernel (vec_dot through the type's activation path) are always the
// pinned ggml reference traits; this decoder adds no quantization arithmetic.
// K-quants (256-element superblocks) require the block-size-aware geometry check
// in Decoder::tensor; 32-element quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) share it.
bool streamed_supported_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// Typed architecture descriptor: replaces scattered "llama"-only string checks.
// prefix selects the metadata key namespace; the flags select the two
// architecture-specific behaviors this decoder now implements.
struct ArchSpec {
    std::string prefix;  // metadata key namespace, e.g. "llama" or "qwen2"
    bool neox_rope;      // NeoX half-split RoPE (qwen2) vs normal consecutive-pair (llama)
    bool qkv_bias;       // separate Q/K/V projection biases present and required
};
ArchSpec resolve_arch(const std::string &name) {
    if (name == "llama") return {"llama", false, false};
    if (name == "qwen2") return {"qwen2", true, true};
    throw std::runtime_error("streamed backend supports llama and qwen2 architectures only");
}

// The vocabulary-only upstream loader allocates tokenizer metadata, never weights.
// Every weight row used by this decoder arrives through the caller's range reader.
struct Decoder {
    const gguf_context *g;
    uint64_t file_bytes;
    size_t chunk;
    llmbridge_range_callback reader;
    llmbridge_lookahead_schedule_callback schedule;
    llmbridge_cancel_poll poll;
    void *userdata;
    size_t dim, hidden, heads, kvheads, head_dim, kvdim, layers, context, max_context;
    float epsilon, rope_base;
    bool neox_rope = false, has_qkv_bias = false, has_rope_freqs = false;
    size_t workspace_required, kv_bytes;
    Tensor embedding, final_norm, output, rope_freqs{};
    std::vector<Layer> blocks;
    std::vector<uint8_t> packed;
    std::vector<float> row, quantized, x, norm, q, k, v, att, projected, gate, up, logits, keys, values, scores, attention_values;
    std::vector<float> qbias, kbias, vbias, freq_factors;

    void alive() { if (poll(userdata) == 0) throw Cancelled{}; }
    Tensor tensor(const std::string &name, size_t columns, size_t rows) {
        auto id = gguf_find_tensor(g, name.c_str());
        require(id >= 0, "required streamed tensor missing");
        auto ne = gguf_get_tensor_ne(g, id);
        auto type = gguf_get_tensor_type(g, id);
        require(streamed_supported_type(type) &&
                ne[0] == static_cast<int64_t>(columns) && ne[1] == static_cast<int64_t>(rows) &&
                ne[2] == 1 && ne[3] == 1, "unsupported streamed tensor type/shape");
        // Block-size-aware geometry: F32 blck=1, 32-element quants blck=32,
        // K-quant superblocks blck=256. The pinned ggml traits own the layout,
        // so one check covers every admitted type with no hardcoded block size.
        const int64_t block = ggml_blck_size(type);
        require(block > 0 && columns % static_cast<size_t>(block) == 0,
                "invalid streamed quantized block geometry");
        // Keep packed row bytes consistent with the pinned ggml dot/type traits.
        size_t rb = static_cast<size_t>(ggml_row_size(type, columns));
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
            size_t ctx, size_t vocab, uint64_t budget, bool allocate_state = true,
            llmbridge_lookahead_schedule_callback lookahead = nullptr)
        : g(metadata), file_bytes(size), chunk(max_chunk), reader(source), schedule(lookahead), poll(cancel), userdata(state), context(ctx) {
        auto architecture = gguf_find_key(g, "general.architecture");
        require(architecture >= 0 && gguf_get_kv_type(g, architecture) == GGUF_TYPE_STRING,
                "missing/invalid streamed model architecture");
        const ArchSpec spec = resolve_arch(gguf_get_val_str(g, architecture));
        neox_rope = spec.neox_rope; has_qkv_bias = spec.qkv_bias;
        const std::string pre = spec.prefix + ".";
        // Reject unimplemented architecture options rather than silently approximate them.
        const std::set<std::string> known = {pre+"context_length", pre+"embedding_length", pre+"block_count",
            pre+"feed_forward_length", pre+"rope.dimension_count", pre+"rope.freq_base",
            pre+"attention.head_count", pre+"attention.head_count_kv", pre+"attention.layer_norm_rms_epsilon",
            pre+"attention.key_length", pre+"attention.value_length", pre+"vocab_size"};
        for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
            if (i % 256 == 0) alive();
            std::string key = gguf_get_key(g, i);
            require(key.rfind(pre, 0) != 0 || known.count(key), "unsupported architecture metadata option in streamed backend");
        }
        dim = number(g, (pre+"embedding_length").c_str()); hidden = number(g, (pre+"feed_forward_length").c_str());
        heads = number(g, (pre+"attention.head_count").c_str()); layers = number(g, (pre+"block_count").c_str());
        auto kh = gguf_find_key(g, (pre+"attention.head_count_kv").c_str());
        kvheads = kh < 0 ? heads : number(g, (pre+"attention.head_count_kv").c_str());
        max_context = number(g, (pre+"context_length").c_str());
        require(dim && dim <= 8192 && hidden && hidden <= 32768 && heads && kvheads && heads % kvheads == 0 &&
                dim % heads == 0 && layers && layers <= 128 && vocab && vocab <= 262144 &&
                context <= max_context, "streamed model configuration exceeds supported limits");
        head_dim = dim / heads; kvdim = head_dim * kvheads;
        // Optional K/V head width (llama-3 records it) must equal the single head
        // dimension this decoder rotates and stores; reject mismatches rather than
        // silently mis-shape attention.
        for (const char *k : {"attention.key_length", "attention.value_length"}) {
            if (gguf_find_key(g, (pre + k).c_str()) >= 0)
                require(number(g, (pre + k).c_str()) == head_dim, "streamed backend requires key/value length == head dim");
        }
        if (gguf_find_key(g, (pre + "vocab_size").c_str()) >= 0)
            require(number(g, (pre + "vocab_size").c_str()) == vocab, "streamed vocab_size disagrees with tokenizer inventory");
        // RoPE dimension count is optional; absent means the full head (qwen2 omits it).
        // When present it must equal head_dim: this decoder only rotates the full head.
        auto rope_dim_key = gguf_find_key(g, (pre+"rope.dimension_count").c_str());
        const size_t rope_dim = rope_dim_key < 0 ? head_dim : number(g, (pre+"rope.dimension_count").c_str());
        require(head_dim % 2 == 0 && rope_dim == head_dim, "streamed backend requires full-head RoPE");
        epsilon = real(g, (pre+"attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
        rope_base = real(g, (pre+"rope.freq_base").c_str(), 10000.f);
        // Fixed workspace: O(layers * context * KV width), no weight-matrix allocation.
        auto kvcount = checked_mul(checked_mul(layers, context), kvdim);
        auto floats = checked_mul(kvcount, 2) + checked_mul(dim, 7) + checked_mul(kvdim, 2) +
                      checked_mul(hidden, 2) + vocab + 2 * (context + 255) + 2 * std::max(dim, hidden);
        auto bytes = checked_mul(floats, sizeof(float));
        kv_bytes = checked_mul(kvcount, 2 * sizeof(float));
        workspace_required = bytes + chunk + checked_mul(layers, sizeof(Layer));
        require(bytes <= budget && chunk <= budget - bytes && checked_mul(layers, sizeof(Layer)) <= budget - bytes - chunk,
                "streamed KV/activation workspace exceeds budget");
        embedding = tensor("token_embd.weight", dim, vocab);
        final_norm = tensor("output_norm.weight", dim, 1);
        auto out_id = gguf_find_tensor(g, "output.weight");
        output = out_id < 0 ? embedding : tensor("output.weight", dim, vocab);
        // Optional precomputed per-pair RoPE frequency factors (llama-3 "llama3"
        // scaling, baked into the GGUF at conversion time). head_dim/2 F32 values.
        auto freqs_id = gguf_find_tensor(g, "rope_freqs.weight");
        has_rope_freqs = freqs_id >= 0;
        if (has_rope_freqs) rope_freqs = tensor("rope_freqs.weight", head_dim / 2, 1);
        blocks.reserve(layers);
        for (size_t l = 0; l < layers; ++l) {
            alive();
            std::string p = "blk." + std::to_string(l) + ".";
            Layer layer{tensor(p+"attn_norm.weight", dim, 1), tensor(p+"attn_q.weight", dim, dim),
                tensor(p+"attn_k.weight", dim, kvdim), tensor(p+"attn_v.weight", dim, kvdim),
                tensor(p+"attn_output.weight", dim, dim), tensor(p+"ffn_norm.weight", dim, 1),
                tensor(p+"ffn_gate.weight", dim, hidden), tensor(p+"ffn_up.weight", dim, hidden),
                tensor(p+"ffn_down.weight", hidden, dim)};
            if (has_qkv_bias) {
                layer.qb = tensor(p+"attn_q.bias", dim, 1);
                layer.kb = tensor(p+"attn_k.bias", kvdim, 1);
                layer.vb = tensor(p+"attn_v.bias", kvdim, 1);
            }
            blocks.push_back(layer);
        }
        // Refuse unrecognized tensors (adapters, unexpected biases, etc.).
        // qwen2 adds exactly three Q/K/V bias tensors per layer; llama has none.
        // rope_freqs.weight is the only admitted optional global extra tensor.
        const int64_t per_layer = has_qkv_bias ? 12 : 9;
        const int64_t extra = has_rope_freqs ? 1 : 0;
        require(gguf_get_n_tensors(g) == static_cast<int64_t>(layers * per_layer + (out_id < 0 ? 2 : 3) + extra),
                "unsupported extra tensors in streamed model");
        // Tensor descriptors own their offsets/shapes; no metadata is used by forward().
        g = nullptr;
        if (allocate_state) allocate_workspace(vocab);
    }
    void allocate_workspace(size_t vocab) {
        const auto kvcount = checked_mul(checked_mul(layers, context), kvdim);
        packed.resize(chunk); row.resize(std::max(dim, hidden)); quantized.resize(std::max(dim, hidden));
        x.resize(dim); norm.resize(dim); q.resize(dim); k.resize(kvdim); v.resize(kvdim);
        att.resize(dim); projected.resize(dim); gate.resize(hidden); up.resize(hidden);
        logits.resize(vocab); keys.resize(kvcount); values.resize(kvcount); scores.resize(context + 255); attention_values.resize(context + 255);
        if (has_qkv_bias) { qbias.resize(dim); kbias.resize(kvdim); vbias.resize(kvdim); }
        if (has_rope_freqs) freq_factors.resize(head_dim / 2);
    }
    // Read a projection bias row and add it elementwise (qwen2 Q/K/V bias).
    void add_bias(const Tensor &bias, std::vector<float> &target, std::vector<float> &scratch) {
        require(bias.columns == target.size() && scratch.size() == target.size(), "streamed bias shape mismatch");
        read_row(bias, 0, scratch.data());
        for (size_t i = 0; i < target.size(); ++i) {
            target[i] += scratch[i];
            require(std::isfinite(target[i]), "nonfinite streamed biased projection");
        }
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
        // Group whole consecutive rows in the existing bounded packed buffer.
        // No row crosses a tensor boundary and the final batch is exact, without padding.
        const size_t rows_per_read = chunk / t.row_bytes;
        require(rows_per_read > 0, "streamed matrix row exceeds bounded buffer");
        for (size_t first = 0; first < t.rows;) {
            alive();
            const size_t rows = std::min(rows_per_read, t.rows - first);
            const size_t bytes = rows * t.row_bytes; // <= chunk, validated tensor geometry.
            // Tell Rust the exact next row batch from this tensor descriptor.
            // Admission refusal is an explicit synchronous fallback.
            if (schedule && first + rows < t.rows) {
                const size_t next_rows = std::min(rows_per_read, t.rows - first - rows);
                const uint64_t next_offset = t.offset + (first + rows) * t.row_bytes;
                const uint64_t tensor_bytes = t.row_bytes * t.rows;
                (void)schedule(userdata, next_offset, next_rows * t.row_bytes,
                               t.offset, tensor_bytes);
            }
            require(reader(userdata, packed.data(), t.offset + first * t.row_bytes, bytes) == bytes,
                    "streamed matrix row batch read failed");
            for (size_t local = 0; local < rows; ++local) {
                alive(); // Cancellation remains observable before every dot product.
                float sum = 0;
                traits->vec_dot(static_cast<int>(t.columns), &sum, 0,
                    packed.data() + local * t.row_bytes, 0, activation, 0, 1);
                require(std::isfinite(sum), "streamed matvec overflow/nonfinite weight");
                result[first + local] = sum;
            }
            first += rows;
        }
    }

    void rms(const Tensor &weight) {
        read_row(weight, 0, row.data());
        // Upstream rounds each square to FP32 before double accumulation.
        double sum = 0;
        for (auto z : x) { require(std::isfinite(z), "nonfinite RMS input"); sum += static_cast<double>(z * z); }
        float scale = 1.f / std::sqrt(static_cast<float>(sum / dim) + epsilon);
        for (size_t i = 0; i < dim; ++i) { norm[i] = x[i] * scale * row[i]; require(std::isfinite(norm[i]), "RMS overflow"); }
    }
    void rope(std::vector<float> &vector, size_t count, size_t pos) {
        // Match the pinned ggml incremental FP32 angle sequence (theta *= theta_scale
        // per pair, starting at pos). NORM and NEOX share this angle schedule and
        // differ only in which two head elements each angle rotates.
        const float theta_scale = std::pow(rope_base, -2.0f / head_dim);
        for (size_t h = 0; h < count; ++h) {
            float angle = static_cast<float>(pos);
            for (size_t j = 0; j < head_dim / 2; ++j) {
                // Optional per-pair frequency factor (llama-3 scaling): theta / ff[j].
                // Matches ggml rope_cache_init, which divides theta by freq_factors.
                const float theta = has_rope_freqs ? angle / freq_factors[j] : angle;
                const float c = std::cos(theta), s = std::sin(theta);
                // NeoX rotates (j, j+head_dim/2); normal rotates consecutive (2j, 2j+1).
                const size_t a = neox_rope ? h * head_dim + j : h * head_dim + 2 * j;
                const size_t b = neox_rope ? a + head_dim / 2 : a + 1;
                const float first = vector[a], second = vector[b];
                vector[a] = first * c - second * s;
                vector[b] = first * s + second * c;
                angle *= theta_scale;
            }
        }
    }
    void forward(llama_token token, size_t pos, bool need_logits) {
        require(pos < context && token >= 0 && static_cast<size_t>(token) < embedding.rows, "streamed token/context out of range");
        read_row(embedding, token, x.data());
        // Position/layer-independent RoPE frequency factors: read once per token.
        if (has_rope_freqs) read_row(rope_freqs, 0, freq_factors.data());
        for (size_t l = 0; l < layers; ++l) {
            auto &b = blocks[l];
            rms(b.an); matvec(b.q, norm, q); matvec(b.k, norm, k); matvec(b.v, norm, v);
            if (has_qkv_bias) { add_bias(b.qb, q, qbias); add_bias(b.kb, k, kbias); add_bias(b.vb, v, vbias); }
            rope(q, heads, pos); rope(k, kvheads, pos);
            auto base = (l * context + pos) * kvdim;
            std::copy(k.begin(), k.end(), keys.begin() + base);
            for (size_t i = 0; i < kvdim; ++i) values[(l * kvdim + i) * context + pos] = v[i];
            // Match the pinned non-flash graph's padded FP32 attention operations.
            const size_t n_att = ((pos + 1 + 255) / 256) * 256;
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            for (size_t h = 0; h < heads; ++h) {
                auto kh = h / (heads / kvheads);
                float maximum = -std::numeric_limits<float>::infinity();
                for (size_t t = 0; t <= pos; ++t) {
                    auto ki = (l * context + t) * kvdim + kh * head_dim;
                    float dot = 0;
                    ggml_vec_dot_f32(static_cast<int>(head_dim), &dot, 0, q.data() + h * head_dim, 0, keys.data() + ki, 0, 1);
                    scores[t] = dot * scale;
                    require(std::isfinite(scores[t]), "attention score overflow");
                    maximum = std::max(maximum, scores[t]);
                }
                std::fill(scores.begin() + pos + 1, scores.begin() + n_att, -std::numeric_limits<float>::infinity());
                double denominator = ggml_vec_soft_max_f32(static_cast<int>(n_att), scores.data(), scores.data(), maximum);
                require(std::isfinite(denominator) && denominator > 0, "streamed attention softmax overflow");
                float inverse = static_cast<float>(1.0 / denominator);
                for (size_t t = 0; t < n_att; ++t) scores[t] *= inverse;
                for (size_t i = 0; i < head_dim; ++i) {
                    auto vi = (l * kvdim + kh * head_dim + i) * context;
                    std::copy(values.begin() + vi, values.begin() + vi + pos + 1, attention_values.begin());
                    std::fill(attention_values.begin() + pos + 1, attention_values.begin() + n_att, 0.0f);
                    ggml_vec_dot_f32(static_cast<int>(n_att), &att[h*head_dim+i], 0, attention_values.data(), 0, scores.data(), 0, 1);
                }
            }
            matvec(b.o, att, projected);
            for (size_t i = 0; i < dim; ++i) x[i] += projected[i];
            rms(b.fn); matvec(b.gate, norm, gate); matvec(b.up, norm, up);
            ggml_vec_swiglu_f32(static_cast<int>(hidden), gate.data(), gate.data(), up.data());
            matvec(b.down, gate, projected);
            for (size_t i = 0; i < dim; ++i) { x[i] += projected[i]; require(std::isfinite(x[i]), "streamed residual overflow"); }
        }
        if (need_logits) { rms(final_norm); matvec(output, norm, logits); }
    }
};
} // namespace

static int32_t generate_streamed_cpu(
    const char *path, const char *prompt, uint32_t context, uint32_t max_tokens,
    uint64_t file_bytes, size_t max_chunk, uint64_t workspace_budget, uint64_t tokenizer_prefix,
    llmbridge_range_callback reader, llmbridge_lookahead_schedule_callback schedule,
    llmbridge_token_callback emit,
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
        auto tokenizer = gguf_find_key(metadata.get(), "tokenizer.ggml.tokens");
        require(tokenizer >= 0 && gguf_get_kv_type(metadata.get(), tokenizer) == GGUF_TYPE_ARRAY &&
                gguf_get_arr_type(metadata.get(), tokenizer) == GGUF_TYPE_STRING,
                "missing/invalid tokenizer token inventory");
        const auto vocabulary_size = gguf_get_arr_n(metadata.get(), tokenizer);
        Decoder decoder(metadata.get(), file_bytes, max_chunk, reader, poll, userdata, context,
                        vocabulary_size, workspace_budget, false, schedule);
        // Resolve owned tensor descriptors first, then retire the duplicate GGUF token
        // inventory before the upstream vocabulary and decoder workspace are allocated.
        metadata.reset();
        decoder.alive();
        auto mp = llama_model_default_params(); mp.vocab_only = true; mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> vocabulary(llama_model_load_from_file(path, mp), llama_model_free);
        require(static_cast<bool>(vocabulary), "streamed tokenizer load failed");
        auto vocab = llama_model_get_vocab(vocabulary.get());
        require(static_cast<size_t>(llama_vocab_n_tokens(vocab)) == vocabulary_size, "streamed tokenizer inventory changed");
        int count = llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)), nullptr, 0, true, true);
        require(count < 0 && count != INT32_MIN, "streamed prompt tokenization failed"); count = -count;
        require(static_cast<uint64_t>(count) + max_tokens <= context, "streamed prompt and response exceed context");
        std::vector<llama_token> tokens(count);
        require(llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)), tokens.data(), count, true, true) == count,
                "streamed prompt tokenization changed");
        decoder.alive();
        decoder.allocate_workspace(vocabulary_size);
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

extern "C" int32_t llmbridge_generate_streamed_cpu_v2(
    const char *path, const char *prompt, uint32_t context, uint32_t max_tokens,
    uint64_t file_bytes, size_t max_chunk, uint64_t workspace_budget, uint64_t tokenizer_prefix,
    llmbridge_range_callback reader, llmbridge_token_callback emit,
    llmbridge_logits_callback emit_logits, llmbridge_cancel_poll poll, void *userdata,
    char *error, size_t capacity) {
    return generate_streamed_cpu(path, prompt, context, max_tokens, file_bytes, max_chunk,
        workspace_budget, tokenizer_prefix, reader, nullptr, emit, emit_logits, poll,
        userdata, error, capacity);
}

extern "C" int32_t llmbridge_generate_streamed_cpu_v3(
    const char *path, const char *prompt, uint32_t context, uint32_t max_tokens,
    uint64_t file_bytes, size_t max_chunk, uint64_t workspace_budget, uint64_t tokenizer_prefix,
    llmbridge_range_callback reader, llmbridge_lookahead_schedule_callback schedule,
    llmbridge_token_callback emit, llmbridge_logits_callback emit_logits,
    llmbridge_cancel_poll poll, void *userdata, char *error, size_t capacity) {
    if (!schedule) {
        if (error && capacity) std::snprintf(error, capacity, "%s", "lookahead callback is required for ABI v3");
        return -1;
    }
    return generate_streamed_cpu(path, prompt, context, max_tokens, file_bytes, max_chunk,
        workspace_budget, tokenizer_prefix, reader, schedule, emit, emit_logits, poll,
        userdata, error, capacity);
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

// Teacher-forced streamed logits: prefill prompt, then feed fixed continuation
// tokens, emitting the full logit vector at the final prompt position and after
// each fed token. Setup mirrors generate_streamed_cpu but samples nothing.
extern "C" int32_t llmbridge_streamed_forced_logits_v1(
    const char *path, const char *prompt,
    const int32_t *forced_tokens, uint32_t forced_count, uint32_t context,
    uint64_t file_bytes, size_t max_chunk, uint64_t workspace_budget, uint64_t tokenizer_prefix,
    llmbridge_range_callback reader, llmbridge_logits_callback emit_logits,
    llmbridge_cancel_poll poll, void *userdata, char *error, size_t capacity) {
    if (error && capacity) error[0] = '\0';
    try {
        require(std::endian::native == std::endian::little, "streamed backend requires little endian");
        require(path && prompt && reader && emit_logits && poll && userdata && context >= 64 && context <= 8192 &&
                (forced_count == 0 || forced_tokens) && forced_count <= 2048 && max_chunk >= 34 &&
                max_chunk <= 1024*1024 && workspace_budget && workspace_budget <= 1024ull*1024*1024 &&
                std::strlen(prompt) <= 65536, "invalid streamed forced-logits arguments");
        if (!poll(userdata)) return 2;
        gguf_init_params gp{}; gp.no_alloc = true; gp.ctx = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_callback(reader, userdata, max_chunk, file_bytes, gp), gguf_free);
        require(static_cast<bool>(metadata), "streamed GGUF parse failed");
        if (tokenizer_prefix) require(gguf_get_data_offset(metadata.get()) == tokenizer_prefix, "authenticated tokenizer metadata prefix mismatch");
        if (!poll(userdata)) return 2;
        auto tokenizer = gguf_find_key(metadata.get(), "tokenizer.ggml.tokens");
        require(tokenizer >= 0 && gguf_get_kv_type(metadata.get(), tokenizer) == GGUF_TYPE_ARRAY &&
                gguf_get_arr_type(metadata.get(), tokenizer) == GGUF_TYPE_STRING,
                "missing/invalid tokenizer token inventory");
        const auto vocabulary_size = gguf_get_arr_n(metadata.get(), tokenizer);
        Decoder decoder(metadata.get(), file_bytes, max_chunk, reader, poll, userdata, context,
                        vocabulary_size, workspace_budget, false, nullptr);
        metadata.reset();
        decoder.alive();
        auto mp = llama_model_default_params(); mp.vocab_only = true; mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> vocabulary(llama_model_load_from_file(path, mp), llama_model_free);
        require(static_cast<bool>(vocabulary), "streamed tokenizer load failed");
        auto vocab = llama_model_get_vocab(vocabulary.get());
        require(static_cast<size_t>(llama_vocab_n_tokens(vocab)) == vocabulary_size, "streamed tokenizer inventory changed");
        int count = llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)), nullptr, 0, true, true);
        require(count < 0 && count != INT32_MIN, "streamed prompt tokenization failed"); count = -count;
        require(static_cast<uint64_t>(count) + forced_count <= context, "streamed prompt and forced tokens exceed context");
        std::vector<llama_token> tokens(count);
        require(llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)), tokens.data(), count, true, true) == count,
                "streamed prompt tokenization changed");
        for (uint32_t i = 0; i < forced_count; ++i)
            require(forced_tokens[i] >= 0 && static_cast<size_t>(forced_tokens[i]) < vocabulary_size, "forced token outside vocabulary");
        decoder.alive();
        decoder.allocate_workspace(vocabulary_size);
        for (size_t i = 0; i < tokens.size(); ++i) decoder.forward(tokens[i], i, i+1 == tokens.size());
        if (!emit_logits(decoder.logits.data(), decoder.logits.size(), userdata)) return 2;
        size_t position = tokens.size();
        for (uint32_t i = 0; i < forced_count; ++i) {
            decoder.alive();
            decoder.forward(static_cast<llama_token>(forced_tokens[i]), position++, true);
            if (!emit_logits(decoder.logits.data(), decoder.logits.size(), userdata)) return 2;
        }
        return 0;
    } catch (const Cancelled &) { return 2;
    } catch (const std::exception &e) { if (error && capacity) std::snprintf(error, capacity, "%s", e.what()); return -1;
    } catch (...) { if (error && capacity) std::snprintf(error, capacity, "unknown streamed forced-logits exception"); return -1; }
}

extern "C" int32_t llmbridge_inspect_streamed_plan_v1(
    uint64_t file_bytes, size_t chunk, llmbridge_range_callback reader,
    llmbridge_cancel_poll poll, void *userdata, llmbridge_streamed_shape_v1 *out,
    char *error, size_t capacity) {
    if (error && capacity) error[0] = '\0';
    if (out) *out = {};
    try {
        require(reader && poll && userdata && out && file_bytes >= 24 && chunk >= 34 && chunk <= 1024*1024,
                "invalid streamed plan probe arguments");
        if (!poll(userdata)) return 2;
        gguf_init_params gp{}; gp.no_alloc = true;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_callback(reader, userdata, chunk, file_bytes, gp), gguf_free);
        require(static_cast<bool>(metadata), "streamed plan GGUF parse failed");
        if (!poll(userdata)) return 2;
        auto g = metadata.get();
        auto tokenizer = gguf_find_key(g, "tokenizer.ggml.tokens");
        require(tokenizer >= 0 && gguf_get_kv_type(g, tokenizer) == GGUF_TYPE_ARRAY &&
                gguf_get_arr_type(g, tokenizer) == GGUF_TYPE_STRING, "missing/invalid tokenizer token inventory");
        auto vocab = gguf_get_arr_n(g, tokenizer);
        Decoder d(g, file_bytes, chunk, reader, poll, userdata, 64, vocab, UINT64_MAX, false);
        if (!poll(userdata)) return 2;
        uint64_t payload = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            if (!poll(userdata)) return 2;
            auto size = gguf_get_tensor_size(g, i);
            require(size <= UINT64_MAX - payload, "planned tensor payload overflow"); payload += size;
        }
        uint64_t kv_per_token = d.kv_bytes / 64;
        uint64_t workspace_per_token = kv_per_token + 2 * sizeof(float);
        *out = {d.workspace_required - 64*workspace_per_token, workspace_per_token, kv_per_token,
            file_bytes, payload, gguf_get_data_offset(g), d.max_context,
            d.dim, d.heads, d.kvheads, d.layers, d.hidden, vocab, chunk};
        return 0;
    } catch (const Cancelled &) { return 2;
    } catch (const std::exception &e) { if (error && capacity) std::snprintf(error, capacity, "%s", e.what()); return -1;
    } catch (...) { if (error && capacity) std::snprintf(error, capacity, "unknown plan probe exception"); return -1; }
}
