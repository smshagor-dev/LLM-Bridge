#!/usr/bin/env python3
"""Instrument pinned upstream example to dump raw logits; no inference changes."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    text = args.source.read_text()
    needle = '            new_token_id = llama_sampler_sample(smpl, ctx, -1);'
    if text.count(needle) != 1:
        raise SystemExit('Pinned reference sampling site changed; refusing instrumentation')
    injection = '''            // Validation-only: dump complete logits before greedy sampling.
            if (const char * path = std::getenv("LLMBRIDGE_REFERENCE_LOGITS")) {
                FILE * file = std::fopen(path, "ab");
                if (!file) return 1;
                size_t count = static_cast<size_t>(llama_vocab_n_tokens(vocab));
                size_t written = std::fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), count, file);
                int closed = std::fclose(file);
                if (written != count || closed != 0) return 1;
            }
'''
    context = '    llama_context_params ctx_params = llama_context_default_params();'
    if text.count(context) != 1:
        raise SystemExit('Pinned reference context site changed')
    text = text.replace(context, context + '\n' + """    if (std::getenv("LLMBRIDGE_REFERENCE_F32_KV")) {
        ctx_params.type_k = GGML_TYPE_F32;
        ctx_params.type_v = GGML_TYPE_F32;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
""")
    args.source.write_text('#include <cstdlib>\n' + text.replace(needle, injection + needle))


if __name__ == '__main__':
    main()
