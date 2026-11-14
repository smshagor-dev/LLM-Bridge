# Bounded session cache for streamed inference

`llmbridge chat ... --backend streamed --cache-mb 1 --metrics` enables an opt-in block cache. Default `--cache-mb 0` retains exact uncached range reads. Both paths execute the same decoder and use the same logits/sampling code.

The cache owns one open source for one generation. A miss reads one aligned source block, at most the configured reader chunk (64 KiB for CLI). Adjacent row callbacks reuse that block. Cross-block requests assemble exactly the requested bytes in fixed private staging; the final block never reads beyond EOF. Least-recently-used blocks are evicted when the fixed slot inventory is full. This is synchronous bounded read-ahead, not a background worker, unbounded queue or speculative layer skipping.

Budget includes fixed payload, response staging and slot arrays; it excludes allocator headers and the reader's separately reported buffer. Cache metadata/lookups are limited to 512 slots and the CLI accepts 0–32 MiB. Workspace/KV and cache budgets are separate and both enter CLI RAM admission. A 1 MiB budget with 64 KiB chunks reserves 983,376 accounted bytes in 14 slots; unused budget is not allocated. A configured budget does not guarantee available process memory: under the tested 24 MiB address-space limit, a 2 MiB cache returned `std::bad_alloc` before token generation. The 1 MiB configuration passed. There is no silent cap increase or automatic cache-size fallback.

Cache hits validate the source descriptor and path identity before and after staging. Cancellation is checked at lookup/refill/publication boundaries. Failed refills invalidate the affected slot. Changed sources return an error rather than cached weights. These are ordinary identity checks, not authenticated immutable model snapshots; an adversary who can rewrite bytes while preserving identity metadata is outside that guarantee. Cache data never survives the generation or switches to another model.

## Metrics and measured result

`LLMBRIDGE_STREAMED` reports source bytes, reader buffer size, source read calls, accounted cache allocation, block hits/misses and evictions. Source calls/bytes measure application file-range I/O, not physical storage operations or OS cache hits. Hit counters are block lookups, so one cross-block row can produce two lookups.

On the pinned stories15M Q8_0 checkpoint (26,671,328 bytes), three eight-step prompt trials with cache disabled and enabled passed complete reference-output/logit comparison under the same Linux 24 MiB RLIMIT_AS ceiling. Cached and uncached raw logit traces were also exactly equal. Read calls fell from 581,046 to 2,293 for two prompts and from 598,915 to 2,392 for the third (about 99.6%). Bounded read-ahead increased total source bytes by about 0.8–0.9%. This does not imply a proportional throughput gain, reduced compute, or support for large models/physical 2GB machines.

Reproduce using the independent reference setup in [STREAMED_DECODER.md](STREAMED_DECODER.md):

```sh
python3 scripts/check_streamed_parity.py --binary target/release/llmbridge \
  --reference reference-build/bin/llama-simple --model models/stories15M-q8_0.gguf \
  --output benchmarks/results/uncached --max-address-space-mb 24 --cache-mb 0
python3 scripts/check_streamed_parity.py --binary target/release/llmbridge \
  --reference reference-build/bin/llama-simple --model models/stories15M-q8_0.gguf \
  --output benchmarks/results/cached --max-address-space-mb 24 --cache-mb 1
python3 scripts/compare_cache_results.py \
  --uncached benchmarks/results/uncached/streamed_parity.json \
  --cached benchmarks/results/cached/streamed_parity.json \
  --output benchmarks/results/cache_comparison.json
```

The comparison gate requires identical model/binary/cap configuration, successful reference parity, identical cached/uncached full output and raw logit files, cache allocation within budget, and fewer actual source range reads. CI retains the raw evidence. Checked-in summaries identify source revision and binary hash.

Regression coverage includes adjacent reads, cross-block/EOF ranges, LRU eviction, disabled mode, budget rejection, changed sources, cancellation on hits/refills, late source changes and invalidated failed refills. Real-model cancellation/workspace/context contracts run with cache both disabled and enabled. Optional Linux authenticated sessions now feed verified blocks into this cache; see [VERIFIED_MODEL_SESSIONS.md](VERIFIED_MODEL_SESSIONS.md). Portable authentication, asynchronous prefetch, persistent caches and adaptive cache scheduling remain future work.

## Current execution and authentication policy

Matrix rows are now grouped within the existing bounded native chunk. Authenticated sessions bypass this LRU and rehash every requested block per public range call. The configured cache reservation can still be released, but it does not provide authenticated cache hits. The numerical/read-call results above are historical measurements of the stated earlier source, not current authenticated-performance claims. See [bounded row batching](BOUNDED_ROW_BATCHING.md).
