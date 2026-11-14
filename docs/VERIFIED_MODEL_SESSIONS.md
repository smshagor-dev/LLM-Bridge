# Verified streamed model sessions (Linux)

`--sha256 <trusted-whole-file-digest>` opts a streamed generation into authenticated model reads. A separate earlier `models verify` command is not sufficient to bind a later path load to those bytes. The default path remains explicitly unverified; `--metrics` reports `LLMBRIDGE_INTEGRITY verified=0` or the verified digest, accounted overhead and authenticated block loads.

```sh
llmbridge chat models/stories15M-q8_0.gguf --prompt 'The sky is' \
  --backend streamed --max-tokens 8 --sha256 \
  2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6 \
  --verification-mb 8 --address-space-mb 24 --metrics
```

The digest must come from an independently trusted source. Matching a caller-supplied checksum is content integrity, not a publisher signature, license check or provenance service. The resident backend rejects `--sha256` rather than silently treating a preflight hash as verified execution. Unsupported platforms fail closed; there is no fallback to an ordinary mutable tokenizer path.

## How the session binds consumed bytes

1. A bounded pass hashes the entire checkpoint and records SHA-256 for each reader-sized block. Metadata/manifest/scratch admission happens before that pass. A mismatch or cancellation returns before native token/logit callbacks.
2. Only the GGUF metadata prefix is copied into an anonymous Linux memfd. Its logical file length matches the checkpoint, but weight regions are sparse zero holes, not copied or allocated model weights. Write/grow/shrink/seal seals are applied and checked. The sealed prefix is rehashed after sealing against the prefix observed during the trusted whole-file pass.
3. Native vocabulary-only loading uses the sealed descriptor path. Native GGUF parsing still uses the authenticated Rust reader; its actual metadata/tensor-data offset must equal the sealed prefix length before vocabulary loading. This prevents an indexing/source race from supplying a different tokenizer prefix.
4. Every newly fetched weight block is hashed against the anchored block manifest before any requested row bytes can reach native computation. A single authenticated block window avoids rehashing/reloading a whole block for each adjacent tiny row. Optional LRU cache entries originate from this verified reader and cannot be changed by the source publisher.
5. Source descriptor/path identity and cancellation checks remain active. Cached original authenticated bytes may remain valid even if a source was rewritten while preserving identity, but newly fetched altered bytes are rejected. A normal detected identity change also fails the session.

The old C entry point remains available. An additive `llmbridge_generate_streamed_cpu_v2` accepts the tokenizer-prefix binding; all C++ exceptions and Rust callback panics remain contained. This API does not change the embedding application's process limits.

## Memory and storage scope

`--verification-mb` budgets page-rounded sealed metadata, the block-digest array, one authentication scratch buffer and the retained digest text. Default is 8 MiB, configurable up to 64 MiB. Large checkpoints or metadata prefixes that exceed it are refused; support is not inferred from file size alone. Allocator headers, parser/tokenizer objects and fixed stack/runtime overhead are outside this component budget and still require process-level admission/enforcement.

On the pinned 26,671,328-byte stories15M Q8 checkpoint the accounted verification overhead was **807,712 bytes**. Three eight-token prompts, with the outer cache both disabled and enabled at 1 MiB, completed under Linux **24 MiB RLIMIT_AS**, matching complete reference output and all 256,000 FP32-KV logits per trial. This is a tiny-model result, not a physical 2GB-host or large-model proof. The metadata memfd has kernel-owned tmpfs pages, which are explicitly included in verification accounting but are not process RSS. RLIMIT_AS does not cap kernel page cache/tmpfs or total machine RAM. `source_bytes_read` includes the preflight whole-file pass plus newly fetched authenticated blocks. The outer cache's `source_read_calls` counts requests into the bounded reader; verified mode has an additional single-block authentication window, so that count is not the number of OS reads. `blocks_verified` counts freshly checked blocks during consumption. Authentication also adds a whole-file read and per-block SHA work; it is not advertised as a speed improvement.

The sealed descriptor is closed and its metadata pages reclaimed when the session ends. No full checkpoint copy, permanent cache file, cloud upload or external inference is introduced. Local same-user process compromise, ptrace, malicious native libraries and compromised trusted digests are not solved by content hashing. File-index/native-parser fuzzing and broader OS worker isolation remain required hardening work.

## Reproduce and test

Use the independent reference setup in [STREAMED_DECODER.md](STREAMED_DECODER.md):

```sh
python3 scripts/check_streamed_parity.py --binary target/release/llmbridge \
  --reference reference-build/bin/llama-simple --model models/stories15M-q8_0.gguf \
  --output benchmarks/results/verified --max-address-space-mb 24 --verify
cargo run --release -p llmbridge-core --features native --example verified_contract \
  -- models/stories15M-q8_0.gguf
```

Tests cover hash/budget/cancellation admission, sealed-prefix content and write refusal, partial-block/EOF reads, and a changed uncached block with original size/inode/mtime deliberately preserved. The real-model contract copies its fixture to a newly created temporary file, mutates a future weight block after the first verified logits, restores mtime, and requires an authenticated-block error. Prior correct streamed tokens cannot be rolled back; no claim of all-or-nothing generation is made. CI retains raw reference/logit evidence.

## Repeated-read correction

Authenticated ranges now rehash requested source blocks on every public read and bypass the optional LRU, including repeated reads of a previously verified block. Historical cache/read-call results precede this correction and are not current authenticated-throughput evidence. See [bounded row batching](BOUNDED_ROW_BATCHING.md) for validation and the cost of fresh verification.
