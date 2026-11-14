# Bounded contiguous weight-row reads

The streamed decoder now fills its existing packed-weight buffer with whole consecutive matrix rows, instead of making one Rust/C++ range callback for each row. A batch holds at most `floor(chunk_bytes / row_bytes)` rows and stops exactly at the tensor end. It never pads or reads another tensor. Embedding/norm reads remain exact. The per-row dot kernel, activation quantization, reduction order and sampling are unchanged.

There is no extra weight buffer, background queue or worker. Packed native scratch, Rust source scratch and optional cache remain charged separately as before. Cancellation is polled before each batch and before every row's dot product. Cache-release signals are consumed at the next bounded-read or logits callback, rather than necessarily at every matrix row; this is cooperative control, not a deadline.

`LLMBRIDGE_STREAMED` and `StreamedMetrics` expose `range_read_calls`, `range_bytes_requested` and `max_range_read_bytes`. These count native range requests, including GGUF metadata, while `weight_read_calls`/`weight_bytes_requested` isolate matrix/embedding/norm payload requests independently of application source reads and SHA block verification. Counters are constant-space and saturate. They do not count physical storage operations.

## Authenticated reads

An authenticated source re-reads and hashes each requested manifest block on **every** public range call. Previously verified scratch and the optional LRU cannot substitute for fresh source verification: a writer can alter a previously read block while restoring the observed size/mtime/inode. Reusing that old verification could miss the change. The authenticated path therefore bypasses the optional LRU. Session checks do not clone digest strings per range.

The configured optional-cache reservation is still allocated/accounted until release or teardown, even when authenticated reads bypass it. It does not imply authenticated cache hits or read-call savings; use zero cache when no such reservation is wanted. Unauthenticated LRU behavior remains available with identity checks, without a cryptographic guarantee. Historical authenticated-cache/read-call measurements from earlier commits do not describe this corrected verification policy.

Native GGUF metadata callbacks read directly into their provided buffer from the already sealed trusted prefix using bounded offset reads. Crossing that prefix is rejected, so sparse weight holes cannot be returned as metadata. This needs no additional metadata copy. Weight requests still re-read and verify the original source.

Batching reduces range/verification callback overhead without reusing stale verification across calls. It is synchronous I/O grouping, not asynchronous prefetch, parallel compute, a complete adaptive scheduler or proof of large-model feasibility. Larger reads can pull unused rows before cancellation; each request still fits one configured chunk. Verified bytes copied into native-owned batch scratch remain trusted checkpoint bytes even if the backing source changes later; the next public read re-verifies its requested blocks.

## Reproduce

```sh
cargo run --release -p llmbridge-core --features native --example row_batch_contract -- models/stories15M-q8_0.gguf
```

The pinned Q8/Q4 contract compares complete-output hashes and SHA-256 over every FP32 logit at 4096, 65536 and odd 65535-byte chunk sizes, with cache reservations disabled/enabled. It checks identical logical requested bytes, chunk bounds, fresh authenticated reads and fewer requests at larger chunks. Independent reference protocols retain full-output/logit tolerances and the existing Q8 Linux 24 MiB address-space cap. Same-identity repeated-block and LRU-hit mutation regressions must reject tampered data. Existing cancellation, exact-budget and cache-release integrity contracts remain required.

Actual runtime/throughput effects depend on model, source/page cache, SHA cost, CPU and chunk geometry. No physical-disk or universal speed claim follows from callback counts. Physical 2GB and large-model validation remain pending.
