# Vendor-neutral bounded range-read storage — design requirements

Status: **design only (Phase 5 preparation); not implemented.** Prepared 2026-11-23 against commit `c942b74`.

## Purpose and scope

Provide a transport-agnostic way to read a GGUF model source in **bounded byte ranges** so the existing out-of-core streamed decoder can stream a model whose bytes live elsewhere (local file, HTTP range server, object store) without changing the decoder or weakening local-source integrity.

Remote storage is a **transport for bytes**, not remote computation. Reading a model remotely in bounded ranges is distinct from, and must not be conflated with, executing inference remotely or splitting compute across devices.

Non-goal: this layer does not add remote compute, distributed tensors, or caching the whole model in RAM (that would break the ~14.5 MB bounded-memory result measured in `benchmarks/evidence/native_streamed_memory_feasibility_2026-11-23.json`).

## Integration point

The decoder consumes bytes through the `BoundedLocalReader` / `read_range_cancellable(offset, len, cancelled)` contract (see `crates/llmbridge-core/src/storage.rs`, `read_cache.rs`). The remote layer must expose the **same** cancellable, bounded, offset/length range-read surface so it is a drop-in `Source` behind a trait, e.g.:

```
trait RangeSource {
    fn len(&self) -> u64;
    fn object_identity(&self) -> &ObjectIdentity;   // immutable version binding
    fn read_range_cancellable(&mut self, offset: u64, len: usize, cancelled: &mut dyn FnMut() -> bool)
        -> io::Result<&[u8]>;   // bounded; len <= chunk; buffer reused
    fn disk_bytes_read(&self) -> u64;
}
```

The existing `authenticate_model` (whole-file SHA-256 + sealed per-block manifest) and `verify_staged_range_cancellable` (mutation detection) operate on top of this trait unchanged.

**Implemented building block (2026-11-24).** Per-range integrity against the sealed manifest can now be checked *without a second read* via `BoundedLocalReader::verify_block_bytes_against_manifest` / `CachedLocalReader::verify_staged_block_snapshot_cancellable` (content-addressed, fail-closed, opt-in). This is the local realization of properties 3–4 below: a fetched range is authenticated by hashing the delivered bytes against the trusted manifest block, so a remote reader never needs to re-fetch a range to verify it. Trust boundary, attacker model, and the reason the primary per-token re-stream cost cannot be removed under the bounded-memory + untrusted-writer constraints are documented in `AUTHENTICATED_SNAPSHOT_INTEGRITY.md`.

**Validated snapshot decode mode (2026-11-24).** The building block is now wired into the local decode path as an opt-in `ReadPolicy::AuthenticatedSnapshot` (CLI `--snapshot-reuse`): authenticated range reads reuse manifest-verified blocks through the bounded cache. On a matched 8-token Qwen2.5-7B workload this cut physical source bytes 48% (107.3 GB → 55.4 GB) and SHA-256-hashed blocks 51% (1.57M → 774k) with bit-identical output and bounded RSS (+32 MiB cache), i.e. the redundant re-read/re-hash of an already-authenticated block within a pass is removed while every delivered byte stays manifest-authentic (`benchmarks/evidence/authenticated_snapshot_decode_comparison_2026-11-24.json`). This sharpens the remote design: a remote `RangeSource` should (a) version-bind the object (property 2), (b) authenticate each fetched range against the sealed manifest exactly once (properties 3–4, no re-fetch to verify), and (c) stage the version-bound whole-model snapshot to fast **local** storage with this same bounded, manifest-verified reuse cache on top. The measured reuse applies to syscalls and hashing of repeated same-block requests, not to total transfer volume: remote viability is still bandwidth-bound at `model_size × tokens` and must not be presented otherwise.

## Required properties

1. **Bounded async range requests.** Each request is `[offset, offset+len)` with `len <= reader chunk` (64 KiB today). At most `max_outstanding` in flight; a bounded reservation (see Admission) gates issuance. No request may allocate model-sized buffers.
2. **Immutable object identity / source version binding.** Every source binds to a stable identity captured at open time: `{uri, size, strong_validator}` where the validator is an immutable version token — object-store generation/version id, content-addressed digest, or HTTP strong `ETag` + `Last-Modified`. Every range response must be checked against this identity (e.g. `If-Match: <etag>` / versioned GET). A changed validator mid-stream is a hard error, never a silent re-fetch.
3. **Integrity verification.** The whole-object SHA-256 is verified against the trusted expected digest exactly as the local path does; per-range reads are verified against the sealed manifest. Remote bytes are untrusted until verified. Never admit unverified bytes to native compute.
4. **TOCTOU / mutation prevention.** Bind reads to the immutable version (property 2) so a range fetched at T1 and a range fetched at T2 provably belong to the same object version. This is the remote analogue of local mutation detection; it must be at least as strong. A short-lived signed URL or a version-pinned GET prevents an object swap between ranges.
5. **Short-read handling.** A transport that returns fewer bytes than requested must loop/stitch within the same range bound until `len` is satisfied or fail; a short read is never treated as EOF unless `offset+len == size`.
6. **Retry and backoff.** Idempotent range GETs retry on transient transport errors (connection reset, 5xx, timeout) with bounded exponential backoff + jitter and a hard attempt cap. Retries must re-assert the version binding (property 2). Non-idempotent or integrity failures do not retry.
7. **Cancellation and deadlines.** The `cancelled` closure and the existing `Instant` deadline are honoured **between and during** transport operations, matching the Phase 2 in-flight deadline contract (no post-expiry publication). A cancelled/expired request releases its reservation and buffers.
8. **Backpressure.** Bounded request/response channels (as in `LocalLookahead`) cap outstanding work; producers block or shed rather than growing unboundedly.
9. **Memory reservations.** Transport buffers go through the same admission accounting as local read buffers (`reserved`/`outstanding`/`actual_staging`), so the enforced RLIMIT_AS / memory budget still bounds the whole process. No separate unbounded transport arena.
10. **Authentication boundaries.** Transport credentials (bearer tokens, signed URLs, object-store keys) are provided by the caller and never logged, never written to evidence, and never mixed with model-integrity material. Integrity (SHA-256/manifest) is independent of transport auth.
11. **Transport error taxonomy.** Distinguish: not-found, permission-denied, precondition-failed (version changed), transient, integrity-mismatch, cancelled, deadline-expired, short-read-exhausted. Map each to a typed error; only transient errors retry.
12. **Provider independence.** A `RangeSource` trait with adapters: (a) local file (existing `BoundedLocalReader`), (b) generic HTTP `Range:` GET with strong validators, (c) object-store adapter (S3/GCS/Azure-style versioned GET). No provider SDK leaks into the core; adapters are thin and selected by URI scheme.

## Failure and resource contract

- Any integrity or version-binding failure aborts the stream with a typed error and releases all reservations/buffers (worker terminated, zero outstanding, zero reserved — the same invariants proven in Phase 2).
- Deadline/cancellation cleanup matches the existing streamed contract.
- No model bytes, credentials, or URLs are logged.

## Testing requirements (when implemented)

- Injected-failure tests: transient error then success; permanent error; short reads; version-changed mid-stream (must reject); integrity mismatch (must reject); cancellation/deadline during a transport op (no post-expiry publication); reservation accounting returns to zero on every path.
- Parity: remote-sourced inference must produce byte-identical output/logits to the local-sourced run for the same object.
- Bounded-memory: peak RSS must stay within the same envelope measured for the local path.

## Explicitly out of scope for Phase 5

Remote compute, tensor sharding, GPU/accelerator offload, and any design that relaxes local-source mutation detection. These are separate capabilities and must not be implied by shipping remote storage.

## Measured I/O workload and remote feasibility (2026-11-23)

Measured on the streamed backend, ctx 128, 2 threads, 8 generated tokens, cache off unless noted (WSL2 ext4; see `benchmarks/evidence/secure_weight_access_io_2026-11-23.json`).

| Model | File size | Source bytes (8 tok) | Disk read calls | Approx amplification |
|---|---:|---:|---:|---:|
| Stories110M Q6_K | 110.8 MB | 999 MB | 144,869 | ~9x |
| Qwen2.5-0.5B Q8_0 | 531 MB | 5.35 GB | 846,160 | ~10x |
| Qwen2.5-1.5B Q4_K_M | 986 MB | 10.21 GB | — | ~10x |
| Llama-3.2-1B Q4_K_M | 808 MB | 8.74 GB | — | ~11x |

**Cost categories.** The dominant cost is repeated per-token weight streaming: each forward pass re-reads the entire weight set (no persistent weight residency), so source bytes ~= (prefill + generated) x model payload. Authentication is a one-time whole-file SHA-256 pass (amortized). Mutation-detection re-reads and the consume read of the same range are intra-pass and local.

**Bounded cache (<=32 MB, existing `CachedLocalReader`).** A 32 MB cache cut disk read *calls* ~9.5x (Stories110M 144,869 -> 15,268; Qwen0.5B 846,160 -> 81,688) at ~99.97% block-hit ratio, by absorbing immediate intra-pass re-requests of the same range. It did **not** reduce total source *bytes* (999 MB -> 1000 MB; 5.35 GB -> 5.35 GB): the per-token working set is the whole model, which exceeds any bounded cache, so blocks are evicted before the next token re-reads them. **Fewer read calls is not fewer physical bytes.** A cache large enough to cut bytes would have to hold the whole model, contradicting the out-of-core bounded-memory design.

**Remote bandwidth lower bounds (transfer only, per generated token = one model payload).**

| Model/token | 10 Mbps | 100 Mbps | 1 Gbps |
|---|---:|---:|---:|
| 531 MB (Qwen0.5B) | ~7.1 min | ~42 s | ~4.2 s |
| 808 MB (Llama-3.2-1B) | ~10.8 min | ~65 s | ~6.5 s |
| 986 MB (Qwen1.5B) | ~13.1 min | ~79 s | ~7.9 s |

These are bandwidth floors; add latency. At ~846k range operations for a 531 MB / 8-token run, even a 1 ms RTT serialized would add ~846 s; requests therefore must be coalesced (few large ranges per tensor, not per row) and pipelined up to `max_outstanding`.

**Conclusion.** Under the current re-stream-every-token access pattern, remote range-read is bandwidth-bound at model_size x tokens and latency-bound at range_count x RTT. It is impractical over 10/100 Mbps for interactive decoding and marginal at 1 Gbps. Viability requires one of: (a) an immutable, version-bound whole-model snapshot staged to fast local storage (disk, not RAM) with the bounded RAM cache on top — keeping out-of-core RAM bounds while paying the download once; (b) coalesced, pipelined large-range transport with a per-model-version integrity proof; or (c) accepting multi-second per-token latency. The in-RAM bounded cache addresses syscalls, not transfer volume, and must not be presented as a remote-bandwidth optimization.
