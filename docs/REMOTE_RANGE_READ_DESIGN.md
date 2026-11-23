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
