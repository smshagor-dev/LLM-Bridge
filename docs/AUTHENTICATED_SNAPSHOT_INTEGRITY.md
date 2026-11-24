# Authenticated read path, immutable-snapshot trust boundary, and manifest range verification

Status: **audit + trust-boundary design, with one implemented and tested primitive.** Prepared 2026-11-24 on branch `local/streamed-metadata-admission`. Grounded in the real source at this commit; no numbers in this document are hypothetical unless labelled as bounds.

This document answers three things the way the code actually behaves:

1. the complete authenticated read path and where its cost is (Part I audit);
2. what an "immutable model snapshot" can and cannot mean here, against an explicit attacker model (Part II);
3. the one integrity-preserving optimization that is safe under the project's own constraints, which is implemented and tested in this change (Part III), and why the larger win is a documented blocker rather than a shipped feature.

---

## Part I — The authenticated read path (audit)

### Flow

```
open source (regular file, symlink rejected, identity captured)
  -> authenticate_model (Linux):
       whole-file SHA-256 over bounded chunks  == trusted expected digest  (fail closed)
       per-block SHA-256 list  hashes[i]        (one digest per chunk-sized block)
       tokenizer-metadata prefix copied into a sealed memfd (F_SEAL_WRITE|GROW|SHRINK|SEAL), rehashed
       weight region is NOT copied (sparse holes) -> out-of-core
  -> per range request during decode:
       CachedLocalReader::read_range_cancellable
         if authenticated: BYPASS the LRU entirely
       BoundedLocalReader::read_authenticated_range
         for each covered block: validate_source (identity) ; seek+read_exact ; SHA-256 ; compare hashes[i]
         (cached_block reset to None on entry => re-hash on every public call)
  -> bytes published to GGML tensor decode
```

Source of truth: `crates/llmbridge-core/src/storage.rs`, `storage_auth.rs`, `read_cache.rs`.

### The per-block hash list is a valid trust root

`hashes[i]` is computed over exactly the bytes that produced the whole-file digest, in the same pass, and is accepted only after that digest equals the trusted expected value. A flat per-block hash list anchored to a verified whole-file digest is a (degenerate) Merkle commitment: given the trusted root, each block hash authenticates that block's bytes. The list lives in process memory after a verified pass, so its integrity is the process TCB — the same trust boundary that holds the model bytes and the GGML kernels. **A match against `hashes[i]` proves a block's bytes are authentic, independent of the current on-disk source state.** This fact is what Part III exploits.

### Separated cost counters (do not conflate)

| Quantity | Where | On the authenticated streamed path |
|---|---|---|
| Logical tensor bytes | GGML demand | model payload per forward pass |
| Application source bytes | `disk_bytes_read` | re-read of every covered block on **every** public range call; the LRU is bypassed for authenticated sources (`read_cache.rs` `is_authenticated()` branch) |
| SHA-256 hashed bytes | `blocks_verified` x block size | equals application source bytes — every byte read is re-hashed |
| Mutation-validation work | `validate_source` | two `open`+`stat` identity checks per range (not byte reads) |
| Cache hits / misses | `ReadCacheMetrics` | 0 / 0 — the LRU is bypassed under authentication |

Measured amplification for the re-stream-every-token pattern is recorded in `benchmarks/evidence/secure_weight_access_io_2026-11-23.json` and summarized in `REMOTE_RANGE_READ_DESIGN.md` (≈9–11x source bytes vs file size). For Qwen2.5-7B a 2-token authenticated run on 2026-11-24 measured 54.2 GB source read, 755,540 blocks verified, 79.1 MB peak RSS (`benchmarks/evidence/authenticated_snapshot_manifest_verification_2026-11-24.json`).

### Hashing is already hardware-accelerated

`sha2 0.10.9` selects the x86 SHA-NI backend at runtime via `cpufeatures`; the validation host reports the `sha_ni` CPU flag. The re-hash CPU is therefore already on the hardware path — there is **no** untapped "enable SHA-NI" optimization. The re-hash cost that remains is intrinsic to re-reading bytes from a mutable source.

---

## Part II — Immutable-snapshot trust boundary

### Attacker model

A local process that can write the model file in place (same path, size, inode, and mtime) between two reads, but cannot write this process's memory or the kernel objects it holds (sealed memfd, open descriptors). This is exactly the attacker the existing tests encode (`storage_auth.rs`: `unchanged_identity_cannot_hide_a_modified_uncached_block`).

### What does and does not establish immutability

- **File path, size, mtime, inode, ETag:** auxiliary identity only. Not proof of immutability against the attacker above. (The code already treats identity as a *change detector*, not an authenticity proof.)
- **Read-only permissions / naming convention:** not sufficient against a local writer with rights, per the master specification's own warning.
- **Sealed memfd (`F_SEAL_WRITE|GROW|SHRINK|SEAL`):** genuine OS-backed immutability — but RAM-resident. Used today only for the bounded tokenizer-metadata prefix. Sealing the whole 4.68 GB model would make it resident, defeating the out-of-core / bounded-RAM result and still costing the read.
- **Content-addressed verified copy on local disk:** gives a stable *identity* and a one-time verification, but an ordinary on-disk file is still writable by the attacker, so reads from it must still be re-authenticated. It removes download/transfer cost (relevant to remote), not the re-hash on a mutable local file.
- **Per-block hash manifest anchored to the trusted digest (already built):** authenticates bytes *by content*, so it does not depend on source immutability at all. This is the usable trust root.

### Consequence (the honest architectural result)

Under the two stated constraints together — **bounded memory (≤256 MiB address space) and an untrusted local writer** — neither the per-token weight **re-read** nor the weight **re-hash** on the primary streamed path can be safely eliminated:

- eliminating the re-read needs residency (a cache/snapshot large enough to hold the working set = whole model) → violates bounded memory;
- eliminating the re-hash needs provable immutability between reads → needs a sealed (RAM) snapshot → violates bounded memory, or an on-disk file that the attacker can still mutate → unsafe.

This is a correct negative result, not a failure to try: it is the same conclusion the project already recorded in `IMPLEMENTATION_STATUS.md` ("the synchronous source reread and consumption-time checks remain mandatory … depends on an immutable source snapshot") and in `local_lookahead_regression_profile_2026-11-23.json`. A performance "win" that removed these would weaken integrity or the memory bound, which the specification forbids.

---

## Part III — Implemented: manifest range verification without re-read (opt-in)

There is exactly one place where the manifest trust root yields a safe, bounded win **without** touching the primary decode path or the memory bound: authenticating **already-staged** bytes (produced by the opt-in `LocalLookahead` worker) against the manifest, instead of re-reading the source a second time to re-compare.

### Primitive

`BoundedLocalReader::verify_block_bytes_against_manifest(offset, bytes) -> io::Result<bool>` (`storage_auth.rs`):

- `Ok(true)` — `bytes` is exactly one full, block-aligned manifest block and its SHA-256 matches `hashes[block]`; performs **no source I/O**; increments `blocks_verified`.
- `Ok(false)` — not a single full aligned block; the caller must fall back to the mutable-source re-read path.
- `Err` — a full aligned block whose bytes do **not** match the manifest → **fail closed**.

### Consumer (opt-in)

`CachedLocalReader::verify_staged_block_snapshot_cancellable(offset, staged, cancelled)` (`read_cache.rs`): on an authenticated source with a full aligned staged block, verifies against the manifest with no second read; otherwise falls back to the existing `verify_staged_range_cancellable`, which keeps its mutable-source mutation-detection semantics.

### Why this is safe, and why it is additive / opt-in

The fast path's contract is *immutable-snapshot (content-addressed)*: staged bytes are accepted iff they are cryptographically authentic, so a source mutation **after** prefetch is intentionally not detected (the bytes are already proven correct). That is a **different** guarantee from the default consumption-time path, which asserts "the staged bytes still equal the current source". The default guarantee is what `local_lookahead.rs::consumption_time_authentication_rejects_mutation_after_prefetch` verifies. Because the two guarantees differ, the manifest path is kept **additive and opt-in** (per the specification: "any new mode must be opt-in until fully validated", "if existing source-mutation guarantees cannot be preserved, do not enable the new path in production"). The default path and its test are unchanged.

### Tests (adversarial, deterministic)

`storage_auth.rs`:
- `manifest_block_verification_authenticates_without_source_io` — authentic full/short blocks verify with `disk_bytes_read` unchanged; `blocks_verified` advances.
- `manifest_block_verification_fails_closed_on_tampered_full_block` — tampered full block → `InvalidData`; counter not advanced.
- `manifest_block_verification_signals_fallback_for_partial_or_unaligned_ranges` — unaligned/partial/over-length/EOF → `Ok(false)`.
- `manifest_block_verification_trusts_bytes_despite_in_place_source_mutation` — after in-place mutation (same size+mtime), authentic bytes still verify with no read, while the default `read_range` reports "block changed".
- `manifest_block_verification_requires_authentication` — unauthenticated reader → error.

`read_cache.rs`:
- `staged_block_snapshot_verifies_without_reread_and_fails_closed` — full aligned block: zero source bytes on accept and on tampered-reject; after in-place mutation the manifest path accepts the authentic bytes while `verify_staged_range_cancellable` rejects.
- `staged_block_snapshot_falls_back_and_reads_for_partial_ranges` — partial staged range routes through the default path and does read the source.
- `staged_block_snapshot_without_auth_uses_default_verification` — unauthenticated falls back and detects a size change.

### Measured effect and scope

The win is the elimination of the **consumption-time duplicate read** of staged ranges on the authenticated lookahead path: a full aligned staged block is verified with **0** source bytes instead of one covering-block re-read (proven by the `disk_bytes_read` assertions above). The same manifest trust root is now also exploited by a production decode mode — **Part IV** — which reuses already-authenticated blocks through the bounded cache and yields a measured end-to-end 7B improvement while preserving integrity and the memory bound.

---

## Part IV — Production integration: two explicit read modes (2026-11-24)

The manifest trust root is now wired into the real streamed decode path as an
**opt-in** read policy on `CachedLocalReader`, selected by `StreamedConfig.snapshot_reuse`
and the CLI flag `--snapshot-reuse` (which requires the streamed backend, `--sha256`
authentication, and `--cache-mb > 0`).

### Mode A — Mutable Source (default, unchanged)

`ReadPolicy::MutableVerified`. Authenticated sources bypass the LRU and re-read
**and** re-hash every covered block on every range call, so any in-place source
mutation between calls is detected and rejected. This is the behaviour described
in Part I and is the default; its mutation-detection tests are untouched.

### Mode B — Authenticated Snapshot (opt-in)

`ReadPolicy::AuthenticatedSnapshot`. Authenticated range reads are served through
the bounded LRU. Every block entering the cache is authenticated against the
trusted per-block manifest on the miss path (the authenticated source read fails
closed on any mismatch), and cache hits reuse those already-authenticated retained
bytes with **no second source read and no re-hash**. This is the content-addressed
semantics of Part III applied per range: **content authenticity, not
mutation-event detection**. A source mutation after a block is cached is
intentionally not detected on a hit (the retained bytes are already proven to be
the authentic model bytes); a newly fetched changed block still fails closed on a
miss. Because this guarantee differs from Mode A's, it is additive and opt-in; the
default path and its tests are unchanged (`read_cache.rs`:
`default_policy_authenticated_still_bypasses_cache_and_rereads`).

### Why it is safe

Only manifest-verified bytes ever enter the cache (the miss path routes through
`read_authenticated_range`, which hashes and compares to `hashes[i]`). A cache hit
serves those retained bytes after the ordinary identity check, never re-publishing
unverified data. Cancellation and deadline checks are honoured on hits
(`snapshot_policy_cache_hit_still_honors_cancellation`). The bounded cache is the
existing `CachedLocalReader` budget (≤ 32 MiB), so no new unbounded allocation is
introduced and the memory bound is preserved.

### Measured effect (genuine Qwen2.5-7B, matched 8-token greedy workload)

`benchmarks/evidence/authenticated_snapshot_decode_comparison_2026-11-24.json`.
Both runs: identical prompt, context, tokens, threads, CPU affinity, workspace and
2048 MiB address-space cap; Mode B adds only `--cache-mb 32 --snapshot-reuse`.

| Metric | Mode A (default) | Mode B (snapshot) | Δ |
|---|---:|---:|---:|
| Generated text | " Paris. It is located in the northern" | *bit-identical* | **parity** |
| Logical demand (`range_bytes_requested`) | 50.66 GB | 50.66 GB | unchanged |
| Physical source bytes read | 107.31 GB | 55.43 GB | **−48.3%** |
| Blocks SHA-256 verified (hashed) | 1,566,020 | 774,336 | **−50.6%** |
| TTFT | 30.07 s | 22.61 s | **−24.8%** |
| Decode throughput | 0.188 tok/s | 0.265 tok/s | **+41%** |
| Total wall | 67.27 s | 49.00 s | **−27.2%** |
| Peak RSS | 79.3 MB | 112.9 MB | +32 MiB (= cache) |

The bounded cache absorbs the ~2× intra-forward-pass re-reads/re-hashes of
already-authenticated 64 KiB blocks (hit ratio ≈ 50.5%), roughly halving physical
I/O **and** SHA-256 hashing, which is where Mode A spends most of its callback time
(`range_callback_ns` 59.7 s → 41.4 s). This is a genuine end-to-end win, not a
read-call artefact: logical demand and output are unchanged and every served byte
is still manifest-authentic.

**What it does not do.** It does not make I/O model-size-independent: 55.4 GB for 8
tokens still scales with `model_size × tokens`, because the per-token working set is
the whole model and exceeds any bounded cache (evictions ≈ misses). The honest
negative result of Part II stands — the *base* per-token re-stream cost is
intrinsic; Mode B only removes the *redundant* re-read/re-hash of the same block
within a pass. Bounded memory is preserved: Mode B runs under 256/512/2048 MiB
address-space caps with peak RSS 112 MB (the +32 MiB is exactly the opt-in cache).

Mode B is retained as opt-in; the default is unchanged.

---

## Relationship to remote range reads

The same primitive is the remote building block: a remote reader must authenticate each fetched range **against the sealed manifest**, never by a second fetch. See `REMOTE_RANGE_READ_DESIGN.md` properties 3–4; this change makes that verification concrete on the local path first. The remote viability conclusion is unchanged: remote range read is bandwidth-bound at `model_size x tokens` and needs a version-bound whole-model snapshot staged to fast **local** storage (disk, not RAM) with the bounded RAM cache on top.
