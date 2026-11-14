# Cooperative optional-cache release

The streamed decoder can drop its optional block cache while retaining KV/context state, the bounded source buffer, trusted block manifest and sealed tokenizer metadata. This is a one-way session transition from cached to uncached reads. It does not grow/rebuild a cache or change the model, prompt, context, output limit or backend.

```sh
llmbridge chat model.gguf --backend streamed --prompt 'Example' \
  --cache-mb 1 --cache-pressure-mb 512 --metrics
```

`--cache-pressure-mb` is an optional **available-RAM floor**, not a process RSS limit or an additional allocation budget. At logits boundaries, the CLI samples effective available RAM (including known Linux cgroup headroom). Below the floor, or if a sample becomes unknown, it sends one release request and stops sampling. With no allocated cache there is no pressure sampling. The default has no pressure monitor. Resident mode, zero thresholds and MiB overflow are rejected.

Monitoring is cooperative and has sampling overhead. The configured cache still needs to pass initial admission and allocation; use zero cache or `--auto-cache` when that initial allocation cannot fit. Neither monitoring nor release prevents every OOM or enforces total process RSS. More uncached I/O may follow release; no throughput improvement is claimed.

## Library control and lifetime

`generate_streamed_controlled` accepts `StreamedControl { expected_sha256, cache }`. `expected_sha256: Some(trusted_digest)` selects the same authenticated path as `generate_streamed_verified`; `None` remains explicitly unverified. Existing generation functions keep their signatures and default behavior.

Create one `CacheControl` per session. `request_release()` can be called by the consumer or another thread. Requests coalesce and are consumed at safe row/logits callback boundaries. Source bytes have already been copied into native-owned row storage before the previous callback returns; no cache slice crosses those boundaries. A request from the logits consumer is handled before native generation resumes. Borrowed controls are not retained after the synchronous generation call. This is not a deadline or a multi-session broadcast signal.

Releasing drops payload, staging and slot vectors using empty replacements, without allocating replacement buffers or copying cache contents. Their capacities become zero. The source reader and authentication state remain alive, so future reads keep cancellation, identity and authenticated-block checks. Cache cannot regrow in the session. Repeated requests are idempotent.

Completion snapshots (before reader teardown) distinguish current `cache_allocated_bytes`, `cache_peak_allocated_bytes`, `cache_released_bytes` and `cache_release_events`. Peak/released bytes describe the cache-owned allocation accounting, excluding allocator headers, source buffer, KV/workspace, integrity metadata and OS page cache. The allocator can retain freed pages; zero current cache allocation does **not** assert an immediate RSS decrease, physical erasure or a whole-process memory bound.

## Validation scope

Unit tests check zero remaining vector capacities, idempotence, source-read/cancellation/identity checks, disabled-cache behavior, atomic control and floor boundaries. The real-model contract compares complete output and streaming SHA-256 of every logit, checks final/peak cache accounting, then requires cancellation and same-size/inode/mtime late-mutation rejection after cache release.

Golden Q8/Q4 trials use an intentionally oversized floor to trigger the policy deterministically, and require exactly one release, zero final optional-cache allocation and released bytes equal to the previous cache peak. They compare every output/logit with the same independent numerical reference. The Q8 trial keeps the existing native 24 MiB process cap; Q4 remains uncapped because its file is smaller. These are forced-signal correctness experiments, not physical low-RAM-pressure, 2GB-machine, RSS-reclamation or performance results. The AVX2 default-repacked Q4 reference limitation is unchanged.
