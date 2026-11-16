# Verified local batch resume

This is completed-job resume. Unfinished inference restarts from its prompt; KV state or token-level continuation is not restored. Ordinary batch behavior remains unchanged unless a checkpoint mode is selected.

```sh
llmbridge batch model.gguf --input prompts.txt --output-dir new-results \
  --sha256 TRUSTED_MODEL_SHA256 --resumable

llmbridge batch model.gguf --input prompts.txt --output-dir new-results \
  --sha256 TRUSTED_MODEL_SHA256 --resume
```

`--resumable` requires a new output directory and records identities/completions. `--resume` requires an existing compatible checkpoint. Both require an independently trusted `--sha256`, and cannot be combined. The first implementation requires Linux advisory file locking and a filesystem supporting hard links and directory synchronization. This does not affect ordinary batch support on other platforms.

Before any completion is skipped, the current model is fully SHA-256 verified, even if every job is already complete. The input file is hashed with bounded reads, and the actual running executable is hashed through `/proc/self/exe`. A versioned `batch.identity` binds model, exact input bytes (including line endings), binary artifact, context, token limit and prompt-line limit. Worker count, slot/reserve sizing, waiting/pressure policy and per-job deadline are operational controls and can change. An identical binary/model/input may move paths; changed binary bytes, model bytes, input bytes or generation settings refuse resume. Verification adds full-file I/O without loading the whole files into RAM; it is not instant activation.

`job-N.done` binds the manifest identity, job ID, parsed prompt hash, exact output hash and byte length. A completed job is skipped only after reading and hashing its actual output and comparing the entire bounded receipt. No stat-only shortcut is used. Completed output and receipt files are not rewritten. New completions sync output data, publish `.txt` with no-replace hard links, then sync/publish the receipt and directory metadata. Result hashing and record writing use bounded buffers and do not retain whole responses.

An unfinished `job-N.partial` is preserved as `job-N.retry-K.partial` without replacing an existing archive; the job then starts again in a new partial file. Missing/corrupt completion data, a final `.txt` without a receipt, an incomplete receipt publication or nonregular job/record paths refuse reuse rather than overwrite uncertain data. Old ordinary batches have no checkpoint identity and cannot be retroactively trusted for skipping. Unknown unrelated files are retained.

A persistent `batch.lock` is held with a nonblocking kernel `flock` lease throughout checkpoint use. Resume acquires this lease before expensive scans. Concurrent cooperating resume processes refuse; closing the process descriptors releases the lease, including after SIGKILL. Do not remove/replace the lock file manually while a process may be active. This is local coordination, not a distributed/object-storage lease.

The input file descriptor used for hashing is rewound and reused for prompt parsing. The raw byte stream is hashed again while it is consumed; an EOF mismatch fails the batch. Completed receipts additionally bind each actual prompt. This detects inconsistent input use without claiming an immutable filesystem snapshot. Records are ordinary local integrity records, not signed attestations: somebody able to rewrite both records and outputs is outside this guarantee. Adversarial concurrent filesystem writes and power-loss recovery are not established. Inconsistent publication states fail closed for review; this is not an exactly-once transaction system.

Checkpoint preflight, skip verification, result hashing, fsync and publication are outside native/admission budgets and can delay return. Job cancellation remains cooperative. A per-job zero native budget may still verify/skip already-completed work; unfinished work is not generated. The final checkpoint-mode summary adds `skipped=N`, while `completed=N` counts newly finished jobs. Completed results and archived partials remain available after failures.

## Validation

Unit tests cover byte/prompt/job receipt binding, output corruption, bounded metadata reads, orphan-final refusal, partial archival, exclusive lock/release, exact input line-ending hashes and mode/trust validation. `scripts/check_batch_resume.py` uses the real pinned Q8 model: interrupt after verified completion plus partial native output, resume with another worker count/wait/deadline budget, preserve existing output/receipt bytes and inode/mtime, archive partials and compare every final output against direct streamed chat. It verifies all-complete read-only resume, model/output corruption with unchanged size/mtime, input/configuration/binary mismatch, identical-binary relocation, malformed/orphan receipts, concurrent locking and kernel lock release after SIGKILL, plus retry after an expired per-job deadline. No physical RSS, throughput, large-model or low-RAM feasibility is claimed.
