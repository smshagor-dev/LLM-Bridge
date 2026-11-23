# Chat and batch execution deadline options

`--deadline-ms <u64>` opts into a cooperative monotonic deadline. Omit it to retain earlier behavior. Values are parsed as unsigned milliseconds and converted with `Instant::checked_add`; an unrepresentable instant is an error rather than an unlimited fallback. Zero means immediate expiry.

```sh
llmbridge chat model.gguf --backend streamed --prompt "The sky is" --deadline-ms 5000
llmbridge batch model.gguf --input prompts.txt --output-dir new-results \
  --wait-timeout-ms 1000 --deadline-ms 5000 --sha256 TRUSTED_MODEL_SHA256
```

| Command | Budget starts | Budget covers |
|---|---|---|
| Streamed `chat` | After argument validation, before model inspection/preflight | Preflight/planning time and the same absolute deadline in authenticated native execution |
| `batch` | Once per job, after reading its prompt and opening its partial output, immediately before executor admission | Queue/pressure waiting plus authenticated native execution; admission never restarts it |

Chat checks expiry before opening a model or creating a logits trace for zero budget, between preflight stages, and in the timed native API. With adaptive planning enabled, the same deadline is polled between Rust GGUF metadata fields, during bounded reads for trusted hashing, and during native metadata callbacks and validation. Resident chat rejects the option even with zero budget. Existing trusted SHA, cache release, logit trace create-new protection, CPU compatibility and process-admission guards remain active.

Batch computes a fresh deadline inside the worker loop for each job; this is not one global batch deadline. `--wait-timeout-ms` remains independent and waiting-only. A shorter waiting timeout reports admission timeout; a shorter whole-job deadline reports expiry. Expiry releases the job's ticket/slot through the library's existing cleanup. Batch records the first error, cooperatively cancels other work and joins workers. Completed files remain complete; failed jobs retain `.partial` data and are not published as `.txt`. A zero budget can therefore create an empty partial file, while still refusing model opening. CLI batch does not resume or retry failed work automatically.

Chat streams bytes to stdout and may leave a valid partial prefix on failure; an opted-in logit trace may also be partial. Its legacy trailing-newline behavior is retained when native generation has been entered. Neither CLI removes already-delivered bytes. No path preempts in-flight I/O, planning, native operations or consumers. The budget is not a hard wall-clock process-exit/maximum reaction-latency guarantee and does not enforce RSS.

`scripts/check_cli_deadlines.py` runs actual pinned Q8 CLI subprocesses. Future-deadline chat must match untimed complete stdout and every FP32 logit, including with trusted adaptive planning and cache release enabled. It checks zero-budget pre-open refusal, unsupported backend/invalid values, protected existing traces, and wrong-digest refusal. Adaptive fixture budgets require actual native logits before chat expiry and nonempty batch partial output before batch expiry, with zero published results and zero final pending/reserved charge. Positive per-job batch output matches direct chat. The protocol also checks waiting-timeout/deadline ordering under pressure. Selected fixture budgets exercise expiry on the test host.

On Linux, add `--probe-active-planning` to test expiry while authentication is in progress. The probe appends sparse zero padding to a temporary copy of the Q8 model while passing the original trusted digest. It requires procfs to show a read beyond the original EOF and verifies that expiry returns before publishing a plan, creating a logits trace, or writing to stdout.

```sh
python3 scripts/check_cli_deadlines.py --binary target/release/llmbridge \
  --model models/stories15M-q8_0.gguf --probe-active-planning \
  --output build/metadata-evidence/cli-deadlines.json
```
