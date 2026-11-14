# Bounded streamed batch CLI

`llmbridge batch` runs newline-delimited UTF-8 prompts through one shared CPU/RAM-planned streamed executor. It creates a fixed set of worker threads and reads one bounded prompt per worker rather than loading the input file or creating one thread per line. The original fixed-slot executor still controls each admitted native session. Results are streamed directly to separate files; response bytes are never accumulated as an entire batch in RAM.

```sh
llmbridge batch model.gguf --input prompts.txt --output-dir new-results \
  --workers 2 --slot-mb 64 --os-reserve-mb 512 \
  --wait-timeout-ms 30000 --sha256 TRUSTED_MODEL_SHA256
```

The output directory must be new and its parent must exist. Each input line gets a stable one-based job number: `job-00000001.txt`, etc. Completion order may differ. Native token bytes are preserved exactly, without adding a trailing newline. Output begins as `.partial` using create-new semantics. A successful generation is published via an atomic, no-replace hard link and its partial name is removed; this requires filesystem hard-link support. Failed/cancelled work retains its partial file, and completed results remain available. No existing directory or output file is overwritten. Publication is not a power-loss durability guarantee.

Options are explicit key/value pairs:

| Option | Default | Meaning |
|---|---:|---|
| `--workers` | 1 | Requested workers, capped by the existing CPU/RAM plan |
| `--slot-mb` | 64 | Full cooperative charge per admitted session |
| `--os-reserve-mb` | 512 | Planner reserve outside session slots |
| `--wait-timeout-ms` | 30000 | Per-job admission wait, not native execution time; zero refuses |
| `--max-prompt-bytes` | 65536 | Maximum bytes per line, including LF/CRLF when present |
| `--max-tokens` | 16 | Native output-token limit, 1–2048 |
| `--ctx` | 128 | Native context, 64–8192 |
| `--sha256` | absent | Optional caller-supplied trusted model digest; verified per session |
| `--pressure-pause-mb` | absent | Optional pressure floor; must cover OS reserve plus one slot |
| `--pressure-resume-mb` | absent | Required with pause floor, strictly higher recovery threshold |

Pressure sampling, when selected, uses the existing OS/cgroup-aware source with a shared 100 ms minimum interval. Unknown headroom fails closed; queued calls recover using the existing hysteresis policy. Input is a local file. LF, CRLF and a final line without newline are supported; blank, invalid UTF-8 and oversized lines fail the batch. There is no total-job limit apart from the checked u64 numbering range and available storage; memory admission still uses conservative, finite slots and per-line limits. Separate prompt buffers, input buffering, worker stacks, filesystem and callbacks are outside the slot accounting, so this is not a physical RSS bound.

The first worker error is recorded before signalling shared cancellation. In-flight/waiting jobs observe that flag through the existing library cancellation path; new input stops. Ctrl+C signals the same flag. Worker-creation errors and caught worker panics also cancel/join existing workers. The command reports per-job progress on stderr, final completed/worker/pending/reserved counts on stdout, and exits nonzero on error/cancellation. Input I/O, synchronous sampling, native operations and scheduling can delay reaction; there is no hard cancellation/timeout return deadline. Already-completed files are preserved. There is no automatic resume/retry/checkpoint import or HTTP server.

`scripts/check_batch.py` executes the actual CLI with two pinned Q8 fixture jobs and compares exact output bytes against separate direct streamed `chat` calls. It also verifies existing-output preservation, bounded input refusal, wrong-digest refusal, pressure timeout cleanup and POSIX SIGINT cleanup. These are tiny-model functionality/byte-parity checks, not measured worker overlap, throughput, process memory or large-model feasibility. The library's separate concurrent-session and all-logit contracts retain their existing coverage.
