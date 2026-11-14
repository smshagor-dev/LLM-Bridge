# Measured streamed-session performance

`chat --backend streamed --metrics` now emits one `LLMBRIDGE_PERFORMANCE` JSON record after successful generation. Library callers opt in with `StreamedConfig.collect_performance`; the default is disabled. Collection uses a fixed-size set of counters, no per-token trace or model-sized allocation. It does add clock/counter overhead when enabled.

| Field | Measurement |
| --- | --- |
| `total_ns` | Wall time inside the generation call, including Rust index admission, optional authentication, native metadata/vocabulary loading, prefill, decode, callbacks and native cleanup |
| `first_logits_ready_ns` | Time from generation entry to the first logits callback, including admission/prefill |
| `first_piece_ns` | Time to the first nonempty piece callback entry; this is readiness, before the consumer writes/displays it |
| `sampling_steps` | Logits callbacks, including a possible final end-of-generation decision |
| `emitted_pieces`, `emitted_bytes` | Non-EOG piece callbacks and payload bytes; empty pieces count as pieces, not visible output |
| `decode_intervals`, `decode_elapsed_ns` | Intervals between first and last logits callbacks: N sampling steps give N−1 decode intervals |
| `decode_steps_per_second` | Intervals divided by their measured wall time; null when fewer than two steps or zero elapsed time |
| `range_callback_ns` | Aggregate bounded range callback time, including identity checks, cache lookup/refills, block authentication and copying |
| `consumer_callback_ns` | Aggregate user piece/logits callback time; includes CLI write/flush and optional logits export |

Decode intervals include consumer work between logits callbacks. Range time includes native metadata reads and cannot be subtracted from the total to claim isolated CPU compute time. Whole-file authentication before native generation contributes to total/first-readiness times but is outside range-callback time. CLI auto-planning before the generation call and child startup/teardown are outside `total_ns`; the benchmark reports separate `child_wall_ns`. Failed/cancelled sessions return errors and have no successful performance record. Memory observations remain in `LLMBRIDGE_MEMORY`.

## Reproducible workload measurements

```sh
python3 scripts/benchmark_streamed.py --binary target/release/llmbridge \
  --model models/stories15M-q8_0.gguf --output /tmp/session-performance.json \
  --repetitions 3 --address-space-mb 24 --cache-mb 1
```

The runner executes one uninstrumented output-validation session, followed by fresh measured child sessions. Every trial must exit successfully, provide a structurally consistent timing record and complete byte-counted output. Output hashes must match each other and the uninstrumented session. EOG and empty pieces do not invent visible-output latency or token throughput. Reports contain hashes of outputs, diagnostics and the prompt, configuration, source/binary/model hashes and every trial; they do not save prompt or response text. Existing output paths (including symlinks/models) are refused before trials.

Optional `--sha256 <independently-trusted-digest>` runs authenticated sessions. `--target-decode-steps-per-second N` and `--target-first-piece-ms N` check measured workload targets. **Every trial** must meet every requested target; failure/timeout/missing metrics or insufficient decode intervals fails the command after saving the report. A one-step EOG can be a valid completed session, but cannot pass a requested decode-speed or visible-piece target.

These are current workload observations. The filesystem page-cache state is uncontrolled, and the runner's bounded model hashing and output-validation session may warm caches. Separate child processes are not cold-storage experiments. There is no physical-disk-bandwidth result, instrumentation-overhead speed comparison, isolated compute rate, cross-model extrapolation, scheduler guarantee, large-model feasibility or real 2GB-hardware claim. No universal speed threshold is imposed in CI. Full automatic cache/thread scheduling remains future work.
