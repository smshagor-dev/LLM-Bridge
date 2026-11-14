# Native-derived streamed memory planning

`llmbridge plan <model.gguf> --max-ram-mb 64 --json` inspects a supported dense Llama GGUF and returns an explainable CPU plan. It reuses the native decoder's metadata, tensor-shape and workspace admission code. It does not tokenize prompts, read model weight payloads, allocate activations/KV, verify publisher identity, or run inference. Native callback reads are restricted to the validated GGUF metadata prefix. Unsupported shapes/quantization/options fail with the same decoder restrictions.

The JSON distinguishes native workspace, FP32 KV, range-reader buffer, cache/verification reservations, estimated metadata expansion and runtime reserve. Its `workspace_fixed_bytes + context * workspace_per_token_bytes` equals the decoder's workspace admission boundary. GQA/MQA KV cost comes from actual layers, key/value heads and head width, rather than a generic per-token constant. The current streamed decoder is serial (`selected_threads: 1`); reporting hardware logical CPUs is not a claim of adaptive parallel execution.

## Budget meaning

`--max-ram-mb` limits **estimated additional allocations/reservations**, not total process RSS, virtual size or physical machine RAM. It intersects measured available RAM after OS reserve. Defaults include a 512 MiB OS reserve, a 32 MiB runtime/allocator/tokenizer uncertainty reserve, twice the metadata prefix size, and the full configured cache/verification budgets. The native workspace part is derived from actual native admission math; the runtime and metadata reserves are estimates. A conservative plan can refuse a job that might run manually. Do not treat a successful plan as proof that every native allocation fits or that a 2GB machine survives.

Use `chat --address-space-mb` for separate Linux RLIMIT_AS enforcement. If auto-planning is combined with that limit, the planner also intersects its incremental budget with remaining virtual-address headroom sampled **after** metadata inspection. Unknown virtual size fails closed. This extra conservatism may reject a 24 MiB auto-planned job even when the manually configured tiny model passed the separate 24 MiB experiment.

`plan --reserve-verification --verification-mb 8` reserves verification overhead, but does not verify model content. JSON explicitly says `model_integrity: not_verified_by_plan` and `process_limit_enforced_by_plan: false`. `chat --auto-context --sha256 <trusted-hash>` authenticates blocks before native plan inspection, then performs a fresh authenticated generation admission. That can require two bounded whole-file hash passes; it never silently substitutes an unauthenticated native inspection.

## Context selection

A request is constrained by model training context, native workspace ceiling, available RAM, configured reserves and optional incremental budget. The minimum supported context is 64. Requests fail if the minimum cannot fit. A smaller context is selected only with explicit `--auto-context`. Generation prints the applied JSON plan as `LLMBRIDGE_PLAN` on stderr before running. Prompts are **not** silently truncated; actual tokenizer/context checks still reject a prompt/response that does not fit the selected context.

```sh
llmbridge plan model.gguf --ctx 256 --auto-context --max-ram-mb 64 --json
llmbridge chat model.gguf --backend streamed --prompt 'The sky is' \
  --ctx 256 --auto-context --max-ram-mb 64 --metrics
```

The current native planner probes the supported graph at 64 tokens without allocating runtime state, derives linear workspace coefficients, and uses those coefficients to find the largest admissible context. The native constructor still independently checks actual execution inputs/budget. File changes or resource-pressure changes after a plan are not trusted as permanent admission guarantees.

## Validation protocol

```sh
cargo run --release -p llmbridge-core --features native --example streamed_plan_contract \
  -- models/stories15M-q8_0.gguf
python3 scripts/check_streamed_plan.py --binary target/release/llmbridge \
  --model models/stories15M-q8_0.gguf --output benchmarks/results/streamed_plan.json
```

The real-model contract asks for 128 tokens with a native workspace budget calculated for 96, requires selection of exactly 96, executes generation at that exact budget, then requires rejection with no token/logit delivery at one byte less. Its two-core/2GB profile is synthetic admission input, not a real 2GB-host experiment. The CLI protocol checks model-context reduction only with opt-in, too-small workspace/reserve rejection, adaptive/manual output agreement, trusted-hash adaptive mode and wrong-hash refusal. Independent reference output/logit protocols remain in CI. Broader hardware/model tiers, bandwidth/compute feasibility, dynamic thread selection and full M3 scheduling remain incomplete.

## Reproducible CPU baseline

The Rust native build and CI reference explicitly disable host-specific `-march=native` and select the same upstream SSE4.2/AVX/AVX2/FMA/F16C/BMI2 x86 target flags. This measured x86-64 build requires those CPU capabilities; it is not a generic pre-AVX2 binary. Earlier CI cached runs hit `std::bad_alloc` under 24 MiB on a host-specific AMX-capable runner even though local runs passed. Different instruction-set builds change mapped binary size and allocator headroom. This build repair removes that host-dependent optimization input; the process ceiling stays 24 MiB and all real-model checks must still pass. It does not establish a portable whole-process RSS guarantee or a universal CPU compatibility result. Generic pre-AVX2 and other SIMD build profiles still need independent numerical and memory validation.

A preliminary scalar-only profile preserved generated bytes but exceeded the existing logits tolerance on one of the three prompts. That profile is not treated as validated; the comparison threshold was not relaxed. The validated-profile work therefore uses matching explicit AVX2 build flags, with generic pre-AVX2 numerical support still pending.

## Adaptive cache admission

`plan --auto-cache` and `chat --backend streamed --auto-cache` treat the cache budget as an optional ceiling. The CLI ceiling defaults to 1 MiB; an explicit `--cache-mb 0` disables it and `--cache-mb N` sets its maximum (up to 32 MiB). Library callers set `StreamedPlanRequest.adapt_cache` with their explicit ceiling.

The policy admits context/workspace and required reader, metadata, runtime, OS and verification reserves first. It then selects the largest complete cache-slot allocation within the remaining incremental budget and ceiling, including staging and slot metadata. Less than staging plus one block selects zero. Cache is never a reason to shrink context. A context that cannot fit still fails unless separately authorized by `--auto-context`. Manual cache reservations remain mandatory.

JSON reports selected `cache_budget_bytes`, requested `cache_ceiling_bytes` and `cache_adaptive`; generation uses the selected byte budget, including non-MiB values. Planner and real cache construction share allocation geometry. The existing source-identity, cancellation and authenticated-block checks remain on every cache path. Required verification reservations are never traded for cache.

This is a deterministic memory-admission policy, not online cache resizing, throughput optimization, asynchronous prefetch, bandwidth-aware scheduling or proof that more cache is faster. Estimates still do not enforce total RSS; separate Linux limits can reject allocations. The CLI's conservative preliminary host admission can also refuse before adaptive inspection.
