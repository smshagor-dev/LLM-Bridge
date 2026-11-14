# LLM Bridge

**A standalone, vendor-agnostic Rust/C++ inference runtime exploring memory-aware execution of local large language models.**

LLM Bridge is an open engineering effort to make **local AI inference more accessible on machines with constrained RAM, CPU capacity, and optional or no GPU**. It is not tied to OpenMindAI, a cloud service, a specific GPU vendor, or a particular model provider.

> **Current reality:** Real CPU/GGUF generation and an **experimental complete streamed dense-Llama decoder** are implemented. The pinned 15M Q8 checkpoint completed reference-matched generation under a Linux 24 MiB address-space cap smaller than its model file. This is a tiny-model result; larger models, real 2 GB hardware, portable per-request limits, remote storage, API and desktop remain incomplete. See [streamed decoder validation](docs/STREAMED_DECODER.md).

## The problem we are solving

Running a large model locally is often limited by **model-weight storage, RAM residency, intermediate activations, KV cache, memory bandwidth, compute throughput, and operating-system pressure**. Downloading a model onto disk does not mean its inference workload fits in RAM. Likewise, memory mapping a model does not guarantee a bounded working set, acceptable latency, or survival under a tight memory limit.

Our research and engineering question is:

**Can an inference engine plan and execute model operations using explicitly budgeted tensor reads, reusable workspaces, adaptive scheduling, and measured memory admission, while preserving correct model outputs?**

The intended benefit is wider hardware access, including CPU-only systems. The goal is **not** to claim that arbitrary model sizes can run at useful speeds on arbitrary machines: disk I/O, bandwidth, computation, KV cache, and numerical correctness still impose real limits.

## What we want to build

1. **Portable runtime:** Rust control plane with a carefully versioned C ABI to native C++ kernels; CPU-first, with vendor-neutral acceleration considered only after measured CPU correctness.
2. **Model understanding:** Parse and validate model metadata, tensor types, shapes, quantization, offsets, alignment, file identity, and integrity before execution.
3. **Bounded tensor access:** Load only required file ranges into reusable, explicitly budgeted buffers; authenticate data when trusted hashes are supplied; reject corrupt or changing sources.
4. **Graph execution:** Connect streamed tensors to *real model graph operations*, including attention, feed-forward layers, and eventually complete prefill/decode, without silently reverting to unrestricted full-weight residency.
5. **Adaptive planning:** Admit or reject workloads based on available memory, CPU characteristics, context/KV-cache requirements, and measurable overhead; choose safe chunking and scheduling.
6. **Safe lifecycle:** Cancellation, transactional output behavior, cache isolation and eviction, failure containment across FFI, and deterministic resource cleanup.
7. **Reproducible proof:** Compare logits/tokens with trusted reference implementations; measure RSS/PSS, cgroup limits, latency, throughput, I/O, and behavior under constrained hardware.

These are **engineering targets**, not a list of completed features. See [AGENTS.md](AGENTS.md) for contribution rules.

## Architecture: current components and intended connection

```text
                  CLI / future applications
                            |
                  Rust control plane
             hardware / admission / cancellation
                            |
       GGUF index ---- source identity / integrity
                            |
        bounded local reader / tensor range provider
                            |
         verified chunk cache / reusable staging
                            |
          native C ABI ---- C++ / ggml kernels
                            |
              model graph / prefill / decode
                            |
                  tokens / logits / metrics
```

**Execution paths:** Normal generation defaults to the pinned llama.cpp resident backend. `--backend streamed` uses the experimental complete dense-Llama decoder with bounded Rust tensor reads, native ggml kernels and FP32 KV. It loads only tokenizer metadata through upstream's vocabulary-only path. The full cache/prefetch/scheduler integration shown above remains a target.

### Where to find the code

| Location | Responsibility |
| --- | --- |
| `crates/llmbridge-cli/` | CLI entry point and developer commands |
| `crates/llmbridge-core/src/inference.rs` | Resident/native inference and operator boundary |
| `crates/llmbridge-core/src/streamed.rs`, `native/src/streamed.cpp` | Experimental complete streamed dense-Llama decoder |
| `crates/llmbridge-core/src/gguf_index.rs` | GGUF metadata and tensor inventory |
| `crates/llmbridge-core/src/storage.rs` | Bounded local file reads and source identity |
| `crates/llmbridge-core/src/tensor_batch.rs` | Bounded indexed tensor-range operations |
| `crates/llmbridge-core/src/tensor_cache.rs` | Budgeted chunk cache and revision isolation |
| `crates/llmbridge-core/src/f32_stream.rs` | Streamed F32 matrix-vector operations |
| `crates/llmbridge-core/src/f32_chain.rs` | Two-stage F32 composition with caller-owned buffers |
| `crates/llmbridge-core/src/f32_residual.rs` | Streamed F32 residual composition |
| `crates/llmbridge-core/src/q8_batch.rs` | Native Q8 tensor operator experiments |
| `crates/llmbridge-core/src/memory_plan.rs` | Early memory/admission planning |
| `crates/llmbridge-core/src/runtime_memory.rs` | Process-memory measurement helpers |
| `docs/`, `benchmarks/`, `scripts/` | Validation protocols, benchmarks and reproducible checks |

Some components are experimental and may change as graph integration progresses. Code on open draft PR branches is **not** necessarily part of `main`.

## What works today vs. what remains open

| Area | Current state | Remaining proof or implementation |
| --- | --- | --- |
| M0 foundation | Rust workspace, CLI, native build and basic inspection | Continue portability and safety checks |
| M1 CPU baseline | Real GGUF tokenizer and native CPU generation; one small Q8 checkpoint matched pinned upstream greedy output in CI | Broader model families, cancellation and numerical validation |
| GGUF inventory | Indexed tensor shapes/types/offsets and checked ranges; independent 57-tensor comparison recorded | More malformed-format and compatibility coverage |
| Bounded storage | Range reader, source identity, optional integrity checks, cache budgeting | Connect reads to the **complete** native execution graph |
| Streamed decoder | Complete F32/Q8_0 dense-Llama path; pinned 15M output/logit and Linux address-space validation | Broader models, contexts, cache/prefetch and robustness |
| Memory accounting | Linux RSS/PSS and cgroup-related instrumentation | **Enforced** process memory bounds under real model loads |
| Adaptive planner | Early planning helpers | Full hardware-aware scheduling, KV-cache admission and benchmarks |
| Large-model low-RAM inference | **Not verified** | Correctness, OS survival, I/O and latency evidence on actual constrained hardware |
| GPU and other formats | **Not advertised as supported** | Separate implementation and reproducible validation |

**Evidence:** [M1 test protocol](docs/M1_TEST_PROTOCOL.md), [compatibility matrix](docs/MODEL_COMPATIBILITY.md), and [implementation status](IMPLEMENTATION_STATUS.md). An M1 CI parity result is **not** proof of M2 streaming or strict-RSS behavior.

## Build and run

**Prerequisites:** Pinned Rust 1.85.1 (declared library minimum 1.82), CMake 3.22+, a C++20 compiler, Git, and network access on the initial build (CMake FetchContent retrieves pinned llama.cpp v0.9.0).

```sh
cargo build --release -p llmbridge
cargo run --release -p llmbridge -- doctor
cargo run --release -p llmbridge -- hardware --json
cargo run --release -p llmbridge -- models inspect /models/model.gguf
cargo run --release -p llmbridge -- chat /models/model.gguf --prompt "The sky is" --max-tokens 16 --ctx 128 --threads 2 --stream
```

The current `chat` command performs **raw prompt continuation**, not chat-template-validated conversational inference. It uses native tokenization, attention, CPU kernels, and greedy sampling; it does not send prompts to a remote service. Terminal Ctrl+C requests cooperative cancellation through native polling. A single blocking native operation remains non-preemptible.

No model weights are bundled. For a known small GGUF model, its pinned hash, source, and reproducible test procedure, consult [M1 test protocol](docs/M1_TEST_PROTOCOL.md). Respect each model's license.

### Inspect a streamed memory plan

```sh
cargo run --release -p llmbridge -- plan /models/model.gguf --ctx 128 --max-ram-mb 64 --json
```

This budget is an incremental admission estimate, not a process-RSS cap. Context reduction requires `--auto-context`; generation prints its chosen plan. [Planning protocol and limits](docs/STREAMED_MEMORY_PLANNING.md).

### Experimental streamed generation

```sh
cargo run --release -p llmbridge -- chat /models/model.gguf --prompt "The sky is" \
  --backend streamed --ctx 128 --max-tokens 8 --workspace-mb 8 --metrics
```

On Linux, `--sha256 <trusted-digest>` binds streamed generation to authenticated blocks and sealed tokenizer metadata. `--verification-mb` bounds verification overhead. [Verified session protocol](docs/VERIFIED_MODEL_SESSIONS.md).

Optional `--cache-mb 1` enables bounded session-local block caching/read-ahead, with source identity and cancellation checks on hits. Default caching is disabled. [Measured cache behavior](docs/STREAMED_CACHE.md).

For Linux standalone CLI use, `--address-space-mb 24` enforces `RLIMIT_AS`. This is a virtual-address limit, distinct from an explicit workspace budget, cgroup memory or physical machine RAM. Only the pinned tiny checkpoint has been verified end to end. [Protocol, evidence and limits](docs/STREAMED_DECODER.md).

### Run checks before proposing changes

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Native dependencies and test coverage may vary by platform; consult CI workflows for the exact environment and required model-parity jobs. Do not treat a successful unit test as a full model-parity or low-memory proof.

## Roadmap and acceptance criteria

**M2 — current priority:** Build on the experimental full-decoder checkpoint: extend verified sessions and bounded caches, prefetch, scheduling, portable worker limits, broader correctness and memory validation. The current Linux/tiny-model proof does **not** complete every M2 or product requirement.

**M3 — adaptive hardware/memory planning:** Calculate feasible context, KV-cache, activation and staging budgets; choose execution strategies based on measured hardware; fail safely when a workload cannot be admitted.

**Later milestones:** Expand execution coverage, supported model families and quantization, optional acceleration, integration interfaces, and production hardening. Their detailed requirements come from the project master specification and must be validated individually; this README does not claim they are complete.

### What would count as convincing evidence?

- **Correctness:** Same model checksum, tokenizer, prompt, sampling settings and reference version; report logits/tokens, tolerance where applicable, and mismatches.
- **Memory:** Record current and peak RSS/PSS, cgroup constraints, allocator/staging budgets, KV cache, OS and available RAM; demonstrate behavior under an enforced limit, not only an estimate.
- **Performance:** Record hardware, model size/quantization, context, cold/warm cache, first-token latency, tokens/sec, disk reads, and repeated trials.
- **Robustness:** Exercise cancellation during actual reads/compute, corrupt or replaced files, short reads, cache invalidation, malformed metadata, and FFI failures.
- **Reproducibility:** Pin source revisions and model hashes, publish commands and raw results, and clearly separate verified observations from hypotheses.

## Contributing

Contributions should advance **working, testable runtime functionality**, not only scaffolding or claims.

1. Read [AGENTS.md](AGENTS.md), [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md), and the relevant module/tests.
2. Check existing issues and open PRs to avoid duplicate work; branch from current `main`.
3. Implement a small reviewable slice with regression tests and, where possible, a reference comparison.
4. Run relevant checks and include **actual commands and results** in the PR. If tests were not run, say so.
5. Open a **draft PR** while validation is incomplete. Do not merge untested work or publish unsupported performance/model-support claims.

For architecture questions, prefer concrete proposals specifying tensor flow, buffer ownership, cancellation boundaries, resource limits, and independent validation.

## Project boundaries

- **Independent project:** Not an OpenMindAI-only feature or cloud-backed inference service.
- **CPU-first:** No assumption of NVIDIA, CUDA, GPU availability, or a particular model vendor.
- **Security:** Untrusted model files require strict parsing, bounds checks, identity/integrity validation and safe native boundaries.
- **Honest claims:** `mmap` is not out-of-core proof; a small-model CI parity result is not large-model parity; a logical chunk-cache cap is not a process RSS cap.
- **Upstream:** The baseline C++ backend is pinned to [llama.cpp v0.9.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.9.0), commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8`, behind a versioned Rust/C ABI.

**Status of this README:** Describes the repository's direction and confirmed checkpoints, not a claim that all roadmap capabilities are available on `main`.
