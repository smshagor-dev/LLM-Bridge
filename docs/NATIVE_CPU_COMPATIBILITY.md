# Native CPU admission

The default measured x86 native build explicitly enables SSE4.2, AVX, AVX2, FMA, F16C and BMI2. Low RAM or two CPU cores does not imply that those instruction sets are present. CPU-only inference is not a claim that this binary runs on every CPU.

`llmbridge doctor --json` reports the build profile, architecture, detected CPU/OS capabilities, missing capabilities and `native_compatible`. Plain `doctor` includes the same CPU diagnostic. These commands do not invoke native functions. A diagnostic command can exit successfully while reporting an incompatible native backend.

Rust's runtime feature checks include the OS extended-state requirements for AVX. Hardware CPUID bits alone are not sufficient. Cargo stamps the native build as `x86-avx2` on x86 targets; no-native binaries report `disabled`, and other architectures report `unvalidated` and refuse native execution. An explicit x86-64 `baseline` build instead requires SSE2; unknown profiles fail closed. A compatible report does not establish model support, available RAM or performance.

Safe Rust native generation, streamed generation (including authenticated sessions), native metadata probes and Q8 operator execution check compatibility before their first native call. CLI generation checks before model inspection/address-space changes. CPU incompatibility produces a normal error listing missing CPU/OS capabilities; there is no unvalidated scalar fallback. Pure Rust GGUF inspection/hashing remains available. Pre-existing cancellation/configuration errors can still take precedence.

This guard protects the supported Rust entry paths in a baseline-targeted Rust build. Direct C ABI consumers must perform their own CPU/OS admission. Custom compiler flags that raise the Rust binary's own minimum ISA can execute unsupported instructions before a runtime guard; this check cannot undo such build flags. The native SIMD flags and stamped profile must be updated together when adding a future build profile. The explicit SSE2 profile is covered by a separate emulated-inference protocol; ARM, dynamic SIMD dispatch and physical old-CPU tests remain pending.

## Emulated incompatible-host contract

```sh
cargo build --release --locked
cargo build --release -p llmbridge-core --features native --example cpu_guard_contract
python3 scripts/check_native_cpu.py --binary target/release/llmbridge \
  --contract target/release/examples/cpu_guard_contract \
  --emulator /usr/bin/qemu-x86_64 --output /tmp/native-cpu-guard.json
```

Install the distribution's `qemu-user` package to supply the emulator. The protocol runs the actual binaries with QEMU's `core2duo` CPU model. A second `Haswell,-xsave` diagnostic requires AVX to be unavailable without extended-state support. Doctor must start and report incompatibility; resident/streamed/authenticated generation and planning must return exit 1 with the CPU error and no output. A separate contract calls twelve Rust library paths directly, requires CPU rejection before native parsing/generation and verifies zero operator range reads. It does not rely solely on the CLI's guard. Emulator version/hash, binary hashes, exit codes and output hashes are recorded. Missing emulator, timeout, SIGILL, unexpected diagnostics or a failed library contract fails the command and is recorded.

Native real-model output/logit protocols on the compatible current host remain separate. Emulated rejection is not successful old-CPU inference, a physical Core 2 Duo/2GB test, performance measurement or proof for every CPU/OS combination.

For the explicit lower-ISA artifact, build/cache rules and positive inference evidence, see [baseline CPU builds](BASELINE_CPU_BUILD.md). The AVX2 negative-admission contract still requires an AVX2 binary and remains separate.
