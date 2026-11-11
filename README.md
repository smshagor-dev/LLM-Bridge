# LLM Bridge

Standalone, vendor-agnostic Rust/C++ inference runtime under active implementation.

**No 550B-on-2GB claim.** Standard mmap in llama.cpp is not proven memory-bounded streaming.

The M0 baseline includes bounded GGUF header inspection and a hardware resource probe. Native M1 inference uses pinned C++ CPU kernels, developed on `feat/m0-m1-cpu-inference`; see [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for exact validation limits.

Toolchain: Rust 1.82+, CMake 3.22+, C++20, Git. No weights, credentials or telemetry bundled.
