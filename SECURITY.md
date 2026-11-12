# Security policy

Report potential vulnerabilities privately to the maintainers through GitHub's security advisory workflow if enabled. Do not post exploitable issues or user prompts publicly.

M0/M1 is experimental. Untrusted model files may reach complex native parsers. Prefer offline model inspection and authorized checkpoints. The C ABI catches exceptions, and the Rust inspector limits header parsing; these safeguards do **not** constitute a sandbox. No unauthenticated network API is exposed. No telemetry or model uploads are implemented.

Before production releases: fuzz GGUF parsers, implement isolated workers with OS limits, signed/hash-verified manifests, C ABI audits, dependency/license scans, and platform memory-pressure tests.
