# Contributing

Start with AGENTS.md and IMPLEMENTATION_STATUS.md. Use topic branches and small reviewable PRs. A supported model claim needs the checkpoint hash, architecture, host profile, reproducible command, raw output, and trusted reference comparison. Tests and docs must describe exact limitations.

Minimum local checks: cargo fmt --all -- --check; cargo test --workspace --no-default-features; cargo clippy --workspace --all-targets --no-default-features -- -D warnings; cargo test --workspace --all-targets. Native tests require Git/CMake/C++20 and fetch pinned upstream source.

Never commit model weights, secrets, user data, private prompts or fabricated benchmark results.
