#!/bin/sh
set -eu
# Only run when user explicitly provides the model. Never silently download weights.
test "$#" -eq 1 || { echo "usage: sh scripts/m1_smoke.sh /path/model.gguf" >&2; exit 2; }
cargo run --release -p llmbridge -- doctor
cargo run --release -p llmbridge -- models inspect "$1"
cargo run --release -p llmbridge -- chat "$1" --prompt "The sky is" --max-tokens 16 --ctx 128 --threads 2 --stream
