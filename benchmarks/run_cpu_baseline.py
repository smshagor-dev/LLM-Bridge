#!/usr/bin/env python3
"""Capture raw end-to-end process results. No synthetic throughput estimates."""
import argparse
import json
import platform
import subprocess
import time
import re
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--binary", required=True)
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
p.add_argument("--repetitions", type=int, default=3)
args = p.parse_args()
if not 1 <= args.repetitions <= 30:
    p.error("repetitions must be 1..30")
command = [args.binary, "chat", args.model, "--prompt", "The sky is", "--max-tokens", "16", "--threads", "2", "--ctx", "128", "--metrics"]
runs = []
for index in range(args.repetitions):
    start = time.perf_counter_ns()
    result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)

    report = {"run": index + 1, "wall_seconds": (time.perf_counter_ns()-start)/1e9, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    metrics = re.search(r"^LLMBRIDGE_MEMORY (\{.*\})$", result.stderr, re.MULTILINE)
    if metrics:
        report["memory"] = json.loads(metrics.group(1))
    runs.append(report)
report = {"host": platform.platform(), "command": command, "runs": runs, "note": "Wall time includes startup/model load; VmHWM is process lifetime peak RSS; RSS/PSS and cgroup usage are sampled after run. Not tokens/s or end-to-end quality."}
Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(args.output)
