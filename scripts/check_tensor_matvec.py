#!/usr/bin/env python3
"""Independent Q8_0 reference versus actual native ggml row-streamed operator.

Loads one real GGUF tensor row at a time. No model-wide weight allocation,
no synthetic result, and no assumption that this proves full transformer
inference or process RSS bounds.
"""
import argparse
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path

from check_tensor_regions import parse

def reference(path, tensor):
    cols, rows = tensor["dims"]
    row_bytes = cols // 32 * 34
    outputs = []
    with path.open("rb") as f:
        f.seek(tensor["offset"])
        for _ in range(rows):
            data = f.read(row_bytes)
            if len(data) != row_bytes:
                raise AssertionError("independent reference saw truncated tensor")
            total = 0.0
            for block in range(0, row_bytes, 34):
                scale = struct.unpack_from("<e", data, block)[0]
                weights = struct.unpack_from("<32b", data, block + 2)
                total += sum(scale * w for w in weights)
            outputs.append(total)
    return outputs

def tensor_digest(path, tensor):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(tensor["offset"])
        remaining = tensor["size"]
        while remaining:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                raise AssertionError("unexpected EOF when verifying tensor SHA-256")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()

def actual(binary, path, tensor, expected_digest):
    result = subprocess.run(
        [str(binary), "models", "tensor-matvec", str(path), tensor["name"], expected_digest],
        capture_output=True, text=True, timeout=120, check=True,
    )
    rows = []
    bytes_read = None
    for line in result.stdout.splitlines():
        if line.startswith("row "):
            _, index, value = line.split()
            if int(index) != len(rows):
                raise AssertionError("native matrix row order differs")
            rows.append(float(value))
        if line.startswith("weight bytes read: "):
            bytes_read = int(line.removeprefix("weight bytes read: "))
    if bytes_read != tensor["size"]:
        raise AssertionError(f"unexpected native I/O count: {bytes_read} not {tensor['size']}")
    if len(rows) != tensor["dims"][1]:
        raise AssertionError("missing native output rows")
    return rows, bytes_read

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.model).resolve()
    binary = Path(args.binary).resolve()
    candidates = [
        t for t in parse(path)
        if t["type"] == 8 and len(t["dims"]) == 2
        and t["dims"][0] % 32 == 0
        and 0 < t["dims"][0] <= 16384 and 0 < t["dims"][1] <= 65536
    ]
    if not candidates:
        raise AssertionError("checkpoint has no supported Q8_0 matrix")
    tensor = min(candidates, key=lambda t: t["size"])
    expected = reference(path, tensor)
    digest = tensor_digest(path, tensor)
    observed, bytes_read = actual(binary, path, tensor, digest)
    differences = [abs(a - b) for a, b in zip(expected, observed)]
    for idx, (want, got) in enumerate(zip(expected, observed)):
        if not math.isfinite(got) or not math.isclose(want, got, rel_tol=1e-5, abs_tol=1e-4):
            raise AssertionError(f"row {idx}: expected {want}, native returned {got}")
    result = {
        "model": path.name, "tensor": tensor, "verified_tensor_sha256": digest,
        "rows": tensor["dims"][1],
        "columns": tensor["dims"][0], "bytes_read": bytes_read,
        "max_abs_difference": max(differences, default=0.0),
        "relative_tolerance": 1e-5, "absolute_tolerance": 1e-4,
        "native_ggml_q8_0_operator_matches_independent_reference": True,
        "limitations": "single streamed operator, not complete transformer execution or bounded process RSS",
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("native streamed Q8_0 matvec parity: PASS", tensor["name"], tensor["dims"])

if __name__ == "__main__":
    main()
