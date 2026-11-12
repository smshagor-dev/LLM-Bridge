#!/usr/bin/env python3
"""Independently parse a GGUF tensor table and verify the Rust CLI's exact range hashes.

This script uses Python only in CI and experiments, never in the production inference path.
"""
import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

PRIMITIVE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
LAYOUT = {
    0: (1, 4), 1: (1, 2),
    2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 40),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292),
    30: (1, 2),
}

def fixed(reader, fmt):
    size = struct.calcsize(fmt)
    raw = reader.read(size)
    if len(raw) != size:
        raise ValueError("truncated GGUF metadata")
    return struct.unpack("<" + fmt, raw)[0]

def string(reader):
    size = fixed(reader, "Q")
    if size > 4 * 1024 * 1024:
        raise ValueError("oversized metadata string")
    raw = reader.read(size)
    if len(raw) != size:
        raise ValueError("truncated metadata string")
    return raw.decode("utf-8")

def skip_value(reader, kind):
    if kind == 8:
        string(reader)
        return
    if kind == 9:
        elem_kind = fixed(reader, "I")
        count = fixed(reader, "Q")
        if elem_kind == 8:
            for _ in range(count):
                string(reader)
        else:
            reader.seek(count * PRIMITIVE[elem_kind], 1)
        return
    reader.seek(PRIMITIVE[kind], 1)

def parse(path):
    tensor_records = []
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("bad GGUF magic")
        version = fixed(f, "I")
        if version not in (2, 3):
            raise ValueError("unsupported GGUF version")
        count = fixed(f, "Q")
        kv_count = fixed(f, "Q")
        alignment = 32
        for _ in range(kv_count):
            key = string(f)
            kind = fixed(f, "I")
            if key == "general.alignment":
                if kind != 4:
                    raise ValueError("alignment type is not uint32")
                alignment = fixed(f, "I")
            else:
                skip_value(f, kind)
        for _ in range(count):
            name = string(f)
            rank = fixed(f, "I")
            dims = [fixed(f, "Q") for _ in range(rank)]
            kind = fixed(f, "I")
            offset = fixed(f, "Q")
            tensor_records.append((name, dims, kind, offset))
        data_offset = (f.tell() + alignment - 1) // alignment * alignment
    source_size = path.stat().st_size
    records = []
    for name, dims, kind, relative in tensor_records:
        block, size_per_block = LAYOUT[kind]
        elements = 1
        for dimension in dims:
            elements *= dimension
        if dims[0] % block or elements % block:
            raise ValueError("invalid quantization block geometry")
        bytes_count = (elements // block) * size_per_block
        absolute = data_offset + relative
        if absolute + bytes_count > source_size:
            raise ValueError("tensor range beyond file end")
        records.append({
            "name": name, "dims": dims, "type": kind,
            "offset": absolute, "size": bytes_count
        })
    return records

def run_binary(binary, *args):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True,
                            text=True, check=True, timeout=90)
    return result.stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.model).resolve()
    binary = Path(args.binary).resolve()
    actual = parse(path)
    cli = run_binary(binary, "models", "tensor-index", path)
    expected_count = re.search(r"^tensor count: (\d+)$", cli, re.MULTILINE)
    if expected_count is None or int(expected_count.group(1)) != len(actual):
        raise AssertionError("Rust tensor count differs from independent GGUF parser")
    for region in actual:
        line = "{} type={} dims={} offset={} size={}".format(
            region["name"], region["type"], region["dims"], region["offset"], region["size"])
        if line not in cli.splitlines():
            raise AssertionError("tensor byte region differs from independent parser: " + region["name"])
    chosen = actual[0]
    sha = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(chosen["offset"])
        remaining = chosen["size"]
        while remaining:
            block = f.read(min(65536, remaining))
            if not block:
                raise AssertionError("truncated file during independent tensor digest")
            sha.update(block)
            remaining -= len(block)
    expected_sha = sha.hexdigest()
    output = run_binary(binary, "models", "tensor-sha256",
                        path, chosen["name"], expected_sha)
    if ("sha256: " + expected_sha) not in output:
        raise AssertionError("Rust bounded tensor streaming hash mismatch")
    # Explicit wrong-hash test: the CLI must refuse instead of returning success.
    reject = subprocess.run([str(binary), "models", "tensor-sha256", str(path),
                             chosen["name"], "0" * 64], capture_output=True, text=True)
    if reject.returncode == 0:
        raise AssertionError("corrupted tensor digest was incorrectly accepted")
    result = {
        "model": path.name, "tensor_count": len(actual),
        "first_tensor": chosen, "first_tensor_sha256": expected_sha,
        "matches_all_tensor_regions": True,
        "verified_streamed_tensor_sha256": True,
        "rejects_bad_sha256": True,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("M2 tensor index / verified byte-range parity: PASS", len(actual), "tensors")

if __name__ == "__main__":
    main()
