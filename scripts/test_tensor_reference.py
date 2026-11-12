#!/usr/bin/env python3
"""Unit checks for the independent Q8_0 numerical reference used in CI.

Run from the repository root:
    python3 -m unittest discover -s scripts -p 'test_tensor_reference.py'
"""
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).with_name("check_tensor_matvec.py")
_SPEC = importlib.util.spec_from_file_location("check_tensor_matvec", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class IndependentQ8ReferenceTests(unittest.TestCase):
    def test_signed_weights_and_half_precision_scales(self):
        # One 34-byte Q8_0 block per row, including signed int8 extremes.
        rows = [
            struct.pack("<e32b", 0.5, *([-128, 127] + [1] * 30)),
            struct.pack("<e32b", -0.25, *([4] * 32)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensor.bin"
            path.write_bytes(b"HEADER" + b"".join(rows))
            tensor = {"dims": [32, 2], "offset": 6, "size": 68}
            values = _MODULE.reference(path, tensor)
            self.assertEqual(values, [0.5 * (-128 + 127 + 30), -0.25 * (4 * 32)])

    def test_truncated_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.bin"
            path.write_bytes(struct.pack("<e32b", 1.0, *([1] * 32)))
            with self.assertRaisesRegex(AssertionError, "truncated tensor"):
                _MODULE.reference(path, {"dims": [32, 2], "offset": 0, "size": 68})

    def test_digest_is_limited_to_tensor_region(self):
        import hashlib
        block = struct.pack("<e32b", 1.0, *([1] * 32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(b"PREFIX" + block + b"TRAILER")
            tensor = {"dims": [32, 1], "offset": 6, "size": 34}
            self.assertEqual(_MODULE.tensor_digest(path, tensor), hashlib.sha256(block).hexdigest())


if __name__ == "__main__":
    unittest.main()
