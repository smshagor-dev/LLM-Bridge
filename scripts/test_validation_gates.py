import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from check_m1_parity import compare

ROOT = Path(__file__).resolve().parents[1]


class ValidationGateTests(unittest.TestCase):
    def test_complete_generation_required(self):
        self.assertTrue(compare(b'Hello world\n', b'PromptHello world\n', b'Prompt'))
        for ours in [b'world\n', b'Hello\n', b'Hello world\nextra', b'', b'\n']:
            self.assertFalse(compare(ours, b'PromptHello world\n', b'Prompt'))
        self.assertFalse(compare(b'Hello\n', b'WrongHello\n', b'Prompt'))
        self.assertTrue(compare(b'Hello\n', b'Hello\n', b''))

    def test_failed_benchmark_is_a_failed_gate_with_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake = directory / 'fake'
            fake.write_text('#!/bin/sh\necho inference-failed >&2\nexit 7\n')
            fake.chmod(0o755)
            model = directory / 'model'; model.write_bytes(b'fixture')
            output = directory / 'results.json'
            result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/run_cpu_baseline.py'),
                                     '--binary', str(fake), '--model', str(model),
                                     '--output', str(output), '--repetitions', '1'], capture_output=True)
            self.assertEqual(result.returncode, 1)
            data = json.loads(output.read_text())
            self.assertFalse(data['valid'])
            self.assertEqual(data['runs'][0]['exit_code'], 7)
            self.assertIn('inference-failed', data['runs'][0]['stderr'])

    def test_missing_memory_is_not_a_valid_linux_trial(self):
        if sys.platform != 'linux':
            self.skipTest('Linux-specific measurement gate')
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake = directory / 'fake'
            fake.write_text('#!/bin/sh\necho generated-text\n')
            fake.chmod(0o755)
            output = directory / 'results.json'
            result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/run_cpu_baseline.py'),
                                     '--binary', str(fake), '--model', str(fake),
                                     '--output', str(output), '--repetitions', '1'], capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(output.read_text())['valid'])
