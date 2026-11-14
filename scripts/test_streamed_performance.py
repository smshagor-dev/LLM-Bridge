import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from benchmark_streamed import main, validate_result


class PerformanceProtocolTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {
            'total_ns': 2_000_000_000, 'first_logits_ready_ns': 900_000_000,
            'first_piece_ns': 910_000_000, 'sampling_steps': 3,
            'emitted_pieces': 3, 'emitted_bytes': 3, 'decode_intervals': 2,
            'decode_elapsed_ns': 1_000_000_000, 'decode_steps_per_second': 2.0,
            'range_callback_ns': 500_000_000, 'consumer_callback_ns': 100,
        }

    def result(self, metrics=None, code=0, output=b'abc\n'):
        payload = self.metrics if metrics is None else metrics
        return subprocess.CompletedProcess([], code, output,
            b'LLMBRIDGE_PERFORMANCE ' + json.dumps(payload).encode() + b'\n')

    def test_exact_target_boundary_and_exceeded_target(self):
        _, met = validate_result(self.result(), 8, 2.0, 910.0)
        self.assertTrue(met)
        self.assertFalse(validate_result(self.result(), 8, 2.001)[1])
        self.assertFalse(validate_result(self.result(), 8, target_first_ms=909.0)[1])

    def test_failed_or_missing_or_duplicate_metrics_cannot_pass(self):
        with self.assertRaises(ValueError):
            validate_result(self.result(code=1), 8)
        r = self.result()
        for stderr in [b'', r.stderr * 2]:
            with self.assertRaises(ValueError):
                validate_result(subprocess.CompletedProcess([], 0, b'abc\n', stderr), 8)

    def test_inconsistent_times_counts_and_nonfinite_rate_fail(self):
        for key, value in [('decode_intervals', 3), ('sampling_steps', 9),
                           ('decode_elapsed_ns', 2_000_000_000),
                           ('range_callback_ns', 2_000_000_001),
                           ('first_piece_ns', None), ('total_ns', True),
                           ('decode_steps_per_second', float('nan')),
                           ('emitted_bytes', 4), ('emitted_pieces', 0)]:
            p = copy.copy(self.metrics)
            p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_result(self.result(p), 8)

    def test_single_eog_step_does_not_invent_latency_or_throughput(self):
        p = dict(self.metrics, sampling_steps=1, emitted_pieces=0,
                 emitted_bytes=0, decode_intervals=0, decode_elapsed_ns=0,
                 first_piece_ns=None, decode_steps_per_second=None)
        r = self.result(p, output=b'\n')
        self.assertTrue(validate_result(r, 8)[1])
        self.assertFalse(validate_result(r, 8, target_rate=1)[1])
        self.assertFalse(validate_result(r, 8, target_first_ms=10000)[1])

    def test_truncated_output_cannot_pass(self):
        with self.assertRaises(ValueError):
            validate_result(self.result(output=b'ab\n'), 8)
        with self.assertRaises(ValueError):
            validate_result(self.result(output=b'abc'), 8)

    def test_one_slow_trial_fails_target_and_retains_all_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, model, output = [root / name for name in ('binary', 'model', 'report.json')]
            binary.write_bytes(b'binary')
            model.write_bytes(b'model')
            slow = dict(self.metrics, decode_elapsed_ns=2_000_000_000,
                        decode_steps_per_second=1.0, total_ns=3_000_000_000)
            trials = iter([self.result(), self.result(), self.result(slow)])
            def run(command, **_):
                if command[0] == 'git':
                    return subprocess.CompletedProcess(command, 0, '', '')
                if '--metrics' not in command:
                    return subprocess.CompletedProcess(command, 0, b'abc\n', b'')
                return next(trials)
            argv = ['benchmark_streamed.py', '--binary', str(binary), '--model', str(model),
                    '--output', str(output), '--target-decode-steps-per-second', '2']
            with patch('sys.argv', argv), patch('benchmark_streamed.subprocess.run', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report['pass'])
            self.assertEqual([t['target_met'] for t in report['trials']], [True, True, False])
            self.assertTrue(report['uninstrumented_output_matches'])

    def test_existing_report_is_preserved_before_any_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'existing'
            output.write_bytes(b'preserve')
            argv = ['benchmark_streamed.py', '--binary', 'missing', '--model', 'missing',
                    '--output', str(output)]
            with patch('sys.argv', argv), self.assertRaises(FileExistsError):
                main()
            self.assertEqual(output.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main()
