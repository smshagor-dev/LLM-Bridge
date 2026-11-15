#!/usr/bin/env python3
"""Subprocess checks for bounded batch CLI; reports hashes, never prompt/output text."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sha256', default='2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6')
    args = parser.parse_args()
    binary, model = str(Path(args.binary).resolve()), str(Path(args.model).resolve())
    digest = lambda data: hashlib.sha256(data).hexdigest()
    assert digest(Path(model).read_bytes()) == args.sha256
    cases = {}
    with tempfile.TemporaryDirectory(prefix='llmbridge-batch-') as tmp:
        root = Path(tmp)
        source = root / 'prompts.txt'
        prompts = ['The sky is', 'Once upon a time']
        source.write_text('\n'.join(prompts) + '\n', encoding='utf-8')

        def command(out, *extra):
            return [binary, 'batch', model, '--input', str(source), '--output-dir', str(root / out), '--workers', '2', '--os-reserve-mb', '0', '--max-tokens', '4', '--sha256', args.sha256, *extra]

        def execute(out, *extra):
            return subprocess.run(command(out, *extra), capture_output=True, timeout=60)

        result = execute('success')
        assert result.returncode == 0, result.stderr.decode(errors='replace')
        assert b'completed=2 ' in result.stdout and b'pending=0 reserved_bytes=0' in result.stdout
        outputs = []
        for index, prompt in enumerate(prompts, 1):
            data = (root / 'success' / f'job-{index:08}.txt').read_bytes()
            reference = subprocess.run([binary, 'chat', model, '--backend', 'streamed', '--prompt', prompt, '--max-tokens', '4', '--sha256', args.sha256], capture_output=True, timeout=60)
            assert reference.returncode == 0
            assert data + b'\n' == reference.stdout
            outputs.append(digest(data))
        assert not list((root / 'success').glob('*.partial'))
        cases['two_jobs_match_direct_bytes'] = {'pass': True, 'output_sha256': outputs, 'summary': result.stdout.decode().strip()}

        before = {p.name: digest(p.read_bytes()) for p in (root / 'success').iterdir()}
        result = execute('success')
        assert result.returncode == 1 and b'cannot create new output directory' in result.stderr
        assert before == {p.name: digest(p.read_bytes()) for p in (root / 'success').iterdir()}
        cases['existing_outputs_preserved'] = {'pass': True}

        result = execute('oversized', '--max-prompt-bytes', '4')
        assert result.returncode == 1 and b'prompt exceeds' in result.stderr
        assert not list((root / 'oversized').glob('*.txt'))
        assert b'pending=0 reserved_bytes=0' in result.stdout
        cases['oversized_input_fails_cleanly'] = {'pass': True}

        result = execute('invalid-prompt-limit', '--max-prompt-bytes', '65537')
        assert result.returncode == 1 and b'invalid batch' in result.stderr
        assert not (root / 'invalid-prompt-limit').exists()
        cases['native_prompt_limit_rejected_before_output_creation'] = {'pass': True}

        result = execute('wrong-hash', '--sha256', '0' * 64)
        assert result.returncode == 1
        assert not list((root / 'wrong-hash').glob('*.txt'))
        assert b'pending=0 reserved_bytes=0' in result.stdout
        cases['untrusted_model_publishes_no_result'] = {'pass': True}

        pressure = ['--pressure-pause-mb', '8796093022208', '--pressure-resume-mb', '8796093022209']
        result = execute('pressure-timeout', *pressure, '--wait-timeout-ms', '20')
        assert result.returncode == 1 and b'admission wait timed out' in result.stderr
        assert not list((root / 'pressure-timeout').glob('*.txt'))
        assert b'pending=0 reserved_bytes=0' in result.stdout
        cases['pressure_timeout_cleanup'] = {'pass': True}

        if os.name == 'posix':
            process = subprocess.Popen(command('cancelled', *pressure), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                start = time.monotonic()
                while not list((root / 'cancelled').glob('*.partial')):
                    assert process.poll() is None
                    assert time.monotonic() - start < 30
                    time.sleep(0.005)
                process.send_signal(signal.SIGINT)
                stdout, stderr = process.communicate(timeout=30)
                assert process.returncode == 1 and b'cancelled' in stderr
                assert b'pending=0 reserved_bytes=0' in stdout
                assert not list((root / 'cancelled').glob('*.txt'))
                cases['sigint_cleanup'] = {'pass': True}
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
    report = {'pass': True, 'binary_sha256': digest(Path(binary).read_bytes()), 'model_sha256': args.sha256, 'cases': cases, 'note': 'Real tiny-model CLI byte parity and failure cleanup, not measured physical memory/concurrency overlap or throughput.'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('batch CLI byte parity, bounded input, output protection, trust refusal, timeout and cancellation: PASS')


if __name__ == '__main__':
    main()
