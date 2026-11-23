#!/usr/bin/env python3
"""Real CLI deadline checks, storing hashes/counts rather than prompt/output text."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def model_read_position(pid, model):
    process = Path(f'/proc/{pid}')
    try:
        for descriptor in (process / 'fd').iterdir():
            try:
                if descriptor.readlink() == model:
                    info = (process / 'fdinfo' / descriptor.name).read_text()
                    return int(info.splitlines()[0].split()[1])
            except (FileNotFoundError, PermissionError):
                continue
    except (FileNotFoundError, PermissionError):
        pass
    return 0


def check_active_planning(chat, model, root):
    # Leave the model's valid header and tensor locations intact. A descriptor
    # observed past the original EOF proves the whole-file hash pass started.
    padded = (root / 'padded.gguf').resolve()
    original_bytes = Path(model).stat().st_size
    shutil.copyfile(model, padded)
    with padded.open('r+b') as fixture:
        fixture.truncate(original_bytes + 512 * 1024 * 1024)
    trace = root / 'interrupted-plan.f32'
    for budget in [50, 100, 200, 400, 800, 1600, 3200]:
        command = chat('--auto-context', '--deadline-ms', str(budget), '--logits', str(trace), path=str(padded))
        furthest = 0
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            stop = time.monotonic() + 8
            while process.poll() is None and time.monotonic() < stop:
                furthest = max(furthest, model_read_position(process.pid, padded))
                if furthest > original_bytes:
                    break
                time.sleep(0.005)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
        if furthest > original_bytes:
            assert process.returncode == 1 and b'deadline expired' in stderr, stderr.decode(errors='replace')
            assert not stdout and not trace.exists() and b'LLMBRIDGE_PLAN ' not in stderr
            return {
                'pass': True,
                'selected_budget_ms': budget,
                'original_model_bytes': original_bytes,
                'fixture_bytes': padded.stat().st_size,
                'observed_file_position': furthest,
            }
    raise AssertionError('No adaptive planning run was observed during whole-file hashing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sha256', default='2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6')
    parser.add_argument('--probe-active-planning', action='store_true', help='Use Linux procfs to confirm expiry during planning authentication')
    args = parser.parse_args()
    binary, model = str(Path(args.binary).resolve()), str(Path(args.model).resolve())
    sha = lambda data: hashlib.sha256(data).hexdigest()
    assert sha(Path(model).read_bytes()) == args.sha256
    cases = {}
    with tempfile.TemporaryDirectory(prefix='llmbridge-cli-deadline-') as temp:
        root = Path(temp)

        def chat(*extra, path=model, prompt='The sky is'):
            return [binary, 'chat', path, '--prompt', prompt, '--backend', 'streamed', '--max-tokens', '4', '--sha256', args.sha256, *extra]

        def run(command):
            return subprocess.run(command, capture_output=True, timeout=60)

        settings = ['--cache-mb', '1', '--cache-pressure-mb', '8796093022208', '--metrics']
        plain = run(chat(*settings, '--logits', str(root / 'plain.f32')))
        timed = run(chat(*settings, '--deadline-ms', '30000', '--logits', str(root / 'timed.f32')))
        assert plain.returncode == timed.returncode == 0
        assert plain.stdout == timed.stdout
        plain_logits, timed_logits = [(root / n).read_bytes() for n in ['plain.f32', 'timed.f32']]
        assert plain_logits == timed_logits and plain_logits
        assert b'cache_release_events=1' in timed.stderr
        cases['future_chat_matches_bytes_logits_and_cache_release'] = {'pass': True, 'output_sha256': sha(timed.stdout), 'logits_sha256': sha(timed_logits)}

        adaptive_trace = root / 'adaptive.f32'
        adaptive = run(chat(*settings, '--auto-context', '--deadline-ms', '30000', '--logits', str(adaptive_trace)))
        assert adaptive.returncode == 0 and b'LLMBRIDGE_PLAN ' in adaptive.stderr, adaptive.stderr.decode(errors='replace')
        assert adaptive.stdout == plain.stdout and adaptive_trace.read_bytes() == plain_logits
        cases['future_verified_adaptive_plan_matches_bytes_and_logits'] = {'pass': True, 'output_sha256': sha(adaptive.stdout), 'logits_sha256': sha(plain_logits)}
        if args.probe_active_planning:
            cases['active_adaptive_plan_authentication_expiry'] = check_active_planning(chat, model, root)

        zero_trace = root / 'zero.f32'
        result = run(chat('--deadline-ms', '0', '--logits', str(zero_trace), path='/unopened-zero-deadline.gguf'))
        assert result.returncode == 1 and b'deadline expired' in result.stderr
        assert not result.stdout and not zero_trace.exists()
        cases['zero_chat_refuses_before_model_and_trace_creation'] = {'pass': True}
        result = run([binary, 'chat', '/unopened-resident-deadline.gguf', '--prompt', 'fixture', '--deadline-ms', '100'])
        assert result.returncode == 1 and b'requires streamed backend' in result.stderr and not result.stdout
        for invalid in ['-1', 'not-a-number', '18446744073709551616']:
            result = run(chat('--deadline-ms', invalid, path='/unopened-invalid-deadline.gguf'))
            assert result.returncode == 1 and b'invalid deadline-ms' in result.stderr and not result.stdout
        cases['unsupported_backend_and_invalid_chat_values_refuse'] = {'pass': True}

        protected = root / 'protected.f32'
        protected.write_bytes(b'preserve-existing-trace')
        result = run(chat('--deadline-ms', '30000', '--logits', str(protected)))
        assert result.returncode == 1 and protected.read_bytes() == b'preserve-existing-trace'
        assert not result.stdout
        cases['deadline_preserves_existing_logit_trace'] = {'pass': True}
        wrong_trace = root / 'wrong.f32'
        result = run(chat('--deadline-ms', '30000', '--sha256', '0' * 64, '--logits', str(wrong_trace)))
        assert result.returncode == 1 and b'SHA-256' in result.stderr
        assert wrong_trace.stat().st_size == 0 and result.stdout in [b'', b'\n']
        cases['timed_chat_wrong_digest_refuses_delivery'] = {'pass': True}

        active = None
        for budget in [1, 5, 10, 20, 50, 100, 250, 500, 1000, 2000]:
            trace = root / f'active-chat-{budget}.f32'
            result = run(chat('--max-tokens', '96', '--deadline-ms', str(budget), '--logits', str(trace)))
            expired = result.returncode == 1 and b'deadline expired' in result.stderr
            assert expired or result.returncode == 0, result.stderr.decode(errors='replace')
            if expired and trace.exists() and trace.stat().st_size > 0:
                active = {'pass': True, 'selected_budget_ms': budget, 'logits_bytes_before_expiry': trace.stat().st_size, 'partial_stdout_bytes': len(result.stdout), 'partial_stdout_sha256': sha(result.stdout)}
                break
        assert active, 'No deadline-expired chat demonstrated native logits before expiry'
        cases['active_native_chat_expiry'] = active

        source = root / 'input.txt'
        prompts = ['The sky is', 'Once upon a time']
        source.write_text('\n'.join(prompts) + '\n')
        def batch(name, *extra, path=model):
            return [binary, 'batch', path, '--input', str(source), '--output-dir', str(root / name), '--workers', '1', '--os-reserve-mb', '0', '--sha256', args.sha256, '--max-tokens', '4', *extra]
        result = run(batch('future-batch', '--deadline-ms', '30000'))
        assert result.returncode == 0 and b'completed=2 ' in result.stdout
        assert b'pending=0 reserved_bytes=0' in result.stdout
        for index, prompt in enumerate(prompts, 1):
            reference = run(chat(prompt=prompt))
            data = (root / 'future-batch' / f'job-{index:08}.txt').read_bytes()
            assert reference.returncode == 0 and data + b'\n' == reference.stdout
        cases['per_job_future_batch_matches_direct_bytes'] = {'pass': True}
        result = run(batch('wrong-timed-batch', '--deadline-ms', '30000', '--sha256', '0' * 64))
        assert result.returncode == 1 and b'SHA-256' in result.stderr
        assert not list((root / 'wrong-timed-batch').glob('*.txt'))
        assert b'pending=0 reserved_bytes=0' in result.stdout
        cases['timed_batch_wrong_digest_publishes_nothing'] = {'pass': True}

        for name, extra, expected in [
            ('zero-batch', ['--deadline-ms', '0'], b'deadline expired'),
            ('wait-first', ['--deadline-ms', '30000', '--wait-timeout-ms', '0'], b'admission wait timed out'),
            ('deadline-first', ['--deadline-ms', '20', '--wait-timeout-ms', '30000', '--pressure-pause-mb', '8796093022208', '--pressure-resume-mb', '8796093022209'], b'deadline expired'),
        ]:
            result = run(batch(name, *extra, path='/unopened-batch-deadline.gguf'))
            assert result.returncode == 1 and expected in result.stderr
            assert b'pending=0 reserved_bytes=0' in result.stdout
            assert not list((root / name).glob('*.txt'))
            assert all(p.stat().st_size == 0 for p in (root / name).glob('*.partial'))
        cases['batch_zero_and_independent_wait_deadline_cleanup'] = {'pass': True}
        for invalid in ['-1', 'not-a-number', '18446744073709551616']:
            result = run(batch('invalid-batch', '--deadline-ms', invalid))
            assert result.returncode == 1 and not (root / 'invalid-batch').exists()
        cases['invalid_batch_values_refuse_before_output_creation'] = {'pass': True}

        source.write_text(prompts[0] + '\n')
        active = None
        for budget in [1, 10, 50, 100, 250, 500, 1000, 2000]:
            name = f'active-batch-{budget}'
            result = run(batch(name, '--max-tokens', '96', '--deadline-ms', str(budget)))
            expired = result.returncode == 1 and b'deadline expired' in result.stderr
            assert expired or result.returncode == 0, result.stderr.decode(errors='replace')
            partials = list((root / name).glob('*.partial'))
            if expired and any(p.stat().st_size > 0 for p in partials):
                assert not list((root / name).glob('*.txt'))
                assert b'pending=0 reserved_bytes=0' in result.stdout
                active = {'pass': True, 'selected_budget_ms': budget, 'partial_bytes': sum(p.stat().st_size for p in partials), 'published_results': 0}
                break
        assert active, 'No deadline-expired batch demonstrated generated bytes before expiry'
        cases['active_batch_preserves_partial_without_publication'] = active
    report = {'pass': True, 'binary_sha256': sha(Path(binary).read_bytes()), 'model_sha256': args.sha256, 'cases': cases, 'note': 'Adaptive fixture budgets prove active CLI wiring, not maximum deadline latency, RSS, physical2GB or throughput.'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('CLI deadlines: chat bytes/logits parity, active expiry, per-job batch limits, protected partials and independent waiting timeout: PASS')


if __name__ == '__main__':
    main()
