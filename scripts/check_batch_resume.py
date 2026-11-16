#!/usr/bin/env python3
"""Real pinned-model interruption/resume and checkpoint integrity protocol (Linux)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
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
    assert os.name == 'posix' and Path('/proc/self/exe').exists(), 'Linux checkpoint protocol required'
    binary, model = str(Path(args.binary).resolve()), str(Path(args.model).resolve())
    digest = lambda b: hashlib.sha256(b).hexdigest()
    assert digest(Path(model).read_bytes()) == args.sha256
    cases = {}
    with tempfile.TemporaryDirectory(prefix='llmbridge-batch-resume-') as temp:
        root = Path(temp)
        source = root / 'input.txt'
        prompts = ['The sky is', 'Once upon a time', 'A little bird']
        source.write_text('\n'.join(prompts) + '\n')
        def command(out, mode, *extra, path=model, inp=source):
            return [binary, 'batch', path, '--input', str(inp), '--output-dir', str(root / out), '--workers', '1', '--os-reserve-mb', '0', '--max-tokens', '64', '--sha256', args.sha256, mode, *extra]
        def run(cmd):
            return subprocess.run(cmd, capture_output=True, timeout=180)
        def contents(directory):
            return {p.name: digest(p.read_bytes()) for p in directory.iterdir() if p.is_file()}
        def wait(process, condition):
            start = time.monotonic()
            while not condition():
                assert process.poll() is None, 'batch exited before interruption point'
                assert time.monotonic() - start < 120, 'batch never reached interruption point'
                time.sleep(0.005)

        process = subprocess.Popen(command('results', '--resumable'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            result_dir = root / 'results'
            wait(process, lambda: bool(list(result_dir.glob('*.done'))) and any(p.stat().st_size > 0 for p in result_dir.glob('job-????????.partial')))
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=120)
            assert process.returncode == 1 and b'cancelled' in stderr
            done = sorted(result_dir.glob('*.done'))
            assert 0 < len(done) < len(prompts)
            completed = {p.name: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) for p in result_dir.iterdir() if p.suffix in ['.done', '.txt']}
            partials = {p.name: p.read_bytes() for p in result_dir.glob('job-????????.partial')}
            assert any(partials.values())
            before_done = len(done)
        finally:
            if process.poll() is None:
                process.kill(); process.communicate()
        resumed = run(command('results', '--resume', '--workers', '2', '--wait-timeout-ms', '60000', '--deadline-ms', '30000'))
        assert resumed.returncode == 0, resumed.stderr.decode(errors='replace')
        assert f'skipped={before_done}'.encode() in resumed.stdout
        assert b'pending=0 reserved_bytes=0' in resumed.stdout
        for name, (data, inode, modified) in completed.items():
            p = result_dir / name
            assert p.read_bytes() == data and p.stat().st_ino == inode and p.stat().st_mtime_ns == modified
        for name, data in partials.items():
            archives = list(result_dir.glob(name.replace('.partial', '.retry-*.partial')))
            assert archives and any(p.read_bytes() == data for p in archives)
        for index, prompt in enumerate(prompts, 1):
            reference = run([binary, 'chat', model, '--backend', 'streamed', '--prompt', prompt, '--max-tokens', '64', '--sha256', args.sha256])
            assert reference.returncode == 0
            assert (result_dir / f'job-{index:08}.txt').read_bytes() + b'\n' == reference.stdout
        cases['interrupted_native_batch_resumes_verified_jobs'] = {'pass': True, 'skipped_existing_jobs': before_done, 'total_completed_jobs': len(prompts), 'completed_files_unchanged': True, 'unfinished_partials_archived': True}

        before = contents(result_dir)
        all_done = run(command('results', '--resume', '--deadline-ms', '0'))
        assert all_done.returncode == 0 and b'completed=0 ' in all_done.stdout and b'skipped=3' in all_done.stdout
        assert contents(result_dir) == before
        cases['fully_completed_resume_is_read_only'] = {'pass': True}

        def reject_copy(name, mutate, *extra, path=model, inp=source, engine=binary):
            directory = root / name
            shutil.copytree(result_dir, directory)
            mutate(directory)
            before = contents(directory)
            arguments = command(name, '--resume', *extra, path=path, inp=inp)
            arguments[0] = engine
            result = run(arguments)
            assert result.returncode == 1, name
            assert contents(directory) == before, name
            return result
        result = reject_copy('wrong-config', lambda _: None, '--ctx', '256')
        assert b'identity mismatch' in result.stderr
        changed_input = root / 'changed-input.txt'; changed_input.write_text(source.read_text() + 'changed\n')
        result = reject_copy('wrong-input', lambda _: None, inp=changed_input)
        assert b'identity mismatch' in result.stderr
        wrong_model = root / 'changed-model.gguf'; shutil.copy2(model, wrong_model)
        stat = wrong_model.stat()
        with wrong_model.open('r+b') as f:
            f.seek(-1, os.SEEK_END); value = f.read(1); f.seek(-1, os.SEEK_END); f.write(bytes([value[0] ^ 1]))
        os.utime(wrong_model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        result = reject_copy('wrong-model', lambda _: None, path=str(wrong_model))
        assert b'SHA-256' in result.stderr
        cases['changed_model_input_configuration_refuse_unchanged'] = {'pass': True, 'model_size_mtime_preserved': True}
        changed_binary = root / 'changed-binary'
        shutil.copy2(binary, changed_binary)
        copied_command = command('results', '--resume')
        copied_command[0] = str(changed_binary)
        copied_result = run(copied_command)
        assert copied_result.returncode == 0 and b'skipped=3' in copied_result.stdout
        cases['identical_binary_copy_can_resume'] = {'pass': True}
        with changed_binary.open('ab') as executable:
            executable.write(b'\x00')  # A different ELF artifact with the same executable code.
        result = reject_copy('wrong-binary', lambda _: None, engine=str(changed_binary))
        assert b'identity mismatch' in result.stderr
        cases['changed_binary_artifact_refuses_unchanged'] = {'pass': True}

        def corrupt_output(directory):
            path = directory / 'job-00000001.txt'; data = path.read_bytes(); assert data
            stat = path.stat(); path.write_bytes(bytes([data[0] ^ 1]) + data[1:]); os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        result = reject_copy('corrupt-output', corrupt_output)
        assert b'receipt mismatch' in result.stderr
        result = reject_copy('corrupt-receipt', lambda d: (d / 'job-00000001.done').write_bytes(b'bad receipt'))
        assert b'receipt mismatch' in result.stderr
        result = reject_copy('orphan-result', lambda d: (d / 'job-00000001.done').unlink())
        assert b'unrecorded final' in result.stderr
        cases['corrupt_and_orphan_results_never_overwritten'] = {'pass': True, 'output_size_mtime_preserved': True}

        lock_input = root / 'lock-input.txt'; lock_input.write_text('The sky is\n')
        pressure = ['--pressure-pause-mb', '8796093022208', '--pressure-resume-mb', '8796093022209']
        process = subprocess.Popen(command('locked', '--resumable', '--max-tokens', '4', *pressure, inp=lock_input), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            wait(process, lambda: (root / 'locked' / 'job-00000001.partial').exists())
            result = run(command('locked', '--resume', '--max-tokens', '4', inp=lock_input))
            assert result.returncode == 1 and b'locked' in result.stderr
            process.kill(); process.communicate(timeout=30)
        finally:
            if process.poll() is None:
                process.kill(); process.communicate()
        expired = run(command('locked', '--resume', '--max-tokens', '4', '--deadline-ms', '0', inp=lock_input))
        assert expired.returncode == 1 and b'deadline' in expired.stderr
        assert not (root / 'locked' / 'job-00000001.txt').exists()
        result = run(command('locked', '--resume', '--max-tokens', '4', '--deadline-ms', '30000', inp=lock_input))
        assert result.returncode == 0 and b'completed=1 ' in result.stdout
        cases['exclusive_lock_and_sigkill_release'] = {'pass': True, 'expired_deadline_retry': True}

        no_digest = [binary, 'batch', model, '--input', str(source), '--output-dir', str(root / 'missing-digest'), '--resumable']
        result = run(no_digest)
        assert result.returncode == 1 and b'trusted --sha256' in result.stderr
        assert not (root / 'missing-digest').exists()
        cases['checkpoint_requires_trusted_digest'] = {'pass': True}
    report = {'pass': True, 'binary_sha256': digest(Path(binary).read_bytes()), 'model_sha256': args.sha256, 'cases': cases, 'note': 'Real Linux interruption/resume and integrity protocol; not signed records, adversarial concurrent-write resistance, power-loss or physical-memory/throughput proof.'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('verified batch resume: native interruption, skip integrity, partial retry, model/input/config checks, output protection and kernel lock recovery: PASS')


if __name__ == '__main__':
    main()
