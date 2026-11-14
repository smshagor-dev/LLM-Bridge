#!/usr/bin/env python3
"""Exercise CPU admission under a genuinely incompatible emulated x86 CPUID."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(65536), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--emulator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    def git(*parts):
        return subprocess.run(['git', '-C', str(repository), *parts], capture_output=True,
                              text=True, check=True).stdout.strip()
    source = {'commit': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}'),
              'working_tree_dirty': bool(git('status', '--porcelain'))}
    report = {'pass': False, 'source': source, 'guest_cpu': 'core2duo', 'cases': {},
              'note': 'QEMU user-mode negative admission test; no inference speed, old physical CPU or 2GB-hardware claim.'}
    with args.output.open('x') as destination:
        try:
            binary, contract, emulator = [str(p.resolve()) for p in (args.binary, args.contract, args.emulator)]
            report['binary_sha256'] = sha256(args.binary)
            report['contract_sha256'] = sha256(args.contract)
            report['emulator_sha256'] = sha256(args.emulator)
            report['emulator_version'] = subprocess.run([emulator, '--version'], capture_output=True,
                text=True, check=True, timeout=30).stdout.splitlines()[0]
            def run(name, executable, arguments, guest_cpu='core2duo', environment=None):
                result = subprocess.run([emulator, '-cpu', guest_cpu, executable, *arguments],
                                        capture_output=True, check=False, timeout=30, env=environment)
                report['cases'][name] = {'exit_code': result.returncode, 'guest_cpu': guest_cpu,
                    'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(),
                    'stderr_sha256': hashlib.sha256(result.stderr).hexdigest()}
                return result
            doctor = run('doctor', binary, ['doctor', '--json'])
            cpu = json.loads(doctor.stdout)['cpu']
            report['emulated_cpu'] = cpu
            assert doctor.returncode == 0 and cpu['native_profile'] == 'x86-avx2'
            assert not cpu['native_compatible'] and 'avx2' in cpu['missing_cpu_os_capabilities']
            spoofed = run('runtime_profile_cannot_override_build', binary, ['doctor', '--json'],
                          environment={**os.environ, 'LLMBRIDGE_CPU_PROFILE': 'baseline'})
            spoofed_cpu = json.loads(spoofed.stdout)['cpu']
            assert spoofed.returncode == 0 and spoofed_cpu['native_profile'] == 'x86-avx2'
            assert not spoofed_cpu['native_compatible']
            no_xsave = run('avx_without_os_state', binary, ['doctor', '--json'], 'Haswell,-xsave')
            no_xsave_cpu = json.loads(no_xsave.stdout)['cpu']
            report['avx_without_os_state_cpu'] = no_xsave_cpu
            assert no_xsave.returncode == 0 and not no_xsave_cpu['native_compatible']
            assert not no_xsave_cpu['capabilities']['avx']
            for name, command in [
                ('resident', ['chat', '/absent-cpu-fixture.gguf', '--prompt', 'fixture']),
                ('streamed', ['chat', '/absent-cpu-fixture.gguf', '--prompt', 'fixture', '--backend', 'streamed']),
                ('verified_streamed', ['chat', '/absent-cpu-fixture.gguf', '--prompt', 'fixture', '--backend', 'streamed', '--sha256', '0' * 64]),
                ('plan', ['plan', '/absent-cpu-fixture.gguf'])]:
                result = run(name, binary, command)
                assert result.returncode == 1 and not result.stdout
                assert b'native CPU incompatible' in result.stderr
            result = run('ten_library_paths', contract, [])
            assert result.returncode == 0
            assert b'ten Rust native execution/probe paths rejected before FFI: PASS' in result.stdout
            report['pass'] = True
        except (OSError, subprocess.SubprocessError, ValueError, AssertionError, KeyError, IndexError) as error:
            report['error'] = type(error).__name__ + ': ' + str(error)
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps({'pass': report['pass'], 'report': str(args.output)}))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
