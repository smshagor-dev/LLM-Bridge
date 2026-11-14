#!/usr/bin/env python3
"""Real derived Q4_0 inventory/admission plus unsupported Q4_1 refusal."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from check_streamed_parity import validate_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--unsupported-model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    validate_checkpoint('stories15M-derived-q4_0', hashlib.sha256(args.model.read_bytes()).hexdigest())
    report = {'pass': False, 'cases': {}, 'note': 'Format/admission contract; no language-quality or memory-cap proof.'}
    with args.output.open('x') as output:
        def run(name, *command):
            r = subprocess.run([str(args.binary.resolve()), *map(str, command)],
                               capture_output=True, timeout=90, check=False)
            report['cases'][name] = {'exit_code': r.returncode,
                'stdout_sha256': hashlib.sha256(r.stdout).hexdigest(),
                'stderr_sha256': hashlib.sha256(r.stderr).hexdigest()}
            return r
        try:
            inventory = run('tensor_inventory', 'models', 'tensor-index', args.model)
            types = [int(t) for t in re.findall(rb' type=(\d+) dims=', inventory.stdout)]
            assert inventory.returncode == 0 and types.count(2) == 44 and types.count(0) == 13
            assert len(types) == 57
            report['tensor_types'] = {'Q4_0': 44, 'F32': 13}
            planned = run('adaptive_plan', 'plan', args.model, '--auto-cache', '--json')
            assert planned.returncode == 0
            report['plan'] = json.loads(planned.stdout)
            assert report['plan']['selected_context'] == 128
            # Ensure the negative fixture really is an indexed unsupported format,
            # rather than an unreadable file or a bad-checkpoint/hash error.
            wrong = run('unsupported_inventory', 'models', 'tensor-index', args.unsupported_model)
            assert wrong.returncode == 0 and b'type=3 dims=' in wrong.stdout
            for name, command in [('unsupported_plan', ['plan', args.unsupported_model]),
                                  ('unsupported_execution', ['chat', args.unsupported_model,
                                   '--backend', 'streamed', '--prompt', 'fixture', '--max-tokens', '8'])]:
                r = run(name, *command)
                assert r.returncode == 1 and not r.stdout.strip()
                assert b'unsupported streamed tensor type/shape' in r.stderr
            report['pass'] = True
        except (OSError, subprocess.SubprocessError, ValueError, AssertionError, KeyError) as error:
            report['error'] = type(error).__name__ + ': ' + str(error)
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({'pass': report['pass'], 'report': str(args.output)}))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
