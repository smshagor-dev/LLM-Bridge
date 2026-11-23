#!/usr/bin/env python3
"""Exercise native streamed-planner refusal of an unknown llama GGUF key."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    report = {'pass': False, 'case': 'unknown_llama_metadata_refused_before_plan'}
    try:
        original = args.model.read_bytes()
        # Preserve the key's encoded byte length and every GGUF offset. The
        # string value stays unchanged; only a benign general key becomes an
        # unsupported llama option.
        old_key = b'general.name'
        new_key = b'llama.foo.xx'
        assert len(old_key) == len(new_key)
        assert original.count(old_key) == 1, 'expected exactly one general.name key'
        mutated = original.replace(old_key, new_key, 1)
        report['model_sha256'] = hashlib.sha256(original).hexdigest()
        report['fixture_sha256'] = hashlib.sha256(mutated).hexdigest()

        with tempfile.TemporaryDirectory(prefix='llmbridge-native-metadata-') as directory:
            fixture = Path(directory) / 'unsupported-llama-key.gguf'
            fixture.write_bytes(mutated)
            result = subprocess.run(
                [str(args.binary.resolve()), 'plan', str(fixture), '--json'],
                capture_output=True, timeout=90, check=False)
            report.update({
                'exit_code': result.returncode,
                'stdout_bytes': len(result.stdout),
                'stderr_sha256': hashlib.sha256(result.stderr).hexdigest(),
                'refusal_message': 'unsupported llama metadata option in streamed backend',
            })
            assert result.returncode == 1, f'unexpected exit code: {result.returncode}'
            assert not result.stdout, 'failed plan must not emit stdout/plan JSON'
            assert b'unsupported llama metadata option in streamed backend' in result.stderr
        report['pass'] = True
    except (OSError, subprocess.SubprocessError, AssertionError) as error:
        report['error'] = f'{type(error).__name__}: {error}'

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pass': report['pass'], 'report': str(args.output)}))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
