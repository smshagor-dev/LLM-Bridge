#!/usr/bin/env python3
"""Record raw CPU trials; failed runs or absent Linux peak RSS fail the gate."""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--repetitions', type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.repetitions <= 30:
        p.error('repetitions must be 1..30')
    command = [args.binary, 'chat', args.model, '--prompt', 'The sky is',
               '--max-tokens', '16', '--threads', '2', '--ctx', '128', '--metrics']
    runs = []
    for index in range(args.repetitions):
        start = time.perf_counter_ns()
        trial = {'run': index + 1, 'valid': False}
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=300, check=False)
            trial.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
            metrics = re.search(r'^LLMBRIDGE_MEMORY (\{.*\})$', result.stderr, re.MULTILINE)
            if metrics:
                trial['memory'] = json.loads(metrics.group(1))
            peak = trial.get('memory', {}).get('peak_rss_bytes')
            measured = isinstance(peak, int) and not isinstance(peak, bool) and peak > 0
            trial['valid'] = result.returncode == 0 and bool(result.stdout.strip()) and (
                platform.system() != 'Linux' or measured)
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            trial['error'] = str(error)
        trial['wall_seconds'] = (time.perf_counter_ns() - start) / 1e9
        runs.append(trial)
    digest = hashlib.sha256()
    try:
        with open(args.model, 'rb') as model:
            for chunk in iter(lambda: model.read(1024 * 1024), b''):
                digest.update(chunk)
        model_sha256 = digest.hexdigest()
    except OSError:
        model_sha256 = None
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    valid = all(trial['valid'] for trial in runs) and model_sha256 is not None
    report = {'valid': valid, 'host': platform.platform(), 'git_sha': revision.stdout.strip(),
              'model_sha256': model_sha256, 'command': command, 'runs': runs,
              'note': 'Wall time includes load/startup; VmHWM is lifetime peak RSS. '
                      'Post-run RSS/PSS are samples. Not tokens/s or quality.'}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(args.output)
    return 0 if valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
