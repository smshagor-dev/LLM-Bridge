#!/usr/bin/env python3
"""Measure real streamed sessions; gate optional workload-specific latency targets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_result(result, max_tokens, target_rate=None, target_first_ms=None):
    if result.returncode != 0:
        raise ValueError('generation failed; no successful performance result')
    lines = [line.split(b' ', 1)[1] for line in result.stderr.splitlines()
             if line.startswith(b'LLMBRIDGE_PERFORMANCE ')]
    if len(lines) != 1:
        raise ValueError('exactly one successful session timing record required')
    p = json.loads(lines[0])
    required = ['total_ns', 'sampling_steps', 'emitted_pieces', 'emitted_bytes',
                'decode_intervals', 'decode_elapsed_ns', 'range_callback_ns',
                'consumer_callback_ns']
    for key in required:
        if type(p.get(key)) is not int or p[key] < 0:
            raise ValueError('invalid timing counter: ' + key)
    if not (0 < p['sampling_steps'] <= max_tokens and p['total_ns'] > 0):
        raise ValueError('invalid total time or sampling count')
    first = p.get('first_logits_ready_ns')
    if type(first) is not int or not 0 <= first <= p['total_ns']:
        raise ValueError('missing/invalid first logits time')
    if (p['decode_intervals'] != p['sampling_steps'] - 1
            or first + p['decode_elapsed_ns'] > p['total_ns']
            or p['range_callback_ns'] + p['consumer_callback_ns'] > p['total_ns']):
        raise ValueError('inconsistent session timing')
    if p['sampling_steps'] - p['emitted_pieces'] not in (0, 1):
        raise ValueError('emission count inconsistent with sampling/EOG')
    if not result.stdout.endswith(b'\n') or p['emitted_bytes'] != len(result.stdout) - 1:
        raise ValueError('emitted byte count differs from complete CLI output')
    visible = p.get('first_piece_ns')
    if p['emitted_bytes'] > 0:
        if type(visible) is not int or not first <= visible <= p['total_ns']:
            raise ValueError('invalid first visible piece time')
    elif visible is not None:
        raise ValueError('empty output cannot have visible first piece')
    rate = p.get('decode_steps_per_second')
    expected = (p['decode_intervals'] * 1e9 / p['decode_elapsed_ns']
                if p['decode_intervals'] > 0 and p['decode_elapsed_ns'] > 0 else None)
    if expected is None:
        if rate is not None:
            raise ValueError('unmeasured decode rate must be null')
    elif (type(rate) not in (int, float) or not math.isfinite(rate)
          or not math.isclose(rate, expected, rel_tol=1e-12)):
        raise ValueError('decode rate differs from measured interval count')
    target_met = ((target_rate is None or (rate is not None and rate >= target_rate))
                  and (target_first_ms is None or
                       (visible is not None and visible / 1e6 <= target_first_ms)))
    return p, target_met


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt', default='The sky is')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--ctx', type=int, default=128)
    parser.add_argument('--cache-mb', type=int, default=0)
    parser.add_argument('--address-space-mb', type=int)
    parser.add_argument('--sha256', help='independently trusted model digest for authenticated sessions')
    parser.add_argument('--target-decode-steps-per-second', type=float)
    parser.add_argument('--target-first-piece-ms', type=float)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 10 or not 1 <= args.max_tokens <= 2048:
        parser.error('repetitions must be 1..10; max-tokens must be 1..2048')
    for target in (args.target_decode_steps_per_second, args.target_first_piece_ms):
        if target is not None and (not math.isfinite(target) or target <= 0):
            parser.error('performance targets must be finite and positive')
    repository = Path(__file__).resolve().parents[1]
    def git(*arguments):
        result = subprocess.run(['git', '-C', str(repository), *arguments],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    provenance = {'commit': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}'),
                  'working_tree_dirty': git('status', '--porcelain') != ''}
    # Exclusive creation refuses existing reports, models and symlinks before trials.
    with args.output.open('x') as destination:
        command = [str(args.binary.resolve()), 'chat', str(args.model.resolve()),
                   '--backend', 'streamed', '--prompt', args.prompt,
                   '--max-tokens', str(args.max_tokens), '--ctx', str(args.ctx),
                   '--cache-mb', str(args.cache_mb)]
        if args.address_space_mb is not None:
            command += ['--address-space-mb', str(args.address_space_mb)]
        if args.sha256 is not None:
            command += ['--sha256', args.sha256]
        report = {'schema': 1, 'pass': False, 'source': provenance,
                  'binary_sha256': digest_file(args.binary),
                  'model_sha256': digest_file(args.model),
                  'prompt_sha256': hashlib.sha256(args.prompt.encode()).hexdigest(),
                  'configuration': {'context': args.ctx, 'max_tokens': args.max_tokens,
                                    'cache_mb': args.cache_mb, 'address_space_mb': args.address_space_mb,
                                    'authenticated': args.sha256 is not None},
                  'targets': {'decode_steps_per_second': args.target_decode_steps_per_second,
                              'first_piece_ms': args.target_first_piece_ms},
                  'note': 'Observed workload only; page-cache state uncontrolled; range-callback time includes identity checks, cache work and block authentication. Decode intervals include consumer callbacks. No physical disk bandwidth, isolated compute, cold-start or larger-model claim.',
                  'trials': []}
        # This run is only an output-equivalence check, not a timing comparison.
        reference = {'valid': False}
        try:
            result = subprocess.run(command,
                                    capture_output=True, timeout=300, check=False)
            reference.update(valid=result.returncode == 0, exit_code=result.returncode,
                             output_sha256=hashlib.sha256(result.stdout).hexdigest())
        except (OSError, subprocess.TimeoutExpired) as error:
            reference['error'] = type(error).__name__
        report['uninstrumented_validation'] = reference
        for index in range(args.repetitions):
            trial = {'run': index + 1, 'valid': False, 'target_met': False}
            start = time.perf_counter_ns()
            try:
                result = subprocess.run([*command, '--metrics'], capture_output=True, timeout=300, check=False)
                trial.update(exit_code=result.returncode,
                             output_sha256=hashlib.sha256(result.stdout).hexdigest(),
                             stderr_sha256=hashlib.sha256(result.stderr).hexdigest())
                performance, met = validate_result(result, args.max_tokens,
                                                  args.target_decode_steps_per_second,
                                                  args.target_first_piece_ms)
                trial.update(valid=True, target_met=met, performance=performance)
            except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as error:
                trial['error'] = type(error).__name__ + ': ' + str(error)
            trial['child_wall_ns'] = time.perf_counter_ns() - start
            report['trials'].append(trial)
        outputs = {trial.get('output_sha256') for trial in report['trials']}
        report['outputs_identical'] = len(outputs) == 1 and None not in outputs
        report['uninstrumented_output_matches'] = (reference['valid'] and
            outputs == {reference.get('output_sha256')})
        report['pass'] = (report['outputs_identical'] and report['uninstrumented_output_matches']
                          and all(t['valid'] and t['target_met'] for t in report['trials']))
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write('\n')
    print(json.dumps({'pass': report['pass'], 'trials': len(report['trials']),
                      'report': str(args.output)}))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
