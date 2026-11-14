#!/usr/bin/env python3
"""Validate identical cached/uncached generations and measured read-call savings."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uncached', type=Path, required=True)
    p.add_argument('--cached', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    a = json.loads(args.uncached.read_text()); b = json.loads(args.cached.read_text())
    valid = a['pass'] and b['pass'] and a['model_sha256'] == b['model_sha256'] and (
        a['binary_sha256'] == b['binary_sha256'] and a['cache_budget_bytes'] == 0 < b['cache_budget_bytes']
        and a['address_space_limit_bytes'] == b['address_space_limit_bytes'])
    trials = []
    valid = valid and len(a['trials']) == len(b['trials']) and bool(a['trials'])
    for index, (cold, warm) in enumerate(zip(a['trials'], b['trials'])):
        calls_a = cold['reader_metrics']['source_read_calls']
        calls_b = warm['reader_metrics']['source_read_calls']
        logits_equal = (args.uncached.parent / f'streamed-{index}.f32').read_bytes() == (
            args.cached.parent / f'streamed-{index}.f32').read_bytes()
        passed = cold['prompt'] == warm['prompt'] and cold['streamed_text'] == warm['streamed_text'] and (
            logits_equal and 0 < calls_b < calls_a and
            0 < warm['reader_metrics']['cache_allocated_bytes'] <= b['cache_budget_bytes'])
        valid = valid and passed
        trials.append({'prompt': cold['prompt'], 'pass': passed, 'logits_identical': logits_equal,
                       'uncached_source_read_calls': calls_a, 'cached_source_read_calls': calls_b,
                       'read_call_reduction_percent': 100 * (1 - calls_b / calls_a) if calls_a else None,
                       'uncached_source_bytes': cold['reader_metrics']['source_bytes_read'],
                       'cached_source_bytes': warm['reader_metrics']['source_bytes_read'],
                       'cached_peak_rss_bytes': warm['memory']['peak_rss_bytes'],
                       'cache_allocated_bytes': warm['reader_metrics']['cache_allocated_bytes']})
    report = {'pass': bool(valid), 'model_sha256': a['model_sha256'], 'binary_sha256': a['binary_sha256'],
              'source_revision': a['git_sha'], 'address_space_limit_bytes': a['address_space_limit_bytes'],
              'cache_budget_bytes': b['cache_budget_bytes'], 'trials': trials,
              'note': 'Source read calls/bytes are application file-range I/O, not physical device I/O. '
                      'Cache includes bounded aligned read-ahead. This is not a speed or real-2GB-host claim.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
