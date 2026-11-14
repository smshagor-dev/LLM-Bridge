#!/usr/bin/env python3
"""Require exact cache transparency and the explicitly selected source-read policy."""
import argparse
import json
from pathlib import Path


def compare_reports(a, b, uncached_directory, cached_directory, authenticated_bypass=False):
    # Keep numerical/raw-output equality independent of the read policy. Never
    # infer a relaxed policy from equal counters or a failed performance check.
    common = ('model_sha256', 'binary_sha256', 'git_sha', 'address_space_limit_bytes',
              'checkpoint', 'generation_max_tokens', 'cpu_profile', 'emulation',
              'reference_revision', 'reference_binary_sha256', 'reference_weight_repacking',
              'kv_precision', 'logit_tolerance', 'cache_pressure_floor_mib', 'cache_release_required')
    valid = bool(a['pass'] and b['pass'] and all(a.get(k) == b.get(k) for k in common)
                 and a['cache_budget_bytes'] == 0 < b['cache_budget_bytes']
                 and a.get('verified_session_requested') is authenticated_bypass
                 and b.get('verified_session_requested') is authenticated_bypass)
    valid = valid and len(a['trials']) == len(b['trials']) and bool(a['trials'])
    trials = []
    for index, (cold, warm) in enumerate(zip(a['trials'], b['trials'])):
        cm = cold['reader_metrics']; wm = warm['reader_metrics']
        calls_a = cm['source_read_calls']; calls_b = wm['source_read_calls']
        logits_a = (uncached_directory / f'streamed-{index}.f32').read_bytes()
        logits_b = (cached_directory / f'streamed-{index}.f32').read_bytes()
        logits_equal = bool(logits_a) and logits_a == logits_b
        if authenticated_bypass:
            integrity = all(t.get('integrity_pass') is True and
                            t.get('integrity_metrics', {}).get('verified') == '1' and
                            t['integrity_metrics'].get('sha256') == a['model_sha256'] and
                            int(t['integrity_metrics'].get('blocks_verified', '0')) >= calls_a > 0
                            for t in (cold, warm))
            policy = integrity and calls_a == calls_b and all(
                m.get('weight_read_calls') == calls_a and m.get('cache_block_hits') == 0
                and m.get('cache_block_misses') == 0 and m.get('cache_evictions') == 0
                and m.get('cache_release_events') == 0 for m in (cm, wm)) and (
                0 < cm['source_bytes_read'] == wm['source_bytes_read'] and
                cm.get('weight_bytes_requested') == wm.get('weight_bytes_requested') and
                cm.get('reader_buffer_bytes') == wm.get('reader_buffer_bytes') and
                cold['integrity_metrics']['blocks_verified'] == warm['integrity_metrics']['blocks_verified'])
        else:
            policy = 0 < calls_b < calls_a
        passed = bool(cold['pass'] and warm['pass'] and cold['prompt'] == warm['prompt'] and
                      cold['streamed_text'] == warm['streamed_text'] and logits_equal and policy and
                      cm['cache_allocated_bytes'] == 0 and
                      0 < wm['cache_allocated_bytes'] <= b['cache_budget_bytes'])
        valid = valid and passed
        trials.append({'prompt': cold['prompt'], 'pass': passed, 'logits_identical': logits_equal,
                       'read_policy_pass': bool(policy), 'uncached_source_read_calls': calls_a,
                       'cached_source_read_calls': calls_b,
                       'read_call_reduction_percent': 100 * (1 - calls_b / calls_a) if calls_a else None,
                       'uncached_source_bytes': cm['source_bytes_read'],
                       'cached_source_bytes': wm['source_bytes_read'],
                       'cached_peak_rss_bytes': warm['memory']['peak_rss_bytes'],
                       'cache_allocated_bytes': wm['cache_allocated_bytes']})
    return {'pass': bool(valid), 'model_sha256': a['model_sha256'], 'binary_sha256': a['binary_sha256'],
            'source_revision': a['git_sha'], 'address_space_limit_bytes': a['address_space_limit_bytes'],
            'cache_budget_bytes': b['cache_budget_bytes'], 'trials': trials,
            'read_policy': 'authenticated_bypass' if authenticated_bypass else 'unauthenticated_read_reduction',
            'note': 'Exact output/logit transparency remains required in both policies. Authenticated '
                    'bypass requires fresh weight-source reads and zero LRU activity, not read savings. '
                    'Application range counts are not physical device I/O or universal speed proof.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uncached', type=Path, required=True)
    p.add_argument('--cached', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expect-authenticated-bypass', action='store_true',
                   help='Require both trusted sessions to bypass LRU and preserve fresh source reads')
    args = p.parse_args()
    report = compare_reports(json.loads(args.uncached.read_text()), json.loads(args.cached.read_text()),
                             args.uncached.parent, args.cached.parent, args.expect_authenticated_bypass)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
