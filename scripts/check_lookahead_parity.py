#!/usr/bin/env python3
"""Compare lookahead output/logits with pinned-reference artifacts."""
import argparse
import array
import json
import math
from pathlib import Path

from check_m1_parity import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lookahead-dir', type=Path, required=True)
    parser.add_argument('--reference-report', type=Path, required=True)
    parser.add_argument('--reference-logits', type=Path, required=True)
    parser.add_argument('--prompt-prefix', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.reference_report.read_text())
    trial = next(t for t in report['trials'] if t['prompt'] == args.prompt_prefix)
    ours = (args.lookahead_dir / 'lookahead.stdout').read_bytes()
    expected = trial['reference_text'].encode()
    a = array.array('f'); a.frombytes((args.lookahead_dir / 'lookahead.logits.f32').read_bytes())
    b = array.array('f'); b.frombytes(args.reference_logits.read_bytes())
    b = b[:len(a)]
    errors = [abs(x-y) for x, y in zip(a, b)]
    # The callback API returns generated token bytes without the CLI's final
    # line terminator; add that protocol framing before the exact comparison.
    text_match = compare(ours + b'\n', expected, ('<s> ' + args.prompt_prefix).encode())
    logits_match = len(a) == len(b) and bool(a) and all(
        math.isfinite(x) and math.isfinite(y) and abs(x-y) <= 0.01 + 0.001*abs(y)
        for x, y in zip(a, b))
    result = {
        'pass': text_match and logits_match,
        'reference_revision': report['reference_revision'],
        'lookahead_logit_count': len(a),
        'reference_logit_count_compared': len(b),
        'complete_output_match': text_match,
        'logits_match': logits_match,
        'logits_max_abs_error': max(errors, default=None),
        'logits_rmse': math.sqrt(sum(x*x for x in errors)/len(errors)) if errors else None,
        'logit_tolerance': {'atol': 0.01, 'rtol': 0.001},
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
