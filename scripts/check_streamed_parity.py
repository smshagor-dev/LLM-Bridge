#!/usr/bin/env python3
"""Real streamed decoder versus independently built pinned llama.cpp."""
import argparse
import array
import hashlib
import json
import math
import os
import re
import platform
import time
import subprocess
import sys
from pathlib import Path
from check_m1_parity import compare


CHECKPOINTS = {
    'stories15M-q8_0': '2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6',
    'stories15M-derived-q4_0': 'b5cd77bb0d1a785a82d7a8a2bcb4170dfe6e43e881b9cbb0d1c8184a6dd99839',
}


def validate_checkpoint(name, digest):
    if name not in CHECKPOINTS or digest != CHECKPOINTS[name]:
        raise ValueError('golden protocol requires the exact explicitly selected pinned checkpoint')


def validate_proof_cap(megabytes, file_bytes, system, emulated=False):
    if megabytes and emulated:
        raise ValueError("emulated trials cannot be claimed as a guest process memory-cap proof")
    if megabytes < 0:
        raise ValueError('address-space limit must be nonnegative')
    if megabytes and (system != 'linux' or file_bytes <= megabytes * 1024 * 1024):
        raise ValueError('constrained proof requires Linux and a model larger than the address-space cap')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--checkpoint', choices=CHECKPOINTS, default='stories15M-q8_0')
    p.add_argument('--reference-no-repack', action='store_true', help='Explicitly disable reference extra weight buffer/repacking kernels')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--emulator', type=Path, help='QEMU user-mode CPU validation; not a memory-cap experiment')
    p.add_argument('--guest-cpu', default='core2duo')
    p.add_argument('--max-tokens', type=int, default=8)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--cache-mb', type=int, default=0)
    p.add_argument('--cache-pressure-mb', type=int)
    p.add_argument('--require-cache-release', action='store_true')
    p.add_argument('--max-address-space-mb', type=int, default=0, help='Linux child RLIMIT_AS; 0 disables')
    args = p.parse_args()
    if args.cache_pressure_mb is not None and args.cache_pressure_mb <= 0:
        p.error('cache pressure floor must be positive')
    if args.require_cache_release and (args.cache_pressure_mb is None or args.cache_mb == 0):
        p.error('required cache release needs a pressure floor and a nonzero cache')
    if not 1 <= args.max_tokens <= 64:
        p.error('max-tokens must be 1..64')
    if sys.byteorder != 'little':
        p.error('reference raw float dump currently requires little endian')
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    try:
        validate_checkpoint(args.checkpoint, digest)
        validate_proof_cap(args.max_address_space_mb, args.model.stat().st_size, sys.platform, args.emulator is not None)
    except ValueError as error:
        p.error(str(error))
    limit_bytes = args.max_address_space_mb * 1024 * 1024
    prefix = []
    emulation = None
    if args.emulator:
        prefix = [str(args.emulator.resolve()), '-cpu', args.guest_cpu]
        emulation = {'cpu': args.guest_cpu,
                     'emulator_sha256': hashlib.sha256(args.emulator.read_bytes()).hexdigest(),
                     'version': subprocess.run([str(args.emulator.resolve()), '--version'], capture_output=True,
                         text=True, check=True, timeout=30).stdout.splitlines()[0]}
    cpu = json.loads(subprocess.run(prefix + [str(args.binary.resolve()), 'doctor', '--json'],
        capture_output=True, text=True, check=True, timeout=30).stdout)['cpu']
    if not cpu['native_compatible']:
        p.error('positive golden protocol requires a compatible native build/CPU profile')
    reference_env = {**os.environ, 'LLMBRIDGE_REFERENCE_F32_KV': '1'}
    reference_env.pop('LLMBRIDGE_REFERENCE_NO_REPACK', None)
    if args.reference_no_repack:
        reference_env['LLMBRIDGE_REFERENCE_NO_REPACK'] = '1'
    def apply_limit():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    args.output.mkdir(parents=True, exist_ok=True)
    trials = []
    binary_digest = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    reference_digest = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    for index, prompt in enumerate(['The sky is', 'Once upon a time', 'A little dog']):
        streamed_logits = args.output / f'streamed-{index}.f32'
        reference_logits = args.output / f'reference-{index}.f32'
        reference_logits.unlink(missing_ok=True)
        streamed_logits.unlink(missing_ok=True)
        base = prefix + [str(args.binary.resolve()), 'chat', str(args.model.resolve()), '--prompt', prompt,
                '--max-tokens', str(args.max_tokens), '--ctx', '128', '--metrics', '--cache-mb', str(args.cache_mb)]
        if args.verify: base += ['--sha256', digest]
        if args.cache_pressure_mb is not None: base += ['--cache-pressure-mb', str(args.cache_pressure_mb)]
        started = time.monotonic()
        actual = subprocess.run(base + ['--backend', 'streamed', '--workspace-mb', '8',
                                       '--logits', str(streamed_logits)], capture_output=True, timeout=240,
                                preexec_fn=apply_limit if limit_bytes else None)
        wall_seconds = time.monotonic() - started
        expected = subprocess.run(prefix + [str(args.reference.resolve()), '-m', str(args.model.resolve()),
                                   '-n', str(args.max_tokens), '-ngl', '0', prompt], capture_output=True, timeout=240,
                                  env={**reference_env, 'LLMBRIDGE_REFERENCE_LOGITS': str(reference_logits.resolve())})
        result = {'prompt': prompt, 'streamed_exit': actual.returncode, 'reference_exit': expected.returncode,
                  'streamed_text': actual.stdout.decode(errors='replace'),
                  'reference_text': expected.stdout.decode(errors='replace'),
                  'streamed_stderr': actual.stderr.decode(errors='replace'),
                  'reference_stderr': expected.stderr.decode(errors='replace')}
        cache_metrics = re.search(r'^LLMBRIDGE_STREAMED (.*)$', result['streamed_stderr'], re.MULTILINE)
        integrity = re.search(r'^LLMBRIDGE_INTEGRITY (.*)$', result['streamed_stderr'], re.MULTILINE)
        result['integrity_metrics'] = dict(re.findall(r'(\w+)=(\w+)', integrity.group(1))) if integrity else {}
        result['integrity_pass'] = not args.verify or (result['integrity_metrics'].get('verified') == '1' and result['integrity_metrics'].get('sha256') == digest and int(result['integrity_metrics'].get('blocks_verified', '0')) > 0)
        result['reader_metrics'] = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', cache_metrics.group(1))} if cache_metrics else {}
        cache = result['reader_metrics']
        result['cache_release_pass'] = not args.require_cache_release or (
            cache.get('cache_allocated_bytes') == 0 and cache.get('cache_release_events') == 1
            and cache.get('cache_peak_allocated_bytes', 0) > 0
            and cache.get('cache_released_bytes') == cache.get('cache_peak_allocated_bytes'))
        result['wall_seconds_including_load'] = wall_seconds
        memory = re.search(r'^LLMBRIDGE_MEMORY (\{.*\})$', result['streamed_stderr'], re.MULTILINE)
        result['memory'] = json.loads(memory.group(1)) if memory else {}
        result['memory_cap_pass'] = not limit_bytes or (
            actual.returncode == 0 and isinstance(result['memory'].get('peak_rss_bytes'), int)
            and 0 < result['memory']['peak_rss_bytes'] <= limit_bytes)
        result['tokens_match'] = actual.returncode == expected.returncode == 0 and compare(actual.stdout, expected.stdout, ("<s> " + prompt).encode())
        a = array.array('f'); b = array.array('f')
        if streamed_logits.exists(): a.frombytes(streamed_logits.read_bytes())
        if reference_logits.exists(): b.frombytes(reference_logits.read_bytes())
        errors = [abs(x-y) for x,y in zip(a,b)]
        # Both paths use pinned ggml activation quantization and dot kernels.
        # Graph scheduling and reduction order can still differ.
        result['logits_count'] = len(a)
        result['logits_max_abs_error'] = max(errors, default=None)
        result['logits_rmse'] = math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None
        result['logits_match'] = bool(a) and len(a) == len(b) and len(a) % 32000 == 0 and len(a) <= args.max_tokens * 32000 and all(
            math.isfinite(x) and math.isfinite(y) and abs(x-y) <= 0.01 + 0.001*abs(y) for x,y in zip(a,b))
        result['reference_repacking_observed'] = 'repack tensor ' in result['reference_stderr']
        result['reference_configuration_pass'] = not args.reference_no_repack or not result['reference_repacking_observed']
        result['pass'] = result['cache_release_pass'] and result['reference_configuration_pass'] and result['tokens_match'] and result['logits_match'] and result['memory_cap_pass'] and result['integrity_pass']
        trials.append(result)
    report = {'pass': all(t['pass'] for t in trials), 'checkpoint': args.checkpoint, 'generation_max_tokens': args.max_tokens, 'cpu_profile': cpu, 'emulation': emulation,
              'reference_weight_repacking': 'explicitly_disabled' if args.reference_no_repack else 'upstream_default', 'model_sha256': digest, 'model_file_bytes': args.model.stat().st_size,
              'binary_sha256': binary_digest, 'reference_binary_sha256': reference_digest,
              'reference_revision': 'cd959bae99ff4bed22bef2604f45241c09bbb8c8',
              'working_tree_dirty': bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout.strip()),
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('native/src/streamed.cpp'), Path('crates/llmbridge-core/src/streamed.rs'), Path('crates/llmbridge-core/src/read_cache.rs'), Path('crates/llmbridge-core/src/storage.rs'), Path('crates/llmbridge-core/src/storage_auth.rs')]},
              'verified_session_requested': args.verify,
              'cache_budget_bytes': args.cache_mb * 1024 * 1024,
              'cache_pressure_floor_mib': args.cache_pressure_mb, 'cache_release_required': args.require_cache_release,
              'address_space_limit_bytes': limit_bytes or None,
              'memory_proof_scope': 'native_model_larger_than_cap' if limit_bytes else 'not_a_memory_cap_proof', 'host': platform.platform(),
              'git_sha': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
              'kv_precision': 'FP32 on both paths; reference flash attention disabled',
              'logit_tolerance': {'atol': 0.01, 'rtol': 0.001},
              'note': 'Checkpoint-specific experimental streamed path with pinned ggml dot kernels versus upstream graph; observed matching logits are checkpoint-specific. RLIMIT_AS is a Linux process address-space cap, not a portable RSS API.',
              'trials': trials}
    (args.output / 'streamed_parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'trials'}))
    for trial in trials: print({k:v for k,v in trial.items() if 'stderr' not in k and 'text' not in k})
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
