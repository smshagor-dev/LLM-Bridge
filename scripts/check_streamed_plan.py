#!/usr/bin/env python3
"""Real-model CLI admission/refusal and explicit adaptive-context regression."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); binary=str(args.binary.resolve()); model=str(args.model.resolve())
    digest=hashlib.sha256(args.model.read_bytes()).hexdigest()
    if digest != '2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6':
        p.error('golden planning protocol requires pinned stories15M Q8_0')
    def run(*command):
        r=subprocess.run([binary,*command],capture_output=True,text=True,timeout=90)
        return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    cases={}
    cases['plan']=run('plan',model,'--max-ram-mb','64','--json')
    normal=json.loads(cases['plan']['stdout'])
    assert cases['plan']['exit_code']==0 and normal['selected_context']==128
    assert normal['native_workspace_bytes']==normal['workspace_fixed_bytes']+128*normal['workspace_per_token_bytes']
    assert normal['kv_bytes']==6*128*288*2*4
    assert not normal['process_limit_enforced_by_plan'] and normal['model_integrity']=='not_verified_by_plan'
    cases['adaptive_plan']=run('plan',model,'--ctx','256','--auto-context','--json')
    adaptive=json.loads(cases['adaptive_plan']['stdout'])
    assert adaptive['selected_context']==128 and adaptive['context_reduced']
    for name, flags in [('context_refusal',['--ctx','256']),('workspace_refusal',['--workspace-mb','1']),('reserve_refusal',['--max-ram-mb','32'])]:
        cases[name]=run('plan',model,*flags)
        assert cases[name]['exit_code']!=0 and not cases[name]['stdout'].strip()
    base=['chat',model,'--backend','streamed','--prompt','The sky is','--max-tokens','8']
    cases['manual_generation']=run(*base,'--ctx','128')
    cases['adaptive_generation']=run(*base,'--ctx','256','--auto-context','--max-ram-mb','64','--address-space-mb','64')
    assert cases['manual_generation']['exit_code']==cases['adaptive_generation']['exit_code']==0
    assert cases['manual_generation']['stdout']==cases['adaptive_generation']['stdout']
    line=next(x for x in cases['adaptive_generation']['stderr'].splitlines() if x.startswith('LLMBRIDGE_PLAN '))
    applied=json.loads(line.removeprefix('LLMBRIDGE_PLAN '))
    assert applied['selected_context']==128 and applied['context_reduced']
    cases['verified_adaptive_generation']=run(*base,'--ctx','256','--auto-context','--max-ram-mb','64','--sha256',digest,'--metrics')
    assert cases['verified_adaptive_generation']['exit_code']==0
    assert cases['verified_adaptive_generation']['stdout']==cases['manual_generation']['stdout']
    assert 'LLMBRIDGE_INTEGRITY verified=1' in cases['verified_adaptive_generation']['stderr']
    cases['bad_hash_adaptive_generation']=run(*base,'--auto-context','--sha256','0'*64)
    assert cases['bad_hash_adaptive_generation']['exit_code']!=0 and not cases['bad_hash_adaptive_generation']['stdout'].strip()
    assert 'SHA-256 mismatch' in cases['bad_hash_adaptive_generation']['stderr']
    cases['infeasible_adaptive_generation']=run(*base,'--workspace-mb','1','--auto-context')
    assert cases['infeasible_adaptive_generation']['exit_code']!=0 and not cases['infeasible_adaptive_generation']['stdout'].strip()
    cases['auto_cache_plan']=run('plan',model,'--auto-cache','--max-ram-mb','64','--json')
    cache_plan=json.loads(cases['auto_cache_plan']['stdout'])
    assert cases['auto_cache_plan']['exit_code']==0 and cache_plan['cache_adaptive']
    assert cache_plan['selected_context']==128 and 0 < cache_plan['cache_budget_bytes'] <= 1024*1024
    cases['auto_cache_zero']=run('plan',model,'--auto-cache','--cache-mb','0','--json')
    assert json.loads(cases['auto_cache_zero']['stdout'])['cache_budget_bytes']==0
    cases['auto_cache_generation']=run(*base,'--ctx','128','--auto-cache','--max-ram-mb','64','--metrics')
    measured=cases['auto_cache_generation']
    assert measured['exit_code']==0 and measured['stdout']==cases['manual_generation']['stdout']
    applied_cache=json.loads(next(x.removeprefix('LLMBRIDGE_PLAN ') for x in measured['stderr'].splitlines() if x.startswith('LLMBRIDGE_PLAN ')))
    assert f"cache_allocated_bytes={applied_cache['cache_budget_bytes']} " in measured['stderr']
    cases['verified_auto_cache_generation']=run(*base,'--auto-cache','--max-ram-mb','64','--sha256',digest,'--metrics')
    assert cases['verified_auto_cache_generation']['exit_code']==0
    assert cases['verified_auto_cache_generation']['stdout']==cases['manual_generation']['stdout']
    assert 'LLMBRIDGE_INTEGRITY verified=1' in cases['verified_auto_cache_generation']['stderr']
    cases['auto_cache_cannot_reduce_context']=run('plan',model,'--ctx','256','--auto-cache')
    assert cases['auto_cache_cannot_reduce_context']['exit_code']!=0
    cases['auto_cache_resident_refusal']=run('chat',model,'--prompt','The sky is','--auto-cache')
    assert cases['auto_cache_resident_refusal']['exit_code']!=0 and not cases['auto_cache_resident_refusal']['stdout'].strip()
    report={'pass':True,'model_sha256':digest,'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            'git_sha':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
            'working_tree_dirty':bool(subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,check=True).stdout.strip()),
            'native_plan':normal,'applied_plan':applied,'cases':cases,
            'note':'Admission budgets are incremental estimates, not total process RSS. Synthetic 2GB profiles are not real hardware tests.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pass':True,'native_plan':normal,'applied_plan':applied},indent=2))


if __name__=='__main__':
    main()
