"""Independent E022 export accounting, prefixes, ancestry and phone edits."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from research.audit import digest,save_json
from reproduction.audit_causal_recipe_completion import distance


def main(out):
    p=json.loads((out/'protocol.json').read_text());run=Path(p['parent_run'])
    assert digest(run/'protocol.json')==p['parent_protocol_sha256']
    assert digest(Path(__file__).with_name('evaluate_causal_recipe_comparison.py'))==p['code_sha256']
    original=json.loads((run/'protocol.json').read_text());splits=json.loads((run/'splits.json').read_text())
    for family in ('source_hashes','implementation_hashes','derived_hashes'):
        for path,h in original[family].items():assert digest(path)==h
    assert not set(p['target_days']) & set(original['forbidden_final_days'])
    assert len(p['conditions'])==130
    files={};phone_checks=0
    for kind,day,budget,mode in p['conditions']:
        target=out/kind/day/str(budget)/mode
        seal=json.loads((target/'export-complete.json').read_text());parent=p['parents'][kind]
        assert digest(parent['path'])==parent['sha256']==seal['parent_sha256']
        cal=splits['calibration'][day];n=len(cal) if budget=='full' else budget
        assert seal['calibration_ids']==[r['id'] for r in cal[:n]]
        assert seal['observed_sentences']==n and seal['labels']==(n if mode=='adapter' else 0)
        assert seal['steps']==len(seal['losses'])==(100 if mode=='adapter' else 0)
        assert np.isfinite(seal['losses']).all()
        for name,h in seal['artifacts_sha256'].items():assert digest(target/name)==h;files[str(target/name)]=h
        fast=torch.load(target/'adapter.pt',map_location='cpu',weights_only=False)
        assert set(fast)=={'log_gain','offset','u','v'} and sum(v.numel() for v in fast.values())==4608
        assert all(torch.isfinite(v).all() for v in fast.values())
        if mode=='normalization':
            assert torch.count_nonzero(fast['log_gain'])==torch.count_nonzero(fast['offset'])==torch.count_nonzero(fast['v'])==0
        stats=json.loads((target/'normalization.json').read_text())
        if n:
            assert stats['ids']==seal['calibration_ids'] and stats['bins']==sum(r['bins'] for r in cal[:n])
        else: assert stats==json.loads((run/(kind+'-normalization.json')).read_text())['global_stats']
        if mode=='adapter':assert stats==json.loads((target.parent/'normalization/normalization.json').read_text())
        rows=json.loads((target/'predictions.json').read_text());expected=splits['evaluation'][day]
        assert [r['id'] for r in rows]==[r['id'] for r in expected]
        assert not set(stats['ids']) & {r['id'] for r in rows}
        with np.load(target/'logits.npz') as arrays:
            assert len(arrays.files)==len(rows)==40
            for i,(r,e) in enumerate(zip(rows,expected)):
                assert r['session']==day and r['text']==e['text']
                logits=arrays[f'trial_{i:04d}'];assert logits.shape==((e['bins']-32)//4+1,41) and np.isfinite(logits).all()
                seq=logits.argmax(-1).tolist();phones=[x for j,x in enumerate(seq) if x!=40 and (j==0 or x!=seq[j-1])]
                assert phones==r['predicted_phones']
                assert distance(r['reference_phones'],phones)==r['per']['errors']==sum(r['per'][k] for k in ('substitutions','deletions','insertions'))
                phone_checks+=1
    save_json(out/'export-audit.json',dict(status='pass',conditions=130,phone_checks=phone_checks,unique_test_trials=200,
        source_and_parent_identities='pass',calibration_prefixes_and_separation='pass',adapter_states='pass',artifacts=files,script_sha256=digest(__file__)))
    print('passed',phone_checks,'phone checks',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);main(a.parse_args().output.resolve())
