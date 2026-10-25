"""Independent CPU-only E016 progress/final audit; never selects on public test."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.audit import digest,save_json
from research.metrics import summarize


def distance(a,b):
    previous=list(range(len(b)+1))
    for i,token in enumerate(a,1):
        current=[i]
        for j,other in enumerate(b,1):
            current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(token!=other)))
        previous=current
    return previous[-1]


def decode(native):
    values=native.argmax(-1).tolist()
    return [p for i,p in enumerate(values) if p!=40 and (i==0 or p!=values[i-1])]


def decode_source_order(native):
    """Preserve author's blank-first argmax priority on exact BF16 ties.

    Export permutation moves blank from zero to forty. Reordering equal maxima
    must not silently change the independently reconstructed runtime PER.
    """
    source=native[:,[40]+list(range(40))].argmax(-1).tolist()
    return [p-1 for i,p in enumerate(source) if p!=0 and (i==0 or p!=source[i-1])]


def audit_export(folder,records):
    rows=json.loads((folder/'predictions.json').read_text())
    expected=json.loads((folder/'metrics.json').read_text())
    assert [r['trial_id'] for r in rows]==[r['id'] for r in records]
    phones=errors=0;loss=0.;native_errors=0;tie_bins=0;changed_trials=[]
    with np.load(folder/'logits.npz') as archive:
        assert len(archive.files)==len(rows)
        for i,(row,record) in enumerate(zip(rows,records)):
            values=archive[f'trial_{i:04d}'];assert values.ndim==2 and values.shape[1]==41
            assert np.isfinite(values).all()
            prediction=decode_source_order(values);reference=[p-1 for p in record['phonemes']]
            native_prediction=decode(values)
            native_errors+=distance(reference,native_prediction)
            tie_bins+=int(((values==values.max(-1,keepdims=True)).sum(-1)>1).sum())
            if prediction!=native_prediction:changed_trials.append(row['trial_id'])
            count=distance(reference,prediction)
            assert prediction==row['phoneme_prediction']
            assert count==row['per']['errors'] and len(reference)==row['per']['reference_units']
            assert sum(row['per'][k] for k in ('substitutions','deletions','insertions'))==count
            phones+=len(reference);errors+=count
            lp=torch.from_numpy(values).unsqueeze(1)
            loss+=torch.nn.functional.ctc_loss(lp,torch.tensor(reference),torch.tensor([len(values)]),
                torch.tensor([len(reference)]),blank=40,reduction='sum',zero_infinity=True).item()
    assert errors==expected['phone_errors'] and phones==expected['reference_phones']
    assert abs(loss/phones-expected['ctc_per_reference_phone'])<1e-5
    return {'passed':True,'errors':errors,'phones':phones,'per':errors/phones,
            'ctc_per_reference_phone':loss/phones,'by_day_and_uncertainty':summarize(rows,repetitions=2000),
            'error_components':{k:sum(r['per'][k] for r in rows) for k in ('substitutions','deletions','insertions')},
            'argmax_tie_audit':{'policy':'Original author class order: blank first, then phones',
                'tied_bins':tie_bins,'native_order_changed_trial_ids':changed_trials,
                'native_order_errors':native_errors,'native_order_per':native_errors/phones},
            'source_compatible_metric':'Runtime-only here; padded logits are not in trimmed exports, so not independently rederived'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--final',action='store_true')
    args=parser.parse_args();run=args.run.resolve();assert not args.output.exists()
    contract=json.loads((run/'contract.json').read_text())
    assert contract['runner_sha256']==digest(ROOT/'reproduction/train_bit_full.py')
    assert contract['helper_sha256']==digest(ROOT/'reproduction/train_bit_ctc_pilot.py')
    assert all(digest(ROOT/'research/vendor/bit-f107ae9'/k)==h for k,h in contract['source_hashes'].items())
    latest=json.loads((run/'latest.json').read_text())
    assert digest(run/latest['checkpoint'])==latest['sha256']
    state=torch.load(run/latest['checkpoint'],map_location='cpu',weights_only=False)
    history=state['history'];epoch=state['epoch'];steps=state['steps']
    assert epoch==latest['epoch'] and steps==epoch*275
    assert [h['epoch'] for h in history]==list(range(1,epoch+1))
    assert state['scheduler']['total_steps']==800*275 and state['scheduler']['last_epoch']==steps
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    assert state['sampler_rng'] and state['torch_rng'].dtype==torch.uint8 and state['mps_rng'] is not None
    assert state['optimizer']['state']
    for item in history:
        assert item['validation']['trials']==440 and item['validation']['reference_phones']==12204
        assert abs(item['validation']['per']-item['validation']['phone_errors']/12204)<1e-12
        assert math.isfinite(item['train_ctc_per_reference_phone']) and math.isfinite(item['validation']['ctc_per_reference_phone'])
        # Actual source OneCycleLR: zero warmup, default final_div_factor10,000.
        expected=2e-9+(.0005-2e-9)*(1+math.cos(math.pi*(item['step']+1)/(800*275)))/2
        assert abs(expected-item['lr'])<1e-12
    audit={'passed':True,'scope':'CPU-only sealed checkpoint/progress audit','epoch':epoch,'steps':steps,
           'checkpoint_sha256':latest['sha256'],'source_hashes_verified':True,
           'finite_model_weights':True,'optimizer_sampler_rng_payload_present':True,'schedule_verified':True,
           'limitation':'Stored optimizer/RNG audit, not interrupted-run replay; source-compatible metric not independently rederived'}
    if args.final:
        assert epoch==800 and json.loads((run/'status.json').read_text())['status']=='training_complete_wer_pending'
        records=json.loads((run/'trials.json').read_text())
        chosen=json.loads((run/'best-per.json').read_text())
        minimum=min(history,key=lambda h:h['validation']['per'])
        assert chosen['epoch']==minimum['epoch'] and chosen['metric']==minimum['validation']['per']
        assert digest(run/chosen['checkpoint'])==chosen['sha256']
        assert not ({r['id'] for r in records['train']} & {r['id'] for r in records['val']+records['test']})
        assert not ({r['id'] for r in records['val']} & {r['id'] for r in records['test']})
        audit['validation']=audit_export(run/'selected-validation',records['val'])
        audit['test']=audit_export(run/'selected-test',records['test'])
        audit['numerical_gap_to_scratch_target']=audit['validation']['per']-.1726
        audit['inside_exploratory_one_pp_band']=abs(audit['numerical_gap_to_scratch_target'])<=.01
        audit['qualification']='Not exact reproduction while source/paper metric, split and selected-hyperparameter correspondence remain unresolved; one seed and historical public-label exposure'
    save_json(args.output,audit);print(json.dumps(audit))


if __name__=='__main__':main()
