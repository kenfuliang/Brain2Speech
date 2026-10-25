"""Independently account E021 decoding and produce paired calibration curves."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from research.audit import save_json,digest
from preprocessing_comparison import TRANSFORMS
from run_preprocessing_comparison import BUDGETS,checked


def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=row
    return prev[-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve();protocol,data=checked(run)
    execution=json.loads((run/'execution-contract.json').read_text());decoder=json.loads((run/'decoder-identity.json').read_text())
    assert decoder['image_id']==execution['decoder_image_id'] and decoder['scoring']==protocol['decoding']['scoring']
    assert all(decoder['lm_files'][Path(p).name]==h for p,h in execution['LM_files'].items())
    days=list(protocol['accounting']['evaluation']);splits=json.loads((run/'splits.json').read_text());result={};by_arm={}
    for arm in TRANSFORMS:
        curves=[];by_arm[arm]={}
        for budget in BUDGETS:
            e=[];u=[];pe=[];pu=[];labels=[]
            for day in days:
                path=run/arm/day/str(budget);rows=json.loads((path/'decoded.json').read_text());ident=json.loads((path/'export-complete.json').read_text())
                assert [r['id']for r in rows]==[r['id']for r in splits['evaluation'][day]]
                n=len(splits['calibration'][day])if budget=='full'else budget
                assert ident['calibration_ids']==[r['id']for r in splits['calibration'][day][:n]]
                for name,h in ident['artifacts_sha256'].items():assert digest(path/name)==h
                with np.load(path/'logits.npz')as archive:
                    for i,row in enumerate(rows):
                        source=data['evaluation'][day][i];assert row['text']==source['text']
                        assert row['reference_phones']==source['labels'].tolist()
                        logits=archive[f'trial_{i:04d}'];assert logits.shape==((source['bins']-32)//4+1,41)
                        assert np.isfinite(logits).all();previous=40;sequence=[]
                        for token in logits.argmax(-1).tolist():
                            if token!=40 and token!=previous:sequence.append(token)
                            previous=token
                        assert sequence==row['predicted_phones']
                        assert distance(row['text'].split(),row['decoded_text'].split())==row['wer']['errors']
                        assert distance(row['reference_phones'],row['predicted_phones'])==row['per']['errors']
                e.append(sum(r['wer']['errors']for r in rows));u.append(sum(r['wer']['reference_units']for r in rows))
                pe.append(sum(r['per']['errors']for r in rows));pu.append(sum(r['per']['reference_units']for r in rows));labels.append(n)
            arrays=np.array([e,u,pe,pu]);by_arm[arm][str(budget)]=arrays
            rng=np.random.default_rng(101);draws=np.array([arrays[:,rng.integers(0,len(days),len(days))].sum(1)for _ in range(2000)])
            curves.append(dict(budget=budget,total_calibration_labels=sum(labels),labels_by_day=dict(zip(days,labels)),
                WER=sum(e)/sum(u),PER=sum(pe)/sum(pu),WER_CI=np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist(),
                per_day_WER=dict(zip(days,(np.array(e)/u).tolist()))))
        first=next((r['budget']for r in curves if r['WER']<=protocol['WER_target']),None)
        result[arm]=dict(curve=curves,first_grid_budget_reaching_target=first if first is not None else 'not reached',
            qualification='Point estimate crossing; no interpolation, fixed five-day conditional uncertainty')
    paired={}
    for arm in ('sqrt','anscombe'):
        paired[arm]={}
        for budget in BUDGETS:
            a=by_arm[arm][str(budget)];b=by_arm['raw'][str(budget)];rng=np.random.default_rng(101);delta=[]
            for rep in range(2000):
                ix=rng.integers(0,len(days),len(days));aa=a[:,ix].sum(1);bb=b[:,ix].sum(1);delta.append(aa[0]/aa[1]-bb[0]/bb[1])
            paired[arm][str(budget)]=dict(delta_WER=a[0].sum()/a[1].sum()-b[0].sum()/b[1].sum(),CI=np.quantile(delta,[.025,.975]).tolist())
    save_json(run/'comparison.json',dict(arms=result,paired_vs_raw=paired,WER_target=protocol['WER_target'],
        uncertainty='2000 paired session-level draws, seed101, only five days/one participant/one model seed; exploratory'))
    save_json(run/'comparison-verification.json',dict(status='pass',cells=105,evaluated_sentence_occurrences=4200,
        independent_word_phone_edit_distance=True,paired_ids_and_prefixes=True,source_identity='checked',
        implementation_sha256=digest(__file__)))
    print('Comparison verified; figures/report reconciliation still required',flush=True)


if __name__=='__main__':main()
