"""Independent scalar trajectory/accounting checks and evidence seal."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from analyze_neural_drift import ROOT, preprocess, fixed_inference
from plot_repeated_latent_trajectory import load_model
from research.audit import digest, save_json


def main(out):
    p=json.loads((out/'protocol.json').read_text());m=json.loads((out/'metrics.json').read_text())
    for path,h in {**p['source_hashes'],**m['source_hashes']}.items(): assert digest(path)==h
    plan=p['comparisons']; records=m['records']; assert len(plan)==len(records)==84
    assert len({r['prompt'] for r in records if r['cross_day']})==77
    manifest=json.loads(Path(next(k for k in p['source_hashes'] if k.endswith('/manifest.json'))).read_text())
    cohort=json.loads(Path(next(k for k in p['source_hashes'] if k.endswith('/cohort.json'))).read_text())
    lookup={r['id']:r for r in cohort}; repeated={r['prompt'] for r in plan}
    assert not {r['session'] for r in cohort}&set(manifest['folds']['final'])
    def read(i):
        r=lookup[i]
        with np.load(out/f"{r['session']}-{r['index']:04d}.npz") as v:
            h=v['hidden'].copy();seconds=v['seconds'].copy()
        assert h.shape==((r['bins']-32)//4+1,1024) and np.isfinite(h).all()
        np.testing.assert_allclose(seconds,(32+4*np.arange(len(h)))*.02,rtol=0,atol=1e-12)
        return h
    torch.set_num_threads(2)
    checkpoint=Path(next(k for k in p['source_hashes'] if k.endswith('.pt')))
    model=load_model(checkpoint)
    norm=json.loads(Path(next(k for k in p['source_hashes'] if k.endswith('/normalization.json'))).read_text())
    example_ids=[plan[0]['left'],plan[0]['right']]
    for i in example_ids:
        r=lookup[i];source=ROOT/manifest['sessions'][r['session']]['train']['path']
        mat=loadmat(source,variable_names=['tx1','spikePow'])
        raw=np.concatenate([mat['tx1'].ravel()[r['index']][:,:128],mat['spikePow'].ravel()[r['index']][:,:128]],axis=1).astype(np.float64)
        x=preprocess(raw,np.array(norm['mean']),np.array(norm['variance'])).astype(np.float64)
        _,h=fixed_inference(model,x,torch.device('cpu'));np.testing.assert_array_equal(h,read(i))
        del mat
    del model
    def independent(a,b,mode):
        if mode=='phase':
            def interp(h):
                t=np.linspace(0,1,len(h));return np.column_stack([np.interp(np.linspace(0,1,64),t,h[:,j]) for j in range(1024)])
            a,b=interp(a),interp(b)
        n=min(len(a),len(b));a,b=a[:n],b[:n]
        squared=sum(float(np.dot(x-y,x-y)) for x,y in zip(a,b))/(n*1024)
        ac=a-a.mean(0);bc=b-b.mean(0)
        dot=sum(float(np.dot(x,y)) for x,y in zip(ac,bc))
        na=sum(float(np.dot(x,x)) for x in ac);nb=sum(float(np.dot(x,x)) for x in bc)
        return np.sqrt(squared),dot/np.sqrt(na*nb)
    checked=0
    for expected,r in zip(plan,records):
        for key in expected: assert r[key]==expected[key]
        a=read(r['left']);b=read(r['right']); target=lookup[r['right']]
        candidates=[x for x in cohort if x['session']==target['session'] and x['block']==target['block'] and x['text'] not in repeated and .9<=x['bins']/target['bins']<=1.1]
        candidates.sort(key=lambda x:(abs(np.log(x['bins']/target['bins'])),x['index']))
        assert r['controls']==[x['id'] for x in candidates[:5]]
        for mode in ('elapsed','phase'):
            g=r['metrics'][mode]
            for h,stored in [(b,g['positive'])]+[(read(i),v) for i,v in zip(r['controls'],g['negatives'])]:
                rms,cos=independent(a,h,mode)
                np.testing.assert_allclose([rms,cos],[stored['RMS'],stored['shape_cosine']],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(stored['RMS']**2,stored['centroid_RMS']**2+stored['centered_RMS']**2,atol=1e-12)
                checked+=1
            if r['controls']:
                np.testing.assert_allclose(g['shape_advantage'],g['positive']['shape_cosine']-np.median([v['shape_cosine'] for v in g['negatives']]),atol=1e-12)
    for mode in ('elapsed','phase'):
        for key,flag in [('cross_day',True),('within_day',False)]:
            group=[r for r in records if r['cross_day']==flag]
            for metric in ('RMS','shape_cosine','velocity_cosine','centroid_RMS'):
                vals=[r['metrics'][mode]['positive'][metric] for r in group]
                np.testing.assert_allclose(m['summary'][mode][key][metric]['median'],np.median(vals),atol=1e-12)
    save_json(out/'verification.json',dict(status='pass',independent_pair_checks=checked,independent_source_reinference=example_ids,accounting='77 cross-day and7 within-day, controls strictly matched',source_hashes='pass',summary_medians='pass',script_sha256=digest(__file__)))
    print('verified',checked,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);main(a.parse_args().output.resolve())
