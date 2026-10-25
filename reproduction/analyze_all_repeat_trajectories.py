"""All 77 cross-day repeats, strict-duration controls; frozen CPU instrument."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_neural_drift import ROOT, E017_SHA, preprocess, fixed_inference
from plot_repeated_latent_trajectory import load_model
from quantify_trajectory_correction import compare, summary
from research.audit import digest, save_json, space_guard


def controls(pair, lookup, rows, repeated):
    target = lookup[pair['right']]
    candidates = [r for r in rows if r['session'] == target['session'] and
                  r['block'] == target['block'] and r['text'] not in repeated and
                  .9 <= r['bins'] / target['bins'] <= 1.1]
    candidates.sort(key=lambda r: (abs(math.log(r['bins']/target['bins'])), r['index']))
    return [r['id'] for r in candidates[:5]]


def phase(h):
    old = np.linspace(0, 1, len(h)); new = np.linspace(0, 1, 64)
    return np.stack([np.interp(new, old, h[:, j]) for j in range(h.shape[1])], axis=1)


def main(out):
    torch.set_num_threads(2); space_guard()
    old = ROOT/'research/runs/e018-neural-drift-20261011-01'
    manifest_path = ROOT/'research/data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    seal = json.loads((old/'seal.json').read_text())
    for name in ('cohort.json','normalization.json','protocol.json','repeated-content.json'):
        assert digest(old/name) == seal['files'][name]
    rows = json.loads((old/'cohort.json').read_text()); lookup = {r['id']: r for r in rows}
    pairs = json.loads((old/'repeated-content.json').read_text())['pairs']
    assert len([p for p in pairs if p['cross_day']]) == 77
    assert len([p for p in pairs if not p['cross_day']]) == 7
    assert len(rows) == 6680 and all(r['partition'] == 'train' for r in rows)
    assert not {r['session'] for r in rows} & set(manifest['folds']['final'])
    repeated = {p['prompt'] for p in pairs}
    plan = [dict(p, controls=controls(p,lookup,rows,repeated)) for p in pairs]
    ids = {p[k] for p in plan for k in ('left','right')} | {i for p in plan for i in p['controls']}
    norm = json.loads((old/'normalization.json').read_text())
    checkpoint = Path(json.loads((old/'protocol.json').read_text())['checkpoint'])
    assert digest(checkpoint) == E017_SHA
    protocol = dict(question='Do all exact cross-day repeats retain content-specific latent trajectory similarity?',
        cohort_trials=6680, cross_day_pairs=77, within_day_pairs=7, unique_extraction_trials=len(ids),
        comparisons=plan, instrument='Frozen E017 finalGRU1024D; adapter0; trial reset; E018 fixed normalization and trailing9bin filter; CPUfloat64/2threads',
        primary='Same elapsed-time frames, cropped to shorter trajectory; full1024D RMS, centred shape and first-difference cosine',
        sensitivity='64-point normalized-duration interpolation; not phoneme alignment, not causal deployment processing',
        controls='Up to5 different non-repeat prompts, same target day/block, duration within +/-10% of target repeat; deterministic nearest-duration order; no relaxed fallback',
        uncertainty='Descriptive distributions and leave-one-session-out sensitivity; sessions/controls reused, no iid pair/bin tests or population CI. Seven within-day prompts are different content/days, not matched noise ceilings.',
        boundary='No retraining, correction, decoder, final-day recording access or biological drift attribution; E017 historically exposed cohort and two final days',
        chart=dict(surface='Existing canonical Markdown research report with PNG',family='Distribution dots plus paired specificity scatter',palette='Blue cross-day, gold within-day, neutral controls; markers distinguish groups'),
        source_hashes={str(p):digest(p) for p in [manifest_path, checkpoint,old/'cohort.json',old/'normalization.json',old/'repeated-content.json',Path(__file__),ROOT/'reproduction/quantify_trajectory_correction.py',ROOT/'reproduction/analyze_neural_drift.py',ROOT/'reproduction/plot_repeated_latent_trajectory.py',ROOT/'reproduction/pytorch_baseline.py']})
    if out.exists():
        assert json.loads((out/'protocol.json').read_text()) == protocol, 'Resume identity mismatch'
    else:
        out.mkdir(parents=True);save_json(out/'protocol.json',protocol)
    if (out/'seal.json').exists(): return
    caches = [ROOT/'research/runs/trajectory-correction-20261012-02',ROOT/'research/runs/repeated-latent-trajectories-20261012-02']
    cache_seals = {p:json.loads((p/'seal.json').read_text())['files'] for p in caches}
    prior=ROOT/'research/runs/all-repeat-trajectories-20261012-01'
    if prior.exists() and prior!=out:
        original=json.loads((prior/'protocol.json').read_text())
        assert original['comparisons']==plan and original['instrument']==protocol['instrument']
        for path,h in original['source_hashes'].items():
            if path!=str(Path(__file__)): assert digest(path)==h
        cache_seals[prior]={p.name:digest(p) for p in prior.glob('*.npz')}
        save_json(out/'preserved-extraction-cache.json',dict(source=str(prior),reason='Prior extraction complete; report serialization failed on NumPy integer, not inference',files=cache_seals[prior]))
        caches.append(prior)
    model = None; histories = {}; hashes = {}; cached = 0; started = time.monotonic()
    for day in sorted({lookup[i]['session'] for i in ids}):
        selected = [r for r in rows if r['session']==day and r['id'] in ids]
        source = ROOT/manifest['sessions'][day]['train']['path']
        assert digest(source)==manifest['sessions'][day]['train']['sha256']; hashes[str(source)]=digest(source)
        mat = None
        for r in selected:
            space_guard(); name=f"{day}-{r['index']:04d}.npz"; dest=out/name
            existing = dest if dest.exists() else next((p/name for p in caches if (p/name).exists()),None)
            if existing:
                if existing != dest: assert digest(existing)==cache_seals[existing.parent][name]
                with np.load(existing) as v: h=v['hidden'].copy()
                cached += 1
            else:
                if mat is None: mat=loadmat(source,variable_names=['tx1','spikePow'])
                if model is None: model=load_model(checkpoint)
                raw=np.concatenate([mat['tx1'].ravel()[r['index']][:,:128],mat['spikePow'].ravel()[r['index']][:,:128]],axis=1).astype(np.float64)
                assert raw.shape==(r['bins'],256) and np.isfinite(raw).all()
                x=preprocess(raw,np.array(norm['mean']),np.array(norm['variance'])).astype(np.float64)
                _,h=fixed_inference(model,x,torch.device('cpu'))
                # First fresh trial: deterministic reload, state reset and future-prefix check.
                if not (out/'pilot.json').exists():
                    _,again=fixed_inference(model,x,torch.device('cpu'));np.testing.assert_array_equal(h,again)
                    reload=load_model(checkpoint);_,again=fixed_inference(reload,x,torch.device('cpu'));np.testing.assert_array_equal(h,again);del reload
                    cut=min(96,len(x)-1);_,prefix=fixed_inference(model,x[:cut],torch.device('cpu'))
                    np.testing.assert_allclose(prefix,h[:len(prefix)],rtol=0,atol=1e-12)
                    save_json(out/'pilot.json',dict(status='pass',id=r['id'],reload='exact',state_reset='exact',prefix_atol=1e-12))
            assert h.shape==((r['bins']-32)//4+1,1024) and np.isfinite(h).all()
            histories[r['id']]=h
            if not dest.exists(): np.savez_compressed(dest,hidden=h,seconds=(32+4*np.arange(len(h)))*.02)
            save_json(out/'progress.json',dict(extracted=len(histories),total=len(ids),seconds=time.monotonic()-started,free_bytes=space_guard()))
        del mat
        print('extracted',day,len(histories),'/',len(ids),flush=True)
    records=[]
    for p in plan:
        a,b=histories[p['left']],histories[p['right']]
        metrics={}
        for mode,transform in [('elapsed',lambda h:h),('phase',phase)]:
            positive=compare(transform(a),transform(b))
            negative=[compare(transform(a),transform(histories[i])) for i in p['controls']]
            metrics[mode]=dict(positive=positive,negatives=negative)
            if negative:
                metrics[mode].update(shape_advantage=positive['shape_cosine']-np.median([n['shape_cosine'] for n in negative]),
                    RMS_advantage=np.median([n['RMS'] for n in negative])-positive['RMS'])
        records.append(dict(**p,metrics=metrics))
    result={}
    for mode in ('elapsed','phase'):
        result[mode]={}
        for name,flag in [('cross_day',True),('within_day',False)]:
            group=[r for r in records if r['cross_day']==flag]
            result[mode][name]={k:summary([r['metrics'][mode]['positive'][k] for r in group]) for k in ('RMS','shape_cosine','velocity_cosine','centroid_RMS')}
            controlled=[r for r in group if r['controls']]
            result[mode][name].update(pairs=len(group),with_controls=len(controlled),with_at_least3_controls=sum(len(r['controls'])>=3 for r in group),
                negative_instances=sum(len(r['controls']) for r in group))
            if controlled:
                for k in ('shape_advantage','RMS_advantage'):
                    result[mode][name][k]=summary([r['metrics'][mode][k] for r in controlled])
                result[mode][name]['positive_shape_advantage_pairs']=int(sum(r['metrics'][mode]['shape_advantage']>0 for r in controlled))
    cross=[r for r in records if r['cross_day'] and r['controls']]
    loo=[]
    for day in sorted({lookup[i]['session'] for i in ids}):
        left=[r for r in cross if day not in (r['left_day'],r['right_day'])]
        if left: loo.append(dict(excluded=day,pairs=len(left),median_shape_advantage=float(np.median([r['metrics']['elapsed']['shape_advantage'] for r in left]))))
    save_json(out/'metrics.json',dict(summary=result,records=records,leave_one_session_out=loo,source_hashes=hashes,cached_trials=cached,wall_seconds=time.monotonic()-started))
    fig,axs=plt.subplots(1,3,figsize=(16,6));fig.subplots_adjust(left=.07,right=.97,top=.75,bottom=.28,wspace=.38)
    rng=np.random.default_rng(101)
    for ax,key,title in [(axs[0],'shape_cosine','Trajectory shape'),(axs[1],'RMS','Absolute trajectory distance')]:
        for j,(flag,label,color,marker) in enumerate([(True,'Cross-day\n77 pairs','#3276ad','o'),(False,'Within-day\n7 pairs','#c58b24','^')]):
            vals=[r['metrics']['elapsed']['positive'][key] for r in records if r['cross_day']==flag]
            ax.scatter(j+rng.uniform(-.12,.12,len(vals)),vals,s=18,color=color,marker=marker,alpha=.65)
            ax.plot([j-.18,j+.18],[np.median(vals)]*2,color='#273644',lw=2)
        ax.set_xticks([0,1],['Cross-day\n77 pairs','Within-day\n7 pairs']);ax.set_title(title);ax.set_ylabel('Mean-centred cosine' if key=='shape_cosine' else 'RMS hidden-unit difference')
        ax.set_ylim((-1,1) if key=='shape_cosine' else (0,None));ax.grid(axis='y',alpha=.15)
    for flag,marker,color in [(True,'o','#3276ad'),(False,'^','#c58b24')]:
        group=[r for r in records if r['cross_day']==flag and r['controls']]
        axs[2].scatter([np.median([n['shape_cosine'] for n in r['metrics']['elapsed']['negatives']]) for r in group],
                       [r['metrics']['elapsed']['positive']['shape_cosine'] for r in group],marker=marker,color=color,s=24,label=('Cross-day' if flag else 'Within-day')+f' n={len(group)}')
    axs[2].plot([-1,1],[-1,1],':',color='#697784');axs[2].set(xlim=(-1,1),ylim=(-1,1),xlabel='Different-sentence median shape cosine',ylabel='Repeated-sentence shape cosine',title='Above diagonal: content similarity');axs[2].legend(fontsize=9)
    fig.text(.07,.93,'All repeated sentences: frozen-encoder trajectory similarity',fontsize=18,weight='bold')
    fig.text(.07,.86,f'19 permitted training days screened • 77 cross-day + 7 within-day pairs • {len(ids)} unique extracted trials',fontsize=11)
    fig.text(.97,.93,'✳',ha='right',fontsize=20,color='#3276ad')
    fig.text(.07,.07,'Full1,024D, shared elapsed time; no phoneme alignment. Controls: same target block, duration ±10%, up to5 per pair.\n'
             'Points and reused controls share sessions: no independent-pair CI. Within-day repeats use different prompts/days, not a matched ceiling.\n'
             'Frozen E017 with fixed adapter0/departed preprocessing; historical exposure. No biological drift or decoding-benefit claim.',fontsize=10,linespacing=1.5)
    fig.savefig(out/'all-repeat-similarity.png',dpi=160,facecolor='white');plt.close(fig)
    assert digest(checkpoint)==E017_SHA
    save_json(out/'provenance.json',dict(status='generated-needs-independent-QA',hashes={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
