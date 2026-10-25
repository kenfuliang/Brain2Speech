"""E019: matched offline normalization x trained-input-adapter diagnostic.

Only existing weights are used. Three conditions are newly inferred; G0 reuses
sealed E018 trial outputs. Daily moments use the whole training partition and
are explicitly transductive, not causal adaptation or an accuracy benchmark.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.io import loadmat
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import analyze_neural_drift as prior
from research.audit import digest,save_json,space_guard
from research.metrics import edits,greedy

CONDITIONS={'G0':('global','reference'),'Gd':('global','native'),
            'D0':('day','reference'),'Dd':('day','native')}
CONTRASTS={'normalization_at_reference':{'D0':1,'G0':-1},
           'normalization_at_native':{'Dd':1,'Gd':-1},
           'adapter_at_global':{'Gd':1,'G0':-1},
           'adapter_at_day':{'Dd':1,'D0':-1},
           'joint_minus_baseline':{'Dd':1,'G0':-1},
           'interaction':{'Dd':1,'D0':-1,'Gd':-1,'G0':1}}


def condition_spec(condition,session,protocol,normalization):
    norm,adapter=CONDITIONS[condition]
    stats=normalization['global'] if norm=='global' else normalization['day'][session]
    index=0 if adapter=='reference' else protocol['day_to_adapter'][session]
    return np.array(stats['mean']),np.array(stats['variance']),index


def infer(model,x,adapter,device):
    captured=[]
    hook=model.layers[-1].register_forward_hook(lambda _,args,result:captured.append(result[0]))
    try:
        with torch.inference_mode():
            y=model(torch.from_numpy(x[None]).to(device),day=adapter)[0].cpu().numpy()
            h=captured[0][0].cpu().numpy()
    finally:hook.remove()
    if h.shape!=(len(y),1024) or not np.isfinite(y).all() or not np.isfinite(h).all():
        raise FloatingPointError('Invalid/nonfinite output')
    return y,h


def destination(output,condition,row):
    return output/'trials'/condition/row['session']/f"{row['index']:04d}.npz"


def parent_trial(protocol,row):
    return prior.shard(Path(protocol['parent']),row)


def check(output):
    p=json.loads((output/'protocol.json').read_text())
    for path,sha in p['input_hashes'].items():
        if digest(path)!=sha:raise ValueError('Changed input '+path)
    for path,sha in p['source_hashes'].items():
        if digest(path)!=sha:raise ValueError('Changed implementation '+path)
    for name,sha in json.loads((output/'derived-inputs.json').read_text()).items():
        if digest(output/name)!=sha:raise ValueError('Changed derived input '+name)
    rows=json.loads((output/'cohort.json').read_text())
    n=json.loads((output/'normalization.json').read_text())
    return p,rows,n


def verify_parent(protocol):
    parent=Path(protocol['parent']);seal=json.loads((parent/'seal.json').read_text())
    if seal['status']!='complete':raise ValueError('Completed E018 required')
    for name,sha in seal['files'].items():
        if digest(parent/name)!=sha:raise ValueError('E018 evidence changed '+name)
    return len(seal['files'])


def prepare(args):
    if args.output.exists():check(args.output);return
    space_guard()
    parent=args.parent.resolve(); pp=json.loads((parent/'protocol.json').read_text())
    identity=json.loads(Path(pp['identity']).read_text());m=json.loads(Path(pp['manifest']).read_text())
    cfg=identity['configuration']['dataset']
    mapping=dict(zip(cfg['sessions'],cfg['datasetToLayerMap']))
    days=[d for d in pp['days'] if d in mapping]
    rows=[r for r in json.loads((parent/'cohort.json').read_text()) if r['session'] in days]
    if len(days)!=17 or len(rows)!=6000 or len({r['id'] for r in rows})!=6000:
        raise ValueError('Matched17day/6000trial cohort required')
    if set(days)&set(m['folds']['final']):raise ValueError('Final recording access forbidden')
    if any(not isinstance(mapping[d],int) or not 0<=mapping[d]<19 for d in days):
        raise ValueError('Invalid trained-adapter mapping')
    protocol=dict(experiment='E019',date_local='2026-10-11',parent=str(parent),
        manifest=pp['manifest'],checkpoint=pp['checkpoint'],days=days,trials=6000,
        day_to_adapter={d:mapping[d] for d in days},
        excluded_no_trained_adapter=[d for d in pp['days'] if d not in mapping],
        excluded_final_days=m['folds']['final'],conditions=CONDITIONS,
        new_inferences=18000,baseline='G0 exact E018 outputs on matched6000 subset, not original6680 aggregate',
        source_hashes={str(path.resolve()):digest(path) for path in [Path(__file__),
            ROOT/'reproduction/analyze_neural_drift.py',ROOT/'reproduction/pytorch_baseline.py',ROOT/'research/metrics.py']},
        input_hashes={path:sha for path,sha in pp['input_hashes'].items()
            if path not in [str(ROOT/m['sessions'][d]['train']['path']) for d in pp['days'] if d not in days]},
        preprocessing=dict(feature_transform='untransformed counts/power',smoothing='E018 fixed causal9bin',
            global_normalization='unchanged E018 equal-trial initial-training stats',
            day='equal-trial moments across all training-partition observations of that day',
            daily_normalization_is_offline_transductive=True,labels_used_for_normalization=False,
            reference_adapter=0,native_adapter='released recipe datasetToLayerMap, trained weights only',
            no_weight_updates=True,no_source_recipe_reproduction=True),
        metrics=dict(primary='matched corpus greedy PER; exact S/D/I and phone denominator',
            secondary='macro/worst/day PER and latent centroid/within distance on frozen E018PCA32',
            comparisons=CONTRASTS,seed=101,repetitions=2000,
            uncertainty='paired whole-block resamples within fixed17days; same draw across all four conditions',
            effect_rule='negative candidate-minus-control PER interval upper bound<0; exploratory, not population/multiplicity proof',
            no_task_drift_claim='PER improvement plus latent stabilization does not prove biological stability'),
        safety=dict(batch_size=1,serial=True,preferred_device='MPS',minimum_free_GiB=30,
            stop='nonfinite, corrupt identity, unsupported operation without explicit CPU fallback, storage guard',
            no_training=True,no_WFST=True,no_v1_restart=True),
        reporting=dict(primary='user-selected canonical Markdown research report, append only; no competing report',
            audience='technical research; source/measurement/model definitions and limits',
            visuals='static PNG four-condition PER, matched day heatmap, same-basis latent distances',
            palette='blue #2563A6,orange #C86D26,neutral; solid/open/dashed distinctions',
            caveat='training-partition inference; historical supervision in native adapters; target-day moments see future observations'))
    verify_parent(protocol)
    day_norm={}
    for day in days:
        means=[];seconds=[]
        for row in rows:
            if row['session']!=day:continue
            with np.load(parent_trial(protocol,row)) as v:
                means.append(v['raw_mean']);seconds.append(v['raw_second'])
        mu=np.mean(means,0);variance=np.maximum(np.mean(seconds,0)-mu**2,1e-4)
        day_norm[day]=dict(mean=mu.tolist(),variance=variance.tolist(),trials=len(means))
    global_norm=json.loads((parent/'normalization.json').read_text())
    args.output.mkdir(parents=True)
    prior.write_once(args.output/'protocol.json',protocol)
    prior.write_once(args.output/'cohort.json',rows)
    prior.write_once(args.output/'normalization.json',dict(global_stats_exposure='initial14days',
        **{'global':global_norm,'day':day_norm}))
    prior.write_once(args.output/'derived-inputs.json',{name:digest(args.output/name)
        for name in ['cohort.json','normalization.json']})
    print('prepared17days/6000trials;18000newcondition inferences',flush=True)


def extract(args,pilot=False):
    out=args.output;p,rows,n=check(out)
    if (out/'seal.json').exists():raise RuntimeError('Completed run sealed')
    if not pilot and not (out/'pilot.json').exists():raise RuntimeError('Pilot required')
    torch.set_num_threads(4)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=prior.PublishedGRU().eval()
    model.load_state_dict(torch.load(p['checkpoint'],map_location='cpu',weights_only=False)['model'])
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.to(device)
    initial_parameters={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    if device.type!='mps':prior.write_once(out/'fallback.json',dict(device='cpu',reason='MPS unavailable'))
    if pilot:
        rows=[r for d in [p['days'][0],p['days'][7],p['days'][-1]]
            for r in [next(x for x in rows if x['session']==d),next(x for x in reversed(rows) if x['session']==d)]]
    started=time.monotonic();done=0;probes=[]
    manifest=json.loads(Path(p['manifest']).read_text())
    for day in p['days']:
        subset=[r for r in rows if r['session']==day]
        if not subset:continue
        mat=loadmat(ROOT/manifest['sessions'][day]['train']['path'],variable_names=['tx1','spikePow'])
        for row in subset:
            raw=prior.trial_raw(mat,row)
            with np.load(parent_trial(p,row)) as base:
                reference=base['reference'].copy()
                baseline_mean=base['latent_mean'].copy();baseline_phones=base['phones'].copy()
            if pilot:
                mu,var,adapter=condition_spec('G0',day,p,n)
                logits,hidden=infer(model,prior.preprocess(raw,mu,var),adapter,device)
                assert np.array_equal(greedy(logits),baseline_phones)
                assert np.allclose(hidden.astype(np.float64).mean(0),baseline_mean,atol=1e-10)
            for condition in ['Gd','D0','Dd']:
                path=destination(out,condition,row);meta=path.with_suffix('.json')
                if path.exists() and not pilot:
                    if not meta.exists() or digest(path)!=json.loads(meta.read_text())['sha256']:
                        raise ValueError('Incomplete/corrupt shard; preserve')
                    done+=1;continue
                space_guard();mu,var,adapter=condition_spec(condition,day,p,n)
                x=prior.preprocess(raw,mu,var);t=time.monotonic()
                try:logits,hidden=infer(model,x,adapter,device)
                except (RuntimeError,NotImplementedError) as error:
                    if device.type!='mps' or not any(s in str(error).lower() for s in ['not implemented','not supported']):raise
                    device=torch.device('cpu');model.to(device)
                    prior.write_once(out/'fallback.json',dict(device='cpu',reason=str(error)))
                    logits,hidden=infer(model,x,adapter,device)
                values=dict(latent_mean=hidden.astype(np.float64).mean(0),
                    latent_variance=hidden.astype(np.float64).var(0),phones=np.array(greedy(logits)),
                    reference=reference,frames=np.array(len(hidden)))
                account=edits(reference.tolist(),values['phones'].tolist())
                if pilot:
                    a,h=infer(model,x,adapter,device)
                    assert np.array_equal(a,logits) and np.array_equal(h,hidden)
                    if day==p['days'][0] and condition=='Gd':
                        assert np.array_equal(values['phones'],baseline_phones)
                    probes.append(dict(id=row['id'],condition=condition,adapter=adapter,deterministic=True))
                if path.exists():
                    with np.load(path) as old:
                        assert all(np.array_equal(old[key],value) for key,value in values.items())
                else:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temporary=path.with_suffix('.partial')
                    with temporary.open('wb') as stream:np.savez_compressed(stream,**values)
                    temporary.replace(path)
                    prior.write_once(meta,dict(id=row['id'],condition=condition,sha256=digest(path),
                        adapter=adapter,per=account,device=str(device),trial_processing_seconds=time.monotonic()-t))
                done+=1
            if done%150==0 or pilot:
                observation=dict(done=done,total=len(rows)*3,elapsed_seconds=time.monotonic()-started,**prior.resource(device))
                save_json(out/'progress.json',observation)
                prior.write_once(out/'resources'/f'{time.time_ns()}.json',observation)
                print(day,done,'/',len(rows)*3,round(observation['elapsed_seconds'],1),'seconds',flush=True)
        del mat
    assert all(torch.equal(initial_parameters[key],value.detach().cpu()) for key,value in model.state_dict().items())
    if pilot:
        reloaded=prior.PublishedGRU().eval().to(device)
        reloaded.load_state_dict(torch.load(p['checkpoint'],map_location='cpu',weights_only=False)['model'])
        a,h=infer(reloaded,x,adapter,device)
        assert np.array_equal(a,logits) and np.array_equal(h,hidden)
        prior.write_once(out/'pilot.json',dict(passed=True,baseline_replay=True,checkpoint_reload=True,
            weights_unchanged=True,probes=probes,device=str(device)))
    else:
        prior.write_once(out/'extraction-complete.json',dict(trials=len(rows),new_inferences=done,
            weights_unchanged=True,parent_unchanged_files=verify_parent(p)))
    print('pilot passed' if pilot else 'collection complete',flush=True)


def read_conditions(output,p,rows):
    means={k:[] for k in CONDITIONS};errors={k:[] for k in CONDITIONS};units=[];accounts={k:[] for k in CONDITIONS}
    for row in rows:
        for condition in CONDITIONS:
            path=parent_trial(p,row) if condition=='G0' else destination(output,condition,row)
            meta=json.loads(path.with_suffix('.json').read_text())
            assert digest(path)==meta['sha256'] and meta['id']==row['id']
            with np.load(path) as value:
                e=edits(value['reference'].tolist(),value['phones'].tolist())
                if condition=='G0':units.append(len(value['reference']));reference=value['reference'].copy()
                else:assert np.array_equal(reference,value['reference'])
                assert e==meta['per']
                means[condition].append(value['latent_mean']);errors[condition].append(e['errors']);accounts[condition].append(e)
    return {k:np.stack(v) for k,v in means.items()},{k:np.array(v) for k,v in errors.items()},np.array(units),accounts


def paired_bootstrap(rows,days,errors,units,repetitions=2000):
    rng=np.random.default_rng(101);denominator=np.zeros(repetitions);numerators={k:np.zeros(repetitions) for k in errors}
    for day in days:
        blocks=sorted({r['block'] for r in rows if r['session']==day})
        indices=[np.array([i for i,r in enumerate(rows) if r['session']==day and r['block']==b]) for b in blocks]
        counts=rng.multinomial(len(blocks),np.ones(len(blocks))/len(blocks),size=repetitions)
        denominator+=counts@np.array([units[ix].sum() for ix in indices])
        for key in errors:numerators[key]+=counts@np.array([errors[key][ix].sum() for ix in indices])
    return {k:v/denominator for k,v in numerators.items()}


def summarize(output):
    p,rows,n=check(output)
    if not (output/'extraction-complete.json').exists():raise RuntimeError('Full collection required')
    means,errors,units,accounts=read_conditions(output,p,rows)
    boots=paired_bootstrap(rows,p['days'],errors,units)
    stats={};latent_stats={}
    with np.load(Path(p['parent'])/'summaries.npz') as old:
        axes=old['pca_axes'].copy();center=old['pca_mean'].copy()
    for key in CONDITIONS:
        by_day={}
        for day in p['days']:
            ix=np.array([i for i,r in enumerate(rows) if r['session']==day])
            by_day[day]=dict(trials=len(ix),errors=int(errors[key][ix].sum()),reference_units=int(units[ix].sum()),
                per=float(errors[key][ix].sum()/units[ix].sum()))
        stats[key]=dict(corpus=float(errors[key].sum()/units.sum()),errors=int(errors[key].sum()),
            reference_units=int(units.sum()),ci95=np.quantile(boots[key],[.025,.975]).tolist(),
            macro_day=float(np.mean([x['per'] for x in by_day.values()])),worst_day=max(x['per'] for x in by_day.values()),
            by_day=by_day,S=sum(x['substitutions'] for x in accounts[key]),
            D=sum(x['deletions'] for x in accounts[key]),I=sum(x['insertions'] for x in accounts[key]))
        projected=(means[key]-center)@axes.T
        latent_stats[key]=prior.day_statistics(projected,rows,p['days'])
    effects={}
    for name,coefficients in CONTRASTS.items():
        value=sum(c*stats[k]['corpus'] for k,c in coefficients.items())
        samples=sum(c*boots[k] for k,c in coefficients.items())
        interval=np.quantile(samples,[.025,.975]).tolist()
        effects[name]=dict(candidate_minus_control=value,ci95=interval,
            lower_PER_supported=interval[1]<0 if name!='interaction' else None)
    result=dict(trials=len(rows),days=p['days'],conditions=stats,effects=effects,latent=latent_stats,
        WER='Not measured',CER='Not measured',loss='Not measured; no training',
        interpretations=['Retrospective training-partition diagnostic; no independent benchmark improvement',
            'Daily moments include target-day future observations; native adapters reflect historical supervised training',
            'Same E018 PCA axes, fit4880initial-training means, applied without refitting',
            'Mean recentering is expected by construction; task PER is needed for functional interpretation',
            'Paired conditional fixed-day block intervals; not population, biological cause or multiplicity-corrected claims'])
    prior.write_once(output/'metrics.json',result)
    if not (output/'summaries.npz').exists():
        np.savez_compressed(output/'summaries.npz',units=units,**{k+'_errors':v for k,v in errors.items()},
            **{k+'_latent_mean':v for k,v in means.items()})
    print(json.dumps({k:round(v['corpus']*100,3) for k,v in stats.items()}),flush=True)


def draw(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    m=json.loads((output/'metrics.json').read_text());keys=list(CONDITIONS)
    texts=['Fixed norm\nReference adapter','Fixed norm\nNative adapter','Day norm\nReference adapter','Day norm\nNative adapter']
    figdir=output/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

    def create(title,subtitle,n=2,width=13,height=6):
        fig,axes=plt.subplots(1,n,figsize=(width,height),squeeze=False)
        fig.subplots_adjust(top=.77,bottom=.26,left=.09,right=.95,wspace=.35)
        fig.text(.09,.96,title,fontsize=16,fontweight='bold',color='#222222')
        fig.text(.09,.87,subtitle,fontsize=10,color='#555555');fig.text(.97,.95,'✿',ha='right',fontsize=15,color='#555555')
        return fig,axes.ravel()

    fig,axes=create('Normalization and input-layer diagnostic',
        'T12 • matched6,000training trials /17days • existing E017 weights only • same trailing smoothing')
    rates=np.array([m['conditions'][k]['corpus']*100 for k in keys]);ci=np.array([m['conditions'][k]['ci95'] for k in keys])*100
    for ax,value,title in zip(axes,[rates,np.array([m['latent'][k]['median_between'] for k in keys])],
                              ['Corpus greedy phoneme error','Between-day latent centroid RMS']):
        bars=ax.bar(range(4),value,color=['#EEEEEE','#2563A6','#FFFFFF','#C86D26'],edgecolor='#555555',linewidth=.8)
        bars[2].set_hatch('//');ax.set_xticks(range(4),texts,fontsize=9);ax.set_title(title)
        ax.set_ylim(0,max(value)*1.2)
        for i,v in enumerate(value):ax.text(i,v+max(value)*.03,f'{v:.2f}',ha='center',fontsize=10)
        if ax is axes[0]:
            ax.vlines(range(4),ci[:,0],ci[:,1],color='#333333',linewidth=1.2);ax.set_ylabel('PER (%)')
        else:ax.set_ylabel('Same frozen E018 PCA32 units')
    fig.text(.09,.07,'PER bars: conditional95%paired whole-block intervals. Latent bars are descriptive; no alignment refitting.\nDay statistics see all training observations; native adapters already supervised. No online or biological-drift claim.',fontsize=9,color='#555555')
    fig.savefig(figdir/'factorial-summary.png',dpi=160);plt.close(fig)
    fig,axes=create('Matched day-by-day phoneme errors',
        '17days • each column shares the same trials, reference phones and encoder • transductive normalization',1,13,5.5)
    ax=axes[0];matrix=np.array([[m['conditions'][k]['by_day'][d]['per']*100 for d in m['days']] for k in keys])
    colors=LinearSegmentedColormap.from_list('blue_white',['#FFFFFF','#2563A6'])
    im=ax.imshow(matrix,aspect='auto',cmap=colors,vmin=0,vmax=100)
    ax.set_yticks(range(4),[t.replace('\n',' / ') for t in texts]);ax.set_xticks(range(17),[d[9:] for d in m['days']],rotation=45)
    for i in range(4):
        for j in range(17):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=8,color='#FFFFFF' if matrix[i,j]>60 else '#222222')
    fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Greedy PER (%)')
    fig.text(.09,.07,'June23/July29 excluded: no trained native adapter. Training-partition rates, not independent future-day accuracy.\nConditional uncertainty and exact phone-edit accounting are saved with the report.',fontsize=9,color='#555555')
    fig.savefig(figdir/'per-by-day.png',dpi=160);plt.close(fig)
    prior.write_once(output/'figure-provenance.json',dict(metrics_sha256=digest(output/'metrics.json'),
        files={p.name:digest(p) for p in sorted(figdir.glob('*.png'))}))


def verify(output):
    p,rows,n=check(output);means,errors,units,accounts=read_conditions(output,p,rows)
    m=json.loads((output/'metrics.json').read_text());boots=paired_bootstrap(rows,p['days'],errors,units)
    for key in CONDITIONS:
        assert int(errors[key].sum())==m['conditions'][key]['errors']
        assert np.allclose(np.quantile(boots[key],[.025,.975]),m['conditions'][key]['ci95'])
    for name,c in CONTRASTS.items():
        interval=np.quantile(sum(v*boots[k] for k,v in c.items()),[.025,.975])
        assert np.allclose(interval,m['effects'][name]['ci95'])
    assert m['conditions']['G0']['reference_units']==m['conditions']['Dd']['reference_units']
    assert np.array_equal(errors['G0'][:280],errors['Gd'][:280])
    preserved=verify_parent(p)
    prior.write_once(output/'verification.json',dict(passed=True,trials=6000,conditions=4,
        original_E018_files_preserved=preserved,paired_intervals_recomputed=True,
        adapter0_native_parity_on_first_day=True,source_checkpoint_hash_checked=True,
        no_final_recording_access=True,assessment='Share with retrospective/transductive caveats'))
    print('verification passed',flush=True)


def seal(output):
    assert json.loads((output/'verification.json').read_text())['passed']
    prior.write_once(output/'status.json',dict(status='complete',trials=6000,days=17,training='not run',v1='paused unchanged'))
    prior.write_once(output/'seal.json',dict(status='complete',files={str(p.relative_to(output)):digest(p)
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='seal.json'}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=ROOT/'research/runs/e018-neural-drift-20261011-01')
    parser.add_argument('--stage',choices=['prepare','pilot','collect','report','verify','seal'],required=True)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args)
    elif args.stage in ('pilot','collect'):extract(args,args.stage=='pilot')
    elif args.stage=='report':summarize(args.output);draw(args.output)
    elif args.stage=='verify':verify(args.output)
    elif args.stage=='seal':seal(args.output)


if __name__=='__main__':main()
