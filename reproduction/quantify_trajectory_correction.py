"""Disjoint-prefix latent moment correction, held repeat/control trajectories."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

from analyze_neural_drift import ROOT, E017_SHA, preprocess, fixed_inference, shard
from plot_repeated_latent_trajectory import load_model
from research.audit import digest, save_json, space_guard

DAYS = ['t12.2022.06.07','t12.2022.06.14']
MODES = ['raw','offset','offset_gain']
FOCAL = "she said in blake's voice"


def moments(means, variances):
    mean = means.mean(0)
    var = np.maximum((variances+means**2).mean(0)-mean**2,0)
    return mean,np.sqrt(var)


def correction(a,b):
    ma,sa = a; mb,sb = b
    ratio = sa/np.maximum(sb,1e-6)
    return dict(reference_mean=ma,target_mean=mb,gain=np.clip(ratio,.5,2.),
                clipped=int(np.sum((ratio<.5)|(ratio>2.))),zero_scale=int(np.sum(sb<1e-6)))


def apply(h,fit,mode):
    if mode == 'raw': return h
    centered = h-fit['target_mean']
    return centered*(fit['gain'] if mode=='offset_gain' else 1)+fit['reference_mean']


def cosine(a,b):
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denominator) if denominator>1e-12 else None


def compare(a,b):
    lengths = (len(a),len(b))
    n = min(lengths);a=a[:n];b=b[:n]
    ac=a-a.mean(0);bc=b-b.mean(0)
    return dict(RMS=float(np.sqrt(np.mean((a-b)**2))),
                shape_cosine=cosine(ac,bc),velocity_cosine=cosine(np.diff(a,axis=0),np.diff(b,axis=0)),
                centroid_RMS=float(np.sqrt(np.mean((a.mean(0)-b.mean(0))**2))),
                centered_RMS=float(np.sqrt(np.mean((ac-bc)**2))),frames=n,
                coverage=[n/length for length in lengths])


def summary(values):
    v=np.array(values,dtype=float)
    return dict(mean=float(v.mean()),median=float(np.median(v)),IQR=np.quantile(v,[.25,.75]).tolist(),range=[float(v.min()),float(v.max())])


def main(out):
    assert not out.exists();space_guard();torch.set_num_threads(2)
    old=ROOT/'research/runs/e018-neural-drift-20261011-01'
    cached=ROOT/'research/runs/repeated-latent-trajectories-20261012-02'
    seal=json.loads((old/'seal.json').read_text())
    for name in ('cohort.json','normalization.json','summaries.npz','protocol.json'):
        assert digest(old/name)==seal['files'][name]
    manifest_path=ROOT/'research/data/manifest.json';manifest=json.loads(manifest_path.read_text())
    assert not set(DAYS)&set(manifest['folds']['final'])
    rows=json.loads((old/'cohort.json').read_text());lookup={r['id']:r for r in rows};ordinal={r['id']:i for i,r in enumerate(rows)}
    dayrows={d:[r for r in rows if r['session']==d] for d in DAYS}
    calibration={d:dayrows[d][:20] for d in DAYS}
    fit_ids={r['id'] for rs in calibration.values() for r in rs};fit_texts={r['text'] for rs in calibration.values() for r in rs}
    bytext=defaultdict(list)
    for d in DAYS:
        for r in dayrows[d]:bytext[r['text']].append(r)
    positive=[]
    for text,rs in sorted(bytext.items()):
        if {r['session'] for r in rs}!=set(DAYS) or text in fit_texts:continue
        a=[r for r in rs if r['session']==DAYS[0]][0];b=[r for r in rs if r['session']==DAYS[1]][0]
        if min(a['index'],b['index'])<20:continue
        positive.append(dict(prompt=text,left=a['id'],right=b['id']))
    assert len(positive)==10 and any(p['prompt']==FOCAL for p in positive)
    positive_texts={p['prompt'] for p in positive}
    comparisons=[]
    for pair in positive:
        b=lookup[pair['right']]
        candidates=[r for r in dayrows[DAYS[1]] if r['index']>=20 and r['block']==b['block'] and r['text'] not in fit_texts|positive_texts]
        candidates.sort(key=lambda r:(abs(math.log(r['bins']/b['bins'])),r['index']))
        controls=candidates[:20];assert len(controls)>=10
        comparisons.append(dict(**pair,controls=[r['id'] for r in controls],duration_ratios=[r['bins']/b['bins'] for r in controls]))
    eval_ids={p[k] for p in positive for k in ('left','right')}|{i for p in comparisons for i in p['controls']}
    assert not fit_ids&eval_ids
    assert not fit_texts&{lookup[i]['text'] for i in eval_ids}
    assert all(lookup[i]['index']>=20 for i in eval_ids)
    checkpoint=Path(json.loads((old/'protocol.json').read_text())['checkpoint']);assert digest(checkpoint)==E017_SHA
    norm=json.loads((old/'normalization.json').read_text())
    with np.load(old/'summaries.npz') as v:
        axes=v['pca_axes'].copy();pca_mean=v['pca_mean'].copy()
        fits={}
        for n in (10,20):
            pairs=[]
            for d in DAYS:
                indices=[ordinal[r['id']] for r in calibration[d][:n]]
                pairs.append(moments(v['latent_mean'][indices],v['latent_variance'][indices]))
            fits[n]=correction(*pairs)
    out.mkdir(parents=True)
    protocol=dict(question='Do independent offset/gain estimates selectively improve repeated-sentence trajectory similarity?',
                  focal=FOCAL,days=DAYS,fit=calibration,evaluation=comparisons,
                  correction='Target June14 to reference June7; unpaired equal-trial latent means/variances from20preceding trials/day; no labels used in numerical fit',
                  map='Offset: h-mu_target+mu_ref; offset_gain: mu_ref+clip(sigma_ref/sigma_target,.5,2)*(h-mu_target)',
                  sensitivity='First10trials/day, same evaluation; no outcome-based selection',
                  encoder='Frozen E017 finalGRU1024D; E018 initial-normalization/trailing-filter/adapter0; CPUfloat64 batch1 reset',
                  metrics='Full1024D RMS, centroid/centered RMS, centered trajectory cosine, first-difference cosine; match elapsed time only, truncate common frames, retain full arrays',
                  control='Up to20different sentences, same target-day block, closest duration to positive target trial; excludes all calibration and positive prompts',
                  specificity='Repeat rank against its controls and median-negative RMS minus positive RMS; higher advantage better, no guarantee reduced distances imply speech recovery',
                  uncertainty='Only one session pair and dependent/reused controls; no population CI or iid-bin tests. Report prompt distributions and leave-one-prompt-out sensitivity, not confidence intervals',
                  boundaries='Retrospective historically exposed model; no biological drift, WER/PER benefit or real-time calibration claim; no final data, decoder, model updates or GPU use',
                  chart=dict(family='Grouped dots + shared-axes trajectories',renderer='Static PNG in canonical research report',palette='Blue reference/raw and orange corrected, grey controls; explicit mode labels'),
                  accounting=dict(fit_trials=len(fit_ids),eval_trials=len(eval_ids),repeat_pairs=len(positive),negative_pair_instances=sum(len(p['controls']) for p in comparisons)),
                  input_hashes={str(p):digest(p) for p in [checkpoint,manifest_path,old/'cohort.json',old/'normalization.json',old/'summaries.npz',ROOT/'reproduction/plot_repeated_latent_trajectory.py',ROOT/'reproduction/analyze_neural_drift.py',ROOT/'reproduction/pytorch_baseline.py']},script_sha256=digest(__file__))
    save_json(out/'protocol.json',protocol)
    np.savez_compressed(out/'correction.npz',**{f'n{n}_{k}':v for n,fit in fits.items() for k,v in fit.items()})
    old_cached_seal=json.loads((cached/'seal.json').read_text());histories={};source_hashes={};cache_count=0
    model=None
    for day in DAYS:
        selected=[r for r in dayrows[day] if r['id'] in eval_ids]
        source=ROOT/manifest['sessions'][day]['train']['path'];expected=manifest['sessions'][day]['train']['sha256']
        assert digest(source)==expected;source_hashes[str(source)]=expected
        mat=loadmat(source,variable_names=['tx1','spikePow'])
        for r in selected:
            space_guard();name=f"{day}-{r['index']:04d}.npz";previous=cached/name
            if previous.exists():
                assert digest(previous)==old_cached_seal['files'][name]
                with np.load(previous) as v:hidden=v['hidden'].copy()
                cache_count+=1
            else:
                if model is None:model=load_model(checkpoint)
                raw=np.concatenate([mat['tx1'].ravel()[r['index']][:,:128],mat['spikePow'].ravel()[r['index']][:,:128]],axis=1).astype(np.float64)
                assert raw.shape==(r['bins'],256) and np.isfinite(raw).all()
                x=preprocess(raw,np.array(norm['mean']),np.array(norm['variance'])).astype(np.float64)
                _,hidden=fixed_inference(model,x,torch.device('cpu'))
            assert hidden.shape==((r['bins']-32)//4+1,1024) and np.isfinite(hidden).all()
            histories[r['id']]=hidden
            np.savez_compressed(out/name,hidden=hidden,seconds=(32+4*np.arange(len(hidden)))*.02)
        del mat
        print('extracted',day,len(selected),flush=True)
    result={}
    for n,fit in fits.items():
        result[n]={}
        for mode in MODES:
            records=[]
            for pair in comparisons:
                a=histories[pair['left']];b=apply(histories[pair['right']],fit,mode)
                same=compare(a,b)
                controls=[compare(a,apply(histories[i],fit,mode)) for i in pair['controls']]
                negative_distance=np.array([v['RMS'] for v in controls]);negative_shape=np.array([v['shape_cosine'] for v in controls])
                rank=float(np.mean(negative_distance>same['RMS'])+.5*np.mean(negative_distance==same['RMS']))
                shape_rank=float(np.mean(negative_shape<same['shape_cosine'])+.5*np.mean(negative_shape==same['shape_cosine']))
                records.append(dict(prompt=pair['prompt'],positive=same,negatives=controls,RMS_rank=rank,shape_rank=shape_rank,
                                    RMS_advantage=float(np.median(negative_distance)-same['RMS']),
                                    shape_advantage=float(same['shape_cosine']-np.median(negative_shape))))
            group={key:summary([r['positive'][key] for r in records]) for key in ('RMS','shape_cosine','velocity_cosine','centroid_RMS','centered_RMS')}
            group.update({key:summary([r[key] for r in records]) for key in ('RMS_rank','shape_rank','RMS_advantage','shape_advantage')})
            group['negative_RMS']=summary([np.median([v['RMS'] for v in r['negatives']]) for r in records])
            group['negative_shape_cosine']=summary([np.median([v['shape_cosine'] for v in r['negatives']]) for r in records])
            result[n][mode]=dict(records=records,summary=group,focal=next(r for r in records if r['prompt']==FOCAL))
        for a,b in zip(result[n]['raw']['records'],result[n]['offset']['records']):
            assert np.isclose(a['positive']['shape_cosine'],b['positive']['shape_cosine'],atol=1e-12)
            assert np.isclose(a['positive']['velocity_cosine'],b['positive']['velocity_cosine'],atol=1e-12)
    metrics=dict(results=result,source_hashes=source_hashes,correction_gain_flags={n:dict(clipped=fit['clipped'],zero_scale=fit['zero_scale']) for n,fit in fits.items()},
                 cache_trials=cache_count,checkpoint_unchanged=digest(checkpoint)==E017_SHA,
                 inference='CPUfloat64',interpretation='Distance reduction and content-specific improvement are separate; no biological or decoding benefit established',
                 uncertainty=protocol['uncertainty'])
    save_json(out/'metrics.json',metrics)
    fig,axs=plt.subplots(2,2,figsize=(14,10));fig.subplots_adjust(left=.10,right=.95,top=.78,bottom=.26,wspace=.27,hspace=.42)
    labels=['Raw','Offset','Offset + gain'];focals=[result[20][m]['focal'] for m in MODES]
    x=np.arange(3)
    for ax,key,negkey,title in [(axs[0,0],'RMS','RMS','Full-space distance (lower is closer)'),(axs[0,1],'shape_cosine','shape_cosine','Mean-centred shape (higher is closer)')]:
        main=[r['positive'][key] for r in focals];control=[np.median([v[negkey] for v in r['negatives']]) for r in focals]
        ax.scatter(x-.045,main,marker='o',s=45,color='#3276ad',label='Repeated sentence')
        ax.scatter(x+.045,control,marker='x',s=45,color='#697784',label='Different-sentence median')
        ax.set_xticks(x,labels);ax.set_title(title,fontsize=12);ax.legend(fontsize=9)
        ax.set_ylim(0,max(main+control)*1.27)
        ax.set_ylabel('RMS hidden-unit difference' if key=='RMS' else 'Centred cosine')
        for j,v in enumerate(main):ax.annotate(f'{v:.3f}',(j,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    focal_pair=next(p for p in comparisons if p['prompt']==FOCAL);a=histories[focal_pair['left']];b=histories[focal_pair['right']]
    reference=(a-pca_mean)@axes[:2].T
    for ax,mode in zip(axs[1],('raw','offset_gain')):
        target=(apply(b,fits[20],mode)-pca_mean)@axes[:2].T
        ax.plot(reference[:,0],reference[:,1],'-',color='#3276ad',lw=1.5,label='June7 reference')
        ax.plot(target[:,0],target[:,1],'--',color='#ca762d',lw=1.5,label='June14 '+('raw' if mode=='raw' else 'corrected'))
        ax.scatter([reference[0,0],target[0,0]],[reference[0,1],target[0,1]],marker='^',color=['#3276ad','#ca762d'],s=40)
        ax.set(title='Shared axes: '+('before' if mode=='raw' else 'after offset + gain'),xlabel='Initial-training PC1',ylabel='Initial-training PC2');ax.legend(fontsize=9)
    allpoints=np.vstack([reference,(b-pca_mean)@axes[:2].T,(apply(b,fits[20],'offset_gain')-pca_mean)@axes[:2].T])
    lo=allpoints.min(0);hi=allpoints.max(0);pad=(hi-lo)*.07
    for ax in axs[1]:ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
    for ax in axs.flat:ax.grid(alpha=.15)
    fig.text(.10,.95,'Independent correction: similarity versus generic alignment',fontsize=19,fontweight='bold',color='#20303f')
    fig.text(.10,.90,f'“{FOCAL}” • June7 / June14',fontsize=15)
    fig.text(.10,.85,'Fit: 20 preceding trials/day • test: 10 repeat pairs + within-block different-sentence controls • frozen E017',fontsize=10.5)
    fig.text(.97,.95,'✳',ha='right',fontsize=20,color='#3276ad')
    fig.text(.10,.10,'Corrections use separate, earlier trial moments—not this sentence or its evaluation prompts. Offset-only cannot change centred shape.\n'
             'Top panels show focal pair / its controls; full10-prompt cohort and10-trial sensitivity are in metrics.json. No bin-level inference.\n'
             'Same elapsed time is not phoneme alignment. Raw2D retains only6.1%of paired frame variance; scoring uses all1,024dimensions.\n'
             'Retrospective, historically trained encoder with departed preprocessing; no WER/PER improvement or biological drift claim.',fontsize=9.5,linespacing=1.5,color='#43515e')
    image=out/'offset-gain-comparison.png';fig.savefig(image,dpi=170,facecolor='white');plt.close(fig)
    save_json(out/'provenance.json',dict(status='generated-needs-QA',hashes={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(accounting=protocol['accounting'],focal={m:result[20][m]['focal']['positive'] for m in MODES},
                          cohort={m:{k:v['median'] for k,v in result[20][m]['summary'].items()} for m in MODES}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
