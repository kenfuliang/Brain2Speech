"""Frozen E017 full-frame trajectories; CPU only, sealed outputs, no final data."""
import argparse
import gc
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch

from analyze_neural_drift import ROOT, E017_SHA, preprocess, fixed_inference, shard
from pytorch_baseline import PublishedGRU
from research.audit import digest, save_json, space_guard
from research.metrics import edits, greedy


def load_model(path):
    model = PublishedGRU().double().eval()
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def pair_distance(left, right):
    n = min(len(left), len(right))
    squared = np.mean((left[:n].astype(np.float64)-right[:n])**2, axis=1)
    return dict(common_frames=n, rms=float(np.sqrt(squared.mean())),
                curve=np.sqrt(squared).tolist(),
                compared_fraction=[n/len(left), n/len(right)])


def main(out):
    assert not out.exists(), 'Use a fresh immutable output directory'
    space_guard()
    torch.set_num_threads(2)
    old = ROOT / 'research/runs/e018-neural-drift-20261011-01'
    example = ROOT / 'research/runs/repeated-sentence-activity-20261012-03'
    contract = json.loads((example/'chart-contract.json').read_text())
    original = json.loads((old/'protocol.json').read_text())
    manifest_path = ROOT/'research/data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rows = json.loads((old/'cohort.json').read_text())
    lookup = {r['id']: r for r in rows}
    pairs = [p for p in json.loads((old/'repeated-content.json').read_text())['pairs'] if not p['cross_day']]
    assert len(pairs) == 7 and len({p['prompt'] for p in pairs}) == 7
    focal = [lookup[r['id']] for r in contract['selected']]
    chosen_ids = {r['id'] for r in focal} | {p[k] for p in pairs for k in ('left', 'right')}
    selected = [r for r in rows if r['id'] in chosen_ids]
    assert len(selected) == 16 and {r['partition'] for r in selected} == {'train'}
    assert not {r['session'] for r in selected} & set(manifest['folds']['final'])
    checkpoint = Path(original['checkpoint'])
    assert digest(checkpoint) == E017_SHA
    seal = json.loads((old/'seal.json').read_text())
    for name in ('cohort.json','normalization.json','summaries.npz','repeated-content.json','protocol.json'):
        assert digest(old/name) == seal['files'][name]
    # Source seals are checked below; derived source artifacts have their own markers.
    for name in ('cohort.json', 'normalization.json'):
        assert digest(old/name) == json.loads((old/(name+'.sha256.json')).read_text())['sha256']
    with np.load(old/'summaries.npz') as summaries:
        pca_mean = summaries['pca_mean'].copy()
        axes = summaries['pca_axes'].copy()
    norm = json.loads((old/'normalization.json').read_text())
    assert norm['fit_days'] == manifest['folds']['initial_train'] and norm['equal_trial_weight']
    assert np.allclose(axes @ axes.T, np.eye(32), atol=1e-8)
    out.mkdir(parents=True)
    chart = dict(question='When do frozen-encoder trajectories differ during a repeated sentence?',
                 takeaway='Descriptive instrument trajectories, not identified biological dynamics or proof of drift',
                 family='Shared-coordinate trajectory + temporal curves + contextual pair-distance dots',
                 renderer='Static PNG for existing canonical research report',
                 focal=focal, prompt=contract['prompt'], within_day_context_pairs=pairs,
                 selection='Reuse prior duration-selected example; all seven available within-day repeats',
                 encoder='Frozen E017 final GRU1024D, eval, adapter0, reset each trial; no adaptation',
                 preprocessing=original['preprocessing'],
                 PCA='Reuse E018 first32 initial-training TRIAL-MEAN axes; first2 display, no new fit or per-day axes',
                 time='Patch-end available bin timestamps: (32+4*i)*.02 seconds; no word alignment or time warping',
                 distance='1024D RMS difference at shared elapsed-time frames; truncate only pair comparison to common length',
                 controls='Seven different within-day prompts (six April28,oneJuly5); not focal-prompt or day-matched; no CI or inference',
                 resources='Explicit CPU2threads to avoid GPU contention; storage30GiB guard; no final recordings or decoder',
                 numerical_gate='CPU float64 forward of unchanged float32 weights; exact reload; future-perturbation prefix atol1e-12; cached MPS mean/4frame agreement atol.001,rtol.0001',
                 precision_departure='FP32 exploratory extraction failed bitwise prefix equality (max1.22e-6 hidden); FP64 reduces difference to3.39e-15; not model training or biological drift',
                 palette='Hard two-root cap: blue solid circles vs orange dashed crosses; neutral contextual points',
                 source_hashes={str(p):digest(p) for p in [checkpoint, manifest_path, old/'cohort.json',old/'normalization.json',old/'summaries.npz',example/'chart-contract.json']},
                 script_sha256=digest(__file__))
    save_json(out/'chart-contract.json', chart)
    model = load_model(checkpoint)
    histories = {}
    inputs = {}
    metadata = {}
    source_hashes = {}
    maximum_mps_difference = 0.
    for day in sorted({r['session'] for r in selected}):
        source = ROOT/manifest['sessions'][day]['train']['path']
        expected = manifest['sessions'][day]['train']['sha256']
        assert digest(source) == expected
        source_hashes[str(source)] = expected
        mat = loadmat(source, variable_names=['tx1','spikePow'])
        for r in [r for r in selected if r['session'] == day]:
            space_guard()
            raw = np.concatenate([mat['tx1'].ravel()[r['index']][:,:128],mat['spikePow'].ravel()[r['index']][:,:128]], axis=1).astype(np.float64)
            assert raw.shape == (r['bins'],256) and np.isfinite(raw).all()
            assert (raw >= 0).all() and np.array_equal(raw[:,:128],np.rint(raw[:,:128]))
            x = preprocess(raw,np.array(norm['mean']),np.array(norm['variance'])).astype(np.float64)
            logits, hidden = fixed_inference(model,x,torch.device('cpu'))
            assert len(hidden) == (r['bins']-32)//4+1
            old_shard = shard(old,r)
            meta = json.loads(old_shard.with_suffix('.json').read_text())
            assert digest(old_shard) == meta['sha256']
            with np.load(old_shard) as cached:
                sample = hidden[np.linspace(0,len(hidden)-1,4,dtype=int)]
                assert np.allclose(hidden.astype(np.float64).mean(0),cached['latent_mean'],atol=.001,rtol=.0001)
                assert np.allclose(sample,cached['latent_frames'],atol=.001,rtol=.0001)
                maximum_mps_difference = max(maximum_mps_difference,float(np.max(np.abs(sample-cached['latent_frames']))))
                phone_edits = edits(cached['reference'].tolist(),greedy(logits))
                assert phone_edits == meta['per']
            histories[r['id']] = hidden
            if r['id'] == focal[0]['id']:
                inputs[r['id']] = x
            coordinates = (hidden-pca_mean) @ axes.T
            times = (32+4*np.arange(len(hidden)))*.02
            np.savez_compressed(out/f"{r['session']}-{r['index']:04d}.npz",hidden=hidden,logits=logits,pca=coordinates,seconds=times)
            metadata[r['id']] = dict(**r,frames=len(hidden),PER=phone_edits['errors']/phone_edits['reference_units'],phone_edits=phone_edits)
        del mat
    # Fresh independent model reload, after other trials, proves no carried trial state.
    del model
    gc.collect()
    reloaded = load_model(checkpoint)
    _, again = fixed_inference(reloaded,inputs[focal[0]['id']],torch.device('cpu'))
    assert np.array_equal(again,histories[focal[0]['id']])
    changed = inputs[focal[0]['id']].copy()
    changed[128:] += 10
    _, altered = fixed_inference(reloaded,changed,torch.device('cpu'))
    prefix = np.flatnonzero(32+4*np.arange(len(again)) <= 128)
    prefix_difference = float(np.max(np.abs(altered[prefix]-again[prefix])))
    assert np.allclose(altered[prefix],again[prefix],rtol=0,atol=1e-12)
    left,right = [histories[r['id']] for r in focal]
    distance = pair_distance(left,right)
    controls = [dict(prompt=p['prompt'],left=p['left'],right=p['right'],**pair_distance(histories[p['left']],histories[p['right']])) for p in pairs]
    combined = np.concatenate([left,right]).astype(np.float64)
    centered = combined-combined.mean(0)
    variance = np.sum(centered**2)
    retained = [float(np.sum((centered@axes[:n].T)**2)/variance) for n in (2,32)]
    metrics = dict(trials=metadata, focal_ids=[r['id'] for r in focal], focal_pair=distance,within_day_context=controls,
                   frame_variance_retained=dict(PC2=retained[0],PC32=retained[1]),
                   verification=dict(exact_CPU_reload=True,prefix_invariance=True,padding='none,batch1',
                     fixed_adapter=0,CPU_precision='float64',prefix_max_absolute_difference=prefix_difference,max_cached_MPS_frame_absolute_difference=maximum_mps_difference,
                     shared_PCA=True,source_hashes=source_hashes,checkpoint_unchanged=digest(checkpoint)==E017_SHA),
                   verdict='Trajectories measured; content-conditioned/biological drift inconclusive',
                   warning='E018 fixed preprocessing differs from E017 original training and has poor decoding; same elapsed time is not same phoneme')
    save_json(out/'metrics.json',metrics)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig = plt.figure(figsize=(14,10))
    grid = fig.add_gridspec(2,2,left=.08,right=.96,bottom=.29,top=.78,wspace=.26,hspace=.37,width_ratios=[1.3,1])
    path_ax = fig.add_subplot(grid[:,0]);time_ax = fig.add_subplot(grid[0,1]);distance_ax = fig.add_subplot(grid[1,1])
    colors=['#3276ad','#ca762d'];styles=['-','--'];markers=['o','x']
    for i,(r,h) in enumerate(zip(focal,[left,right])):
        xy = (h-pca_mean)@axes[:2].T
        times = (32+4*np.arange(len(h)))*.02
        label=r['session'][4:].replace('.','-')
        path_ax.plot(xy[:,0],xy[:,1],color=colors[i],linestyle=styles[i],linewidth=1.5,label=label)
        path_ax.scatter(xy[::4,0],xy[::4,1],color=colors[i],marker=markers[i],s=18)
        path_ax.scatter(xy[0,0],xy[0,1],color=colors[i],marker='^',s=80)
        path_ax.scatter(xy[-1,0],xy[-1,1],color=colors[i],marker='s',s=60)
        for idx in range(8,len(h)-1,12):
            path_ax.annotate('',xy=xy[idx+1],xytext=xy[idx-1],arrowprops=dict(arrowstyle='->',color=colors[i],lw=1.5))
        time_ax.plot(times,xy[:,0],color=colors[i],linestyle=styles[i],linewidth=1.5,label=label)
    path_ax.set(title='Shared latent trajectory',xlabel='Initial-training PC1',ylabel='Initial-training PC2')
    path_ax.legend(loc='best',fontsize=10)
    path_ax.text(.02,.02,'▲ start (0.64s)   ■ end (4.88s)\nArrows show forward time',transform=path_ax.transAxes,fontsize=9)
    time_ax.set(title='PC1 through the sentence',xlabel='Elapsed trial time (s)',ylabel='Shared PC1')
    times=(32+4*np.arange(distance['common_frames']))*.02
    distance_ax.plot(times,distance['curve'],color='#43515e',linewidth=1.5)
    distance_ax.set(title='Separation in the full 1,024D space',xlabel='Elapsed trial time (s)',ylabel='RMS hidden-unit difference')
    for ax in (path_ax,time_ax,distance_ax):
        ax.grid(alpha=.15)
    fig.text(.08,.95,'Frozen encoder: same sentence, two temporal trajectories',fontsize=19,fontweight='bold',color='#20303f')
    fig.text(.08,.90,f'“{contract["prompt"]}”',fontsize=16,color='#20303f')
    fig.text(.08,.85,f'June7 / June14 • 54 frames each • frozen E017 / adapter0 • shared initial-training axes • 2D retains {retained[0]*100:.1f}% of paired frame variance',fontsize=10.5)
    fig.text(.97,.95,'✳',ha='right',fontsize=20,color=colors[0])
    context_values=[c['rms'] for c in controls]
    fig.text(.08,.105,f'Unaligned full-space pair RMS: {distance["rms"]:.3f}. Seven within-day contextual pairs: median {np.median(context_values):.3f}, range {min(context_values):.3f}–{max(context_values):.3f}.\n'
             'Those controls contain DIFFERENT prompts/days; they are not matched evidence for this sentence. No inferential interval.\n'
             'Patch-end elapsed time is not phoneme alignment. No time warping, per-day centering or encoder updates.\n'
             'Fixed global normalization and trailing smoothing depart from E017 training; this measuring instrument decodes poorly.\n'
             'Descriptive encoder-state trajectories—not proof of biological drift. CPU extraction; final recordings untouched.',fontsize=9.5,linespacing=1.5,color='#43515e')
    figure=out/'same-sentence-latent-trajectories.png'
    fig.savefig(figure,dpi=170,facecolor='white');plt.close(fig)
    save_json(out/'provenance.json',dict(status='generated-needs-visual-QA',hashes={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(figure=str(figure),focal_pair=distance['rms'],control_median=float(np.median(context_values)),retained=retained),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
