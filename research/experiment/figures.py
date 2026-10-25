"""Static research figures from collected rows only; chart contracts travel with exports."""
import json
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.audit import save_json, digest
from research.metrics import aggregate

COLORS = {'gru':'#3B6FB6','supervised':'#D89C29','dynamics':'#CE6F37','predictive':'#7C8F4F'}
MARKERS = {'gru':'o','supervised':'s','dynamics':'^','predictive':'D'}
SEED_COLORS = {17:'#3B6FB6',29:'#D89C29',43:'#555B63'}
SEED_STYLES = {17:'-',29:'--',43:':'}


def build_figures(root, freeze, qa_only=False):
    root = Path(root)
    source = root/'comparative-summary.json'
    data = json.loads(source.read_text())
    rows = data['per_day_rows']
    folder = root/'figures'
    folder.mkdir(exist_ok=True)
    contracts = []
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#292D32','text.color':'#292D32'})

    def export(fig, name, question, family, subtitle):
        if qa_only:
            subtitle = 'SYNTHETIC QA — NOT RESEARCH RESULTS. '+subtitle
        fig.text(.05,.02,textwrap.fill(subtitle,150),fontsize=8,color='#555B63')
        fig.tight_layout(rect=[0,.09,1,.96])
        fig.savefig(folder/(name+'.png'),dpi=170)
        fig.savefig(folder/(name+'.pdf'))
        plt.close(fig)
        contracts.append({'name':name,'question':question,'family':family,'surface':'Canonical Markdown research report / static PNG and PDF',
                          'source':str(source.relative_to(root)),'source_sha256':digest(source),
                          'subtitle':subtitle,'palette':'Four fixed category roots with distinct markers; gray references',
                          'takeaway':'To be written from measured findings; no improvement inferred by renderer',
                          'qa_status':'Exported; human visual inspection still required',
                          'branding':'Third-party Brain2Speech research; no OpenAI branding', 'synthetic_qa':qa_only})

    def points(selected, key, field):
        keys = sorted({r[key] for r in selected},key=str)
        means, dispersion = [], []
        for item in keys:
            seed_values = [np.mean([r[field] for r in selected if r[key]==item and r['seed']==seed])*100
                           for seed in freeze['seeds']]
            means.append(np.mean(seed_values))
            dispersion.append(np.std(seed_values,ddof=1) if len(seed_values)>1 else 0.)
        return keys, means, dispersion

    # Discrete observed budgets: dots/intervals, not invented continuous calibration trajectories.
    for mode in ['adapter','full']:
        fig, axes = plt.subplots(1,2,figsize=(11,4.5))
        for axis,field in zip(axes,['per','wer']):
            for kind in freeze['selected']:
                selected = [r for r in rows if r['kind']==kind and r['setting'].endswith('/evaluation')
                            and r['setting'].split('/')[0]==mode and '/same-day-control/' not in r['setting']]
                if not selected:
                    continue
                budgets = [b for b in freeze['budgets'] if any(r['setting']==f'{mode}/{b}/evaluation' for r in selected)]
                means, errors = [], []
                for b in budgets:
                    values = [np.mean([r[field] for r in selected if r['seed']==s and r['setting']==f'{mode}/{b}/evaluation'])*100
                              for s in freeze['seeds']]
                    means.append(np.mean(values)); errors.append(np.std(values,ddof=1) if len(values)>1 else 0.)
                axis.errorbar(np.arange(len(budgets)),means,yerr=errors,fmt=MARKERS[kind],
                              color=COLORS[kind],capsize=3,label=kind)
                axis.set_xticks(np.arange(len(budgets)),[str(b) for b in budgets])
            axis.set(xlabel='Tested labeled sentences per day',ylabel=field.upper()+' (%)')
            axis.grid(axis='y',color='#ECEFF2'); axis.legend(fontsize=8)
        axes[1].axhline(freeze['wer_target']*100,color='#555B63',ls='--',label='Development-fixed target')
        fig.suptitle('Calibration budgets — '+mode+' updates')
        export(fig,'calibration-'+mode,'Do dynamics models need fewer calibration labels?','Dot and interval',
               'Mean macro-day errors; bars are seed SD, not confidence intervals. Full budgets vary by day; one participant.')

    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    for axis,field in zip(axes,['per','wer']):
        for kind in freeze['selected']:
            selected = [r for r in rows if r['kind']==kind and r['setting']=='zero-shot']
            if selected:
                days,means,sd = points(selected,'day',field)
                offset = list(freeze['selected']).index(kind)*.13
                axis.errorbar(np.arange(len(days))+offset,means,yerr=sd,fmt=MARKERS[kind],
                              color=COLORS[kind],capsize=3,label=kind)
                axis.set_xticks(np.arange(len(days))+.2,[d[-10:] for d in days],rotation=25,ha='right')
        axis.set(xlabel='Future evaluation day',ylabel=field.upper()+' (%)'); axis.legend(fontsize=8)
        axis.grid(axis='y',color='#ECEFF2')
    fig.suptitle('Zero-shot transfer to future days')
    export(fig,'future-days','Which future days fail without calibration?','Grouped dot and interval',
           'Five discrete evaluation days, not a continuous drift estimate. Bars are seed SD; per-day trial counts in source table.')

    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    conditions = ['zero-shot','robustness/dropout-0.1','robustness/dropout-0.25','robustness/dropout-0.5']
    for axis,field in zip(axes,['per','wer']):
        for kind in freeze['selected']:
            selected = [r for r in rows if r['kind']==kind and r['setting'] in conditions]
            if not selected:
                continue
            means = [np.mean([r[field] for r in selected if r['setting']==c])*100 for c in conditions]
            axis.plot([0,.1,.25,.5],means,marker=MARKERS[kind],color=COLORS[kind],label=kind)
        axis.set(xlabel='Removed channel fraction',ylabel=field.upper()+' (%)'); axis.legend(fontsize=8)
        axis.grid(axis='y',color='#ECEFF2')
    fig.suptitle('Paired channel-loss stress test')
    export(fig,'channel-loss','How sensitive are models to the same lost electrodes?','Dose-response comparison',
           'Counts and power removed together; mean over seeds and days. Four measured stress levels, no interpolation claim.')

    fig, axis = plt.subplots(figsize=(9,4.5))
    conditions = ['zero-shot','robustness/gain-offset','robustness/shuffled','robustness/zero']
    for i,kind in enumerate(freeze['selected']):
        selected = [r for r in rows if r['kind']==kind and r['setting'] in conditions]
        if selected:
            means = [np.mean([r['wer'] for r in selected if r['setting']==c])*100 for c in conditions]
            axis.bar(np.arange(4)+i*.18,means,width=.17,color=COLORS[kind],edgecolor='#292D32',label=kind)
    axis.set_xticks(np.arange(4)+.27,['Normal','Gain / offset','Shuffled','Zero neural'])
    axis.set(ylabel='WER (%)',ylim=(0,max(100,axis.get_ylim()[1]))); axis.legend(fontsize=8)
    fig.suptitle('Neural-dependence and amplitude controls')
    export(fig,'neural-controls','Do outputs depend on neural evidence rather than only the LM?','Grouped bars',
           'Same evaluation references and fixed LM; pooled day/seed averages. Uncertainty remains in the companion metrics.')

    trajectory_groups = {}
    for path in sorted((root/'sequential').glob('**/trajectory.json')):
        kind,seed,mode,budget,replay = path.relative_to(root/'sequential').parts[:5]
        trajectory_groups.setdefault((kind,mode,budget,replay),[]).append((int(seed),path,json.loads(path.read_text())))
    for (kind,mode,budget,replay), items in trajectory_groups.items():
        name = '-'.join([kind,mode,budget,replay])
        days = [r['day'] for r in items[0][2]['transitions']]
        count = len(days)
        fig, axes = plt.subplots(1,3,figsize=(13,4.5))
        for seed,path,trajectory in items:
            ref = json.loads((path.parent/'initial-probe-reference.json').read_text())
            initial = json.loads((root/ref['path']/'metrics.json').read_text())['wer']['macro_day']
            transition = trajectory['transitions']
            style = {'color':SEED_COLORS[seed], 'ls':SEED_STYLES[seed], 'label':str(seed)}
            axes[0].plot(np.arange(count+1),[0]+[(t['historical_wer']-initial)*100 for t in transition],marker='o',**style)
            axes[1].plot(np.arange(count),[t['consecutive_update_cosine'] for t in transition],marker='s',**style)
            axes[2].plot(np.arange(count),[t['fixed_probe_embedding_rms'] for t in transition],marker='^',**style)
        axes[0].set(xlabel='Daily updates',ylabel='Historical WER change (pp)'); axes[0].axhline(0,color='#555B63',ls='--')
        axes[1].set(xlabel='Transition',ylabel='Consecutive update cosine',ylim=(-1.05,1.05))
        axes[2].set(xlabel='Transition',ylabel='Fixed-probe embedding RMS')
        for axis in axes:
            axis.legend(title='Seed',fontsize=7); axis.grid(axis='y',color='#ECEFF2')
        fig.suptitle(f'Sequential diagnostics — {kind}, {mode}, budget {budget}, replay {replay}')
        export(fig,'trajectory-'+name,'Do updates change function or cause forgetting?','Repeated-measures small multiples',
               f'{count} updates; shared historical probes/initial normalization. Positive WER change means forgetting; not biological drift.')

        layers = sorted(items[0][2]['transitions'][0]['normalized_layer_updates'])
        matrices = [np.array([[np.nan if t['normalized_layer_updates'][layer] is None else t['normalized_layer_updates'][layer]
                              for t in item['transitions']] for layer in layers]) for _,_,item in items]
        stack = np.stack(matrices)
        valid = np.sum(np.isfinite(stack),axis=0)
        matrix = np.divide(np.nansum(stack,axis=0),valid,out=np.full(valid.shape,np.nan),where=valid>0)
        fig, axes = plt.subplots(1,2,figsize=(12,6))
        image = axes[0].imshow(np.log10(np.maximum(matrix,1e-12)),aspect='auto',cmap='Blues',interpolation='nearest')
        axes[0].set_yticks(np.arange(len(layers)),layers,fontsize=7)
        axes[0].set_xticks(np.arange(count),[d[-5:] for d in days],rotation=30)
        axes[0].set(xlabel='Update day',title='log10 normalized update norm')
        fig.colorbar(image,ax=axes[0],shrink=.7)
        for seed,_,item in items:
            axes[1].plot(np.arange(1,len(item['cumulative_energy'])+1),item['cumulative_energy'],marker='o',
                         color=SEED_COLORS[seed],ls=SEED_STYLES[seed],label=f"{seed}: rank {item['effective_rank']:.2f}")
        axes[1].set(xlabel='Update-vector singular component',ylabel='Cumulative squared-singular-value energy',ylim=(0,1.05))
        axes[1].legend(fontsize=7)
        fig.suptitle('Encoder update geometry — '+name)
        export(fig,'geometry-'+name,'Are encoder updates compact and layer-specific?','Heatmap and spectrum',
               f'Seed-mean norms; zero parent norms undefined/masked; defined zero updates displayed at log10 floor -12. Energy-entropy effective rank; only {count} transitions.')

        matrices = []
        for seed,path,item in items:
            baseline = [next(r['wer'] for r in rows if r['kind']==kind and r['seed']==seed and r['day']==d and r['setting']=='zero-shot') for d in days]
            matrix = [baseline]
            for d in days:
                predictions = json.loads((path.parent/d/'transfer/predictions.json').read_text())
                matrix.append([aggregate([r for r in predictions if r['session']==target],'wer') for target in days])
            matrices.append(matrix)
        fig, axis = plt.subplots(figsize=(7,5))
        image = axis.imshow(np.mean(matrices,axis=0)*100,cmap='Blues',aspect='auto',vmin=0,
                            vmax=max(100,float(np.max(matrices)*100)))
        axis.set_xticks(np.arange(count),[d[-5:] for d in days],rotation=25)
        axis.set_yticks(np.arange(count+1),['Initial']+[d[-5:] for d in days])
        axis.set(xlabel='Evaluation day',ylabel='Checkpoint after calibration day')
        fig.colorbar(image,ax=axis,label='WER (%)')
        fig.suptitle('Cross-day checkpoint transfer — '+name)
        export(fig,'transfer-'+name,'Which updates transfer to other days?','Cohort heatmap',
               'Mean across seeds; five days after development freeze. Earlier evaluation labels never enter replay or model selection.')
    save_json(folder/'figure-manifest.json',{'figures':contracts,'figures_generated':len(contracts),
              'limitations':'Five sessions; no additional time points fabricated. Research results require rendered QA and interpretation.'})
    return len(contracts)
