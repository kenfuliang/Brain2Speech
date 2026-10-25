"""Independent text audit and paired E022 calibration comparison, saved outputs only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.audit import digest,save_json,space_guard
from reproduction.audit_causal_recipe_completion import distance

FIELDS=('per','wer','cer')
BUDGETS=(0,5,10,25,50,100,'full')


def aggregate(rows):
    by_day={d:[r for r in rows if r['session']==d] for d in sorted({r['session'] for r in rows})}
    result=dict(trials=len(rows),sessions=len(by_day),by_day={})
    for field in FIELDS:
        total={k:sum(r[field][k] for r in rows) for k in ('errors','reference_units','substitutions','deletions','insertions')}
        rates={d:sum(r[field]['errors'] for r in rs)/sum(r[field]['reference_units'] for r in rs) for d,rs in by_day.items()}
        result[field]=dict(**total,corpus=total['errors']/total['reference_units'],macro_day=float(np.mean(list(rates.values()))),worst_day=max(rates.values()))
        for d,v in rates.items():result['by_day'].setdefault(d,{})[field]=v
    return result


def paired(raw,anscombe,repetitions=2000):
    assert [r['id'] for r in raw]==[r['id'] for r in anscombe]
    days=sorted({r['session'] for r in raw});rng=np.random.default_rng(101)
    arrays={f:(np.array([r[f]['errors'] for r in raw]),np.array([r[f]['errors'] for r in anscombe]),np.array([r[f]['reference_units'] for r in raw])) for f in FIELDS}
    groups={d:np.array([i for i,r in enumerate(raw) if r['session']==d]) for d in days};draws={f:[] for f in FIELDS}
    for _ in range(repetitions):
        indices=np.concatenate([rng.choice(groups[d],size=len(groups[d]),replace=True) for d in rng.choice(days,len(days),replace=True)])
        for f,(a,b,n) in arrays.items():draws[f].append(float((b[indices].sum()-a[indices].sum())/n[indices].sum()))
    return {f:dict(anscombe_minus_raw=float((b.sum()-a.sum())/n.sum()),ci95=np.quantile(draws[f],[.025,.975]).tolist()) for f,(a,b,n) in arrays.items()}


def target_budget(curve,target):
    return next((r['labels'] for r in curve if r['wer']<=target),'not reached')


def main(run,out):
    space_guard();assert not out.exists()
    assert json.loads((run/'status.json').read_text())['status']=='decoded-independent-comparison-audit-pending'
    p=json.loads((run/'protocol.json').read_text());audit=json.loads((run/'export-audit.json').read_text());assert audit['status']=='pass'
    for path,h in audit['artifacts'].items():assert digest(path)==h
    ident=json.loads((run/'decoder-identity.json').read_text())
    from research.experiment.evaluator import IMAGE_ID,SCORING
    assert ident['image_id']==IMAGE_ID and ident['scoring']==SCORING
    assert ident['worker_sha256']==digest(Path(__file__).resolve().parents[1]/'research/experiment/wfst_worker.py')
    for name,h in ident['lm_files'].items():assert digest(Path(__file__).resolve().parents[1]/'workspace/languageModel'/name)==h
    records={};hashes={};checked=0;seconds=[]
    for kind,day,budget,mode in p['conditions']:
        folder=run/kind/day/str(budget)/mode
        source=json.loads((folder/'predictions.json').read_text());rows=json.loads((folder/'decoded.json').read_text())
        assert len(rows)==len(source)==40
        for original,r in zip(source,rows):
            assert all(r[k]==v for k,v in original.items())
            for field,reference,hypothesis in [('wer',r['text'].split(),r['decoded_text'].split()),('wer_legacy',r['text'].split(' '),r['decoded_text'].split(' ')),('cer',list(r['text']),list(r['decoded_text']))]:
                assert distance(reference,hypothesis)==r[field]['errors']==sum(r[field][k] for k in ('substitutions','deletions','insertions'))
                assert len(reference)==r[field]['reference_units'];checked+=1
            assert np.isfinite(r['decode_ms']) and r['decode_ms']>=0;seconds.append(r['decode_ms']/1000)
        metric=json.loads((folder/'metrics.json').read_text());computed=aggregate(rows)
        for field in FIELDS:
            assert metric[field]['corpus']==computed[field]['corpus']
            assert metric[field]['worst_day']==computed[field]['worst_day']
        records.setdefault((kind,str(budget),mode),[]).extend(rows)
        for name in ('decoded.json','metrics.json','export-complete.json'):hashes[str(folder/name)]=digest(folder/name)
    result={};deltas={}
    for (kind,budget,mode),rows in records.items():result[f'{kind}/{budget}/{mode}']=aggregate(rows)
    for budget in BUDGETS:
        for mode in (['normalization'] if budget==0 else ['normalization','adapter']):
            a=records[('raw',str(budget),mode)];b=records[('anscombe',str(budget),mode)]
            assert [r['id'] for r in a]==[r['id'] for r in b]
            deltas[f'{budget}/{mode}']=paired(a,b)
    curves={};labels={}
    for kind in ('raw','anscombe'):
        for day in p['target_days']:
            curve=[]
            for budget in BUDGETS:
                mode='normalization' if budget==0 else 'adapter';key=f'{kind}/{budget}/{mode}'
                e=json.loads((run/kind/day/str(budget)/mode/'export-complete.json').read_text())
                curve.append(dict(budget=budget,labels=e['labels'],observed_sentences=e['observed_sentences'],**result[key]['by_day'][day]))
            curves.setdefault(kind,{})[day]=curve;labels.setdefault(kind,{})[day]=target_budget(curve,p['WER_target'])
    out.mkdir(parents=True)
    warning='One participant/paired seed, five historically exposed development days; conditional paired session-then-trial bootstrap2000/seed101, not training/calibration/selection uncertainty or independent final confirmation.13 budget/control comparisons, no multiplicity correction; interpret exploratory.'
    protocol=dict(parent=str(run),parent_protocol_sha256=digest(run/'protocol.json'),script_sha256=digest(__file__),
        uncertainty=warning,WER_target=p['WER_target'],selection='All frozen conditions; no outcome-based parent/budget selection; target crossings descriptive, no extrapolation',
        chart=dict(surface='Canonical Markdown report and PNG',family='Grouped calibration-budget dots + per-day full-budget dots',palette='Blue circles raw; orange crosses Anscombe; neutral target',rows='7 calibration categories and5day categories per arm; all same200unique trials'))
    save_json(out/'protocol.json',protocol)
    metrics=dict(conditions=result,paired_deltas=deltas,calibration_curves=curves,labels_to_25percent_WER=labels,
        target=p['WER_target'],uncertainty=warning,resource=dict(total_sentence_decode_seconds=sum(seconds),mean_sentence_decode_seconds=float(np.mean(seconds)),initialization='Separate graph load, not included in sentence timing',true_peak_memory='Not measured'),source_hashes=hashes)
    save_json(out/'metrics.json',metrics);save_json(out/'verification.json',dict(status='pass',conditions=130,independent_text_edit_checks=checked,unique_test_trials=200,decoder_identity='pass',export_seals='pass',matched_pair_ids='pass'))
    fig,axs=plt.subplots(2,2,figsize=(14,10));fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.16,hspace=.45,wspace=.27)
    for arm,color,marker,offset in [('raw','#3276ad','o',-.06),('anscombe','#ca762d','x',.06)]:
        for ax,field in zip(axs[0],('wer','per')):
            vals=[100*result[f'{arm}/{b}/{"normalization" if b==0 else "adapter"}'][field]['corpus'] for b in BUDGETS]
            ax.scatter(np.arange(7)+offset,vals,color=color,marker=marker,s=40,label=arm.capitalize());ax.set_xticks(np.arange(7),['0','5','10','25','50','100','Full']);ax.set_ylabel(field.upper()+' (%)');ax.set_title('Corpus '+field.upper()+' vs calibration labels');ax.set_ylim(bottom=0)
        for ax,field in zip(axs[1],('wer','per')):
            vals=[100*result[f'{arm}/full/adapter']['by_day'][d][field] for d in p['target_days']]
            ax.scatter(np.arange(5)+offset,vals,color=color,marker=marker,s=40,label=arm.capitalize());ax.set_xticks(np.arange(5),[d[9:] for d in p['target_days']]);ax.set_ylabel(field.upper()+' (%)');ax.set_title('Per-day '+field.upper()+' • full calibration');ax.set_ylim(bottom=0)
    axs[0,0].axhline(25,color='#697784',ls=':',label='25% WER target')
    for ax in axs.flat:ax.grid(axis='y',alpha=.15);ax.legend(fontsize=9)
    fig.text(.08,.95,'Raw versus Anscombe: later-day decoding and calibration',fontsize=18,weight='bold')
    fig.text(.08,.89,'200 unique test sentences • five development days • one paired seed • causal encoder + fixed3gram WFST',fontsize=11)
    fig.text(.97,.95,'✳',ha='right',fontsize=20,color='#3276ad')
    fig.text(.08,.04,'Full =400labels/day except July29(200); zero uses training-global moments; other points use prefix moments plus100adapter steps.\n'
        'Dots are measured budgets, not interpolated learning curves. Normalization-only controls and paired intervals are in metrics.json.\n'
        'Historically inspected development days, not independent final confirmation; no final-day access or biological drift claim.',fontsize=10,linespacing=1.5)
    fig.savefig(out/'raw-anscombe-calibration.png',dpi=160,facecolor='white');plt.close(fig)
    print(json.dumps(dict(full={a:result[f'{a}/full/adapter'] for a in ('raw','anscombe')},paired_full=deltas['full/adapter'],labels=labels),indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();main(v.run.resolve(),v.output.resolve())
