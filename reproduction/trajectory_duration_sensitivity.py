"""Declared post-hoc duration sensitivity; reuse fixed fits, never tune them."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.audit import digest,save_json


def main(run):
    p=json.loads((run/'protocol.json').read_text());m=json.loads((run/'metrics.json').read_text())
    result={}
    for mode,data in m['results']['20'].items():
        records=[]
        for pair,r in zip(p['evaluation'],data['records']):
            keep=[i for i,v in enumerate(pair['duration_ratios']) if .9<=v<=1.1]
            controls=[r['negatives'][i] for i in keep]
            records.append(dict(prompt=pair['prompt'],controls_count=len(keep),control_ids=[pair['controls'][i] for i in keep],
                         positive=r['positive'],negative_median={key:float(np.median([v[key] for v in controls])) if controls else None for key in ['RMS','shape_cosine']},
                         enough_context=len(controls)>=3))
        eligible=[r for r in records if r['enough_context']]
        result[mode]=dict(records=records,eligible_prompts=len(eligible),
                        summary={key:dict(repeated_median=float(np.median([r['positive'][key] for r in eligible])),
                                          control_median_of_prompt_medians=float(np.median([r['negative_median'][key] for r in eligible])),
                                          median_paired_advantage=float(np.median([(r['negative_median'][key]-r['positive'][key])*(1 if key=='RMS' else -1) for r in eligible]))) for key in ['RMS','shape_cosine']},
                        focal=next(r for r in records if r['prompt']==p['focal']))
    artifact=run/'duration-sensitivity-v03.json';assert not artifact.exists()
    save_json(artifact,dict(status='post-hoc-descriptive',reason='Primary nearest20 duration ratios span0.629–1.785; audit requires stricter matching sensitivity',
              threshold='Target trial durations within±10%; require≥3controls for cohort summary; no correction refitting',
              uncertainty='One day pair, sparse/reused controls; no intervals or significance tests',results=result,
              hashes={str(x):digest(x) for x in [run/'protocol.json',run/'metrics.json',Path(__file__)]}))
    fig,axs=plt.subplots(2,2,figsize=(13,9));fig.subplots_adjust(left=.09,right=.95,top=.77,bottom=.25,wspace=.25,hspace=.40)
    modes=['raw','offset','offset_gain'];labels=['Raw','Offset','Offset + gain'];x=np.arange(3)
    for row in range(2):
        for col,key in enumerate(['RMS','shape_cosine']):
            ax=axs[row,col]
            if row==0:
                a=[result[mode]['focal']['positive'][key] for mode in modes]
                b=[result[mode]['focal']['negative_median'][key] for mode in modes]
            else:
                a=[result[mode]['summary'][key]['repeated_median'] for mode in modes]
                b=[result[mode]['summary'][key]['control_median_of_prompt_medians'] for mode in modes]
            ax.scatter(x-.055,a,color='#3276ad',s=45,marker='o',label='Repeated sentence'+('s' if row else ''))
            ax.scatter(x+.055,b,color='#697784',s=45,marker='x',label='Different-sentence median')
            ax.set_xticks(x,labels);ax.set_ylim(0,max(a+b)*1.25)
            ax.set_title(('Focal pair / 3 controls' if row==0 else '9 pairs / ≥3 controls each')+': '+('distance ↓' if col==0 else 'shape similarity ↑'),fontsize=11)
            ax.set_ylabel('Full1,024D RMS' if col==0 else 'Mean-centred cosine');ax.grid(alpha=.15)
            for j,v in enumerate(a):ax.annotate(f'{v:.3f}',(j-.055,v),xytext=(-8,11),textcoords='offset points',ha='center',fontsize=9)
            for j,v in enumerate(b):ax.annotate(f'{v:.3f}',(j+.055,v),xytext=(8,-16),textcoords='offset points',ha='center',fontsize=9,color='#586674')
    handles,labels_legend=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels_legend,loc='upper center',bbox_to_anchor=(.52,.845),ncol=2,fontsize=9,frameon=False)
    fig.text(.09,.95,'Duration-controlled sensitivity: mostly position correction',fontsize=18,fontweight='bold',color='#20303f')
    fig.text(.09,.89,'Independent20-trial/day fit • frozen E017 • same target-day blocks • controls within±10%duration',fontsize=10.5)
    fig.text(.09,.84,'Post-hoc sensitivity, not a replacement for the original10-pair / nearest20-control analysis',fontsize=10)
    fig.text(.97,.95,'✳',ha='right',fontsize=20,color='#3276ad')
    fig.text(.09,.10,'Focal: “she said in Blake’s voice”. One cohort prompt has only2strict controls and is excluded from the9-pair summary.\n'
             'Corrections unchanged: no evaluation labels or trajectories used in fitting. Scoring matches elapsed time, not phonemes.\n'
             'Same encoder and historical preprocessing limitations; lower distance alone is not restored speech decoding.\n'
             'Two sessions, sparse/dependent controls: descriptive only, no confidence interval or biological-drift conclusion.',fontsize=9.5,linespacing=1.5,color='#43515e')
    image=run/'duration-sensitivity-v03.png';assert not image.exists();fig.savefig(image,dpi=170,facecolor='white');plt.close(fig)
    print(json.dumps({mode:result[mode]['focal'] for mode in modes},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);main(parser.parse_args().run.resolve())
