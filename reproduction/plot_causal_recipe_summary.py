"""Final chart layout: include both arms before fixing limits, no clipped outcomes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.audit import digest,save_json


def main(out):
    m=json.loads((out/'metrics.json').read_text());days=list(m['calibration_curves']['raw']);budgets=[0,5,10,25,50,100,'full']
    fig,axs=plt.subplots(2,2,figsize=(14,10));fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.17,hspace=.48,wspace=.28)
    for row in range(2):
        for ax,field in zip(axs[row],('wer','per')):
            maximum=0
            for arm,color,marker,offset in [('raw','#3276ad','o',-.06),('anscombe','#ca762d','x',.06)]:
                vals=([100*m['conditions'][f'{arm}/{b}/{"normalization" if b==0 else "adapter"}'][field]['corpus'] for b in budgets] if row==0 else
                      [100*m['conditions'][f'{arm}/full/adapter']['by_day'][d][field] for d in days])
                maximum=max(maximum,max(vals));ax.scatter(np.arange(len(vals))+offset,vals,color=color,marker=marker,s=40,label=arm.capitalize())
            ax.set_ylim(0,maximum*1.12);ax.set_xticks(np.arange(7 if row==0 else 5),[str(b).capitalize() for b in budgets] if row==0 else [d[9:] for d in days]);ax.set_ylabel(field.upper()+' (%)')
            ax.set_title(('Corpus '+field.upper()+' vs calibration labels') if row==0 else ('Per-day '+field.upper()+' • full calibration'));ax.grid(axis='y',alpha=.15);ax.legend(loc='lower left',fontsize=9)
    axs[0,0].axhline(25,color='#697784',ls=':');axs[0,0].text(.98,25,'25% WER target',transform=axs[0,0].get_yaxis_transform(),ha='right',va='bottom',fontsize=9,color='#43515e')
    fig.text(.08,.95,'Raw versus Anscombe: later-day decoding and calibration',fontsize=18,weight='bold');fig.text(.97,.95,'✳',ha='right',fontsize=20,color='#3276ad')
    fig.text(.08,.89,'200 unique test sentences • five development days • one paired seed • causal encoder + fixed3gram WFST',fontsize=11)
    fig.text(.08,.04,'Full =400labels/day except July29(200); zero uses training-global moments; other points use prefix moments plus100adapter steps.\n'
        'Measured-budget dots, no interpolation. Normalization-only controls and paired intervals are in metrics.json.\n'
        'Historically inspected development days, not independent final confirmation; no final-day access or biological drift claim.',fontsize=10,linespacing=1.5)
    image=out/'raw-anscombe-calibration-v02.png';fig.savefig(image,dpi=160,facecolor='white');plt.close(fig)
    save_json(out/'chart-provenance-v02.json',dict(question='Does Anscombe improve later-day errors or reduce calibration labels?',family='Grouped budget dots and per-day paired dots',surface='Existing Markdown report/PNG',data_sha256=digest(out/'metrics.json'),image_sha256=digest(image),script_sha256=digest(__file__),revision='Display-only: limits include both arms; legend moved to avoid points. Original retained.'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);main(a.parse_args().output.resolve())
