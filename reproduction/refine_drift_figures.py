"""Presentation-only E018 revision; preserve all original PNGs and metrics."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.audit import digest,save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run;target=run/'figures-v02';assert not target.exists()
    target.mkdir()
    metrics=json.loads((run/'metrics.json').read_text());rows=json.loads((run/'cohort.json').read_text())
    days=metrics['days'];ticks=[d[9:] for d in days]
    with np.load(run/'summaries.npz') as data:standard=data['raw_standardized']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

    def create(title,subtitle,n,width=13):
        fig,axes=plt.subplots(1,n,figsize=(width,6),squeeze=False)
        fig.subplots_adjust(top=.77,bottom=.24,left=.09,right=.94,wspace=.4)
        fig.text(.09,.96,title,fontsize=16,fontweight='bold',color='#222222')
        fig.text(.09,.87,subtitle,fontsize=10,color='#555555')
        fig.text(.96,.95,'✿',ha='right',fontsize=15,color='#555555')
        return fig,axes.ravel()

    palette=LinearSegmentedColormap.from_list('blue_white_orange',['#2563A6','#FFFFFF','#C86D26'])
    fig,axes=create('Recorded channel means vary across days',
        'T12 • 19 training-partition days • 6,680 trials • initial-training-reference bin SD units',2)
    means=np.stack([standard[[r['session']==d for r in rows]].mean(0) for d in days])
    saturated=int((np.abs(means)>1).sum())
    for ax,start,title in zip(axes,[0,128],['Threshold-crossing counts','Spike-band power']):
        im=ax.imshow(means[:,start:start+128],aspect='auto',cmap=palette,vmin=-1,vmax=1)
        ax.set_title(title);ax.set_yticks(range(19),ticks);ax.set_xlabel('Released channel index (0–127)')
        fig.colorbar(im,ax=ax,fraction=.035,pad=.025,extend='both',label='Mean shift / reference bin SD')
    fig.text(.09,.07,f'Color saturates at ±1 SD for readability ({saturated}/4,864 cells exceed the display range); no data excluded.\nRaw-unit values and uncertainty remain in evidence; content/recording changes are not controlled by this plot.',fontsize=9,color='#555555')
    fig.savefig(target/'channel-shifts.png',dpi=160);plt.close(fig)
    fig,axes=create('Distribution shift and greedy phoneme error',
        'T12 • 19 matched day aggregates • frozen E017 + adapter0 / fixed normalization; not baseline performance',3,15)
    y=np.array([metrics['per']['by_day'][d]['per']*100 for d in days])
    labels=sorted({0,1,len(days)-1,int(np.argmax(y))})
    for ax,key,title in zip(axes,['counts','power','latent'],['Counts','Power','Latent PCA32']):
        x=np.array(metrics['statistics'][key]['matrix'])[0]
        ci=np.array(metrics['statistics'][key]['reference_distance_ci95'])
        ax.hlines(y,ci[:,0],ci[:,1],color='#AAAAAA',linewidth=.8)
        ax.scatter(x,y,color='#2563A6',s=22)
        for i in labels:
            ax.annotate(ticks[i],(x[i],y[i]),fontsize=8,xytext=(5,5),textcoords='offset points')
        ax.set_title(f"{title}: descriptive ρ={metrics['correlations'][key]['spearman_rho']:.2f}",fontsize=11)
        ax.set_xlabel('Centroid distance from April 28');ax.set_ylabel('Greedy PER (%)')
        ax.margins(x=.1,y=.1)
    fig.text(.09,.07,'Horizontal bars: conditional 95% distance CI. Labels: first two days, final included day and worst PER.\nCorrelations exclude reference day (n=18); no significance/causality claim. No WFST WER was measured.',fontsize=9,color='#555555')
    fig.savefig(target/'shift-versus-per.png',dpi=160);plt.close(fig)
    for name in ['day-distances.png','repeated-prompts.png']:
        shutil.copyfile(run/'figures'/name,target/name)
    save_json(run/'figure-provenance-v02.json',dict(
        supersedes_display='figure-provenance.json; originals retained',
        reason='Prevent scatter-label overlap and improve heatmap contrast; no numerical/methodological changes',
        display_clipped_cells=saturated,renderer_sha256=digest(Path(__file__)),
        metrics_sha256=digest(run/'metrics.json'),summaries_sha256=digest(run/'summaries.npz'),
        files={p.name:digest(p) for p in sorted(target.glob('*.png'))}))


if __name__=='__main__':main()
