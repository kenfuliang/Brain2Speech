"""Presentation-only E019 fix: preserve originals, make full row labels visible."""
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
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    target=run/'figures-v02';assert not target.exists();target.mkdir()
    m=json.loads((run/'metrics.json').read_text());keys=['G0','Gd','D0','Dd']
    matrix=np.array([[m['conditions'][k]['by_day'][d]['per']*100 for d in m['days']] for k in keys])
    fig,ax=plt.subplots(figsize=(13,5.5));fig.subplots_adjust(top=.77,bottom=.27,left=.24,right=.94)
    fig.text(.09,.96,'Matched day-by-day phoneme errors',fontsize=16,fontweight='bold',color='#222222')
    fig.text(.09,.87,'T12 • 17 days / 6,000 matched training trials • unchanged encoder • offline/transductive moments',fontsize=10,color='#555555')
    fig.text(.97,.95,'✿',ha='right',fontsize=15,color='#555555')
    colors=LinearSegmentedColormap.from_list('white_blue',['#FFFFFF','#2563A6'])
    im=ax.imshow(matrix,aspect='auto',cmap=colors,vmin=0,vmax=100)
    ax.set_yticks(range(4),['Fixed norm\nReference layer','Fixed norm\nNative day layer',
        'Per-day norm\nReference layer','Per-day norm\nNative day layer'],fontsize=10)
    ax.set_xticks(range(17),[d[9:] for d in m['days']],rotation=45,fontsize=9)
    for i in range(4):
        for j in range(17):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=8,
            color='#FFFFFF' if matrix[i,j]>60 else '#222222')
    fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Greedy PER (%)')
    fig.text(.09,.07,'June 23 / July 29 excluded: no trained native layer. These are training-partition errors, not new-day accuracy.\nAll conditions share exact trials/phones; uncertainty and exact edits are saved. No new training or WFST.',fontsize=9,color='#555555')
    fig.savefig(target/'per-by-day.png',dpi=160);plt.close(fig)
    shutil.copyfile(run/'figures/factorial-summary.png',target/'factorial-summary.png')
    shutil.copyfile(run/'figures/study-architecture-v02.png',target/'study-architecture-v02.png')
    save_json(run/'display-provenance.json',dict(reason='Row-label clipping fixed; numeric values unchanged; original figures retained',
        metrics_sha256=digest(run/'metrics.json'),renderer_sha256=digest(Path(__file__)),
        files={p.name:digest(p) for p in sorted(target.glob('*.png'))}))


if __name__=='__main__':main()
