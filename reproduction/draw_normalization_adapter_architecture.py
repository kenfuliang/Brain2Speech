"""E019's four-way frozen-weight design, as a reusable high-level PNG."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    fig,ax=plt.subplots(figsize=(11,6));ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    fig.subplots_adjust(left=.04,right=.96,top=.86,bottom=.14)
    fig.suptitle('Separate normalization from the day-input layer',fontsize=17,fontweight='bold',y=.97)
    ax.text(.5,.98,'Same 6,000 trials • same frozen E017 backbone • no weight updates',ha='center',fontsize=11)
    ax.text(.35,.83,'Reference input layer (April 28)',ha='center',fontsize=11)
    ax.text(.77,.83,'Trained input layer for that day',ha='center',fontsize=11)
    for x,y,name,text in [( .35,.65,'G0','Fixed normalization\nReference layer'),
                          (.77,.65,'Gd','Fixed normalization\nNative day layer'),
                          (.35,.34,'D0','Per-day normalization\nReference layer'),
                          (.77,.34,'Dd','Per-day normalization\nNative day layer')]:
        ax.text(x,y,name+'\n'+text,ha='center',va='center',fontsize=12,
            bbox=dict(boxstyle='round,pad=.7',facecolor='#EEEEEE',edgecolor='#666666'))
    ax.text(.56,.72,'Change input layer',ha='center',fontsize=9)
    ax.annotate('',xy=(.63,.65),xytext=(.47,.65),
        arrowprops=dict(arrowstyle='->',color='#666666'))
    ax.annotate('Change normalization',xy=(.13,.36),xytext=(.13,.62),ha='center',fontsize=9,
        arrowprops=dict(arrowstyle='->',color='#666666'))
    fig.text(.5,.06,'G0 reuses E018. Other cells are inference only. Per-day statistics use the entire training partition.\nNative layers were historically supervised; this is not online calibration or biological-drift proof.',
        ha='center',fontsize=10,color='#555555')
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160);plt.close(fig)


if __name__=='__main__':main()
