"""High-level E018 study diagram; separate from the four quantitative figures."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    fig,ax=plt.subplots(figsize=(11,5));ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    fig.subplots_adjust(left=.02,right=.98,top=.9,bottom=.12)
    fig.suptitle('Cross-day shifts: one fixed measuring instrument',fontsize=17,fontweight='bold',x=.5,y=.97)
    boxes=[(.5,.83,'19 days of recorded features\nTraining partitions only'),
           (.23,.54,'Counts and power\nTrial-level summaries'),
           (.77,.54,'Fixed normalization + adapter0\nFrozen E017 encoder → latents'),
           (.5,.22,'Shared reference coordinates\nWhole-block controls + repeated content')]
    for x,y,text in boxes:
        ax.text(x,y,text,ha='center',va='center',fontsize=12,
            bbox=dict(boxstyle='round,pad=.7',facecolor='#EEEEEE',edgecolor='#666666'))
    for a,b in [((.42,.75),(.23,.65)),((.58,.75),(.77,.65)),((.23,.44),(.41,.32)),((.77,.44),(.59,.32))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#555555',lw=1.5))
    fig.text(.5,.025,'Gray = frozen/read-only. No training, adaptation, WFST or final-day recordings. Not proof of biological drift.',
        ha='center',fontsize=10,color='#555555')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160);plt.close(fig)


if __name__=='__main__':main()
