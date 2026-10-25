"""Consistent high-level PNG architecture views for the decision-oriented report."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT/'docs/assets/research-architecture-v1'
FILL = {'data':'#ffffff','train':'#e3effc','frozen':'#eef1f4','mixed':'#eee9fa'}


def render(name,title,nodes,subtitle,training=None):
    output = DEST/(name+'.png')
    if output.exists():
        raise FileExistsError(output)
    fig,ax = plt.subplots(figsize=(16,4.8),dpi=160)
    ax.set(xlim=(0,16),ylim=(0,4.8)); ax.axis('off')
    ax.text(.3,4.3,title,fontsize=21,weight='bold',color='#182b43')
    w = 2.75
    for i,(label,role,changed) in enumerate(nodes):
        x=.3+i*3.15
        ax.add_patch(FancyBboxPatch((x,2.25),w,1.15,boxstyle='round,pad=.02,rounding_size=.12',
                     facecolor=FILL[role],edgecolor='#d98930' if changed else '#bac8d5',
                     linewidth=2.5 if changed else 1.2))
        ax.text(x+w/2,2.825,label,fontsize=14,color='#182b43',ha='center',va='center',linespacing=1.45)
        if i<len(nodes)-1:
            ax.add_patch(FancyArrowPatch((x+w+.05,2.825),(x+3.15-.09,2.825),
                         arrowstyle='-|>',mutation_scale=17,color='#506579',linewidth=1.5))
    if training:
        index,label=training
        x=.3+index*3.15
        ax.add_patch(FancyBboxPatch((x,1.05),w,0.67,boxstyle='round,pad=.02,rounding_size=.1',
                     facecolor='#fff7ea',edgecolor='#bba987',linestyle='--'))
        ax.text(x+w/2,1.385,label,ha='center',va='center',fontsize=11,color='#66543d')
        ax.add_patch(FancyArrowPatch((x+w/2,2.2),(x+w/2,1.78),arrowstyle='-|>',
                     linestyle='--',color='#8b795c',mutation_scale=14))
    ax.text(.3,.73,subtitle,fontsize=11.5,color='#506579')
    for x,role,label in [(0.3,'train','Trainable'),(2.4,'frozen','Frozen'),(4.35,'mixed','Frozen base + trainable adapters')]:
        ax.add_patch(FancyBboxPatch((x,.22),.25,.23,boxstyle='round,pad=.01',facecolor=FILL[role],edgecolor='#bac8d5'))
        ax.text(x+.35,.335,label,va='center',fontsize=10,color='#506579')
    ax.text(10.05,.335,'Orange border: changed component · Dashed: training only',fontsize=10,color='#506579',va='center')
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    fig.savefig(output,facecolor='white'); plt.close(fig)
    print(output)


def render_causal_lfads():
    render('causal-lfads-location','E002 · Causal LFADS-inspired encoder replaces the neural representation stage',[
        ('Causal features\n+ adapter / read-in','train',False),
        ('Causal inference GRU\n→ inferred inputs','train',True),
        ('Causal generator GRU\n→ latent factors','train',True),
        ('CTC head\n→ phoneme scores','train',False),
        ('Fixed WFST search\n+ 3-gram → text','frozen',False)],
        'LFADS-inspired, not original LFADS; E000/E008/E009 use a GRU instead. Future targets are training-only.',
        (2,'Training only: reconstruction\n+ KL; optional next-step loss'))


def render_nomad():
    render('nomad-reference','NoMAD reference · Align new-day neural data to reference-day dynamics',[
        ('Target-day\nneural data','data',False),
        ('Trainable target-day\nalignment mapping','train',True),
        ('Fixed reference\nLFADS dynamics','frozen',False),
        ('Fixed reference\nbehavior decoder','frozen',False),
        ('Motor behavior\npredictions','data',False)],
        'Causal sliding-window inference; target-day alignment is unlabeled. Motor prior art, not brain-to-text reproduction.',
        (1,'Training only: unlabeled\nalignment objective'))


def render_candidates():
    render('nbest-diagnostic','W02 diagnostic · Is the right sentence missing, or ranked below the top choice?',[
        ('Preserved neural\nphoneme logits','data',False),
        ('Fixed WFST graph\nN-best extraction','frozen',True),
        ('Ranked sentences\n+ acoustic / LM scores','data',True),
        ('Reference-only\noracle accounting','data',True),
        ('Search / selection\nbottleneck evidence','data',False)],
        'Diagnostic labels never enter decoding. Oracle WER is not deployable WER; no encoder or language model trains.')


def render_complementarity():
    render('checkpoint-complementarity','W04 diagnostic · Do two saved checkpoints make complementary errors?',[
        ('Released + stopped\ncheckpoint transcripts','data',False),
        ('Exact ordered\nreference alignment','data',False),
        ('Paired word-error\naccounting','data',True),
        ('Reference-selected\nbest-of-two ceiling','data',True),
        ('Gate a label-free\nensemble selector','data',False)],
        'Saved outputs only; no model trains. Reference-selected oracle is diagnostic—not deployed ensemble WER.')


def main(willett_only=False, dynamics_only=False, nomad_only=False, candidates_only=False, complementarity_only=False):
    DEST.mkdir(parents=True,exist_ok=True)
    if complementarity_only:
        render_complementarity()
        return
    if candidates_only:
        render_candidates()
        return
    if nomad_only:
        render_nomad()
        return
    if dynamics_only:
        render_causal_lfads()
        return
    render('willett-cascade-wfst','Willett reference · Cascaded neural-to-text decoding',[
        ('Neural\nfeatures','data',False),('GRU neural\nencoder','train',False),
        ('Phoneme\nprobabilities','train',False),('WFST beam search\n+ fixed n-gram LM','frozen',False),('Text','data',False)],
        'Beam-pruned Viterbi-style search (Kaldi/WeNet); E000 replay freezes neural weights; public baseline uses 3-gram.',(2,'Reference phonemes → CTC loss'))
    if willett_only:
        return
    render_causal_lfads()
    render_nomad()
    render('bit-reference','BIT reference · Continuous neural embeddings condition an audio LLM',[
        ('Neural\nfeatures','data',False),('Pretrained\nTransformer','train',True),
        ('Embedding\nprojector','train',True),('Audio LLM\n+ LoRA','mixed',True),('Text','data',False)],
        'Published approach, not locally reproduced: pretrained encoder; phoneme fine-tuning precedes sentence training.',
        (2,'Training only: target text\n+ contrastive alignment'))
    render('e009-neural-prefix','E009 pilot · Reuse a frozen GRU, learn a neural-to-language interface',[
        ('Neural\nfeatures','data',False),('Frozen GRU\n+ pooled embeddings','frozen',False),
        ('Trainable\nprojector','train',True),('Aero language\nbackbone + LoRA','mixed',True),('Text','data',False)],
        'Implemented BIT-inspired feasibility; no phoneme/WFST stage, waveform frontend, contrastive loss or encoder updates.',
        (2,'Training only: target-token\ncross-entropy'))
    render('daily-adaptation','Cross-day adaptation · Change the encoder using permitted calibration only',[
        ('Target-day\ncalibration data','data',True),('Encoder\npre-checkpoint','frozen',False),
        ('Adapter / full\nmodel update','train',True),('Encoder\npost-checkpoint','frozen',True),('Reserved-day\nevaluation','data',False)],
        'Updates use calibration, not evaluation trials; optimizer resets daily; text decoder remains fixed.',
        (2,'Training only: optional\nhistorical-training replay'))
    render('encoder-analysis','Encoder analysis · Compare functions and updates on the same fixed probes',[
        ('Fixed historical\nprobe trials','data',False),('Pre / post\ncheckpoints','frozen',True),
        ('Embedding +\nprediction changes','data',True),('Matched same-day\nupdate control','data',True),('Encoder-adaptation\nevidence','data',False)],
        'Analysis only; weight changes are not automatically biological neural drift; no probe trial enters replay.')


if __name__=='__main__':
    main(willett_only='--willett-only' in sys.argv,
         dynamics_only='--dynamics-only' in sys.argv,
         nomad_only='--nomad-only' in sys.argv,
         candidates_only='--candidates-only' in sys.argv,
         complementarity_only='--complementarity-only' in sys.argv)
