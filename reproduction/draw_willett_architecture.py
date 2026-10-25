"""Render the high-level Willett cascaded decoder as a deterministic PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

root = Path(__file__).resolve().parent.parent
output = root/'docs/assets/willett-2023-architecture-high-level.png'
if output.exists():
    raise FileExistsError('Preserve existing diagram; choose a versioned filename')
output.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 3.4), dpi=180)
fig.patch.set_facecolor('#ffffff')
ax.set(xlim=(0,15), ylim=(0,3.4))
ax.axis('off')
ax.text(.35,2.85,'Willett 2023 · Brain-to-Text', fontsize=22,
        weight='bold', color='#172b43', va='center')

labels = ['Neural\nactivity','Neural decoder\n(GRU)','Phoneme\nprobabilities','Language\ndecoder','Text']
colors = ['#edf3f8','#e4f4f1','#edf3f8','#eeeafb','#edf3f8']
width, y, height = 2.35, 1.05, 1.05
for i, (label, color) in enumerate(zip(labels,colors)):
    x = .35 + i*2.95
    ax.add_patch(FancyBboxPatch((x,y),width,height,
        boxstyle='round,pad=0.02,rounding_size=0.13',
        facecolor=color,edgecolor='#c9d5df',linewidth=1.1))
    ax.text(x+width/2,y+height/2,label,ha='center',va='center',
            fontsize=15,color='#172b43',linespacing=1.5)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x+width+.08,y+height/2),
            (x+2.95-.10,y+height/2),arrowstyle='-|>',
            mutation_scale=18,linewidth=1.6,color='#52697d'))
ax.text(.35,.48,'Cascaded decoding: neural activity → phonemes → words',
        fontsize=13,color='#52697d',va='center')
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
fig.savefig(output,facecolor='white')
plt.close(fig)
print(output)
