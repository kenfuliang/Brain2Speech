"""Full scratch training is separate from BIT-All pretraining and cascade parity."""
from draw_research_architectures import DEST, render

DEST.mkdir(parents=True,exist_ok=True)
render('e016-bit-full-scratch', 'E016 · Full-data BIT-from-scratch phoneme training', [
    ('8,800 T12 trials\nsource normalization','data',False),
    ('Original bidirectional\nBIT Transformer','train',True),
    ('41-class phoneme\nCTC head','train',False),
    ('440 validation trials\ndual PER selection','data',False),
    ('440 test trials\nfinal PER; WER later','data',False),
], '800 epochs; effective author CLI disables masking. No cross-species SSL, 5-gram or OPT reproduction.',
    (2,'Training only: summed CTC\nApple GPU + CPU CTC'))
