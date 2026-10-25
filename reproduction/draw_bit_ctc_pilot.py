"""High-level E015 view; the full published cascade remains a different system."""
from draw_research_architectures import DEST, render

DEST.mkdir(parents=True, exist_ok=True)
render('e015-bit-ctc-pilot', 'E015 · Original BIT neural Transformer with our fixed decoder', [
    ('Selected T12 trials\noffline normalization', 'data', False),
    ('Bidirectional BIT\nTransformer', 'train', True),
    ('Phoneme scores\nCTC head', 'train', False),
    ('Fixed 3-gram\nWFST search', 'frozen', False),
    ('Decoded text\nWER / CER', 'data', False),
], 'From scratch, 512/40 historical sentences. No BIT-All pretraining, 5-gram or OPT rescoring; not causal.',
    (2, 'Training only: phoneme\nCTC loss; report PER'))
