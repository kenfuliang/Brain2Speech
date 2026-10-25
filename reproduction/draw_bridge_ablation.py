"""Render the paired bridge-ablation architecture views without replacing history."""
from draw_research_architectures import DEST, render

DEST.mkdir(parents=True, exist_ok=True)
for name, title, role, label in [
    ('e013-causal-bridge-lora', 'E013 · Frozen causal neural features, trainable bridge and LoRA',
     'mixed', 'Aero language\nbackbone + LoRA'),
    ('e014-causal-bridge-only', 'E014 · Train only the bridge; freeze the entire language model',
     'frozen', 'Frozen Aero\nlanguage backbone'),
]:
    render(name, title, [
        ('Causal neural\nfeatures', 'data', False),
        ('Frozen trained GRU\n+ fixed trailing pool', 'frozen', False),
        ('Trainable neural\nto-language bridge', 'train', True),
        (label, role, True), ('Generated text', 'data', False),
    ], 'Same 512/40 sentences and initial bridge; full-trial offline generation, not causal streaming text.',
        (2, 'Training only: target-token\ncross-entropy'))
