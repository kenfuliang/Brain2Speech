"""Collect report evidence without changing training state or opening final data."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research.audit import ROOT, digest, save_json, space_guard
from research.metrics import edits
from research.models import Encoder


def replay_metrics(replay):
    references, hypotheses = replay['true_transcripts'], replay['decoded_transcripts']
    if len(references) != len(hypotheses) or len(references) != replay['measured']['trials']:
        raise ValueError('Replay sample accounting mismatch')
    rows = [{'wer': edits(r.split(' '), h.split(' ')), 'cer': edits(list(r), list(h))}
            for r, h in zip(references, hypotheses)]
    result = {}
    rng = np.random.default_rng(101)
    for field in ['wer', 'cer']:
        counts = {k: sum(row[field][k] for row in rows) for k in rows[0][field]}
        errors = np.array([r[field]['errors'] for r in rows])
        units = np.array([r[field]['reference_units'] for r in rows])
        bootstraps = []
        for _ in range(10000):
            idx = rng.integers(len(rows), size=len(rows))
            bootstraps.append(errors[idx].sum()/units[idx].sum())
        result[field] = {**counts, 'rate': counts['errors']/counts['reference_units'],
                         'ci95_trial_bootstrap': np.quantile(bootstraps, [.025, .975]).tolist()}
    if not np.isclose(result['wer']['rate'], replay['measured']['wer'][0], atol=1e-12):
        raise ValueError('Derived WER disagrees with preserved E000 result')
    return {'trials': len(rows), 'metrics': result, 'bootstrap_seed': 101,
            'bootstrap_repetitions': 10000,
            'scope': 'Additional diagnostics from preserved transcripts; not a replay rerun',
            'uncertainty': 'Unpaired trial bootstrap, same unit as E000; not session-level v1 uncertainty',
            'edit_tie_policy': 'substitution, deletion, insertion; alternate optimal paths can change components'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'research/runs/v1-initial-accepted4ms')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    space_guard()
    if args.output.exists():
        raise ValueError('Evidence output exists; preserve snapshot and use a new output path')
    manifest_path = ROOT/'research/data/manifest.json'
    replay_path = ROOT/'reproduction/results/released-checkpoint-3gram.json'
    manifest = json.loads(manifest_path.read_text())
    folds = {}
    for name, sessions in manifest['folds'].items():
        folds[name] = {'sessions': sessions, 'partitions': {}}
        for partition in ['train', 'test']:
            rows = [r for s in sessions for r in manifest['sessions'][s][partition]['trials']]
            folds[name]['partitions'][partition] = {
                'trials': len(rows), 'duration_hours': sum(r['duration_seconds'] for r in rows)/3600}
    parameters = {}
    for kind in ['gru', 'supervised', 'dynamics', 'predictive']:
        model = Encoder(kind)
        parameters[kind] = {'total': sum(p.numel() for p in model.parameters()),
                            'adapter': sum(p.numel() for n, p in model.named_parameters() if n.startswith('adapter.'))}
    histories = {}
    for history_path in sorted(args.run.glob('*/history.json')):
        rows = json.loads(history_path.read_text())
        if rows:
            identity_path = history_path.parent/'best.identity.json'
            histories[history_path.parent.name] = {'history': rows,
                'best_checkpoint_identity': json.loads(identity_path.read_text()) if identity_path.exists() else None,
                'complete': (history_path.parent/'complete.json').exists()}
    evidence = {'as_of': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
                'manifest_sha256': digest(manifest_path), 'e000_sha256': digest(replay_path),
                'folds': folds, 'parameters': parameters,
                'e000_diagnostics': replay_metrics(json.loads(replay_path.read_text())),
                'initial_search': histories,
                'status': json.loads((args.run/'status.json').read_text()),
                'final_data_loaded_for_collection': False,
                'provisional_warning': 'Development checkpoint screens are not final results or an improvement claim'}
    save_json(args.output/'evidence.json', evidence)
    if histories:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, item in histories.items():
            ax.plot([r['epoch']+1 for r in item['history']],
                    [r['development_per']*100 for r in item['history']], marker='o', label=name)
        ax.set(xlabel='Completed epoch', ylabel='Development macro-day PER (%)',
               title='Provisional development training — not final evaluation')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output/'development-learning-curve.png', dpi=160)
        plt.close(fig)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
