"""CPU-only, hash-bound learning curves for completed E016/E017 budgets."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.audit import digest, save_json

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = dict(
    question='Did training losses and monitored PER improve over the full frozen budgets?',
    takeaway='Completed learning trajectories are diagnostic evidence, not matched-paper WER reproduction.',
    family='Trend', variant='Two-panel learning-curve small multiples',
    points='E016:800 epochs; E017:10001 batches and201 validation checks',
    surface='Existing canonical Markdown research report with requested static PNG assets',
    renderer='Matplotlib Agg, CPU only',
    palette='Hard two-root cap:blue #2563A6,orange #C86D26,neutral #555555',
    distinction='Train solid, validation dashed; selected-point open circle; axes have explicit units',
    limitation='Different data/normalization/cohorts/loss units; no cross-model loss or PER superiority claim',
    blossom='Omitted for third-party paper-method reproduction charts',
    structure='Existing report summary/definitions/model findings/decision cards retained; add only relevant figures',
    qa='Inspect exported PNG at full width, then in canonical report panel')


def figure(title, subtitle):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(top=.76, bottom=.22, left=.08, right=.97, wspace=.27)
    fig.text(.08, .95, title, fontsize=16, fontweight='bold', color='#222222')
    fig.text(.08, .86, subtitle, fontsize=10, color='#555555')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E4E4E4', linewidth=.7)
        ax.tick_params(labelsize=10)
    return fig, axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    bit = ROOT/'research/runs/e016-bit-full-s42-20261009-01'
    willett = ROOT/'research/runs/e017-willett-continuation-s17-20261009-01'
    parent = ROOT/'reproduction/results/retraining/pytorch-retrain-s17-20261007-01'
    sources = {}

    def read(path):
        sources[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    final = read(bit/'final-audit-20261009-01.json')
    assert final['passed'] and final['epoch'] == 800
    complete = read(willett/'final-training-audit-20261009-01.json')
    assert complete['passed'] and complete['complete_frozen_budget']
    history = read(bit/'history.json')
    assert [row['epoch'] for row in history] == list(range(1, 801))
    chosen = read(bit/'best-per.json')
    assert min(history, key=lambda row: row['validation']['per'])['epoch'] == chosen['epoch']
    journals = [parent/'training.jsonl']+sorted(willett.glob('training-resume-*.jsonl'))
    training = []
    for path in journals:
        sources[str(path.relative_to(ROOT))] = digest(path)
        training.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    assert [row['attempt'] for row in training] == list(range(10001))
    assert all(np.isfinite(row['prediction_loss']) for row in training)
    validations = []
    for index in range(0, 10001, 50):
        folder = parent if index < 3000 else willett
        row = read(folder/f'validation-{index:05d}.json')
        assert row['trials'] == 680 and row['phones'] == 19024
        validations.append(dict(attempt=index, per=row['per'], edits=row['edits'], phones=row['phones']))
    selected = read(willett/'selected.json')
    assert min(validations, key=lambda row: row['per'])['attempt'] == selected['attempt']
    args.output.mkdir(parents=True)
    save_json(args.output/'chart-data.json', dict(bit=history, willett_training=training,
                                                willett_validation=validations))
    blue, orange = '#2563A6', '#C86D26'
    fig, (loss, per) = figure('BIT scratch: full training trajectory',
        '800 epochs /220,000 updates; seed42;8,800 train/440 validation; one participant')
    epochs = [row['epoch'] for row in history]
    loss.plot(epochs, [r['train_ctc_per_reference_phone'] for r in history], color=blue, label='Training')
    loss.plot(epochs, [r['validation']['ctc_per_reference_phone'] for r in history],
              color=orange, linestyle='--', label='Validation')
    loss.set(yscale='log', ylabel='CTC per reference phone (log scale)', xlabel='Epoch')
    loss.legend(frameon=False)
    per.plot(epochs, [100*r['validation']['per'] for r in history], color=blue)
    per.plot(chosen['epoch'], 100*chosen['metric'], 'o', color=orange, markerfacecolor='white')
    per.set(xlabel='Epoch', ylabel='Standard validation PER (%)')
    per.annotate(f"Selected epoch{chosen['epoch']}: {100*chosen['metric']:.2f}%",
                 xy=(chosen['epoch'], 100*chosen['metric']), xytext=(260, 31),
                 arrowprops=dict(arrowstyle='-', color='#555555'), fontsize=10)
    fig.text(.08, .07, 'Different train/validation conditions; no uncertainty band or seed dispersion.\n'
        'Paper/code validation-cohort mismatch prevents a matched performance claim.', fontsize=10, color='#555555')
    fig.savefig(args.output/'bit-learning.png', dpi=160); plt.close(fig)
    fig, (loss, per) = figure('Willett released recipe: full training trajectory',
        '10,001 scheduled attempts; seed17; augmented batch64;680 public monitoring trials')
    values = np.array([row['prediction_loss'] for row in training])
    # Retain all raw batches; smooth only with a trailing50-batch summary.
    means = np.convolve(values, np.ones(50)/50, mode='valid')
    loss.plot(range(10001), values, color='#BBBBBB', linewidth=.35, label='Raw augmented batch')
    loss.plot(range(49, 10001), means, color=blue, linewidth=1.3, label='Trailing50-batch mean')
    loss.set(xlabel='Attempt index (zero based)', ylabel='Mean sentence prediction CTC')
    loss.legend(frameon=False)
    per.plot([r['attempt'] for r in validations], [100*r['per'] for r in validations], color=blue)
    per.plot(selected['attempt'], 100*selected['per'], 'o', color=orange, markerfacecolor='white')
    per.set(xlabel='Attempt index (zero based)', ylabel='Monitored680-trial PER (%)')
    per.annotate(f"Selected{selected['attempt']}: {100*selected['per']:.2f}%",
        xy=(selected['attempt'], 100*selected['per']), xytext=(3500, 45),
        arrowprops=dict(arrowstyle='-', color='#555555'), fontsize=10)
    for ax in (loss, per):
        ax.axvline(2957, color='#555555', linestyle=':', linewidth=1)
    fig.text(.08, .07, 'Dotted line: preserved-parent continuation at2957. Batch variability includes different days/augmentation.\n'
        '680-trial monitoring PER is not600-trial benchmark PER; no WER or causality claim from these curves.',
        fontsize=10, color='#555555')
    fig.savefig(args.output/'willett-learning.png', dpi=160); plt.close(fig)
    save_json(args.output/'provenance.json', dict(contract=CONTRACT, sources_sha256=sources,
        renderer_sha256=digest(Path(__file__)), outputs_sha256={p.name: digest(p) for p in args.output.iterdir()},
        repeated_family_reason='Both questions concern dense optimization trajectories; separate loss/PER axes prevent unit conflation'))
    print('saved-completed-budget-learning-curves')


if __name__ == '__main__':
    main()
