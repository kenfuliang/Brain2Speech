"""Bridge-only accounting, initialization identity and paired exploratory comparison."""
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

from verify_pretrained_bridge import verify
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit import ROOT, digest, save_json


def audit(path):
    result = verify(path)
    read = lambda name: json.loads((path/name).read_text())
    protocol = read('training-protocol.json'); metrics = read('metrics.json')
    parent = Path(protocol['parent'])
    assert digest(Path(__file__).with_name('train_bridge_only.py')) == protocol['source_sha256']
    assert digest(parent/'step-000000/checkpoint.pt') == protocol['parent_initial_checkpoint_sha256']
    for name in ['features.npz', 'trials.json', 'normalization.json', 'data-identity.json']:
        assert digest(path/name) == digest(parent/name)
    original = torch.load(parent/'step-000000/checkpoint.pt', map_location='cpu', weights_only=False)
    initial = torch.load(path/'step-000000/checkpoint.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(value, initial['bridge'][key]) for key, value in original['bridge'].items())
    assert sum(value.numel() for value in initial['bridge'].values()) == protocol['trainable_parameters']
    assert protocol['language_trainable_parameters'] == metrics['language_trainable_parameters'] == 0
    assert not protocol['lora'] and metrics['lora_removed']
    assert read('initial-replay.json')['passed']
    assert metrics['frozen_language_weights_unchanged'] and metrics['reload_consistent']
    assert read('status.json')['status'] == 'complete'
    history = [json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]
    assert [r['step'] for r in history] == list(range(1, metrics['steps_completed']+1))
    assert all(math.isfinite(r[k]) for r in history for k in ['loss', 'gradient_norm'])
    evaluations = [json.loads(p.read_text()) for p in sorted(path.glob('step-*/evaluation.json'))]
    expected = list(range(0, metrics['steps_completed']+1, 100))
    if expected[-1] != metrics['steps_completed']:
        expected.append(metrics['steps_completed'])
    assert [r['step'] for r in evaluations] == expected
    assert all(abs(r['loss']['language-only']-evaluations[0]['loss']['language-only']) < 1e-12
               for r in evaluations), 'Frozen no-prefix language loss changed'
    best = evaluations[0]; best_loss = best['loss']['real']; bad = stale = 0; stop = None
    for row in evaluations[1:]:
        assert stop is None
        stale = 0 if row['wer']['errors'] < best['wer']['errors'] else stale+1
        if (row['wer']['errors'], row['loss']['real']) < (best['wer']['errors'], best['loss']['real']):
            best = row
        best_loss = min(best_loss, row['loss']['real'])
        bad = bad+1 if row['loss']['real'] > 1.25*best_loss else 0
        stop = 'sustained_validation_deterioration' if bad >= 5 else (
            'validation_wer_plateau' if stale >= 10 else None)
    assert metrics['stop_reason'] == (stop or 'matched_step_budget')
    if stop is None:
        assert metrics['steps_completed'] == protocol['steps']
    result['technical'] = dict(initial_bridge_tensors_identical=True,
        finite_updates=len(history), stop_rule_independently_recomputed=True,
        no_prefix_language_loss_unchanged=True,
        sampled_max_mps_driver_bytes=max(r['mps_driver_bytes'] for r in history),
        base_weight_freeze='Runtime full-language digest and zero-gradient checks; no independent model tensor replay')
    comparator = ROOT/'research/runs/e013-pretrained-causal-bridge-s17-20261008-02'
    baseline = json.loads((comparator/'best-real-predictions.json').read_text())
    candidate = read('best-real-predictions.json')
    assert [r['id'] for r in baseline] == [r['id'] for r in candidate]
    assert [r['reference'] for r in baseline] == [r['reference'] for r in candidate]
    difference = np.array([a['wer']['errors']-b['wer']['errors'] for a,b in zip(candidate, baseline)])
    units = np.array([r['wer']['reference_units'] for r in candidate])
    indices = np.random.default_rng(101).integers(0, 40, size=(2000,40))
    draws = difference[indices].sum(1)/units[indices].sum(1)
    matched = []
    for row in evaluations:
        source = parent if row['step'] <= 500 else comparator
        file = source/f"step-{row['step']:06d}/evaluation.json"
        if not file.exists():
            continue
        other = json.loads(file.read_text())
        matched.append(dict(step=row['step'], bridge_only_wer=row['wer']['rate'],
            bridge_lora_wer=other['wer']['rate'], bridge_only_validation_loss=row['loss']['real'],
            bridge_lora_validation_loss=other['loss']['real']))
    result['bridge_lora_comparison'] = dict(bridge_only_minus_lora=float(difference.sum()/units.sum()),
        conditional_paired_trial_ci95=np.quantile(draws,[.025,.975]).tolist(),
        baseline_selected_step=json.loads((comparator/'best.json').read_text())['step'],
        common_update_checkpoints=matched,
        caveat='Selected checkpoints on the same 40 exposed validation sentences, one day/seed; not selection-adjusted or independent test. Equal maximum 1800 updates; actual early-stop budgets may differ. Original LoRA continuation reset loss-monitor at update500, unlike bridge-only uninterrupted monitoring.')
    return result


if __name__ == '__main__':
    path = Path(sys.argv[1]); result = audit(path)
    if len(sys.argv) > 2:
        output = Path(sys.argv[2]); assert not output.exists(); save_json(output, result)
    print(json.dumps(result, indent=2))
