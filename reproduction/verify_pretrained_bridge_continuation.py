"""Independent continuation accounting and historical same-trial context.

Saved WFST outputs have different preprocessing and are not matched decoder controls.
"""
import json
import math
from pathlib import Path
import sys

from verify_pretrained_bridge import verify
from verify_causal_neural_llm import distance
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit import ROOT, digest, save_json


def audit(path):
    result = verify(path)
    read = lambda name: json.loads((path/name).read_text())
    protocol = read('training-protocol.json')
    parent = Path(protocol['parent'])
    assert digest(parent/'step-000500/checkpoint.pt') == protocol['parent_checkpoint_sha256']
    assert digest(parent/'step-000500/lora/adapter_model.safetensors') == protocol['parent_adapter_sha256']
    assert digest(Path(__file__).with_name('continue_pretrained_bridge.py')) == protocol['source_sha256']
    for name in ['features.npz', 'trials.json', 'data-identity.json', 'normalization.json',
                 'causality.json', 'phoneme-predictions.json', 'uncertainty-protocol.json']:
        assert digest(path/name) == digest(parent/name)
    initial = read('initial.json'); metrics = read('metrics.json')
    assert read('resume-validation.json')['passed']
    assert read('parent-verification.json')['passed']
    assert read('status.json')['status'] == 'complete'
    old = json.loads((parent/'metrics.json').read_text())['last']
    assert initial['step'] == protocol['start_step'] == old['step'] == 500
    for key in ['real', 'train_fixed_probe']:
        assert abs(initial['loss'][key]-old['loss'][key]) < 1e-5
    assert initial['wer']['errors'] == old['wer']['errors']
    history = [json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]
    assert [r['step'] for r in history] == list(range(501, metrics['steps_completed']+1))
    assert all(math.isfinite(r[k]) for r in history for k in ['loss', 'gradient_norm'])
    evaluations = [json.loads(p.read_text()) for p in sorted(path.glob('step-*/evaluation.json'))]
    expected = list(range(500, metrics['steps_completed']+1, 100))
    if expected[-1] != metrics['steps_completed']:
        expected.append(metrics['steps_completed'])
    assert [r['step'] for r in evaluations] == expected
    best = initial; best_loss = initial['loss']['real']; bad = stale = 0; stop = None
    for row in evaluations[1:]:
        assert stop is None, 'Training continued past its declared stop'
        stale = 0 if row['wer']['errors'] < best['wer']['errors'] else stale+1
        if (row['wer']['errors'], row['loss']['real']) < (best['wer']['errors'], best['loss']['real']):
            best = row
        best_loss = min(best_loss, row['loss']['real'])
        bad = bad+1 if row['loss']['real'] > 1.25*best_loss else 0
        stop = 'sustained_validation_deterioration' if bad >= 5 else (
            'validation_wer_plateau' if stale >= 10 else None)
    assert evaluations[-1]['step'] == metrics['steps_completed']
    assert metrics['stop_reason'] == (stop or 'absolute_step_review_boundary')
    if stop is None:
        assert metrics['steps_completed'] == protocol['steps']
    result['continuation'] = dict(parent_hashes_verified=True, immutable_inputs_equal=True,
        resume_losses_and_word_edits_replayed=True, source_hash_verified=True,
        finite_updates=len(history), stop_reason=metrics['stop_reason'],
        stop_rule_independently_recomputed=True,
        sampled_max_mps_driver_bytes=max(r['mps_driver_bytes'] for r in history),
        optimizer_tensor_restoration_independently_verified=False)
    released_path = ROOT/'reproduction/results/released-checkpoint-3gram.json'
    stopped_path = ROOT/'research/runs/e008-stopped-wer-20261007-01/decoded-predictions.json'
    assert digest(released_path) == '15309213c3497524b69abf8da16633f363bed1bb2c92c9cf0faf2568df5124e9'
    stopped_verification = json.loads((stopped_path.parent/'verification.json').read_text())
    assert digest(stopped_path) == stopped_verification['predictions_sha256']
    released = json.loads(released_path.read_text()); stopped = json.loads(stopped_path.read_text())
    assert released['true_transcripts'] == [r['text'] for r in stopped]
    validation = {r['id']: r['text'] for r in read('trials.json') if r['split'] == 'validation'}
    counts = {key: dict(errors=0, reference_words=0) for key in ['released_wfst', 'stopped_wfst']}
    selected = []
    for i, row in enumerate(stopped):
        identity = row['session']+'/test/%04d'%row['trial_index']
        if identity not in validation:
            continue
        assert row['text'] == validation[identity]
        reference = ' '.join(row['text'].lower().split()).split()
        selected.append(identity)
        for key, text in [('released_wfst', released['decoded_transcripts'][i]), ('stopped_wfst', row['decoded_text'])]:
            counts[key]['errors'] += distance(reference, ' '.join(text.lower().split()).split())
            counts[key]['reference_words'] += len(reference)
    assert len(selected) == len(set(selected)) == 40 and set(selected) == set(validation)
    for values in counts.values():
        values['wer'] = values['errors']/values['reference_words']
    result['historical_same_trial_context'] = dict(counts=counts, trials=40,
        artifacts={str(released_path): digest(released_path), str(stopped_path): digest(stopped_path)},
        caveat='Historical noncausal WFST pipelines versus causal-feature LLM; not matched decoder controls. E000 IDs inherited only after exact full-reference ordering check. Selected exposed validation, not an independent paper benchmark.')
    return result


if __name__ == '__main__':
    path = Path(sys.argv[1]); result = audit(path)
    if len(sys.argv) > 2:
        output = Path(sys.argv[2]); assert not output.exists(); save_json(output, result)
    print(json.dumps(result, indent=2))
