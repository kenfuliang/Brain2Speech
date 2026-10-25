import copy
import numpy as np
import pytest
import torch

from research.audit import digest
from research.data import CausalFeatures, prepare
from research.train import adapt, batch, fit, load_checkpoint


def trials():
    rng = np.random.default_rng(17)
    return [{'id': f'train-{i}', 'session': 'day1', 'text': 'a b',
             'raw': rng.poisson(2, (64, 256)).astype(np.float32),
             'labels': np.array([1, 2], dtype=np.int64)} for i in range(2)]


def test_native_fit_reload_resume_and_adaptation_ancestry(tmp_path):
    raw = trials()
    stats = CausalFeatures.fit(raw).state()
    prepared, _ = prepare(raw, stats)
    config = {'kind': 'dynamics', 'hidden': 16, 'seed': 17,
              'lr': .0003, 'auxiliary': .1, 'epochs': 1}
    device = torch.device('cpu')
    checkpoint, result = fit(config, prepared, prepared, stats, tmp_path/'fit', device)
    assert result['epochs_completed'] == 1
    assert result['checkpoint_sha256'] == digest(checkpoint)
    model, payload = load_checkpoint(checkpoint, device)
    model.eval()
    with torch.no_grad():
        before = model(batch(prepared, device)[0])['logits']
    resumed, repeated = fit(config, prepared, prepared, stats, tmp_path/'fit', device)
    assert repeated == result and resumed == checkpoint
    restored, _ = load_checkpoint(resumed, device)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(before, restored(batch(prepared, device)[0])['logits'])
    with pytest.raises(ValueError, match='configuration mismatch'):
        fit({**config, 'lr': .001}, prepared, prepared, stats, tmp_path/'fit', device)
    adapted = adapt(checkpoint, prepared, {**config, 'adaptation_steps': 1}, stats,
                    tmp_path/'adapt', device)
    _, after = load_checkpoint(adapted, device)
    assert after['extra']['parent_sha256'] == digest(checkpoint)
    assert after['extra']['calibration_ids'] == ['train-0', 'train-1']
    assert after['extra']['replay_ids'] == []
    for name, value in payload['model'].items():
        if not name.startswith('adapter.'):
            torch.testing.assert_close(value, after['model'][name])
    with pytest.raises(ValueError, match='ancestry/config'):
        adapt(checkpoint, prepared[:1], {**config, 'adaptation_steps': 1}, stats,
              tmp_path/'adapt', device)


def test_unlabeled_batches_do_not_require_feasible_ctc_labels():
    impossible = copy.deepcopy(trials())
    for t in impossible:
        t['x'] = t['raw'].copy()
        t['labels'] = np.ones(100, dtype=np.int64)
    with pytest.raises(ValueError, match='Impossible CTC'):
        batch(impossible, torch.device('cpu'))
    assert batch(impossible, torch.device('cpu'), require_ctc=False)[0].shape == (2, 64, 256)
