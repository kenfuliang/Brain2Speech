import copy
import numpy as np
import pytest
import torch

from research.audit import ACQUISITION_EVIDENCE, normalize_text
from research.data import CausalFeatures, prefix
from research.metrics import edits, greedy, summarize
from research.models import Encoder, objective
from research.run import check_manifest


def test_feature_prefix_invariance_and_past_only_updates():
    rng = np.random.default_rng(4)
    raw = rng.poisson(2, (64, 256)).astype(np.float32)
    altered = raw.copy()
    altered[32:] += 100
    initial = {'mean': np.ones(256), 'variance': np.ones(256), 'samples': 100}
    first, second = CausalFeatures(**initial), CausalFeatures(**initial)
    x = first.process(raw, update=True)
    y = second.process(altered, update=True)
    np.testing.assert_allclose(x[:32], y[:32])
    assert not np.allclose(first.mean, second.mean)


@pytest.mark.parametrize('kind', ['gru', 'supervised', 'dynamics', 'predictive'])
def test_encoder_prefix_and_checkpoint_reload(kind, tmp_path):
    torch.manual_seed(17)
    model = Encoder(kind).eval()
    x = torch.randn(2, 64, 256)
    changed = x.clone()
    changed[:, 32:] += 20
    with torch.no_grad():
        first = model(x)
        second = model(changed)
    torch.testing.assert_close(first['logits'][:, :8], second['logits'][:, :8])
    checkpoint = tmp_path / 'encoder.pt'
    torch.save(model.state_dict(), checkpoint)
    reloaded = Encoder(kind).eval()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(first['logits'], reloaded(x)['logits'])


@pytest.mark.parametrize('kind', ['gru', 'supervised', 'dynamics', 'predictive'])
def test_finite_training_and_gradient(kind):
    model = Encoder(kind).train()
    raw = torch.poisson(torch.ones(2, 64, 256)*2)
    batch = (torch.randn(2, 64, 256), raw, torch.tensor([1, 2, 3, 4]),
             torch.tensor([64, 60]), torch.tensor([2, 2]))
    loss, metrics = objective(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.adapter.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_ctc_collapse_and_edit_accounting():
    logits = np.full((6, 41), -10.)
    logits[np.arange(6), [1, 1, 40, 1, 2, 40]] = 10
    assert greedy(logits) == [1, 1, 2]
    assert edits(['a', 'b'], ['a', 'c', 'd']) == {
        'errors': 2, 'substitutions': 1, 'deletions': 0, 'insertions': 1, 'reference_units': 2}
    assert edits([], ['a'])['insertions'] == 1
    assert edits(['a'], [])['deletions'] == 1


def test_metrics_day_aggregation():
    rows = [{'session': 'day1', 'per': edits([1, 2], [1])},
            {'session': 'day2', 'per': edits([1, 2, 3, 4], [1, 2, 3, 4])}]
    out = summarize(rows, repetitions=20)
    assert out['per']['corpus'] == 1/6
    assert out['per']['macro_day'] == .25
    assert out['per']['worst_day'] == .5


def test_strict_acquisition_gate_cannot_be_bypassed():
    manifest = {'causal_gate': ACQUISITION_EVIDENCE,
                'folds': {'initial_train': ['a'], 'development': ['b'], 'final': ['c']},
                'eligible_sessions': ['a', 'b', 'c'],
                'sessions': {d: {'train': {'trials': [{'block': 1}]},
                                 'test': {'trials': [{'block': 2}]}} for d in 'abc'}}
    assert check_manifest(manifest) == []
    denied = copy.deepcopy(manifest)
    denied['causal_gate'] = {**ACQUISITION_EVIDENCE, 'strict_gate': 'blocked'}
    assert check_manifest(denied) == [ACQUISITION_EVIDENCE['reason']]
    bad = copy.deepcopy(manifest)
    bad['folds']['final'] = ['a']
    assert len(check_manifest(bad)) >= 2


def test_nested_calibration_and_normalization():
    assert prefix(list(range(100)), 5) == list(range(5))
    assert len(prefix(list(range(100)), 'full')) == 100
    assert normalize_text('Hello, WORLD!') == 'hello world'
