"""Pilot interface tests; do not claim complete BIT reproduction from these."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('bit_ctc_pilot', ROOT/'reproduction/train_bit_ctc_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_source_phoneme_order():
    import json
    assert json.loads((pilot.VENDOR/'vocab.json').read_text()) == ['BLANK'] + list(pilot.PHONES)
    author = np.arange(41)[None, :]
    native = author[:, list(range(1,41))+[0]]
    assert native[0,40] == 0 and native[0,39] == 40 and native[0,0] == 1


def test_padding_mask_timestamp_contract():
    rows = [{'id': 'a'}, {'id': 'b'}]
    arrays = {'a': np.ones((7,256), np.float32), 'b': np.ones((3,256), np.float32)}
    x, mask, ts, ds = pilot.batch(rows, arrays, 'cpu')
    assert mask.sum(1).tolist() == [7,3]
    assert torch.all(x[1,3:] == -100)
    assert ts[1].tolist() == [0,1,2,0,0,0,0]
    assert ds.tolist() == [1,1]


def test_original_model_cpu_gradient_and_reload():
    torch.set_num_threads(2); torch.manual_seed(17)
    net = pilot.model('cpu')
    rows = [{'id': 'a', 'phonemes': [1,2,3]}, {'id': 'b', 'phonemes': [2,3]}]
    arrays = {'a': np.ones((50,256), np.float32), 'b': np.ones((40,256), np.float32)}
    net.train()
    loss, _, lengths = pilot.forward(net, rows, arrays, 'cpu')
    assert lengths.tolist() == [10,8]
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    net.eval()
    with torch.no_grad():
        _, before, _ = pilot.forward(net, rows, arrays, 'cpu')
        fresh = pilot.model('cpu').eval()
        fresh.load_state_dict(net.state_dict())
        _, after, _ = pilot.forward(fresh, rows, arrays, 'cpu')
    assert torch.equal(before, after)


def test_independent_edit_accounting():
    spec = importlib.util.spec_from_file_location('bit_decode', ROOT/'reproduction/decode_bit_ctc_pilot.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for a,b in [([],[]), ([1,2],[1,3,4]), ([1,1,2],[2]), (list('abc'),list('xab'))]:
        assert module.independent_distance(a,b) == pilot.edits(a,b)['errors']
