"""Read-only CPU acceptance audit for E022, without opening neural recordings."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(row[j] + 1, nxt[-1] + 1, row[j-1] + (x != y)))
        row = nxt
    return row[-1]


def audit(run, kind):
    protocol = json.loads((run / 'protocol.json').read_text())
    for family in ('source_hashes', 'implementation_hashes', 'derived_hashes'):
        for path, expected in protocol[family].items():
            assert sha(path) == expected, ('Frozen identity changed', path)
    split = json.loads((run / 'splits.json').read_text())
    train = {r['id']: r['session'] for r in split['train']}
    held = {r['id']: r['session'] for r in split['selection']}
    assert len(train) == 4400 and len(held) == 480 and not train.keys() & held.keys()
    assert not set(train.values()) & set(protocol['forbidden_final_days'])
    folder = run / kind
    complete = json.loads((folder / 'training-complete.json').read_text())
    latest = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    history = latest['history']
    assert complete['attempts'] == latest['next_attempt'] == len(history) == 10001
    selection = []
    for i, record in enumerate(history):
        assert record['attempt'] == i and record['optimizer_iteration'] == i + 1
        assert np.isclose(record['lr'], .02 * (1 - i / 10000), rtol=0, atol=1e-15)
        assert np.isfinite([record['CTC'], record['loss']]).all()
        assert len(record['ids']) == 64
        assert all(train.get(t) == record['day'] for t in record['ids'])
        if 'selection_PER' in record:
            assert i % 50 == 0 and np.isfinite(record['selection_PER'])
            selection.append((i, record['selection_PER']))
    assert [i for i, _ in selection] == list(range(0, 10001, 50))
    best_i, best = min(selection, key=lambda x: x[1])
    assert latest['best'] == best and latest['selected'] == complete['selected'] == f'best-{best_i:05d}.pt'
    selected_path = folder / complete['selected']
    assert sha(selected_path) == complete['sha256']
    selected = torch.load(selected_path, map_location='cpu', weights_only=False)
    assert selected['attempt'] == best_i and selected['selection_PER'] == best
    tensors = list(latest['model'].values()) + list(selected['model'].values())
    tensors += [v for state in latest['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v)]
    assert all(np.isfinite(t.numpy()).all() for t in tensors)
    prediction_path = folder / 'selected-held-blocks/predictions.json'
    logits_path = folder / 'selected-held-blocks/logits.npz'
    rows = json.loads(prediction_path.read_text())
    assert len(rows) == len(held) and {r['id'] for r in rows} == held.keys()
    assert len({r['id'] for r in rows}) == len(rows)
    errors = units = 0
    with np.load(logits_path) as archive:
        assert len(archive.files) == len(rows)
        for i, r in enumerate(rows):
            assert r['session'] == held[r['id']]
            logits = archive[f'trial_{i:04d}']
            assert logits.ndim == 2 and logits.shape[1] == 41 and np.isfinite(logits).all()
            sequence = logits.argmax(-1).tolist()
            phones = [p for j, p in enumerate(sequence) if p != 40 and (j == 0 or p != sequence[j-1])]
            assert phones == r['predicted_phones']
            assert all(0 <= p < 40 for p in r['reference_phones'])
            n = distance(r['reference_phones'], phones)
            assert n == r['per']['errors'] == sum(r['per'][k] for k in ('substitutions', 'deletions', 'insertions'))
            assert len(r['reference_phones']) == r['per']['reference_units']
            errors += n
            units += len(r['reference_phones'])
    assert errors / units == best == complete['PER']['corpus']
    if kind == 'raw':
        gate = json.loads((folder / 'raw-performance-gate.json').read_text())
        assert gate['budget'] == 10001 and gate['PER'] == best and gate['checkpoint_sha256'] == complete['sha256']
        assert gate['status'] == ('pass' if best <= protocol['raw_gate']['threshold'] else 'fail')
    return dict(status='pass', arm=kind, budget=10001, selection_attempt=best_i,
                selected_sha256=complete['sha256'], held_trials=len(rows), phone_errors=errors,
                reference_phones=units, held_corpus_PER=best,
                first200_CTC=float(np.mean([r['CTC'] for r in history[:200]])),
                last200_CTC=float(np.mean([r['CTC'] for r in history[-200:]])),
                evidence_hashes={str(p): sha(p) for p in (selected_path, prediction_path, logits_path)},
                scope='CPU accounting/identity audit; no fresh neural inference, WER or final-day recording access')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--kind', choices=['raw', 'anscombe', 'sqrt'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run.resolve(), args.kind)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
