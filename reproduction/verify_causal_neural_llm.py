"""Independent split/hash/scalar edit-distance accounting for E012 artifacts."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit import digest, save_json


def distance(a, b):
    grid = np.zeros((len(a)+1, len(b)+1), dtype=np.int64)
    grid[:, 0] = np.arange(len(a)+1); grid[0] = np.arange(len(b)+1)
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            grid[i, j] = min(grid[i-1, j]+1, grid[i, j-1]+1,
                             grid[i-1, j-1]+int(x != y))
    return int(grid[-1, -1])


def verify(path):
    read = lambda name: json.loads((path/name).read_text())
    metrics, identity, data = read('metrics.json'), read('identity.json'), read('data-identity.json')
    assert read('status.json')['status'] == 'complete'
    assert digest(Path(__file__).with_name('causal_neural_llm.py')) == identity['code_sha256']
    assert digest(path/'trials.json') == data['trials_sha256']
    rows = read('trials.json'); train, validation = rows[:64], rows[64:]
    norm = lambda text: ' '.join(text.lower().split())
    assert len(rows) == 80 and len({r['id'] for r in rows}) == 80
    assert all('/train/' in r['id'] for r in rows)
    assert not {norm(r['text']) for r in train} & {norm(r['text']) for r in validation}
    assert max(r['index'] for r in train) < min(r['index'] for r in validation)
    predictions = read('generated-transcripts.json'); assert len(predictions) == 64
    lookup = {r['id']: r['text'] for r in validation}
    for condition, aggregate in metrics['generation'].items():
        subset = [r for r in predictions if r['condition'] == condition]
        assert len(subset) == 16 and {r['id'] for r in subset} == set(lookup)
        errors = units = 0
        for row in subset:
            assert row['reference'] == lookup[row['id']]
            a, b = norm(row['reference']).split(), norm(row['prediction']).split()
            count = distance(a, b)
            assert count == row['wer']['errors']
            assert len(a) == row['wer']['reference_units']
            assert count == sum(row['wer'][k] for k in ['substitutions', 'deletions', 'insertions'])
            errors += count; units += len(a)
        assert errors == aggregate['errors'] and units == aggregate['reference_words']
        assert abs(errors/units-aggregate['wer']) < 1e-12
    history = [json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]
    assert [r['step'] for r in history] == list(range(1, identity['steps']+1))
    assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in history)
    return dict(passed=True, split_and_prompt_separation=True, scalar_word_edits_recomputed=True,
                component_sums_checked=True, component_backtracking_independently_verified=False,
                complete_finite_steps=len(history), metrics_sha256=digest(path/'metrics.json'),
                checkpoint_sha256=digest(path/'encoder-final.pt'),
                unique_real_predictions=len({r['prediction'] for r in predictions if r['condition']=='real'}),
                limit='Artifact accounting; does not independently repeat full training or prove neural grounding')


if __name__ == '__main__':
    path = Path(sys.argv[1]); output = path/'verification.json'
    assert not output.exists(), 'Preserve existing verification'
    result = verify(path); save_json(output, result); print(json.dumps(result, indent=2))
