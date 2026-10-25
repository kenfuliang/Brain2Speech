"""CPU audit of a sealed E017 checkpoint; progress is not full completion."""
import argparse
import json
import math
from pathlib import Path

import torch
import train_pytorch_recipe as recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    contract = json.loads((args.run/'continuation-contract.json').read_text())
    parent = Path(contract['parent'])
    for name, expected in contract['parent_artifact_sha256'].items():
        assert recipe.sha(parent/name) == expected
    identity = json.loads((args.run/'run-identity.json').read_text())
    for mapping in ('source_hashes', 'tfrecord_hashes'):
        for name, expected in identity[mapping].items():
            assert recipe.sha(recipe.ROOT/name) == expected
    path = args.run/args.checkpoint
    sealed = recipe.sha(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert recipe.sha(path) == sealed, 'Checkpoint changed during audit; recheck, do not restart trainer'
    next_attempt = state['next_attempt']
    assert 2957 <= next_attempt <= 10001 and state['seed'] == 17
    assert len(state['history']) == next_attempt
    assert all(torch.isfinite(value).all() for value in state['model'].values())
    assert state['optimizer']['param_groups'][0]['iterations'] == next_attempt
    for slots in state['optimizer']['state'].values():
        for value in slots.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
    original = torch.load(parent/'latest.pt', map_location='cpu', weights_only=False)
    assert state['history'][:2957] == original['history']
    journals = sorted(args.run.glob('training-resume-*.jsonl'))
    rows = [json.loads(line) for journal in journals for line in journal.read_text().splitlines() if line]
    consumed = sorted((row for row in rows if row['attempt'] < next_attempt), key=lambda row: row['attempt'])
    assert [row['attempt'] for row in consumed] == list(range(2957, next_attempt))
    for row in consumed:
        assert row['update_applied'] and row['optimizer_iterations'] == row['attempt']+1
        assert all(math.isfinite(row[key]) for key in ('loss', 'prediction_loss', 'gradient_norm', 'lr'))
        assert abs(row['lr']-.02*(1-min(row['attempt'], 10000)/10000)) < 1e-12
        assert state['history'][row['attempt']] == dict(day=row['day'], data_hash=row['data_hash'])
    expected_indices = list(range(3000, next_attempt, 50))
    validations = []
    for index in expected_indices:
        result = json.loads((args.run/f'validation-{index:05d}.json').read_text())
        assert result['trials'] == 680
        assert {int(k): value for k, value in result['per_day_trials'].items()} == {
            day: 20 if day < 4 else 40 for day in range(19)}
        assert sum(row['edits'] for row in result['rows']) == result['edits']
        assert sum(row['length'] for row in result['rows']) == result['phones']
        assert result['per'] == result['edits']/result['phones']
        validations.append(result['per'])
    assert state['best_per'] == min([contract['inherited_selection']['per']]+validations)
    complete = next_attempt == 10001
    if complete:
        completion = json.loads((args.run/'completion.json').read_text())
        assert completion['status'] == 'training-complete-evaluation-required' and completion['attempts'] == 10001
    result = dict(passed=True, checkpoint=str(path.resolve()), checkpoint_sha256=sealed,
        next_attempt=next_attempt, complete_frozen_budget=complete,
        continuation_attempts_verified=len(consumed), validation_checks=len(validations),
        best_validation_per=state['best_per'], sources_and_data_verified=True, parent_unchanged=True,
        finite_model_optimizer_and_logged_updates=True, schedule_and_history_verified=True,
        auditor_sha256=recipe.sha(Path(__file__)),
        limitation='Checkpoint-scoped CPU audit; no new reload/stream replay or WER; current live progress may be later')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
