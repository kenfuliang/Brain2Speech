"""Independent exact-state comparison of the current-state two-update resume gate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import train_pytorch_recipe as recipe


def exact(left, right):
    assert type(left) is type(right), 'State types differ'
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right), 'Tensor state differs'
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            exact(x, y)
    else:
        assert left == right, 'Scalar state differs'


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'resumed', 'contract', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    contract = json.loads(args.contract.read_text())
    parent = Path(contract['parent'])
    for relative, expected in contract['parent_artifact_sha256'].items():
        assert recipe.sha(parent / relative) == expected, 'Preserved parent artifact changed'
    reference = torch.load(args.reference / 'latest.pt', map_location='cpu', weights_only=False)
    resumed = torch.load(args.resumed / 'latest.pt', map_location='cpu', weights_only=False)
    assert reference['next_attempt'] == resumed['next_attempt'] == 2959
    assert reference['seed'] == resumed['seed'] == 17
    assert len(reference['history']) == 2959
    original = torch.load(parent / 'latest.pt', map_location='cpu', weights_only=False)
    boundary = torch.load(args.reference / 'resume-boundary-2958.pt', map_location='cpu', weights_only=False)
    assert original['next_attempt'] == 2957 and boundary['next_attempt'] == 2958
    exact(reference['history'][:2957], original['history'])
    exact(reference['history'][:2958], boundary['history'])
    exact(reference, resumed)
    np.testing.assert_array_equal(np.load(args.reference / 'final-probe-logits.npy'),
                                  np.load(args.resumed / 'final-probe-logits.npy'))
    journals = []
    for directory in (args.reference, args.resumed):
        assert json.loads((directory / 'completion.json').read_text())['status'] == 'training-complete-evaluation-required'
        rows = [json.loads(line) for line in (directory / 'training.jsonl').read_text().splitlines()]
        assert rows and all(row['update_applied'] for row in rows)
        assert all(np.isfinite(row[key]) for row in rows for key in ('loss', 'prediction_loss', 'gradient_norm'))
        journals.append(rows)
    assert [row['attempt'] for row in journals[0]] == [2957, 2958]
    assert [row['attempt'] for row in journals[1]] == [2958]
    exact({key: value for key, value in journals[0][-1].items() if key != 'seconds'},
          {key: value for key, value in journals[1][-1].items() if key != 'seconds'})
    identity = json.loads((parent / 'run-identity.json').read_text())
    for relative, expected in identity['source_hashes'].items():
        assert recipe.sha(recipe.ROOT / relative) == expected
    artifacts = [args.reference / 'latest.pt', args.reference / 'resume-boundary-2958.pt',
                 args.resumed / 'latest.pt', args.contract.parent / 'preparation-verification.json',
                 Path(__file__), recipe.ROOT / 'reproduction/willett_resume_probe.py']
    result = dict(status='passed-recipe-and-fresh-resume', next_attempt=2959,
                  parent_checkpoint_sha256=contract['parent_checkpoint_sha256'], parent_unchanged=True,
                  model_and_optimizer_exact=True, data_stream_and_rng_exact=True,
                  source_hashes=identity['source_hashes'],
                  artifact_sha256={str(path.resolve()): recipe.sha(path) for path in artifacts},
                  limitations=['Two-update current-state continuation gate; not convergence or WER',
                               'Historical public selection/evaluation overlap remains'])
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(result['status'])


if __name__ == '__main__':
    main()
