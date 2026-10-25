"""CPU-only independent logit/phone/CTC audit before final WFST scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import train_pytorch_recipe as recipe
from verify_bit_full import decode, distance
from research.metrics import edits, summarize


def audit_trial(row, values):
    assert values.shape == (row['output_bins'], 41) and np.isfinite(values).all()
    reference = row['reference_phones']
    assert reference and all(isinstance(p, int) and 0 <= p < 40 for p in reference)
    prediction = decode(values)
    assert prediction == row['predicted_phones']
    assert distance(reference, prediction) == row['per']['errors']
    assert edits(reference, prediction) == row['per']
    loss = torch.nn.functional.ctc_loss(torch.from_numpy(values).log_softmax(-1).unsqueeze(1),
        torch.tensor(reference), torch.tensor([len(values)]), torch.tensor([len(reference)]),
        blank=40, reduction='sum').item()
    assert np.isclose(loss, row['prediction_loss'], rtol=1e-5, atol=1e-5)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    output = args.run/'export-verification.json'
    assert not output.exists()
    identity = json.loads((args.run/'identity.json').read_text())
    assert identity['full_budget_attempts'] == 10001
    assert identity['fresh_model_reload_logits_exact'] and identity['selected_per_exact']
    assert recipe.sha(Path(identity['checkpoint'])) == identity['checkpoint_sha256']
    for name, expected in identity['artifacts_sha256'].items():
        assert recipe.sha(args.run/name) == expected
    rows = json.loads((args.run/'predictions.json').read_text())
    all_rows = json.loads((args.run/'selection-predictions.json').read_text())
    assert len(rows) == 600 and len(all_rows) == 680
    assert rows == [row for row in all_rows if row['day'] >= 4]
    assert len({row['trial_id'] for row in all_rows}) == 680
    for day in range(19):
        subset = [row for row in all_rows if row['day'] == day]
        assert [row['trial_index'] for row in subset] == list(range(20 if day < 4 else 40))
        for row in subset:
            assert row['trial_id'] == f"{row['session']}/test/{row['trial_index']:04d}"
            assert edits(row['reference_phones'], row['predicted_phones']) == row['per']
    per = sum(row['per']['errors'] for row in all_rows)/sum(row['per']['reference_units'] for row in all_rows)
    assert per == identity['selected']['per']
    with np.load(args.run/'logits.npz') as archive:
        assert set(archive.files) == {f'trial_{i:04d}' for i in range(600)}
        losses = [audit_trial(row, archive[f'trial_{i:04d}']) for i, row in enumerate(rows)]
    metrics = json.loads((args.run/'metrics-before-lm.json').read_text())
    recomputed = summarize(rows, repetitions=2000)
    assert recomputed['per'] == metrics['per'] and recomputed['by_day'] == metrics['by_day']
    assert np.isclose(np.mean(losses), metrics['mean_prediction_loss'], rtol=1e-5)
    result = dict(passed=True, independent_greedy_phone_edits=True, independent_ctc=True,
        selected_680_phone_counts=True, benchmark_trials=600, selection_trials=680,
        export_identity_sha256=recipe.sha(args.run/'identity.json'),
        auditor_sha256=recipe.sha(Path(__file__)),
        limitation='Raw references originate in sealed source TF service; public selection/evaluation overlap remains')
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print('passed-independent-logit-phone-ctc-audit')


if __name__ == '__main__':
    main()
