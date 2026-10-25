import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reproduction'))
from export_willett_final import checked_phones
from verify_willett_export import audit_trial
from research.metrics import edits


def trial():
    values = np.full((3, 41), -10., dtype=np.float32)
    values[[0, 1, 2], [2, 40, 2]] = 10.
    reference = [2, 2]
    loss = torch.nn.functional.ctc_loss(torch.from_numpy(values).log_softmax(-1).unsqueeze(1),
        torch.tensor(reference), torch.tensor([3]), torch.tensor([2]), blank=40, reduction='sum').item()
    return dict(output_bins=3, reference_phones=reference, predicted_phones=reference,
                per=edits(reference, reference), prediction_loss=loss), values


def test_blank_separated_repeat_retained():
    row, values = trial()
    audit_trial(row, values)


def test_changed_logits_rejected():
    row, values = trial()
    values[2, 3] = 20.
    with pytest.raises(AssertionError):
        audit_trial(row, values)


def test_changed_loss_rejected():
    row, values = trial()
    row['prediction_loss'] += 1.
    with pytest.raises(AssertionError):
        audit_trial(row, values)


def test_blank_reference_rejected():
    with pytest.raises(AssertionError):
        checked_phones(torch.tensor([40]), 1)
