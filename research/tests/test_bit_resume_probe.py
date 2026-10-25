import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reproduction'))
from bit_resume_probe import epoch_batches, exact


def test_epoch_order_and_rng_restore():
    sampler = np.random.default_rng(42)
    restored = np.random.default_rng()
    restored.bit_generator.state = sampler.bit_generator.state
    first = epoch_batches(sampler, 65)
    assert sorted(i for group in first for i in group) == list(range(65))
    assert first == epoch_batches(restored, 65)
    exact(sampler.bit_generator.state, restored.bit_generator.state)


def test_exact_state_and_changed_moment():
    first = {'state': [torch.tensor([1.]), np.array([2])], 'step': 3}
    exact(first, {'state': [torch.tensor([1.]), np.array([2])], 'step': 3})
    with pytest.raises(AssertionError):
        exact(first, {'state': [torch.tensor([1.1]), np.array([2])], 'step': 3})


def test_exact_rejects_dtype_or_keys():
    with pytest.raises(AssertionError):
        exact(torch.tensor([1.]), torch.tensor([1.], dtype=torch.float64))
    with pytest.raises(AssertionError):
        exact({'optimizer': 1}, {'scheduler': 1})
