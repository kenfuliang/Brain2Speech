import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reproduction'))
spec = importlib.util.spec_from_file_location('current_resume', ROOT / 'reproduction/verify_willett_current_resume.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_nested_checkpoint_exactness():
    state = {'model': torch.tensor([1., 2.]), 'rng': (np.array([3, 4]), 2), 'history': [{'day': 1}]}
    module.exact(state, {'model': torch.tensor([1., 2.]), 'rng': (np.array([3, 4]), 2), 'history': [{'day': 1}]})


@pytest.mark.parametrize('other', [torch.tensor([1., 3.]), torch.tensor([1., float('nan')])])
def test_weight_changes_rejected(other):
    with pytest.raises(AssertionError):
        module.exact(torch.tensor([1., 2.]), other)


def test_history_changes_rejected():
    with pytest.raises(AssertionError):
        module.exact([{'day': 1}], [{'day': 2}])
