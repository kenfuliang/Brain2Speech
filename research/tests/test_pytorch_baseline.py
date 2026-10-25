import importlib.util
from pathlib import Path
import numpy as np
import torch
import json

ROOT=Path(__file__).absolute().parents[2]

def test_original_adam_update():
    spec=importlib.util.spec_from_file_location('port',ROOT/'reproduction/pytorch_baseline.py')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p=torch.nn.Parameter(torch.tensor([1.,-2.]))
    p.grad=torch.tensor([.3,-.7])
    mod.KerasAdam([p],lr=.02,eps=.1).step()
    with np.load(ROOT/'reproduction/results/retraining/pytorch-fixture-20261007-01.npz') as ref:
        np.testing.assert_allclose(p.detach().numpy(),ref['adam_expected'],rtol=1e-6,atol=1e-7)

def test_adam_matches_original_changing_day_updates():
    spec=importlib.util.spec_from_file_location('port',ROOT/'reproduction/pytorch_baseline.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    parameters=[torch.nn.Parameter(torch.tensor([1.,-2.])),torch.nn.Parameter(torch.tensor([3.,-4.]))]
    opt=mod.KerasAdam(parameters)
    ref=json.loads((ROOT/'reproduction/results/retraining/recipe-adam-20261007-01.json').read_text())
    for gradients,expected in zip(ref['gradients'],ref['weights']):
        for p,g in zip(parameters,gradients):
            # Autograd omits unused day adapters; the original switch-case yields zero gradients.
            p.grad=torch.tensor(g) if any(g) else None
        opt.step()
        np.testing.assert_allclose([p.detach().numpy() for p in parameters],expected,rtol=1e-6,atol=1e-7)
