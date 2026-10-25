import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
from evaluate_causal_recipe_comparison import conditions
from causal_recipe_control import FastInput,moments,standardize


def test_conditions_same_budgets_and_no_zero_label_training():
    p=dict(target_days=['d'],calibration=dict(budgets=[0,5,10,25,50,100,'full']))
    c=list(conditions(p));assert len(c)==26
    assert [x[1:] for x in c if x[0]=='raw']==[x[1:] for x in c if x[0]=='anscombe']
    assert all(x[3]=='normalization' for x in c if x[2]==0)


def test_prefix_moments_and_count_transform_power_unchanged():
    rows=[dict(id=str(i),raw=np.ones((20,256))*i) for i in (1,2,100)]
    stats=moments(rows[:2],'anscombe');assert stats['ids']==['1','2']
    assert stats['mean'][128:]==[1.5]*128
    z=standardize(rows[0]['raw'],'anscombe',stats)
    np.testing.assert_allclose(z[:,128:],-1,atol=1e-6)


def test_fast_identity_parameter_count_and_gradient():
    m=FastInput();x=torch.randn(2,30,256)
    torch.testing.assert_close(m(x),x,rtol=0,atol=0)
    assert sum(p.numel() for p in m.parameters())==4608
    m(x).square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
