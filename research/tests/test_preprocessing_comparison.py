import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
from preprocessing_comparison import SmallAdapter,transform,fit_stats,preprocess


def test_count_transforms_and_power_fixed():
    x=np.tile(np.arange(256),(12,1)).astype(float)
    assert np.array_equal(transform(x,'raw')[:,:128],x[:,:128])
    assert np.allclose(transform(x,'sqrt')[:,:128],np.sqrt(x[:,:128]))
    assert np.allclose(transform(x,'anscombe')[:,:128],2*np.sqrt(x[:,:128]+.375))
    assert np.array_equal(transform(x,'sqrt')[:,128:],transform(x,'anscombe')[:,128:])


def test_preprocessing_prefix_and_source_preservation():
    rng=np.random.default_rng(17);x=rng.poisson(1,(60,256)).astype(float);original=x.copy()
    for kind in ('raw','sqrt','anscombe'):
        stats=fit_stats([dict(id='train',raw=x)],kind)
        a=preprocess(x,stats);y=x.copy();y[30:]+=100
        assert np.array_equal(a[:30],preprocess(y,stats)[:30])
        assert np.array_equal(a[:30],preprocess(x[:30],stats))
    assert np.array_equal(x,original)


def test_adapter_identity_parameter_budget_and_nonzero_gradient():
    torch.manual_seed(17);m=SmallAdapter();x=torch.randn(2,40,256)
    assert torch.equal(m(x),x)
    m.shared.requires_grad_(False);m.adaptation(True)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==4608
    m(x).square().mean().backward()
    assert m.v.grad.abs().sum()>0
    assert m.shared.weight.grad is None


def test_adapter_reload_and_no_cross_trial_state():
    torch.manual_seed(17);a=SmallAdapter();b=SmallAdapter();b.load_state_dict(a.state_dict())
    x=torch.randn(1,40,256);assert torch.equal(a(x),b(x))
    assert torch.equal(a(x)[:,:20],a(x[:,:20]))
