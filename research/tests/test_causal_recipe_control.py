import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
from causal_recipe_control import transform,moments,standardize,smooth,smooth_tensor,DayStream,FastInput


def test_power_unchanged_and_count_transform_before_normalization():
    raw=np.tile(np.arange(256),(32,1)).astype(float)
    for kind in ('raw','sqrt','anscombe'):
        assert np.array_equal(transform(raw,kind)[:,128:],raw[:,128:])
    assert np.allclose(transform(raw,'anscombe')[:,:128],2*np.sqrt(raw[:,:128]+.375))


def test_source_bin_weighted_standardization():
    a=dict(id='a',raw=np.zeros((2,256)));b=dict(id='b',raw=np.ones((6,256))*2)
    stats=moments([a,b],'raw');assert np.allclose(stats['mean'],1.5)
    assert np.allclose(stats['std'],np.sqrt(.75))
    assert np.allclose(standardize(b['raw'],'raw',stats),(2-1.5)/(np.sqrt(.75)+1e-8))


def test_noise_smoothing_is_causal_and_matches_scipy():
    x=np.random.default_rng(17).normal(size=(2,70,256)).astype(np.float32)
    out=smooth_tensor(torch.from_numpy(x)).numpy()
    assert np.allclose(out,np.stack([smooth(t)for t in x]),atol=1e-6)
    y=x.copy();y[:,35:]+=100
    assert np.array_equal(out[:,:35],smooth_tensor(torch.from_numpy(y)).numpy()[:,:35])


def test_shuffle_repeat_before_batch_and_exact_resume():
    groups=[list(range(7))];a=DayStream(groups);day,batch=a.batch(9)
    assert set(batch[:7])==set(range(7)) and len(batch)==9
    state=a.state();b=DayStream(groups);b.restore(state)
    assert a.batch(64)==b.batch(64)


def test_small_adapter_identity_and_gradients():
    adapter=FastInput();x=torch.randn(1,50,256)
    assert sum(p.numel()for p in adapter.parameters())==4608
    assert torch.equal(adapter(x),x)
    adapter(x).square().mean().backward();assert adapter.v.grad.abs().sum()>0
