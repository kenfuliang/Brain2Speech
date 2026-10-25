import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
import numpy as np
import pytest
import torch
from poisson_recipe import CountHead,count_targets,reconstruction


def test_head_initialization_preserves_rng_and_has_positive_rates():
    torch.manual_seed(17);state=torch.get_rng_state().clone();h=CountHead(np.ones(128)*.6)
    assert torch.equal(state,torch.get_rng_state())
    assert sum(p.numel() for p in h.parameters())==524800
    rates=h(torch.zeros(2,5,1024));assert rates.shape==(2,5,4,128)
    torch.testing.assert_close(rates,torch.full_like(rates,.600001))


def test_targets_cover_nonoverlapping_last_four_bins_and_mask_padding():
    rows=[dict(raw=np.broadcast_to(np.arange(40)[:,None],(40,256))),dict(raw=np.ones((32,256)))]
    t,v=count_targets(rows,torch.tensor([3,1]),'cpu')
    assert t[0,:,0,0].tolist()==[28,32,36]
    assert t[0,:,-1,0].tolist()==[31,35,39]
    r=torch.ones_like(t);a=reconstruction(r,t,v)[0];t[1,1:]=10000
    assert reconstruction(r,t,v)[0]==a


def test_exact_poisson_nll_and_fractional_count_rejection():
    t=torch.zeros(1,1,4,128);r=torch.ones_like(t)*2;v=torch.ones(1,1,dtype=torch.bool)
    assert reconstruction(r,t,v)[0]==2
    with pytest.raises(ValueError):count_targets([dict(raw=np.ones((32,256))*.5)],torch.tensor([1]),'cpu')
