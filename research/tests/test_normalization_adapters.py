"""Matched-condition and uncertainty controls for E019; no training."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
import analyze_normalization_adapters as study


def test_four_conditions_change_only_declared_factors():
    p={'day_to_adapter':{'day':7}}
    n={'global':{'mean':[1,2],'variance':[3,4]},'day':{'day':{'mean':[9,8],'variance':[7,6]}}}
    for key,expected in [('G0',([1,2],0)),('Gd',([1,2],7)),('D0',([9,8],0)),('Dd',([9,8],7))]:
        mean,var,adapter=study.condition_spec(key,'day',p,n)
        assert mean.tolist()==expected[0] and adapter==expected[1]
    with pytest.raises(KeyError):study.condition_spec('Dd','untrained',p,n)


def test_native_reference_same_adapter_on_first_day():
    p={'day_to_adapter':{'first':0}}
    n={'global':{'mean':[0],'variance':[1]},'day':{'first':{'mean':[2],'variance':[1]}}}
    assert study.condition_spec('G0','first',p,n)[2]==study.condition_spec('Gd','first',p,n)[2]==0


def test_paired_bootstrap_identical_predictions_zero_effect():
    rows=[{'session':d,'block':b} for d in ['a','b'] for b in [1,1,2,2]]
    errors={'G0':np.arange(8),'Dd':np.arange(8)};units=np.ones(8)*10
    sample=study.paired_bootstrap(rows,['a','b'],errors,units,100)
    assert np.array_equal(sample['G0'],sample['Dd'])


def test_paired_bootstrap_improvement_and_phone_weighting():
    rows=[{'session':d,'block':b} for d in ['a','b'] for b in [1,1,2,2]]
    units=np.array([5,50,5,50]*2)
    errors={'G0':units.copy(),'Dd':np.zeros(8)}
    sample=study.paired_bootstrap(rows,['a','b'],errors,units,100)
    assert np.all(sample['G0']==1) and np.all(sample['Dd']==0)


def test_factorial_interaction_zero_for_additive_effect():
    rates={'G0':.8,'Gd':.7,'D0':.5,'Dd':.4}
    assert abs(sum(v*rates[k] for k,v in study.CONTRASTS['interaction'].items()))<1e-12


def test_fixed_inference_uses_requested_adapter_and_resets_state():
    torch.set_num_threads(2);torch.manual_seed(101)
    model=study.prior.PublishedGRU().eval();seen=[]
    model.adapters[7].register_forward_hook(lambda *args:seen.append(7))
    x=np.ones((36,256),np.float32)
    y,h=study.infer(model,x,7,torch.device('cpu'))
    z,j=study.infer(model,x,7,torch.device('cpu'))
    assert seen==[7,7] and np.array_equal(y,z) and np.array_equal(h,j)
    assert h.shape==(2,1024)
