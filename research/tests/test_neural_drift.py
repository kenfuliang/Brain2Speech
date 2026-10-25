"""Synthetic accounting, geometry and fixed-instrument controls for E018."""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'reproduction'))
spec=importlib.util.spec_from_file_location('drift',ROOT/'reproduction/analyze_neural_drift.py')
drift=importlib.util.module_from_spec(spec); spec.loader.exec_module(drift)


def test_fixed_preprocessing_prefix_and_trial_reset():
    raw=np.random.default_rng(5).normal(size=(80,256))
    mean=np.zeros(256); variance=np.ones(256)
    full=drift.preprocess(raw,mean,variance)
    changed=raw.copy(); changed[40:]+=100
    assert np.array_equal(full[:40],drift.preprocess(changed,mean,variance)[:40])
    assert np.array_equal(full[:40],drift.preprocess(raw[:40],mean,variance))
    assert np.array_equal(full,drift.preprocess(raw,mean,variance))
    assert np.array_equal(mean,np.zeros(256)) and np.array_equal(variance,np.ones(256))


def test_equal_trial_moments_not_duration_weighted():
    a=np.ones((5,256)); b=np.ones((50,256))*3
    values=[drift.raw_statistics(x) for x in [a,b]]
    mean=np.mean([v['raw_mean'] for v in values],axis=0)
    variance=np.mean([v['raw_second'] for v in values],axis=0)-mean**2
    assert np.allclose(mean,2) and np.allclose(variance,1)
    assert not np.isclose(np.concatenate([a,b]).mean(),2)


def test_shape_units_and_bin_accounting():
    row=dict(index=0,bins=36)
    mat={k:np.empty((1,1),dtype=object) for k in ['tx1','spikePow']}
    for k in mat: mat[k][0,0]=np.ones((36,256))
    raw=drift.trial_raw(mat,row)
    assert raw.shape==(36,256)
    mat['tx1'][0,0][0,0]=.5
    with pytest.raises(ValueError,match='units'): drift.trial_raw(mat,row)
    mat['tx1'][0,0][0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'): drift.trial_raw(mat,row)


def test_shared_pca_fit_does_not_use_development():
    rng=np.random.default_rng(8); values=rng.normal(size=(100,12))
    fit=np.arange(100)<70
    a,p=drift.fit_pca(values,fit,8)
    changed=values.copy(); changed[70:]+=10000
    b,q=drift.fit_pca(changed,fit,8)
    assert np.array_equal(p['axes'],q['axes'])
    assert np.array_equal(p['mean'],q['mean'])
    assert np.array_equal(a[:70],b[:70])
    assert np.allclose(a,(values-p['mean'])@p['axes'].T)


def test_block_split_keeps_whole_chronological_blocks():
    rows=[dict(block=b) for b,n in [(3,7),(4,2),(7,10),(8,3)] for _ in range(n)]
    left,right=drift.split_blocks(rows)
    assert set(left).isdisjoint(right) and max(left)<min(right)
    assert drift.split_blocks([dict(block=1)]*20) is None


def test_bootstrap_preserves_block_groups():
    values=np.array([[0.],[0.],[10.],[10.]])
    result=drift.boot_centroids(values,[1,1,2,2],np.random.default_rng(1),2000)
    assert set(result.ravel())=={0.,5.,10.}


def test_zero_distance_for_identical_inputs():
    x=np.random.default_rng(1).normal(size=(30,5))
    assert np.array_equal(drift.distance(x,x),np.zeros(30))


def test_injected_shift_exceeds_within_day_variation():
    rng=np.random.default_rng(4)
    days=['a','b','c']; rows=[]; values=[]
    for day,shift in zip(days,[0,4,8]):
        for block in range(4):
            for _ in range(8):
                rows.append(dict(session=day,block=block))
                values.append(rng.normal(shift,.05,5))
    result=drift.day_statistics(np.array(values),rows,days,repetitions=100)
    assert result['conditional_support']
    assert result['excess_ci95'][0]>0
    # A gain shift on the same positive baseline also changes feature summaries.
    assert drift.distance(np.ones(5),np.ones(5)*2)>0


def test_content_and_coordinate_changes_can_mimic_shift():
    # Same conditional code, different proportions of two speech contents.
    content=np.array([[0.,1.],[4.,2.]])
    a=np.repeat(content,[90,10],axis=0)
    b=np.repeat(content,[10,90],axis=0)
    assert drift.distance(a.mean(0),b.mean(0))>0
    rotation=np.array([[0.,-1.],[1.,0.]])
    assert drift.distance(a.mean(0),a.mean(0)@rotation)>0
    assert np.allclose(np.linalg.norm(content[1]-content[0]),
                       np.linalg.norm((content@rotation)[1]-(content@rotation)[0]))


def test_repeated_content_sparse_intervals_suppressed():
    rows=[dict(id=str(i),text='same',session=d,bins=40) for i,d in enumerate(['a','b','a'])]
    result=drift.repeated_content(rows,dict(latent=np.arange(6).reshape(3,2)))
    assert result['cross_day_prompts']==1 and result['within_day_prompts']==1
    assert result['interval'] is None and result['verdict']=='inconclusive'


def test_cohort_forbids_final_and_bad_counts():
    manifest={'folds':dict(initial_train=['a'],development=['b'],final=['c'])}
    with pytest.raises(ValueError,match='19'): drift.cohort(manifest)


def test_fixed_model_adapter_reload_and_frame_accounting():
    torch.set_num_threads(2); torch.manual_seed(101)
    model=drift.PublishedGRU().eval()
    seen=[]
    hook=model.adapters[0].register_forward_hook(lambda *args:seen.append(0))
    for adapter in model.adapters[1:]:
        adapter.register_forward_hook(lambda *args:seen.append(1))
    x=np.random.default_rng(9).normal(size=(44,256)).astype(np.float32)
    a,h=drift.fixed_inference(model,x,torch.device('cpu'))
    drift.fixed_inference(model,np.zeros_like(x),torch.device('cpu'))
    b,j=drift.fixed_inference(model,x,torch.device('cpu'))
    assert np.array_equal(a,b) and np.array_equal(h,j)
    assert h.shape==((44-32)//4+1,1024) and seen==[0,0,0]
    other=drift.PublishedGRU().eval(); other.load_state_dict(model.state_dict())
    c,k=drift.fixed_inference(other,x,torch.device('cpu'))
    assert np.array_equal(a,c) and np.array_equal(h,k)
    hook.remove()
