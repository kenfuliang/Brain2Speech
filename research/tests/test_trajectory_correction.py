import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
from quantify_trajectory_correction import moments, correction, apply, compare


def test_equal_trial_moments_and_gain_clipping():
    mean,sd=moments(np.array([[0.,1.],[2.,3.]]),np.ones((2,2)))
    np.testing.assert_allclose(mean,[1,2])
    np.testing.assert_allclose(sd,np.sqrt([2,2]))
    fit=correction((np.zeros(3),np.array([1.,4.,.1])),(np.ones(3),np.ones(3)))
    np.testing.assert_array_equal(fit['gain'],[1.,2.,.5])
    assert fit['clipped']==2


def test_independent_affine_correction_recovers_synthetic_sequence():
    rng=np.random.default_rng(101)
    training=rng.normal(size=(100,4))
    target=training*1.5+2
    fit=correction((training.mean(0),training.std(0)),(target.mean(0),target.std(0)))
    evaluation=rng.normal(size=(30,4))
    observed=evaluation*1.5+2
    corrected=apply(observed,fit,'offset_gain')
    np.testing.assert_allclose(corrected,evaluation,atol=1e-12)
    assert compare(evaluation,corrected)['RMS']<1e-12
    assert compare(evaluation,observed)['RMS']>1


def test_offset_preserves_shape_and_velocity_and_coverage():
    rng=np.random.default_rng(102);a=rng.normal(size=(12,5));b=rng.normal(size=(15,5))
    fit=dict(reference_mean=np.ones(5),target_mean=np.ones(5)*3,gain=np.ones(5))
    raw=compare(a,b);shifted=compare(a,apply(b,fit,'offset'))
    assert np.isclose(raw['shape_cosine'],shifted['shape_cosine'],atol=1e-12)
    assert np.isclose(raw['velocity_cosine'],shifted['velocity_cosine'],atol=1e-12)
    assert raw['coverage']==[1.,.8]
    assert compare(a,a)['RMS']==0
