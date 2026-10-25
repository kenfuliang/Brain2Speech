"""Paired agreement needs exact cohorts and shared resampling."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
from paired_reproduction_metrics import paired_wer


def rows():
    return [{'trial_id':f't{i}','session':f'd{i//2}','reference':'one two',
             'transcript':'one two' if i%2 else 'one'} for i in range(4)]


def test_identical_artifacts_have_exact_zero_paired_gap():
    result=paired_wer(rows(),list(reversed(rows())))
    assert result['candidate_wer']==result['baseline_wer']==.25
    assert result['paired_gap_ci95']==[0.,0.]
    assert result['artifact_relative_equivalence']
    assert not result['inside_exploratory_point_band']  # Published target is a separate claim.


@pytest.mark.parametrize('change',['id','duplicate','reference','session'])
def test_rejects_unmatched_identity_or_reference(change):
    a=rows();b=copy.deepcopy(a)
    if change=='id':b[0]['trial_id']='other'
    elif change=='duplicate':b[0]['trial_id']=b[1]['trial_id']
    elif change=='reference':b[0]['reference']='ONE two'
    else:b[0]['session']='other'
    with pytest.raises(ValueError):paired_wer(a,b)


def test_gap_sign_components_and_reproducibility():
    a=rows();b=rows()
    for row in a:row['transcript']='one two'
    result=paired_wer(a,b)
    assert result['paired_gap_candidate_minus_baseline']==-.25
    assert result['error_components']['candidate']['errors']==0
    assert result['error_components']['baseline']['deletions']==2
    assert result==paired_wer(a,b)
    assert not result['artifact_relative_equivalence']


def test_empty_hypothesis_preserves_legacy_component_difference():
    a=rows();a[0]['transcript']=''
    standard=paired_wer(a,a)
    legacy=paired_wer(a,a,tokenization='legacy-literal-space')
    assert standard['candidate_wer']==legacy['candidate_wer']
    assert standard['error_components']['candidate']['deletions']==3
    assert legacy['error_components']['candidate']['substitutions']==1


def test_in_band_point_does_not_override_wide_paired_uncertainty():
    a=[{'trial_id':f't{i}','session':f'd{i//10}','reference':'one',
        'transcript':'two' if i<19 else 'one'} for i in range(100)]
    b=[dict(row,transcript='two' if i>=81 else 'one') for i,row in enumerate(a)]
    result=paired_wer(a,b)
    assert result['candidate_wer']==result['baseline_wer']==.19
    assert result['inside_exploratory_point_band']
    assert not result['artifact_relative_equivalence']
