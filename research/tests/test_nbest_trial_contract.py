import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[2]/'reproduction/run_nbest_diagnostic.py'
spec=importlib.util.spec_from_file_location('nbest_driver',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def value():
    c=dict(text='one two',ac_score=-1.,lm_score=-2.)
    return dict(index=0,outputs={'1':{'candidates':[c.copy()]},'50':{'candidates':[c.copy()]}})


def test_top1_exact_and_no_missing_candidates():
    assert len(module.validate_trial(value(),0,'one two'))==1
    for expected,index in [('other text',0),('one two',1)]:
        with pytest.raises(ValueError):module.validate_trial(value(),index,expected)
    v=value();v['outputs']['50']['candidates']=[]
    with pytest.raises(ValueError):module.validate_trial(v,0,'one two')


def test_scores_and_nbest_first_candidate_fail_closed():
    v=value();v['outputs']['50']['candidates'][0]['text']='other'
    with pytest.raises(ValueError):module.validate_trial(v,0,'one two')
    v=value();v['outputs']['50']['candidates'][0]['lm_score']=float('inf')
    with pytest.raises(ValueError):module.validate_trial(v,0,'one two')
