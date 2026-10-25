import importlib.util
import sys
from pathlib import Path
import pytest
path=Path(__file__).resolve().parents[2]/'reproduction/score_bit_saved_examples.py'
sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('bit_examples',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    trials=[dict(key=str(i),split='evaluation',text='a b') for i in range(16)]
    examples=[dict(condition=c,references=['a b']*4,predictions=['a c']*4)
              for c in ('real','zero','shuffled','language-only')]
    return examples,trials


def test_saved_subset_counts_and_components():
    result=module.score(*fixture())
    assert result['real']['rates']['wer']==.5
    assert result['real']['totals']['wer']['substitutions']==4
    assert result['real']['trials']==4


def test_missing_or_misaligned_examples_fail():
    examples,trials=fixture()
    with pytest.raises(ValueError):module.score(examples[:3],trials)
    examples[0]['references'][0]='other'
    with pytest.raises(ValueError):module.score(examples,trials)


def test_empty_hypotheses_are_deletions_and_duplicate_conditions_fail():
    examples,trials=fixture();examples[0]['predictions']=['']*4
    result=module.score(examples,trials)
    assert result['real']['totals']['wer']['deletions']==8
    examples[1]['condition']='real'
    with pytest.raises(ValueError):module.score(examples,trials)
