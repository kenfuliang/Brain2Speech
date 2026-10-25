import importlib.util
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
from analyze_checkpoint_complementarity import compare


def test_complementary_errors_and_ties_are_diagnostic():
    released=dict(true_transcripts=['a b','c d'],decoded_transcripts=['a b','c x'])
    stopped=[dict(session='day',trial_index=0,text='a b',decoded_text='a x'),
             dict(session='day',trial_index=1,text='c d',decoded_text='c d')]
    result=compare(released,stopped)
    assert result['wer']['released']==result['wer']['stopped']==.25
    assert result['wer']['oracle']==0
    assert result['trial_complementarity']['released_better']==1
    assert result['trial_complementarity']['stopped_better']==1
    assert result['paired_oracle_uncertainty']['ci95'] is None


def test_reference_order_mismatch_fails_closed():
    with pytest.raises(ValueError):
        compare(dict(true_transcripts=['b'],decoded_transcripts=['b']),
                [dict(text='a',decoded_text='a',session='day',trial_index=0)])
