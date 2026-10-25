import pytest
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from research.reporting.nbest_metrics import summarize_candidates


def candidate(text):
    return dict(text=text, ac_score=-3., lm_score=-2.)


def row(identity='a', text='one two', candidates=None, session='day1'):
    return dict(id=identity, session=session, text=text,
                candidates=candidates if candidates is not None else [candidate('one three'), candidate('one two')])


def test_oracle_gap_and_monotonicity():
    one = summarize_candidates([row()], 1)
    two = summarize_candidates([row()], 2)
    assert one['corpus']['top1_wer'] == one['corpus']['oracle_wer'] == .5
    assert two['corpus']['top1_wer'] == .5
    assert two['corpus']['oracle_wer'] == 0
    assert two['corpus']['oracle_gap'] == .5
    assert two['trial_accounting'][0]['oracle_rank'] == 2


def test_whitespace_empty_hypothesis_duplicates_and_tie():
    result = summarize_candidates([row(candidates=[candidate(''), candidate('one  two'), candidate('one two')])], 5)
    trial = result['trial_accounting'][0]
    assert trial['top1']['deletions'] == 2
    assert trial['top1']['insertions'] == 0
    assert trial['oracle_rank'] == 2
    assert trial['unique'] == 2 and trial['returned'] == 3


def test_corpus_is_not_mean_trial_rate():
    result = summarize_candidates([row(text='one', candidates=[candidate('two')]),
                                   row('b', text='one two three', candidates=[candidate('one two three')], session='day2')], 1)
    assert result['corpus']['top1_wer'] == .25
    assert result['by_day']['day1']['top1_wer'] == 1
    assert result['by_day']['day2']['top1_wer'] == 0


@pytest.mark.parametrize('rows,n', [([],1),([row()],0),([row()],True),
    ([row(),row()],2),([row(text=' ')],1),([row(candidates=[])],1),
    ([row(candidates=[dict(text='one',ac_score=float('nan'),lm_score=1.)])],1)])
def test_fail_closed(rows,n):
    with pytest.raises(ValueError):
        summarize_candidates(rows,n)


def test_pinned_author_wrapper_preserves_candidate_scores_and_resets():
    # Execute only the inspected wrapper, avoiding TensorFlow/native imports.
    source = Path(__file__).resolve().parents[2]/'NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='lm_decode')
    events = []
    results = [SimpleNamespace(sentence=' one two',ac_score=-3.,lm_score=-2.),
               SimpleNamespace(sentence=' one three',ac_score=-4.,lm_score=-1.)]
    decoder = SimpleNamespace(FinishDecoding=lambda:events.append('finish'),
                              Reset=lambda:events.append('reset'), result=lambda:results)
    def decode_numpy(decoder,logits,priors,penalty):
        assert logits.shape == (2,41) and priors.shape == (1,41)
        assert penalty == np.log(2)
        events.append('decode')
    scope = {'np':np,'lm_decoder':SimpleNamespace(DecodeNumpy=decode_numpy)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
    actual = scope['lm_decode'](decoder,np.zeros((2,41)),returnNBest=True,blankPenalty=np.log(2))
    assert actual == [(' one two',-3.,-2.),(' one three',-4.,-1.)]
    assert events == ['decode','finish','reset']
