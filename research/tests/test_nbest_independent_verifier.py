import importlib.util
import json
from pathlib import Path
import pytest
from research.reporting.nbest_metrics import summarize_candidates

path=Path(__file__).resolve().parents[2]/'reproduction/verify_nbest_diagnostic.py'
spec=importlib.util.spec_from_file_location('oracle_verifier',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def save(path,value):path.write_text(json.dumps(value))


def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir();run=tmp_path/'run';run.mkdir()
    original=[dict(session='day1',trial_index=0,text='one two',decoded_text='one three')]
    save(source/'decoded-predictions.json',original)
    candidates=[dict(text='one three',ac_score=-1.,lm_score=-2.),dict(text='one two',ac_score=-2.,lm_score=-3.)]
    record=dict(index=0,outputs={'1':dict(candidates=candidates[:1]),'50':dict(candidates=candidates)})
    save(run/'trial-0000.json',record)
    save(run/'protocol.json',dict(indices=[0],scope='technical-smoke',input_sha256={},
                                 original_decoded_sha256=module.sha(source/'decoded-predictions.json')))
    rows=[dict(id='day1/test/0000',session='day1',text='one two',candidates=candidates)]
    save(run/'metrics.json',dict(curves={str(n):summarize_candidates(rows,n) for n in (1,10,50)}))
    seal(run)
    return run,source


def seal(run):
    save(run/'complete.json',dict(trials=1,sha256={name:module.sha(run/name) for name in ('protocol.json','metrics.json')},
                                 trial_sha256={'0':module.sha(run/'trial-0000.json')}))


def test_independent_edit_distance_and_accounting(tmp_path):
    run,source=fixture(tmp_path)
    result=module.verify(run,source)
    assert result['curves']['1']['oracle_wer']==.5
    assert result['curves']['50']['oracle_wer']==0
    assert module.distance(['a','b'],[])==2
    assert module.distance([],['a'])==1


def test_false_metric_fails_even_with_resealed_hash(tmp_path):
    run,source=fixture(tmp_path)
    metrics=json.loads((run/'metrics.json').read_text())
    metrics['curves']['50']['corpus']['oracle_errors']=200
    save(run/'metrics.json',metrics);seal(run)
    with pytest.raises(AssertionError):module.verify(run,source)


def test_changed_raw_candidate_fails_seal(tmp_path):
    run,source=fixture(tmp_path)
    record=json.loads((run/'trial-0000.json').read_text());record['index']=1
    save(run/'trial-0000.json',record)
    with pytest.raises(AssertionError):module.verify(run,source)


def test_wrong_components_fail_even_when_total_and_seal_match(tmp_path):
    run,source=fixture(tmp_path)
    metrics=json.loads((run/'metrics.json').read_text())
    top=metrics['curves']['1']['trial_accounting'][0]['top1']
    top['substitutions']=0;top['deletions']=1
    save(run/'metrics.json',metrics);seal(run)
    with pytest.raises(AssertionError):module.verify(run,source)


def test_nonfinite_score_fails_even_when_resealed(tmp_path):
    run,source=fixture(tmp_path)
    record=json.loads((run/'trial-0000.json').read_text())
    record['outputs']['50']['candidates'][1]['ac_score']=float('nan')
    save(run/'trial-0000.json',record);seal(run)
    with pytest.raises(AssertionError):module.verify(run,source)


def test_independent_components_match_exhaustive_short_sequences():
    import itertools
    from research.metrics import edits
    sequences=[list(x) for n in range(4) for x in itertools.product(('a','b'),repeat=n)]
    for reference in sequences:
        for hypothesis in sequences:
            assert module.components(reference,hypothesis)==edits(reference,hypothesis)
