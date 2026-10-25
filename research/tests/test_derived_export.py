import pytest
import json
from research.audit import digest, save_json
from research.reporting.derived_verification import verify
from research.tests.test_derived_word_metrics import historical_fixture
from research.reporting.derived_export import export_inputs


def test_export_refuses_incomplete_real_matrix_without_creating_view(tmp_path):
    manifest = {'folds':{'initial_train':[],'final':['future']},
        'sessions':{'future':{'test':{'trials':[{'id':'future/test/0'}]}}}}
    freeze = {'selected':{'gru':{}},'seeds':[17,29,43],'budgets':[5]}
    out = tmp_path/'derived'
    with pytest.raises(ValueError,match='Incomplete derived matrix'):
        export_inputs(tmp_path/'raw',tmp_path/'audits',out,manifest,freeze)
    assert not out.exists()


def test_export_refuses_raw_tree_destination(tmp_path):
    with pytest.raises(ValueError,match='outside raw collection'):
        export_inputs(tmp_path/'raw',tmp_path/'audits',tmp_path/'raw'/'derived',{}, {})


def test_export_materializes_corrected_view_and_preserves_raw(tmp_path):
    raw = tmp_path/'raw'
    source = raw/'probes/gru/17'
    historical_fixture(source)
    audits = tmp_path/'audits'
    save_json(audits/'evaluations'/'probe.json',verify(source,repetitions=20))
    manifest = {'folds':{'initial_train':['day'],'final':[]},
        'sessions':{'day':{'test':{'trials':[{'id':'day/test/0'}]}}}}
    freeze = {'selected':{'gru':{}},'seeds':[17],'budgets':[]}
    before = digest(source/'predictions.json')
    out = tmp_path/'derived'
    assert export_inputs(raw,audits,out,manifest,freeze)==1
    destination = out/'probes/gru/17'
    assert json.loads((destination/'metrics.json').read_text())['wer']['corpus']==.75
    assert json.loads((destination/'predictions.json').read_text())[0]['wer']['reference_units']==4
    assert json.loads((destination/'complete.json').read_text())['not_a_raw_evaluator_completion']
    assert digest(source/'predictions.json')==before
    assert not (destination/'logits.npz').exists()
