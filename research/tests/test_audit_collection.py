import pytest
import json
from research.reporting.audit_collection import expected_ids, process_live
from types import SimpleNamespace
from research.reporting import audit_collection


def test_independent_manifest_scope_for_every_evaluation_stage():
    manifest = {'folds':{'initial_train':['a'],'development':['b'],'final':['c','d']},
                'sessions':{day:{'test':{'trials':[{'id':day+'/test/0'}]}} for day in ['a','b','c','d']}}
    assert expected_ids('development/model',manifest)==['b/test/0']
    assert expected_ids('development-adaptation/gru/full/100/b/evaluation',manifest)==['b/test/0']
    assert expected_ids('probes/gru/17',manifest)==['a/test/0']
    assert expected_ids('final/gru/17/c/adapter/5/same-day-control/evaluation',manifest)==['c/test/0']
    assert expected_ids('sequential/gru/17/full/5/0.25/d/fixed-probes',manifest)==['a/test/0']
    assert expected_ids('sequential/gru/17/full/5/0.25/d/transfer',manifest)==['c/test/0','d/test/0']
    with pytest.raises(ValueError,match='Unexpected'):
        expected_ids('sequential/gru/17/full/5/0.25/b/transfer',manifest)


def test_observation_failure_is_not_a_terminal_process_verdict(monkeypatch):
    monkeypatch.setattr(audit_collection.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=2,stdout=''))
    with pytest.raises(RuntimeError,match='observation failed'):
        process_live(1)


def test_scan_preserves_audits_and_skips_training_completions(tmp_path, monkeypatch):
    root, output = tmp_path/'collection', tmp_path/'audit'
    evaluation = root/'development'/'candidate'
    training = root/'confirmation'/'candidate'
    evaluation.mkdir(parents=True)
    training.mkdir(parents=True)
    (evaluation/'complete.json').write_text(json.dumps({'identity':{'ids':['b/test/0']}}))
    (training/'complete.json').write_text(json.dumps({'identity':{'configuration':'training'}}))
    manifest = {'folds':{'development':['b']},
                'sessions':{'b':{'test':{'trials':[{'id':'b/test/0'}]}}}}
    calls = []
    def verify(folder, ids):
        calls.append((folder, ids))
        return {'component_corrections':[{'trial_id':'b/test/0'}]}
    monkeypatch.setattr(audit_collection.verification, 'verify', verify)
    seen = {}
    audit_collection.scan(root, output, manifest, seen)
    records = list((output/'evaluations').glob('*.json'))
    assert len(records)==1
    original = records[0].read_bytes()
    audit_collection.scan(root, output, manifest, seen)
    assert calls==[(evaluation, ['b/test/0'])]
    assert records[0].read_bytes()==original
    status = json.loads((output/'status.json').read_text())
    assert status['completed_evaluations_verified']==1
    assert status['component_corrected_trials']==1
    assert status['matrix_complete'] is False


def test_scan_rejects_failed_verification_before_acceptance(tmp_path, monkeypatch):
    root, output = tmp_path/'collection', tmp_path/'audit'
    evaluation = root/'development'/'candidate'
    evaluation.mkdir(parents=True)
    (evaluation/'complete.json').write_text(json.dumps({'identity':{'ids':['b/test/0']}}))
    manifest = {'folds':{'development':['b']},
                'sessions':{'b':{'test':{'trials':[{'id':'b/test/0'}]}}}}
    def reject(*args):
        raise ValueError('Artifact hash mismatch')
    monkeypatch.setattr(audit_collection.verification, 'verify', reject)
    seen = {}
    with pytest.raises(ValueError, match='hash mismatch'):
        audit_collection.scan(root, output, manifest, seen)
    assert seen=={}
    assert not (output/'status.json').exists()
    assert not list(output.glob('evaluations/*.json'))
