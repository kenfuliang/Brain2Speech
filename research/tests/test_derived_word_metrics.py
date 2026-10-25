import json
import pytest
from research.audit import digest, save_json
from research.metrics import edits
from research.tests.test_verified_metrics import fixture
from research.reporting.derived_verification import verify


def historical_fixture(folder):
    fixture(folder, 'and he did not')
    rows = json.loads((folder/'predictions.json').read_text())
    rows[0]['text'] = 'then he  then what'
    rows[0]['wer'] = edits(rows[0]['text'].split(' '), rows[0]['decoded_text'].split(' '))
    rows[0]['cer'] = edits(list(rows[0]['text']), list(rows[0]['decoded_text']))
    save_json(folder/'predictions.json', rows)
    save_json(folder/'metrics.json', {'trials':1,'sessions':1,
        **{f:{'corpus':rows[0][f]['errors']/rows[0][f]['reference_units']} for f in ['per','wer','cer']}})
    record = json.loads((folder/'complete.json').read_text())
    record['artifact_sha256'] = {n:digest(folder/n) for n in record['artifact_sha256']}
    save_json(folder/'complete.json', record)


def test_historical_double_space_has_explicit_rate_correction(tmp_path):
    historical_fixture(tmp_path)
    before = {n:digest(tmp_path/n) for n in ['complete.json','predictions.json','metrics.json']}
    result = verify(tmp_path, ['day/test/0'], repetitions=20)
    assert result['rate_changed_trials']==1
    assert result['corrected_edit_totals']['wer']['errors']==3
    assert result['corrected_edit_totals']['wer']['reference_units']==4
    assert result['derived_metrics']['wer']['corpus']==.75
    assert result['original_metrics']['wer']['corpus']==.8
    assert result['derived_metrics']['by_day']['day']['wer']==.75
    assert all(digest(tmp_path/n)==v for n,v in before.items())


def test_empty_transcript_corrects_components_without_rate_change(tmp_path):
    fixture(tmp_path)
    result = verify(tmp_path, repetitions=20)
    assert result['rate_changed_trials']==0
    assert result['component_changed_trials']==1
    assert result['corrected_edit_totals']['wer']['deletions']==2


def test_derived_audit_rejects_bad_identity_and_tampering(tmp_path):
    historical_fixture(tmp_path)
    with pytest.raises(ValueError, match='identity'):
        verify(tmp_path, ['wrong'], repetitions=20)
    (tmp_path/'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact hash'):
        verify(tmp_path, repetitions=20)


def test_derived_audit_rejects_non_tokenization_edit_error(tmp_path):
    historical_fixture(tmp_path)
    rows = json.loads((tmp_path/'predictions.json').read_text())
    rows[0]['wer']['errors'] = 99
    save_json(tmp_path/'predictions.json', rows)
    record = json.loads((tmp_path/'complete.json').read_text())
    record['artifact_sha256']['predictions.json'] = digest(tmp_path/'predictions.json')
    save_json(tmp_path/'complete.json', record)
    with pytest.raises(ValueError, match='Original word edit'):
        verify(tmp_path, repetitions=20)
