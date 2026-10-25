import json
import pytest
from research.audit import digest, save_json
from research.reporting.derived_inputs import DerivedEvaluationLoader
from research.reporting.derived_verification import verify
from research.tests.test_derived_word_metrics import historical_fixture


def test_analysis_loader_never_falls_back_to_raw_scoring(tmp_path):
    source = tmp_path/'source'
    historical_fixture(source)
    audits = tmp_path/'audits'
    audit_path = audits/'evaluations'/'historical.json'
    save_json(audit_path,verify(source,repetitions=20))
    loader = DerivedEvaluationLoader(audits)
    rows,metrics,provenance = loader.load(source,['day/test/0'])
    assert rows[0]['wer']['errors']==3 and rows[0]['wer']['reference_units']==4
    assert metrics['wer']['corpus']==.75
    assert provenance['audit_sha256']==digest(audit_path)
    assert json.loads((source/'metrics.json').read_text())['wer']['corpus']==.8
    with pytest.raises(ValueError,match='Missing derived audit'):
        loader.load(tmp_path/'absent')
    with pytest.raises(ValueError,match='trial identity'):
        loader.load(source,['wrong'])
    (source/'metrics.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact hash'):
        loader.load(source)
