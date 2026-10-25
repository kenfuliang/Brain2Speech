import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'reproduction'))
import run_nbest_fixed_diagnostic as driver


def setup_gate(monkeypatch, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'metrics.json').write_text(json.dumps({'decoder': {'lm_files': {}}}))
    (source / 'identity.json').write_text(json.dumps({
        'artifacts_sha256': {}, 'checkpoint_sha256': 'fixture-checkpoint'}))
    (source / 'decoded-predictions.json').write_text(json.dumps([{}] * 600))
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'SOURCE', source)
    # Fixture identities are deliberately not scientific/runtime evidence.
    monkeypatch.setattr(driver, 'sha', lambda p: 'fixture:' + str(p))
    monkeypatch.setattr(driver.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100 * 1024**3))

    def check(command, **kwargs):
        if 'info' in command:
            return str(80 * 1024**3)
        if 'ps' in command:
            return ''
        if 'image' in command:
            return driver.IMAGE_ID
        raise AssertionError('Unexpected external check')

    monkeypatch.setattr(driver.subprocess, 'check_output', check)
    monkeypatch.setattr(driver.subprocess, 'Popen', lambda *a, **k: pytest.fail('Unauthorized launch'))
    out = tmp_path / 'run'
    argv = ['driver', '--smoke', '--memory-gib', '64', '--output', str(out)]
    return out, argv


def test_prepare_freezes_protocol_without_launching(monkeypatch, tmp_path):
    out, argv = setup_gate(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, 'argv', argv + ['--prepare-only'])
    driver.main()
    protocol = json.loads((out / 'protocol.json').read_text())
    assert protocol['indices'] == [0, 40]
    assert protocol['resource_budget']['container_memory'] == '64g'
    assert protocol['helper_sha256']
    assert 'no SetOpt' in protocol['native_option_lifecycle']
    assert not (out / 'authorization.json').exists()


def test_missing_or_wrong_authorization_cannot_launch(monkeypatch, tmp_path):
    out, argv = setup_gate(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, 'argv', argv + ['--prepare-only'])
    driver.main()
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(RuntimeError, match='Explicit user retry authorization'):
        driver.main()
    (out / 'authorization.json').write_text(json.dumps({
        'approved': True, 'protocol_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='exact protocol hash'):
        driver.main()
