import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reproduction'))
import verify_bit_resume_probe as verifier
from research.audit import digest


def fixture(tmp_path, changed=False):
    folders = [tmp_path/'reference', tmp_path/'restarted']
    for index, folder in enumerate(folders):
        folder.mkdir()
        (folder/'status.json').write_text(json.dumps({
            'status': 'probe-complete-comparison-required', 'production_unchanged': True}))
        path = folder/'final.pt'
        batches = [list(range(i, min(i+32, 8800))) for i in range(0, 8800, 32)]
        journal = [dict(batch=i, ids=[str(n) for n in batches[i]], lengths=[1]*32,
                        loss=1., lr=.0001, logp_sha256='a'*64) for i in range(2)]
        torch.save({'steps': 219727, 'cursor': 2, 'journal': journal, 'batches': batches,
            'model': {'weight': torch.tensor([1. if not changed or index == 0 else 2.])},
            'optimizer': {'state': {0: {'exp_avg': torch.tensor([1.])}}, 'param_groups': [{}]},
            'scheduler': {}, 'sampler_rng': {}, 'torch_rng': torch.tensor([1]),
            'mps_rng': torch.tensor([2])}, path)
        path.with_suffix('.json').write_text(json.dumps({'sha256': digest(path),
            'parent_sha256': 'sealed-parent',
            'probe_sha256': digest(Path(verifier.__file__).parent/'bit_resume_probe.py')}))
    return folders


def invoke(monkeypatch, folders, output):
    monkeypatch.setattr(sys, 'argv', ['verify', '--reference', str(folders[0]),
        '--restarted', str(folders[1]), '--output', str(output)])
    verifier.main()


def test_complete_comparison(tmp_path, monkeypatch):
    folders = fixture(tmp_path)
    output = tmp_path/'verification.json'
    invoke(monkeypatch, folders, output)
    assert json.loads(output.read_text())['passed']


def test_different_states_rejected(tmp_path, monkeypatch):
    folders = fixture(tmp_path, changed=True)
    output = tmp_path/'verification.json'
    with pytest.raises(AssertionError):
        invoke(monkeypatch, folders, output)
    assert not output.exists()


def test_changed_archive_rejected(tmp_path, monkeypatch):
    folders = fixture(tmp_path)
    torch.save({'altered': True}, folders[1]/'final.pt')
    with pytest.raises(AssertionError):
        invoke(monkeypatch, folders, tmp_path/'verification.json')
