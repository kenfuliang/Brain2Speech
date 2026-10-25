"""Do not let the prepared decoder compete with live training."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
import decode_bit_full as module


def test_rejects_live_bit_trainer(tmp_path,monkeypatch):
    (tmp_path/'launch.json').write_text(json.dumps({'pid':123}))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='python train_bit_full.py --resume'))
    with pytest.raises(AssertionError,match='BIT trainer still live'):
        module.require_stopped_training(tmp_path)


def test_rejects_live_willett_trainer(tmp_path,monkeypatch):
    (tmp_path/'launch.json').write_text(json.dumps({'pid':123}))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=''))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:'python willett_fail_stop.py --output child')
    with pytest.raises(AssertionError,match='Competing research workload'):
        module.require_stopped_training(tmp_path)


def test_accepts_no_training_workload(tmp_path,monkeypatch):
    (tmp_path/'launch.json').write_text(json.dumps({'pid':123}))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=''))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:'python unrelated_service.py')
    module.require_stopped_training(tmp_path)
