import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'reproduction'))
from launch_willett_verified import verify_gate, safety


def fixture(tmp_path):
    artifact = tmp_path / 'checkpoint'
    artifact.write_bytes(b'sealed proof')
    gate = tmp_path / 'gate.json'
    payload = dict(status='passed-recipe-and-fresh-resume', next_attempt=2959,
                   parent_unchanged=True, model_and_optimizer_exact=True,
                   data_stream_and_rng_exact=True, parent_checkpoint_sha256='parent',
                   artifact_sha256={str(artifact): safety.recipe.sha(artifact)}, source_hashes={})
    gate.write_text(json.dumps(payload))
    contract = dict(guarded_gpu_resume_gate_sha256=safety.recipe.sha(gate), parent_checkpoint_sha256='parent')
    return contract, gate, artifact, payload


def test_sealed_gate_accepts(tmp_path):
    contract, gate, _, _ = fixture(tmp_path)
    verify_gate(contract, gate)


def test_artifact_mutation_rejected(tmp_path):
    contract, gate, artifact, _ = fixture(tmp_path)
    artifact.write_bytes(b'changed')
    with pytest.raises(AssertionError):
        verify_gate(contract, gate)


def test_gate_mutation_rejected(tmp_path):
    contract, gate, _, _ = fixture(tmp_path)
    gate.write_text('{}')
    with pytest.raises(AssertionError):
        verify_gate(contract, gate)


def test_different_parent_rejected(tmp_path):
    contract, gate, _, _ = fixture(tmp_path)
    contract['parent_checkpoint_sha256'] = 'different'
    with pytest.raises(AssertionError):
        verify_gate(contract, gate)
