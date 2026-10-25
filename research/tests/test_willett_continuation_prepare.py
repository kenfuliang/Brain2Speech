"""Preparation preserves parent evidence and cannot confer launch approval."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
import prepare_willett_continuation as module


@pytest.fixture
def inputs(tmp_path):
    root=tmp_path/'root';(root/'reproduction').mkdir(parents=True)
    (root/'source').write_text('frozen');(root/'data').write_text('frozen-data')
    (root/'reproduction/willett_fail_stop.py').write_text('frozen-wrapper')
    parent=tmp_path/'parent';parent.mkdir()
    identity={'source_hashes':{'source':module.sha(root/'source')},
              'tfrecord_hashes':{'data':module.sha(root/'data')}}
    module.save(parent/'run-identity.json',identity)
    module.save(parent/'selected.json',{'attempt':2950,'per':.24,'checkpoint':'best-02950.pt'})
    state={'seed':17,'next_attempt':2957,'history':[{}]*2957,'best_per':.24,
        'model':{'weight':torch.ones(1)},'optimizer':{'param_groups':[{'iterations':2957}],
        'state':{0:{'step':2957,'m':torch.zeros(1),'v':torch.zeros(1)}}}}
    torch.save(state,parent/'latest.pt')
    torch.save(dict(state,next_attempt=2951),parent/'best-02950.pt')
    replay=tmp_path/'replay.json'
    module.save(replay,{'status':'passed-full-cpu-stream-replay','records':2957,
        'checkpoint_sha256':module.sha(parent/'latest.pt'),
        'parent_unchanged':True,'optimizer_and_rng_payload_verified':True})
    return parent,tmp_path/'child',replay,root


def test_prepare_keeps_inherited_best_and_disallows_launch(inputs):
    parent,child,replay,root=inputs
    before=module.sha(parent/'latest.pt')
    contract=module.prepare(parent,child,replay,root)
    assert contract['status']=='prepared-not-approved'
    assert not contract['serial_launch_gate_passed']
    assert module.sha(parent/'latest.pt')==before==module.sha(child/'latest.pt')
    assert module.sha(parent/'best-02950.pt')==module.sha(child/'best-02950.pt')
    verification=json.loads((child/'preparation-verification.json').read_text())
    assert verification['copies_not_hardlinks'] and verification['parent_unchanged']
    assert json.loads((child/'run-identity.json').read_text())==json.loads((parent/'run-identity.json').read_text())
    with pytest.raises(FileExistsError):module.prepare(parent,child,replay,root)


def test_changed_source_stops_before_child_creation(inputs):
    parent,child,replay,root=inputs
    (root/'source').write_text('changed')
    with pytest.raises(RuntimeError,match='Frozen input changed'):
        module.prepare(parent,child,replay,root)
    assert not child.exists()


def test_missing_inherited_best_is_not_silently_lost(inputs):
    parent,child,replay,root=inputs
    (parent/'best-02950.pt').rename(parent/'unavailable.pt')
    with pytest.raises(FileNotFoundError):module.prepare(parent,child,replay,root)
    assert not child.exists()


def test_replay_checkpoint_mismatch_stops_before_copy(inputs):
    parent,child,replay,root=inputs
    evidence=json.loads(replay.read_text());evidence['checkpoint_sha256']='wrong'
    module.save(replay,evidence)
    with pytest.raises(AssertionError):module.prepare(parent,child,replay,root)
    assert not child.exists()
