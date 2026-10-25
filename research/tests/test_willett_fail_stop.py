"""Safety wrapper must preserve finite KerasAdam arithmetic and reject NaNs."""
import sys
import json
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
import willett_fail_stop as guard


@pytest.mark.parametrize('command',[
    'python train_bit_full.py --resume',
    'python willett_fail_stop.py --output other',
    'python -m research.experiment.pipeline --stage resume',
])
def test_serial_guard_rejects_other_training(command,monkeypatch):
    monkeypatch.setattr(guard.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(guard.subprocess,'check_output',lambda *a,**k:'999999 '+command)
    with pytest.raises(RuntimeError,match='competing trainer'):
        guard.require_serial_resources()


def test_serial_guard_storage_decoder_and_empty_inventory(monkeypatch):
    monkeypatch.setattr(guard.shutil,'disk_usage',lambda _:SimpleNamespace(free=29*1024**3))
    with pytest.raises(RuntimeError,match='30 GiB'):
        guard.require_serial_resources()
    monkeypatch.setattr(guard.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(guard.subprocess,'check_output',lambda args,**k:
        '999999 python unrelated.py' if args[0]=='ps' else 'decoder123')
    with pytest.raises(RuntimeError,match='resident fixed decoder'):
        guard.require_serial_resources()
    monkeypatch.setattr(guard.subprocess,'check_output',lambda args,**k:
        f'{guard.os.getpid()} python willett_fail_stop.py --output own' if args[0]=='ps' else '')
    guard.require_serial_resources()


def test_replay_evidence_is_sealed_to_parent(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();(parent/'latest.pt').write_bytes(b'original')
    evidence=tmp_path/'replay.json'
    evidence.write_text(json.dumps({'status':'passed-full-cpu-stream-replay',
        'parent_unchanged':True,'optimizer_and_rng_payload_verified':True,
        'checkpoint_sha256':guard.recipe.sha(parent/'latest.pt'),'records':2957}))
    contract={'parent':str(parent),'parent_checkpoint_sha256':guard.recipe.sha(parent/'latest.pt'),
        'parent_next_attempt':2957,'full_stream_replay_path':str(evidence),
        'full_stream_replay_sha256':guard.recipe.sha(evidence)}
    guard.require_replay_evidence(contract)
    (parent/'latest.pt').write_bytes(b'changed')
    with pytest.raises(AssertionError,match='Parent checkpoint changed'):
        guard.require_replay_evidence(contract)
    (parent/'latest.pt').write_bytes(b'original');evidence.write_text('{}')
    with pytest.raises(AssertionError,match='Replay evidence changed'):
        guard.require_replay_evidence(contract)


def test_finite_and_nonfinite_values():
    assert guard.require_finite(torch.tensor(1.),'test').item()==1
    with pytest.raises(FloatingPointError):
        guard.require_finite(torch.tensor(float('nan')),'test')


def test_guarded_adam_parity_and_nan_rejection(monkeypatch):
    recipe=guard.recipe
    # Register restorations before installing process-local guards.
    monkeypatch.setattr(recipe,'ctc',recipe.ctc)
    monkeypatch.setattr(recipe.KerasAdam,'step',recipe.KerasAdam.step)
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',torch.nn.utils.clip_grad_norm_)
    monkeypatch.setattr(recipe,'_finite_guards_installed',False,raising=False)
    a=torch.nn.Parameter(torch.tensor([1.,2.]));b=torch.nn.Parameter(a.detach().clone())
    baseline=recipe.KerasAdam([a]);wrapped=recipe.KerasAdam([b])
    a.grad=torch.tensor([.1,.2]);b.grad=a.grad.clone()
    baseline.step();guard.install_guards();wrapped.step()
    assert torch.equal(a,b)
    assert torch.equal(baseline.state[a]['m'],wrapped.state[b]['m'])
    assert torch.equal(baseline.state[a]['v'],wrapped.state[b]['v'])
    b.grad=torch.tensor([float('nan'),.1])
    before=b.detach().clone();iterations=wrapped.param_groups[0]['iterations']
    with pytest.raises(FloatingPointError):wrapped.step()
    assert torch.equal(b,before) and wrapped.param_groups[0]['iterations']==iterations
    with pytest.raises(RuntimeError):torch.nn.utils.clip_grad_norm_([b],10)
