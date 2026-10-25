"""Isolated two-update/fresh-process E016 restart proof; never extend production.

Reference runs the first two batches of epoch800 from epoch799 and saves a
boundary after the first. Restart runs only the second batch from that boundary.
Run these serially, after all other GPU/decoder workloads exit.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
import torch

from export_willett_final import require_idle
from train_bit_full import forward, upstream, VENDOR
from utils.config import DictConfig
from research.audit import digest, save_json, space_guard


def epoch_batches(sampler, count):
    indices = np.arange(count)
    sampler.shuffle(indices)
    batches = [indices[i:i+32].tolist() for i in range(0, count, 32)]
    sampler.shuffle(batches)
    return batches


def require_probe_idle():
    require_idle()
    for line in subprocess.check_output(['ps', '-ax', '-o', 'pid=,command='], text=True).splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        command = shlex.split(fields[1])
        if command and Path(command[0]).name.lower().startswith(('python', 'pypy')):
            if any(name in fields[1] for name in ('bit_resume_probe.py', 'audit_bit_source_metric.py')):
                raise RuntimeError('BIT probe/evaluator is still live')


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for first, second in zip(a, b):
            exact(first, second)
    else:
        assert type(a) is type(b) and a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boundary', type=Path)
    args = parser.parse_args()
    require_probe_idle(); space_guard()
    assert not args.output.exists()
    run = args.run.resolve()
    contract = json.loads((run/'contract.json').read_text())
    assert contract['epochs'] == 800 and contract['seed'] == 42
    for name, sha in contract['source_hashes'].items():
        assert digest(VENDOR/name) == sha
    assert digest(Path(__file__).parent/'train_bit_full.py') == contract['runner_sha256']
    assert digest(Path(__file__).parent/'train_bit_ctc_pilot.py') == contract['helper_sha256']
    assert digest(run/'prepared.pt') == json.loads((run/'cache-identity.json').read_text())['sha256']
    original = run/'epoch-0799.pt'
    original_sha = digest(original)
    data = torch.load(run/'prepared.pt', map_location='cpu', weights_only=False)
    assert len(data['records']['train']) == 8800
    cfg = DictConfig(json.loads((run/'effective-config.json').read_text()))
    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    net = upstream.NDT(cfg.model, method_name='ctc', features='all', vocab_size=41,
                       blank_id=0, zero_infinity=True).float().to('mps')
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5, eps=1e-8)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=220000,
        max_lr=5e-4, pct_start=0., div_factor=25)
    sampler = np.random.default_rng(42)
    if args.boundary:
        seal = json.loads(args.boundary.with_suffix('.json').read_text())
        assert digest(args.boundary) == seal['sha256'] and seal['parent_sha256'] == original_sha
        assert seal['probe_sha256'] == digest(Path(__file__))
        state = torch.load(args.boundary, map_location='cpu', weights_only=False)
        assert state['steps'] == 219726 and state['cursor'] == 1
    else:
        state = torch.load(original, map_location='cpu', weights_only=False)
        assert state['epoch'] == 799 and state['steps'] == 219725
    net.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    schedule.load_state_dict(state['scheduler']); sampler.bit_generator.state = state['sampler_rng']
    torch.set_rng_state(state['torch_rng']); torch.mps.set_rng_state(state['mps_rng'])
    if args.boundary:
        batches = state['batches']; cursor = state['cursor']; journal = state['journal']
    else:
        batches = epoch_batches(sampler, 8800); cursor = 0; journal = []
    assert sorted(i for group in batches for i in group) == list(range(8800))
    args.output.mkdir(parents=True)
    net.train()
    for index in range(cursor, 2):
        space_guard(); optimizer.zero_grad(set_to_none=True)
        rows = [data['records']['train'][i] for i in batches[index]]
        loss, logp, lengths = forward(net, rows, data['arrays'], 'mps')
        assert torch.isfinite(loss) and torch.isfinite(logp).all()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        optimizer.step(); schedule.step()
        assert all(torch.isfinite(p).all() for p in net.parameters())
        journal.append(dict(batch=index, ids=[r['id'] for r in rows], loss=loss.item(),
                            lr=optimizer.param_groups[0]['lr'], lengths=lengths.tolist(),
                            logp_sha256=hashlib.sha256(logp.detach().numpy().tobytes()).hexdigest()))
        saved = dict(model={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            optimizer=copy.deepcopy(optimizer.state_dict()), scheduler=schedule.state_dict(),
            sampler_rng=copy.deepcopy(sampler.bit_generator.state), torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state(), batches=batches, cursor=index+1,
            steps=219725+index+1, journal=copy.deepcopy(journal))
        destination = args.output/('boundary.pt' if index == 0 else 'final.pt')
        torch.save(saved, destination)
        save_json(destination.with_suffix('.json'), dict(sha256=digest(destination),
            parent_sha256=original_sha, probe_sha256=digest(Path(__file__))))
    assert digest(original) == original_sha
    save_json(args.output/'status.json', dict(status='probe-complete-comparison-required',
        production_unchanged=True, isolated_steps=2-cursor, parent_sha256=original_sha,
        scope='Technical epoch800 first-two-batch restart only; no production training extension'))
    print('completed-isolated-BIT-resume-probe')


if __name__ == '__main__':
    main()
