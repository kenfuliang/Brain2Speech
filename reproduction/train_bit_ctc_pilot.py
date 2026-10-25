"""E015: pinned author BIT NDT, bounded from-scratch offline CTC pilot.

No vendor edits. MPS encoder/decoder, differentiable CPU CTC; original BF16
attention retained, float32 weights/buffers. Never changes frozen v1 artifacts.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'research/vendor/bit-f107ae9'
sys.path.insert(0, str(ROOT))
from research.audit import digest, save_json, space_guard
from research.data import PHONES, labels
from research.metrics import edits, greedy
sys.path.insert(0, str(VENDOR))
import models.ndt as upstream

upstream.DEFAULT_CONFIG = str(VENDOR / 'configs/models/encoder/ndt.yaml')
COHORT = ROOT / 'research/runs/e013-pretrained-causal-bridge-s17-20261008-01/trials.json'


def model(device):
    return upstream.NDT({}, method_name='ctc', features='all', vocab_size=41,
                        blank_id=0, zero_infinity=True).float().to(device)


def prepare(output):
    records = json.loads(COHORT.read_text())
    manifest = json.loads((ROOT / 'research/data/manifest.json').read_text())
    vocab = json.loads((VENDOR / 'vocab.json').read_text())
    assert vocab == ['BLANK'] + list(PHONES)
    assert len(records) == 552
    assert len({r['id'] for r in records}) == 552
    assert {r['session'] for r in records} <= set(manifest['folds']['initial_train'])
    arrays, sources, stats = {}, {}, {}
    for day, partition in sorted({(r['session'], r['partition']) for r in records}):
        path = ROOT / manifest['sessions'][day][partition]['path']
        sources[str(path.relative_to(ROOT))] = digest(path)
        mat = loadmat(path, variable_names=['tx1', 'spikePow'])
        rows = [r for r in records if (r['session'], r['partition']) == (day, partition)]
        raw = {r['id']: np.concatenate([mat['tx1'].ravel()[r['index']][:, :128],
                                       mat['spikePow'].ravel()[r['index']][:, :128]], 1).astype(np.float32)
               for r in rows}
        for block in sorted({r['block'] for r in rows}):
            group = [r for r in rows if r['block'] == block]
            joined = np.concatenate([raw[r['id']] for r in group]).astype(np.float64)
            mean, sd = joined.mean(0), joined.std(0)
            assert np.isfinite(joined).all() and (sd > 0).all(), 'Invalid source block variance'
            stats[f'{day}/{partition}/{block}'] = {'mean': mean.tolist(), 'sd': sd.tolist()}
            for r in group:
                arrays[r['id']] = ((raw[r['id']] - mean) / sd).astype(np.float32)
    day_stats = {}
    for day in sorted({r['session'] for r in records}):
        joined = np.concatenate([arrays[r['id']] for r in records
                                 if r['session'] == day and r['split'] == 'train']).astype(np.float64)
        mean, sd = joined.mean(0), joined.std(0)
        assert (sd > 0).all()
        day_stats[day] = {'mean': mean.tolist(), 'sd': sd.tolist()}
        for r in records:
            if r['session'] == day:
                arrays[r['id']] = ((arrays[r['id']] - mean) / sd).astype(np.float32)
    for r in records:
        x = arrays[r['id']]
        y = labels(r['text']) + 1  # Explicit native phones 0..39 -> author phones 1..40.
        assert len(x) < 2500 and np.isfinite(x).all()
        minimum = len(y) + int((y[1:] == y[:-1]).sum())
        assert (len(x) + 4) // 5 >= minimum, f'Infeasible CTC: {r["id"]}'
        r['phonemes'] = y.tolist()
    save_json(output / 'trials.json', records)
    save_json(output / 'normalization.json', {'block': stats, 'training_day': day_stats})
    save_json(output / 'data-identity.json', {'cohort_sha256': digest(COHORT), 'sources': sources})
    return records, arrays


def batch(rows, arrays, device):
    lengths = [len(arrays[r['id']]) for r in rows]
    x = torch.full((len(rows), max(lengths), 256), -100., device=device)
    mask = torch.zeros(x.shape[:2], dtype=torch.long, device=device)
    timestamps = torch.zeros_like(mask)
    for i, r in enumerate(rows):
        n = lengths[i]
        x[i, :n] = torch.from_numpy(arrays[r['id']]).to(device)
        mask[i, :n] = 1
        timestamps[i, :n] = torch.arange(n, device=device)
    return x, mask, timestamps, torch.ones(len(rows), dtype=torch.long, device=device)


def forward(net, rows, arrays, device):
    x, mask, ts, ds = batch(rows, arrays, device)
    hidden, mask, _, _, _ = net.encoder(x, mask, ts, ds, None, net.training)
    logp = net.decoder(hidden, ds)
    lengths = mask.sum(1).cpu()
    target = torch.tensor([p for r in rows for p in r['phonemes']], dtype=torch.long)
    target_lengths = torch.tensor([len(r['phonemes']) for r in rows], dtype=torch.long)
    # CPU operation retains autograd across MPS transfers, avoiding unsupported Metal CTC.
    loss = torch.nn.functional.ctc_loss(logp.transpose(0, 1).cpu(), target, lengths,
                                       target_lengths, blank=0, reduction='sum', zero_infinity=True)
    return loss, logp, lengths


@torch.no_grad()
def evaluate(net, rows, arrays, device, destination=None):
    net.eval()
    loss_sum, phones, errors, details, logits = 0., 0, 0, [], {}
    for start in range(0, len(rows), 4):
        group = rows[start:start+4]
        loss, logp, lengths = forward(net, group, arrays, device)
        loss_sum += loss.item()
        for r, values, length in zip(group, logp.cpu().numpy(), lengths.tolist()):
            # Native evaluator/WFST: phones 0..39, blank40 (author blank0).
            native = values[:length, list(range(1, 41)) + [0]]
            count = edits([p-1 for p in r['phonemes']], greedy(native))
            errors += count['errors']; phones += count['reference_units']
            details.append({'trial_id': r['id'], 'session': r['session'], 'reference': r['text'],
                            'per': count, 'phoneme_prediction': greedy(native)})
            logits[f'trial_{len(details)-1:04d}'] = native
    result = {'ctc_per_reference_phone': loss_sum / phones, 'per': errors / phones,
              'phone_errors': errors, 'reference_phones': phones, 'trials': len(rows)}
    if destination:
        destination.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(destination / 'logits.npz', **logits)
        save_json(destination / 'predictions.json', details)
        save_json(destination / 'metrics.json', result)
    assert np.isfinite(result['ctc_per_reference_phone'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    assert args.steps >= 50 and args.steps % 50 == 0
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(17); np.random.seed(17)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    source_hashes = {str(p.relative_to(VENDOR)): digest(p) for p in VENDOR.rglob('*.py')}
    source_hashes['configs/models/encoder/ndt.yaml'] = digest(VENDOR / 'configs/models/encoder/ndt.yaml')
    save_json(output / 'contract.json', {
        'experiment': 'E015', 'status': 'bounded technical pilot, not published-result reproduction',
        'source_commit': 'f107ae988d782609b78d1d7488490cca9fb7a103', 'source_hashes': source_hashes,
        'runner_sha256': digest(__file__), 'seed': 17, 'steps': args.steps, 'batch_size': 8,
        'architecture': 'Original NDT: 7 layers, width384, 6 heads, patch5, bidirectional, centered smoothing',
        'data': 'Existing historical 512/40 cohort; eight initial-fold days; no final-fold access',
        'normalization': 'Selected-cohort partition/block zscore, then training-only day moments. Validation block observations transductive; no labels in normalization.',
        'optimizer': 'AdamW lr5e-4 wd1e-5 eps1e-8; author OneCycleLR pct_start0 div_factor25 compressed to pilot steps; clip1 added',
        'loss': 'Sum CTC for optimization as source; reporting divided by reference phone count',
        'selection': 'Lowest validation PER every50 updates; historical selected validation, no independent test',
        'stops': 'Hard pilot budget; nonfinite loss/gradient; free storage below30GiB',
        'port': 'FP32 weights/buffers, author BF16 attention, differentiable CPU CTC; native MPS encoder/decoder',
        'wer': 'Fixed existing3gram exploratory pilot pending controlled v1 pause; no5gram/OPT claim',
        'device': device, 'torch': torch.__version__})
    records, arrays = prepare(output)
    train = [r for r in records if r['split'] == 'train']
    val = [r for r in records if r['split'] != 'train']
    assert len(train) == 512 and len(val) == 40
    net = model(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5, eps=1e-8)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.steps, max_lr=5e-4,
                                                   pct_start=0., div_factor=25)
    rng = np.random.default_rng(17)
    initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    history = []; best = float('inf'); started = time.monotonic()

    def record(step):
        nonlocal best
        probe = evaluate(net, train[:16], arrays, device)
        validation = evaluate(net, val, arrays, device, output / f'evaluation-{step:04d}')
        checkpoint = output / f'checkpoint-{step:04d}.pt'
        torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': schedule.state_dict(), 'step': step, 'sampler_rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(),
                    'mps_rng': torch.mps.get_rng_state() if device == 'mps' else None}, checkpoint)
        row = {'step': step, 'train_probe': probe, 'validation': validation,
               'elapsed_seconds': time.monotonic()-started, 'checkpoint_sha256': digest(checkpoint)}
        history.append(row)
        if validation['per'] < best:
            best = validation['per']
            save_json(output / 'selection.json', {'step': step, 'per': best, 'checkpoint': checkpoint.name,
                                                  'checkpoint_sha256': digest(checkpoint)})
        save_json(output / 'history.json', history)
        print(json.dumps(row), flush=True)

    record(0)
    gradient_checks = 0
    for step in range(1, args.steps+1):
        space_guard(); net.train(); optimizer.zero_grad(set_to_none=True)
        group = [train[i] for i in rng.choice(len(train), 8, replace=False)]
        loss, _, _ = forward(net, group, arrays, device)
        if not torch.isfinite(loss).item():
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        assert float(grad_norm) > 0
        gradient_checks += 1
        optimizer.step(); schedule.step()
        if step % 50 == 0:
            record(step)
    selected = json.loads((output / 'selection.json').read_text())
    selected_state = torch.load(output / selected['checkpoint'], map_location='cpu', weights_only=False)
    net.load_state_dict(selected_state['model']); net.eval()
    replay = evaluate(net, val, arrays, device)
    expected = next(r['validation'] for r in history if r['step'] == selected['step'])
    assert replay['phone_errors'] == expected['phone_errors']
    assert abs(replay['ctc_per_reference_phone'] - expected['ctc_per_reference_phone']) < 1e-5
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v.cpu(), initial[k])]
    assert changed
    save_json(output / 'verification.json', {'passed': True, 'finite_nonzero_gradient_updates': gradient_checks,
                                             'reload': replay, 'changed_tensors': changed,
                                             'vendor_hashes_unchanged': all(digest(VENDOR/k) == v for k,v in source_hashes.items())})
    save_json(output / 'status.json', {'status': 'training_complete_wer_pending', 'selected': selected,
                                       'parameters': sum(p.numel() for p in net.parameters()),
                                       'elapsed_seconds': time.monotonic()-started})


if __name__ == '__main__':
    main()
