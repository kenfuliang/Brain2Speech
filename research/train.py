"""Native training, adaptation, inference and immutable checkpoint evidence."""
from __future__ import annotations
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from research.audit import digest, save_json, space_guard
from research.data import prepare, CausalFeatures
from research.metrics import edits, greedy, summarize
from research.models import Encoder, objective


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device_for_run():
    if not torch.backends.mps.is_available():
        return torch.device('cpu')
    try:
        Encoder('predictive').to('mps')(torch.zeros(1, 16, 256, device='mps'))
        return torch.device('mps')
    except (RuntimeError, NotImplementedError) as error:
        if 'not implemented' not in str(error).lower() and 'not supported' not in str(error).lower():
            raise
        return torch.device('cpu')


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def batch(trials, device, require_ctc=True):
    lengths = [len(t['x']) for t in trials]
    for trial, length in zip(trials, lengths):
        ids = trial['labels']
        minimum = len(ids) + int(np.sum(ids[1:] == ids[:-1]))
        if require_ctc and length//4 < minimum:
            raise ValueError(f"Impossible CTC path: {trial['id']}: {length//4} frames < {minimum}")
    x = np.zeros((len(trials), max(lengths), 256), np.float32)
    raw = np.zeros_like(x)
    for i, t in enumerate(trials):
        x[i, :lengths[i]] = t['x']
        raw[i, :lengths[i]] = t['raw']
    return (torch.from_numpy(x).to(device), torch.from_numpy(raw).to(device),
            torch.from_numpy(np.concatenate([t['labels'] for t in trials])).to(device),
            torch.tensor(lengths, device=device),
            torch.tensor([len(t['labels']) for t in trials], device=device))


def write_checkpoint(path, model, config, stats, optimizer=None, **extra):
    space_guard()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
               'config': config, 'stats': stats, 'extra': extra,
               'torch_rng': torch.get_rng_state(),
               'mps_rng': torch.mps.get_rng_state().cpu() if torch.backends.mps.is_available() else None,
               'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
               'optimizer': optimizer.state_dict() if optimizer else None}
    temp = path.with_suffix('.partial')
    torch.save(payload, temp)
    temp.replace(path)
    save_json(path.with_suffix('.identity.json'), {'sha256': digest(path), 'config': config,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'total_parameters': sum(p.numel() for p in model.parameters()), **extra})


def load_checkpoint(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = Encoder(payload['config']['kind'], payload['config'].get('hidden', 256)).to(device)
    model.load_state_dict(payload['model'])
    return model, payload


def predict(model, trials, device, output=None, condition='normal', training_texts=()):
    model.eval()
    rows = []
    logits = []
    durations = []
    embeddings = []
    ordered = list(trials)
    for index, trial in enumerate(ordered):
        space_guard()
        x = trial['x'].copy()
        if condition == 'shuffled':
            # Cross-trial replacement, preserve target duration and independent reference.
            donor = ordered[(index+1) % len(ordered)]['x']
            x = donor[np.arange(len(x)) % len(donor)]
        elif condition == 'zero':
            x[:] = 0
        elif condition.startswith('dropout-'):
            count = round(128*float(condition.split('-')[1]))
            lost = np.random.default_rng(101).permutation(128)[:count]
            x[:, lost] = 0
            x[:, lost+128] = 0
        elif condition == 'gain-offset':
            x = 1.5*x + .5
        start = time.perf_counter()
        with torch.no_grad():
            result = model(torch.from_numpy(x[None]).to(device))
        synchronize(device)
        durations.append((time.perf_counter()-start)*1000)
        logit = result['logits'][0].detach().cpu().numpy()
        predicted = greedy(logit)
        rows.append({'id': trial['id'], 'session': trial['session'], 'text': trial['text'],
                     'reference_phones': trial['labels'].tolist(), 'predicted_phones': predicted,
                     'per': edits(trial['labels'].tolist(), predicted),
                     'content_seen_in_training': trial['text'] in training_texts,
                     'neural_duration_seconds': len(x)*.02})
        logits.append(logit)
        embeddings.append(result['factors'][0].detach().cpu().numpy().mean(0))
    summary = summarize(rows, repetitions=200)
    summary.update({'condition': condition, 'backend': str(device),
                    'mean_encoder_time_ms_per_trial': float(np.mean(durations)),
                    'encoder_real_time_factor': sum(durations)/1000/sum(t['neural_duration_seconds'] for t in rows),
                    'acquisition_delay_ms': 4, 'neural_bin_ms': 20, 'output_interval_ms': 80,
                    'downstream_lookahead_ms': 0,
                    'algorithmic_latency_note': '4 ms acquisition + bin completion + 0–60 ms output scheduling; compute reported separately',
                    'process_rss_bytes': psutil.Process().memory_info().rss})
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / 'predictions.json', rows)
        save_json(output / 'metrics.json', summary)
        np.savez_compressed(output / 'logits.npz', **{f'trial_{i:04d}': x for i, x in enumerate(logits)})
        np.save(output / 'embeddings.npy', np.stack(embeddings))
    return summary


def fit(config, train_trials, dev_trials, stats, output, device):
    output = Path(output)
    if (output / 'complete.json').exists():
        prior = json.loads((output / 'complete.json').read_text())
        if prior['config'] != config:
            raise ValueError('Resume configuration mismatch')
        return output / 'best.pt', prior
    seed_all(config['seed'])
    model = Encoder(config['kind'], config.get('hidden', 256)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    epoch_start, step, best, stale = 0, 0, float('inf'), 0
    latest = output / 'resume.pt'
    if latest.exists():
        model, payload = load_checkpoint(latest, device)
        if payload['config'] != config:
            raise ValueError('Resume configuration mismatch')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
        optimizer.load_state_dict(payload['optimizer'])
        epoch_start = payload['extra']['epoch']+1
        step, best, stale = [payload['extra'][k] for k in ('step', 'best', 'stale')]
        torch.set_rng_state(payload['torch_rng'])
        if payload['mps_rng'] is not None and device.type == 'mps':
            torch.mps.set_rng_state(payload['mps_rng'])
        np.random.set_state(payload['numpy_rng'])
        random.setstate(payload['python_rng'])
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    history = json.loads((output/'history.json').read_text()) if (output/'history.json').exists() else []
    for epoch in range(epoch_start, config['epochs']):
        model.train()
        permutation = np.random.permutation(len(train_trials))
        losses = []
        for start in range(0, len(permutation), 16):
            space_guard()
            group = [train_trials[i] for i in permutation[start:start+16]]
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostic = objective(model, batch(group, device), config['auxiliary'], step)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}, step {step}: {diagnostic}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            losses.append(float(loss.detach()))
            if step % 20 == 0:
                save_json(output/'progress.json', {'epoch': epoch, 'step': step, 'batch_loss': losses[-1],
                          'elapsed_seconds_this_process': time.perf_counter()-began,
                          'process_rss_bytes': psutil.Process().memory_info().rss})
                print(json.dumps({'run': output.name, 'epoch': epoch, 'step': step, **diagnostic}), flush=True)
        dev = predict(model, dev_trials, device)
        score = dev['per']['macro_day']
        improved = score < best
        if improved:
            best, stale = score, 0
            write_checkpoint(output/'best.pt', model, config, stats, epoch=epoch, step=step, dev_per=score)
        else:
            stale += 1
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'development_per': score})
        save_json(output/'history.json', history)
        write_checkpoint(latest, model, config, stats, optimizer, epoch=epoch, step=step, best=best, stale=stale)
        if stale >= 5:
            break
    complete = {'config': config, 'best_development_per': best, 'epochs_completed': len(history),
                'checkpoint_sha256': digest(output/'best.pt'), 'backend': str(device),
                'wall_seconds_this_process': time.perf_counter()-began}
    save_json(output/'complete.json', complete)
    return output/'best.pt', complete


def adapt(parent, calibration, config, stats, output, device, mode='adapter', replay=()):
    output = Path(output)
    if (output/'post.pt').exists():
        identity = json.loads((output/'post.identity.json').read_text())
        if (identity['parent_sha256'] != digest(parent) or identity['config'] != config
                or identity['mode'] != mode
                or identity['calibration_ids'] != [t['id'] for t in calibration]
                or identity['replay_ids'] != [t['id'] for t in replay]):
            raise ValueError('Adaptation ancestry/config mismatch')
        return output/'post.pt'
    seed_all(config['seed'])
    model, _ = load_checkpoint(parent, device)
    for name, p in model.named_parameters():
        p.requires_grad_(mode == 'full' or name.startswith('adapter.'))
    if not calibration:
        raise ValueError('Empty calibration set')
    selected = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(selected, lr=config['lr'], weight_decay=.01)
    write_checkpoint(output/'pre.pt', model, config, stats, parent_sha256=digest(parent))
    began = time.perf_counter()
    model.train()
    for step in range(config['adaptation_steps']):
        space_guard()
        n = 12 if replay else 16
        group = [calibration[i] for i in np.random.randint(len(calibration), size=n)]
        if replay:
            group += [replay[i] for i in np.random.randint(len(replay), size=4)]
        optimizer.zero_grad(set_to_none=True)
        tensors = batch(group, device, require_ctc=mode != 'unlabeled')
        if mode == 'unlabeled':
            out = model(tensors[0], sample=True)
            lengths = tensors[3]
            mask = torch.arange(tensors[0].shape[1], device=device)[None] < lengths[:, None]
            count = torch.nn.functional.poisson_nll_loss(out['count_lograte'], tensors[1][:, :, :128], reduction='none').mean(-1)
            power = (out['power_mean']-torch.log1p(tensors[1][:, :, 128:])).square().mean(-1)
            loss = (count+power+.01*out['kl'])[mask].mean()
        else:
            loss, _ = objective(model, tensors, config['auxiliary'], step+1000)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite adaptation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selected, 5., error_if_nonfinite=True)
        optimizer.step()
    write_checkpoint(output/'post.pt', model, config, stats, optimizer,
                     parent_sha256=digest(parent), mode=mode,
                     calibration_ids=[t['id'] for t in calibration],
                     replay_ids=[t['id'] for t in replay],
                     adaptation_seconds=time.perf_counter()-began)
    return output/'post.pt'
