"""E013: usability gate for frozen E008 GRU under downstream-causal features.

Source raw feature units, training-only moments, trailing smoothing, right-edge
timestamps, fixed-stride trailing latent pooling. Not a causal weight retraining.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
import torch
import yaml

from bit_feasibility import ENCODER, ENCODER_HASH
from pytorch_baseline import PublishedGRU
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit import ROOT, digest, save_json, space_guard
from research.data import labels
from research.metrics import edits, greedy


def normalized(raw, mean, variance):
    kernel = np.exp(-.5*(np.arange(9)/2)**2); kernel /= kernel.sum()
    return lfilter(kernel, [1.], (raw-mean)/np.sqrt(np.maximum(variance, 1e-4)), axis=0).astype(np.float32)


def trailing_pool(hidden):
    n = hidden.shape[0]//5
    return hidden[:n*5].reshape(n, 5, hidden.shape[-1]).mean(1)


def prepare(output):
    space_guard(); torch.set_num_threads(4)
    assert torch.backends.mps.is_available()
    assert digest(ENCODER) == ENCODER_HASH
    manifest = json.loads((ROOT/'research/data/manifest.json').read_text())
    recipe = yaml.safe_load((ROOT/'workspace/derived/rnns/baselineRelease/args.yaml').read_text())
    days = recipe['dataset']['sessions'][:8]
    assert set(days) <= set(manifest['folds']['initial_train'])
    target = days[-1]
    validation = [dict(r, session=target, split='validation', partition='test')
                  for r in manifest['sessions'][target]['test']['trials'] if r['text']]
    assert len(validation) == 40
    norm = lambda text: ' '.join(text.lower().split())
    forbidden = {norm(r['text']) for r in validation}
    seen, training = set(), []
    for day in days:
        count = 0
        for row in manifest['sessions'][day]['train']['trials']:
            text = norm(row['text'])
            if not text or text in forbidden or text in seen:
                continue
            seen.add(text); count += 1
            training.append(dict(row, session=day, split='train', partition='train'))
            if count == 64:
                break
        assert count == 64
    records = training+validation
    save_json(output/'trials.json', records)
    stats, source_hashes, arrays = {}, {}, {}
    for day in days:
        entry = manifest['sessions'][day]['train']; source = ROOT/entry['path']
        source_hashes[entry['path']] = digest(source)
        data = loadmat(source, variable_names=['tx1', 'spikePow'])
        indices = [r['index'] for r in training if r['session'] == day]
        raw = [np.concatenate([data['tx1'].ravel()[i][:, :128], data['spikePow'].ravel()[i][:, :128]], 1).astype(np.float32)
               for i in indices]
        merged = np.concatenate(raw).astype(np.float64)
        assert np.isfinite(merged).all()
        stats[day] = dict(mean=merged.mean(0).tolist(), variance=np.maximum(merged.var(0), 1e-4).tolist(), samples=len(merged))
        arrays.update({r['id']: x for r, x in zip([r for r in training if r['session']==day], raw)})
    entry = manifest['sessions'][target]['test']; source = ROOT/entry['path']
    source_hashes[entry['path']] = digest(source)
    data = loadmat(source, variable_names=['tx1', 'spikePow'])
    for row in validation:
        i = row['index']
        arrays[row['id']] = np.concatenate([data['tx1'].ravel()[i][:, :128], data['spikePow'].ravel()[i][:, :128]], 1).astype(np.float32)
    save_json(output/'normalization.json', stats)
    model = PublishedGRU().to('mps').eval()
    state = torch.load(ENCODER, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model']); del state
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured = []
    hook = model.layers[-1].register_forward_hook(lambda _, args, result: captured.append(result[0]))

    def infer(x, day):
        captured.clear()
        with torch.no_grad():
            logits = model(torch.tensor(x, device='mps')[None], days.index(day))[0].cpu().numpy()
        return logits, captured[0][0].detach().cpu()

    # Prefix timestamps are patch RIGHT edges: 31,35,..., not their starts.
    row = validation[0]; raw = arrays[row['id']]
    mean, variance = np.asarray(stats[target]['mean']), np.asarray(stats[target]['variance'])
    x = normalized(raw, mean, variance); cut = 192
    changed = raw.copy(); changed[cut:] += 1
    assert np.array_equal(x[:cut], normalized(changed, mean, variance)[:cut])
    assert np.array_equal(x[:cut], normalized(raw[:cut], mean, variance))
    a, hidden = infer(x, target); b, _ = infer(normalized(changed, mean, variance), target)
    c, _ = infer(x[:cut], target); n = (cut-32)//4+1
    prefix_error = max(float(np.max(np.abs(a[:n]-b[:n]))), float(np.max(np.abs(a[:n]-c))))
    assert prefix_error < 1e-4, prefix_error
    latent = trailing_pool(hidden)
    causal_identity = dict(prefix_max_error=prefix_error, acquisition_delay_ms=4, bin_ms=20,
        patch_bins=32, patch_stride=4, patch_timestamp='right edge', latent_pool_frames=5,
        neural_token_interval_ms=400, first_token_bin_right_edge=47,
        first_token_available_ms=964, incomplete_final_pool='discarded',
        generation='offline after full trial; streaming text not claimed')
    save_json(output/'causality.json', causal_identity)
    phone_rows = []; features = {}
    # Gate before collecting the larger training feature set.
    for i, row in enumerate(validation):
        x = normalized(arrays[row['id']], mean, variance)
        logits, hidden = infer(x, target)
        target_phones = labels(row['text']).tolist()
        phones = greedy(logits)
        phone_rows.append(dict(id=row['id'], reference=target_phones, prediction=phones,
                               per=edits(target_phones, phones)))
        features[row['id']] = trailing_pool(hidden).numpy()
    errors = sum(r['per']['errors'] for r in phone_rows)
    units = sum(r['per']['reference_units'] for r in phone_rows)
    gate = dict(per=errors/units, errors=errors, reference_phones=units, trials=40,
        passed=errors/units <= .5, usability_threshold=.5,
        threshold_meaning='Prespecified exploratory usability screen, not source reproduction agreement',
        encoder='E008 stopped native Willett-recipe checkpoint, not author-released weights',
        source_encoder_original_full_benchmark_per=.22610489,
        comparison_warning='Different subset/preprocessing; source 600-trial PER is not a paired comparator',
        validation_history='E008 previously monitored public-test performance; not untouched encoder holdout')
    save_json(output/'phoneme-predictions.json', phone_rows)
    save_json(output/'encoder-gate.json', gate)
    print(json.dumps({'encoder_gate': gate}), flush=True)
    if not gate['passed']:
        hook.remove()
        save_json(output/'status.json', dict(status='stopped', reason='causal_encoder_usability_gate_failed', gate=gate))
        return False
    for i, row in enumerate(training):
        space_guard(); day = row['session']; moments = stats[day]
        x = normalized(arrays[row['id']], np.asarray(moments['mean']), np.asarray(moments['variance']))
        _, hidden = infer(x, day)
        features[row['id']] = trailing_pool(hidden).numpy()
        assert np.isfinite(features[row['id']]).all()
        if i % 64 == 0:
            save_json(output/'status.json', dict(status='extracting_features', completed_train=i+1, total_train=512))
            print(json.dumps({'features': i+1}), flush=True)
    hook.remove()
    np.savez_compressed(output/'features.npz', **features)
    save_json(output/'data-identity.json', dict(encoder_sha256=ENCODER_HASH,
        manifest_sha256=digest(ROOT/'research/data/manifest.json'), source_hashes=source_hashes,
        features_sha256=digest(output/'features.npz'), trials_sha256=digest(output/'trials.json'),
        train=512, validation=40, training_days=days, validation_day=target,
        duplicate_prompts=0, hidden_dim=1024, encoder_frozen=True, causality=causal_identity,
        historical_validation_exposure=True, feature_units='raw count and power, no log transform',
        normalization='per-day moments fitted on the selected 64 training trials only'))
    save_json(output/'status.json', dict(status='features_ready'))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'protocol.json', dict(train=512, validation=40, seed=17,
        encoder_sha256=ENCODER_HASH, source_sha256=digest(Path(__file__)),
        causal_per_usability_threshold=.5, next_stage='Frozen features to bridge + LoRA; no encoder fine-tuning',
        stop='Failed finite/prefix checks, PER >50%, or free storage <30 GiB'))
    try:
        prepare(args.output)
    except Exception as error:
        save_json(args.output/'status.json', dict(status='failed', error=str(error))); raise
