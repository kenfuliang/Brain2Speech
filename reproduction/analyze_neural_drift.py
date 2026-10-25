"""E018: read-only source study with immutable, resumable per-trial evidence.

No imports launch training, evaluation services or the WFST decoder. Inputs are
restricted to the manifest's initial/development training partitions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import psutil
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.audit import digest, save_json, space_guard
from research.data import labels
from research.metrics import edits, greedy
from pytorch_baseline import PublishedGRU

E017_SHA = 'c60225fd0759e4b591661a4b02d5d2184a10f85502c95410e2c6a1b98cd9f8ae'
SEED, REPS = 101, 2000
KERNEL = np.exp(-.5*(np.arange(9)/2)**2)
KERNEL /= KERNEL.sum()


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Immutable artifact mismatch: {path}')
    else:
        save_json(path, value)


def cohort(manifest):
    days = manifest['folds']['initial_train'] + manifest['folds']['development']
    if len(days) != 19 or len(set(days)) != 19 or days != sorted(days):
        raise ValueError('Expected exactly 19 unique chronological permitted days')
    if set(days) & set(manifest['folds']['final']):
        raise ValueError('Final days forbidden')
    rows = [dict(r, session=d, partition='train') for d in days
            for r in manifest['sessions'][d]['train']['trials']]
    if len(rows) != 6680 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Cohort accounting mismatch')
    if any(not r['text'] for r in rows):
        raise ValueError('Unlabeled/empty training prompt')
    for day in days:
        g = [r for r in rows if r['session'] == day]
        if [r['index'] for r in g] != list(range(len(g))):
            raise ValueError('Source trial ordering mismatch')
        if any(a['block'] > b['block'] for a, b in zip(g, g[1:])):
            raise ValueError('Nonchronological blocks')
    return days, rows


def trial_raw(mat, row):
    i = row['index']
    counts, power = mat['tx1'].ravel()[i], mat['spikePow'].ravel()[i]
    if counts.shape != power.shape or counts.shape[1] < 128:
        raise ValueError('Channel/feature correspondence failure')
    raw = np.concatenate([counts[:, :128], power[:, :128]], axis=1).astype(np.float64)
    if len(raw) != row['bins'] or len(raw) < 32 or not np.isfinite(raw).all():
        raise ValueError('Invalid/short/nonfinite source trial')
    if (raw < 0).any() or not np.allclose(raw[:, :128], np.rint(raw[:, :128]), atol=1e-5):
        raise ValueError('Source count/power units invalid')
    return raw


def preprocess(raw, mean, variance):
    return lfilter(KERNEL, [1.], (raw-mean)/np.sqrt(np.maximum(variance, 1e-4)),
                   axis=0).astype(np.float32)


def shard(output, row):
    return output/'trials'/row['session']/f"{row['index']:04d}.npz"


def raw_statistics(raw):
    return dict(raw_mean=raw.mean(0), raw_second=(raw*raw).mean(0),
                raw_variance=raw.var(0), raw_min=raw.min(0), raw_max=raw.max(0),
                raw_zero=(raw == 0).sum(0), raw_bins=np.array(len(raw)))


def fixed_inference(model, x, device):
    captured = []
    hook = model.layers[-1].register_forward_hook(lambda _, args, result: captured.append(result[0]))
    try:
        with torch.inference_mode():
            y = model(torch.from_numpy(x[None]).to(device), day=0)[0].cpu().numpy()
            hidden = captured[0][0].cpu().numpy()
    finally:
        hook.remove()
    if hidden.shape != (len(y), 1024) or not np.isfinite(y).all() or not np.isfinite(hidden).all():
        raise FloatingPointError('Invalid/nonfinite logits or embeddings')
    return y, hidden


def resource(device):
    value = dict(utc=datetime.now(timezone.utc).isoformat(),
                 rss_bytes=psutil.Process().memory_info().rss,
                 free_storage_bytes=space_guard(), device=str(device))
    if device.type == 'mps':
        value['mps_allocated_bytes'] = torch.mps.current_allocated_memory()
        value['mps_driver_bytes'] = torch.mps.driver_allocated_memory()
    return value


def check_inputs(output):
    p = json.loads((output/'protocol.json').read_text())
    for path, expected in p['input_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Changed sealed input: '+path)
    for path, expected in p['implementation_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Changed sealed implementation: '+path)
    for name in ('normalization.json','cohort.json'):
        marker=output/(name+'.sha256.json')
        if marker.exists() and digest(output/name)!=json.loads(marker.read_text())['sha256']:
            raise ValueError('Changed derived input: '+name)
    m = json.loads(Path(p['manifest']).read_text())
    days, rows = cohort(m)
    return p, m, days, rows


def prepare(args):
    if args.output.exists():
        check_inputs(args.output)
        return
    space_guard()
    manifest = json.loads(args.manifest.read_text())
    identity = json.loads(args.identity.read_text())
    days, rows = cohort(manifest)
    checkpoint = Path(identity['checkpoint'])
    if identity['checkpoint_sha256'] != E017_SHA or digest(checkpoint) != E017_SHA:
        raise ValueError('E017 selected checkpoint mismatch')
    paths = [args.manifest, args.identity, checkpoint]
    paths += [ROOT/manifest['sessions'][d]['train']['path'] for d in days]
    for d, path in zip(days, paths[3:]):
        if digest(path) != manifest['sessions'][d]['train']['sha256']:
            raise ValueError('Source artifact hash mismatch: '+d)
    code = [Path(__file__), ROOT/'reproduction/pytorch_baseline.py',
            ROOT/'research/metrics.py', ROOT/'research/data.py', ROOT/'research/audit.py']
    protocol = dict(experiment='E018', date_local='2026-10-11',
        manifest=str(args.manifest.resolve()), identity=str(args.identity.resolve()),
        checkpoint=str(checkpoint), days=days, initial_days=manifest['folds']['initial_train'],
        excluded_final_days=manifest['folds']['final'], partition='train', trials=len(rows),
        input_hashes={str(p.resolve()): digest(p) for p in paths},
        implementation_hashes={str(p.resolve()): digest(p) for p in code},
        scope='Retrospective, one participant; no biological/casual/future-day claim',
        preprocessing=dict(features='tx1[:,0:128] + spikePow[:,0:128]; untransformed',
            bin_ms=20, acquisition_delay_ms=4, normalization='equal-trial initial-training moments only',
            variance_floor=1e-4, smoothing_kernel=KERNEL.tolist(), smoothing='causal trailing; trial reset',
            adapter=0, model_mode='eval', latent='last GRU, 1024 dimensions',
            departure='Global fixed normalization/trailing smoothing differ from centered/blockwise training'),
        sampling=dict(frame_sample_per_trial=4, PCA_components=32, PCA_fit='initial-training trial means only',
            summaries='each trial equal weight; no pooled-bin pseudo-replication',
            covariance='covariance of trial means, not full time-bin covariance'),
        statistics=dict(seed=SEED, repetitions=REPS,
            distance='RMS difference between day centroids; raw standardized by initial-training bin SD',
            latent_distance='RMS in shared unwhitened PCA coordinates',
            uncertainty='Whole-block resampling within each fixed day/half; shared draws across day pairs',
            effect='median pairwise between-day distance minus median within-day chronological-half distance',
            sensitivity='mean instead of median aggregation and latent PCA16 instead of32',
            evidence='conditional 95% excess CI lower>0 under primary and sensitivity; exploratory, not multiplicity-corrected',
            repeated_content='all exact prompt pairs; no interval for groups<10; shared-session dependence disclosed',
            PER='greedy CTC,41 classes,blank40; corpus and per-day; no WFST/WER'),
        resource_policy=dict(batch_size=1, preferred='MPS', cpu_fallback='only unavailable/unsupported, explicitly logged',
            minimum_free_GiB=30, serial=True, delete_existing=False),
        chart_contract=dict(surface='four requested static PNGs in canonical Markdown report',
            families=['signed channel heatmap','day-distance heatmaps','paired PCA scatter','shift versus PER scatter'],
            grain='trials for summaries; days for distance/PER; exact prompts for paired plot',
            palette='blue #2563A6 / orange #C86D26 / neutral; signed two-root cap',
            uncertainty='conditional block intervals; sparse prompt pairs descriptive',
            QA='inspect all four exported PNGs; evidence hashes and row counts'))
    args.output.mkdir(parents=True)
    write_once(args.output/'protocol.json', protocol)
    write_once(args.output/'cohort.json', rows)
    np_values = []
    seconds = []
    for day in protocol['initial_days']:
        mat = loadmat(ROOT/manifest['sessions'][day]['train']['path'], variable_names=['tx1', 'spikePow'])
        for r in rows:
            if r['session'] == day:
                raw = trial_raw(mat, r)
                np_values.append(raw.mean(0)); seconds.append((raw*raw).mean(0))
    mean = np.mean(np_values, axis=0)
    variance = np.maximum(np.mean(seconds, axis=0)-mean*mean, 1e-4)
    write_once(args.output/'normalization.json', dict(mean=mean.tolist(), variance=variance.tolist(),
        trials=len(np_values), fit_days=protocol['initial_days'], equal_trial_weight=True))
    for name in ('normalization.json','cohort.json'):
        write_once(args.output/(name+'.sha256.json'),dict(sha256=digest(args.output/name)))
    write_once(args.output/'environment.json', dict(python=sys.version, torch=torch.__version__,
        numpy=np.__version__, platform=platform.platform()))
    print('prepared', len(rows), 'trials;', len(np_values), 'initial-training normalization trials', flush=True)


def model_for(protocol):
    torch.set_num_threads(4)
    state = torch.load(protocol['checkpoint'], map_location='cpu', weights_only=False)
    model = PublishedGRU().eval()
    model.load_state_dict(state['model'])
    del state
    for p in model.parameters():
        p.requires_grad_(False)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    return model.to(device), device


def extract(args, pilot=False):
    output = args.output
    p, m, days, rows = check_inputs(output)
    if (output/'seal.json').exists():
        raise RuntimeError('Completed run is sealed; do not mutate')
    if not pilot and not (output/'pilot.json').exists():
        raise RuntimeError('Verified pilot required')
    normalization = json.loads((output/'normalization.json').read_text())
    mean, variance = np.array(normalization['mean']), np.array(normalization['variance'])
    model, device = model_for(p)
    if device.type == 'cpu':
        write_once(output/'device-fallback.json', dict(reason='MPS unavailable', device='cpu'))
    if pilot:
        rows = [r for d in [days[0], days[7], days[13], days[-1]]
                for r in [next(x for x in rows if x['session'] == d),
                          next(x for x in reversed(rows) if x['session'] == d)]]
    started = time.monotonic()
    done = 0
    probes = []
    for day in days:
        selected = [r for r in rows if r['session'] == day]
        if not selected:
            continue
        mat = loadmat(ROOT/m['sessions'][day]['train']['path'], variable_names=['tx1', 'spikePow'])
        for r in selected:
            destination = shard(output, r)
            metadata = destination.with_suffix('.json')
            if destination.exists():
                if not metadata.exists() or digest(destination) != json.loads(metadata.read_text())['sha256']:
                    raise ValueError('Incomplete/corrupt shard; preserve and inspect')
                if not pilot:
                    done += 1
                    continue
            space_guard()
            raw = trial_raw(mat, r)
            x = preprocess(raw, mean, variance)
            t = time.monotonic()
            try:
                logits, hidden = fixed_inference(model, x, device)
            except (RuntimeError, NotImplementedError) as error:
                if device.type != 'mps' or not any(v in str(error).lower() for v in ('not implemented','not supported')):
                    raise
                write_once(output/'device-fallback.json', dict(reason=str(error), device='cpu'))
                device = torch.device('cpu'); model.to(device)
                logits, hidden = fixed_inference(model, x, device)
            values = raw_statistics(raw)
            values.update(latent_mean=hidden.astype(np.float64).mean(0),
                latent_variance=hidden.astype(np.float64).var(0),
                latent_frames=hidden[np.linspace(0, len(hidden)-1, min(4, len(hidden)), dtype=int)],
                phones=np.array(greedy(logits)), reference=labels(r['text']),
                hidden_length=np.array(len(hidden)))
            per = edits(values['reference'].tolist(), values['phones'].tolist())
            if pilot:
                again, other = fixed_inference(model, x, device)
                if not (np.array_equal(logits, again) and np.array_equal(hidden, other)):
                    raise ValueError('Trial-state reset/determinism failed')
                independent_mean = sum(hidden[i].astype(np.float64) for i in range(len(hidden)))/len(hidden)
                assert np.allclose(independent_mean, values['latent_mean'], atol=1e-10)
                cut = min(len(raw), 128)
                changed = raw.copy(); changed[cut:] += 10
                assert np.array_equal(preprocess(changed, mean, variance)[:cut], x[:cut])
                probes.append(dict(id=r['id'], deterministic=True, state_reset=True,
                    independently_recomputed_mean=True, prefix_preprocessing=True))
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix('.partial')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **values)
                temporary.replace(destination)
                write_once(metadata, dict(id=r['id'], sha256=digest(destination), per=per,
                    device=str(device), inference_seconds=time.monotonic()-t,
                    flags=['zero-variance-channel'] if (values['raw_variance'] == 0).any() else []))
            else:
                with np.load(destination) as saved:
                    for k in values:
                        assert np.array_equal(saved[k], values[k]), (r['id'], k)
            done += 1
            if done % 40 == 0 or pilot:
                observation=dict(status='extracting', done=done,
                    total=len(rows), elapsed_seconds=time.monotonic()-started, **resource(device))
                save_json(output/'progress.json', observation)
                write_once(output/'resources'/f'{time.time_ns()}.json',observation)
                print(day, done, '/', len(rows), 'elapsed', round(time.monotonic()-started, 1), flush=True)
        del mat
    if pilot:
        # Reload from the sealed checkpoint, not from the already resident object.
        original = model
        model, reload_device = model_for(p)
        r = rows[0]
        mat = loadmat(ROOT/m['sessions'][r['session']]['train']['path'], variable_names=['tx1','spikePow'])
        x = preprocess(trial_raw(mat, r), mean, variance)
        a, h = fixed_inference(original, x, device)
        b, q = fixed_inference(model, x, reload_device)
        assert np.allclose(a, b, rtol=1e-5, atol=1e-5) and np.allclose(h, q, rtol=1e-5, atol=1e-5)
        write_once(output/'pilot.json', dict(passed=True, probes=probes, checkpoint_reload=True,
            adapter=0, batch_size=1, device=str(device), checkpoint_sha256=E017_SHA))
    else:
        write_once(output/'extraction-complete.json', dict(trials=done, total=len(rows),
            checkpoint_sha256=E017_SHA, protocol_sha256=digest(output/'protocol.json')))
    print('pilot passed' if pilot else 'extraction complete', flush=True)


def fit_pca(values, fit_mask, components=32):
    mean = values[fit_mask].mean(0)
    centered = values[fit_mask]-mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(components, len(singular), values.shape[1])
    axes = vt[:k]
    # Resolve sign for deterministic visual orientation, not scientific alignment.
    for a in axes:
        if a[np.argmax(np.abs(a))] < 0:
            a *= -1
    return (values-mean)@axes.T, dict(mean=mean, axes=axes,
        explained=singular[:k]**2/np.sum(singular**2))


def split_blocks(rows):
    blocks = sorted({r['block'] for r in rows})
    if len(blocks) < 2:
        return None
    sizes = np.array([sum(r['block'] == b for r in rows) for b in blocks])
    cut = 1+np.argmin(np.abs(np.cumsum(sizes)[:-1]-sizes.sum()/2))
    return blocks[:cut], blocks[cut:]


def boot_centroids(values, blocks, rng, repetitions):
    unique = sorted(set(blocks))
    ns = np.array([sum(b == v for b in blocks) for v in unique])
    totals = np.stack([values[np.array(blocks) == b].sum(0) for b in unique])
    counts = rng.multinomial(len(unique), np.ones(len(unique))/len(unique), size=repetitions)
    return (counts@totals)/(counts@ns)[:, None]


def distance(a, b):
    return np.sqrt(np.mean((a-b)**2, axis=-1))


def day_statistics(values, rows, days, repetitions=REPS):
    rng = np.random.default_rng(SEED)
    means, covariances, boots, within, within_boot, limits = [], [], [], [], [], []
    for d in days:
        ix = np.array([i for i, r in enumerate(rows) if r['session'] == d])
        v = values[ix]; g = [rows[i] for i in ix]; blocks = [r['block'] for r in g]
        means.append(v.mean(0)); covariances.append(np.cov(v, rowvar=False))
        boots.append(boot_centroids(v, blocks, rng, repetitions))
        halves = split_blocks(g)
        if halves is None:
            within.append(np.nan); within_boot.append(np.full(repetitions, np.nan))
            limits.append(dict(day=d, insufficient_blocks=True)); continue
        left, right = [np.array([b in h for b in blocks]) for h in halves]
        within.append(float(distance(v[left].mean(0), v[right].mean(0))))
        wb = distance(boot_centroids(v[left], np.array(blocks)[left], rng, repetitions),
                      boot_centroids(v[right], np.array(blocks)[right], rng, repetitions))
        within_boot.append(wb)
        limits.append(dict(day=d, blocks=len(set(blocks)), left_trials=int(left.sum()),
            right_trials=int(right.sum()), left_blocks=len(halves[0]), right_blocks=len(halves[1]),
            uncertainty_weak=min(map(len, halves)) < 2))
    means = np.stack(means)
    matrix = cdist(means, means)/np.sqrt(values.shape[1])
    covariance_distance = np.array([[np.linalg.norm(a-b, 'fro')/values.shape[1]
        for b in covariances] for a in covariances])
    pair_boot = np.stack([distance(boots[i], boots[j]) for i, j in itertools.combinations(range(len(days)), 2)], axis=1)
    wboot = np.stack(within_boot, axis=1)
    observed_pairs = matrix[np.triu_indices(len(days), 1)]
    excess = np.median(pair_boot, axis=1)-np.nanmedian(wboot, axis=1)
    sensitivity = pair_boot.mean(1)-np.nanmean(wboot, axis=1)
    interval = np.quantile(excess, [.025, .975]).tolist()
    sensitivity_interval = np.quantile(sensitivity, [.025, .975]).tolist()
    ref_boot = np.stack([distance(b, boots[0]) for b in boots], axis=1)
    return dict(matrix=matrix.tolist(), covariance_distance=covariance_distance.tolist(),
        within=within, block_coverage=limits,
        reference_distance_ci95=np.quantile(ref_boot, [.025, .975], axis=0).T.tolist(),
        median_between=float(np.median(observed_pairs)), median_within=float(np.nanmedian(within)),
        excess=float(np.median(observed_pairs)-np.nanmedian(within)), excess_ci95=interval,
        mean_excess_ci95=sensitivity_interval,
        conditional_support=interval[0] > 0 and sensitivity_interval[0] > 0,
        caveat='Fixed 19 days; conditional block resampling, not population/day-pair independence or causal inference')


def repeated_content(rows, representations):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['text']].append(i)
    pairs = []
    for prompt, ids in groups.items():
        for i, j in itertools.combinations(ids, 2):
            a, b = rows[i], rows[j]
            pairs.append(dict(prompt=prompt, left=a['id'], right=b['id'],
                left_day=a['session'], right_day=b['session'], cross_day=a['session'] != b['session'],
                duration_ratio=max(a['bins'], b['bins'])/min(a['bins'], b['bins']),
                distances={k:float(distance(v[i], v[j])) for k,v in representations.items()}))
    return dict(pairs=pairs, cross_day_prompts=len({r['prompt'] for r in pairs if r['cross_day']}),
        within_day_prompts=len({r['prompt'] for r in pairs if not r['cross_day']}),
        verdict='inconclusive', interval=None,
        reason='Only seven within-day repeated prompts; unmatched prompt populations and shared-session dependence. No inferential comparison.')


def summarize(output):
    p, m, days, rows = check_inputs(output)
    if not (output/'extraction-complete.json').exists():
        raise RuntimeError('Full collection required')
    norm = json.loads((output/'normalization.json').read_text())
    mean, sd = np.array(norm['mean']), np.sqrt(norm['variance'])
    arrays = defaultdict(list); phone_rows = []; hashes = {}
    for r in rows:
        path = shard(output, r); meta = json.loads(path.with_suffix('.json').read_text())
        if digest(path) != meta['sha256'] or meta['id'] != r['id']:
            raise ValueError('Trial evidence mismatch')
        hashes[str(path.relative_to(output))] = meta['sha256']
        with np.load(path) as v:
            for key in ('raw_mean','raw_variance','raw_min','raw_max','raw_zero','latent_mean','latent_variance'):
                arrays[key].append(v[key])
            account = edits(v['reference'].tolist(), v['phones'].tolist())
            if account != meta['per']:
                raise ValueError('Independent PER accounting failed')
        phone_rows.append(dict(id=r['id'], session=r['session'], **account))
    arrays = {k:np.stack(v) for k,v in arrays.items()}
    fit = np.array([r['session'] in p['initial_days'] for r in rows])
    latent, pca = fit_pca(arrays['latent_mean'], fit)
    standardized = (arrays['raw_mean']-mean)/sd
    reps = dict(counts=standardized[:, :128], power=standardized[:, 128:], latent=latent)
    results = {k:day_statistics(v, rows, days) for k,v in reps.items()}
    results['latent16'] = day_statistics(latent[:, :16], rows, days)
    repeated = repeated_content(rows, reps)
    if repeated['cross_day_prompts'] != 77 or repeated['within_day_prompts'] != 7:
        raise ValueError('Recorded repeated-content coverage changed')
    per_day = {}
    for d in days:
        g = [r for r in phone_rows if r['session'] == d]
        sums = {k:sum(r[k] for r in g) for k in ('errors','reference_units','substitutions','deletions','insertions')}
        per_day[d] = dict(trials=len(g), per=sums['errors']/sums['reference_units'], **sums)
    correlations = {}
    target = np.array([per_day[d]['per'] for d in days])
    for k in reps:
        x = np.array(results[k]['matrix'])[0]
        rho = float(spearmanr(x[1:], target[1:]).statistic)
        correlations[k] = dict(spearman_rho=rho, days=18, reference_day_excluded=True,
            significance='Not tested; descriptive and same-participant repeated-session dependence')
    zeros = arrays['raw_variance'] == 0
    support = results['counts']['conditional_support'] or results['power']['conditional_support']
    latent_support = results['latent']['conditional_support'] and results['latent16']['conditional_support']
    total_errors = sum(r['errors'] for r in phone_rows); total_units = sum(r['reference_units'] for r in phone_rows)
    metrics = dict(days=days, trials=len(rows), total_bins=sum(r['bins'] for r in rows),
        per=dict(corpus=total_errors/total_units, errors=total_errors, reference_units=total_units,
            macro_day=float(target.mean()), worst_day=float(target.max()), by_day=per_day,
            S=sum(r['substitutions'] for r in phone_rows),D=sum(r['deletions'] for r in phone_rows),
            I=sum(r['insertions'] for r in phone_rows)),
        statistics=results, correlations=correlations,
        PCA=dict(components=len(pca['axes']), retained32=float(pca['explained'].sum()),
            retained2=float(pca['explained'][:2].sum()), fit_trials=int(fit.sum())),
        verdicts=dict(recorded_feature_shift='supported' if support else 'inconclusive',
            fixed_encoder_latent_shift='supported' if latent_support else 'inconclusive',
            content_conditioned_evidence='inconclusive', biological_representational_drift='not established'),
        limitations=['One participant, retrospective model exposed to multiple days including two final days',
            'Fixed global preprocessing and adapter0 differ from original recipe; PER is a usability diagnostic, not baseline reproduction',
            'Different content/task/recording conditions can explain marginal shifts',
            'Conditional bootstrap is not an independent 19-day population sample or multiple-comparison test',
            'Counts/power cannot establish identified-neuron turnover or biological cause',
            'Channel order follows released archive indexing; anatomical stability cannot be independently verified'],
        quality=dict(excluded_trials=0, zero_variance_trial_channels=int(zeros.sum()),
            trials_with_zero_variance=int(zeros.any(1).sum()),
            nonfinite_trials=0, duplicate_trial_ids=0, count_channels=128, power_channels=128,
            trial_mean_quantiles=np.quantile(arrays['raw_mean'],[.01,.25,.5,.75,.99],axis=0).tolist(),
            quantile_grain='Distribution of per-trial means, not pooled time bins',
            duration_quantiles_seconds=np.quantile([r['duration_seconds'] for r in rows],[0,.25,.5,.75,1]).tolist(),
            feature_min=arrays['raw_min'].min(0).tolist(), feature_max=arrays['raw_max'].max(0).tolist(),
            feature_zero_counts=arrays['raw_zero'].sum(0).tolist(),
            note='All anomalies retained; no automatic exclusions'),
        WER='Not measured; WFST not run', CER='Not measured', seed_dispersion='Not applicable; fixed checkpoint')
    write_once(output/'metrics.json', metrics)
    write_once(output/'repeated-content.json', repeated)
    write_once(output/'shard-hashes.json', hashes)
    summary_path = output/'summaries.npz'
    if not summary_path.exists():
        np.savez_compressed(summary_path, **arrays, latent_pca=latent, raw_standardized=standardized,
            pca_mean=pca['mean'], pca_axes=pca['axes'], pca_explained=pca['explained'])
    write_once(output/'phone-accounting.json', phone_rows)
    print(json.dumps(dict(verdicts=metrics['verdicts'], per=metrics['per']['corpus'],
        excesses={k:(v['excess'],v['excess_ci95']) for k,v in results.items()})), flush=True)
    return metrics, rows, arrays, reps


def figures(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    metrics = json.loads((output/'metrics.json').read_text())
    rows = json.loads((output/'cohort.json').read_text())
    repeated = json.loads((output/'repeated-content.json').read_text())
    days = metrics['days']; ticks = [d[8:] for d in days]
    with np.load(output/'summaries.npz') as saved:
        standard = saved['raw_standardized']; latent = saved['latent_pca']
    palette = LinearSegmentedColormap.from_list('blue_white_orange',['#2563A6','#FFFFFF','#C86D26'])
    sequential = LinearSegmentedColormap.from_list('white_blue',['#FFFFFF','#2563A6'])
    figdir = output/'figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'text.color':'#222222','axes.labelcolor':'#222222'})
    sources = {}

    def create(title, subtitle, panels=2, width=13, height=6):
        fig, axes = plt.subplots(1, panels, figsize=(width,height), squeeze=False)
        fig.subplots_adjust(top=.77,bottom=.24,left=.09,right=.94,wspace=.4)
        fig.text(.09,.96,title,fontsize=16,fontweight='bold')
        fig.text(.09,.87,subtitle,fontsize=10,color='#555555')
        # Small neutral blossom at the locked research-header top-right.
        fig.text(.96,.95,'✿',ha='right',fontsize=15,color='#555555')
        return fig, axes.ravel()

    def finish(fig, name, note):
        fig.text(.09,.07,note,fontsize=9,color='#555555')
        fig.savefig(figdir/name,dpi=160)
        plt.close(fig)
        sources[name] = dict(sha256=digest(figdir/name), metrics_sha256=digest(output/'metrics.json'),
            summaries_sha256=digest(output/'summaries.npz'), cohort_sha256=digest(output/'cohort.json'))

    fig, axes = create('Recorded channel means vary across days',
        'T12 • 19 permitted training days • 6,680 trials • equal trial weights; reference-fit bin standard deviations')
    means = np.stack([standard[[r['session']==d for r in rows]].mean(0) for d in days])
    bound = max(float(np.abs(means).max()),1e-6)
    for ax, start, title in zip(axes,[0,128],['Threshold-crossing counts','Spike-band power']):
        im=ax.imshow(means[:,start:start+128],aspect='auto',cmap=palette,vmin=-bound,vmax=bound)
        ax.set_title(title); ax.set_yticks(range(19),ticks); ax.set_xlabel('Released channel index (0–127)')
        fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label='Mean shift / reference bin SD')
    finish(fig,'channel-shifts.png','Unnormalized source features; fixed-reference standardized display. Content and recording changes remain confounds.\nAll trials retained; uncertainty and raw-unit means/variances are in the sealed evidence.')

    fig, axes = create('Between-day distances versus within-day variability',
        'T12 • 19 days / 171 descriptive pairs • centroid RMS; axes shared within each representation',3,15,6.8)
    for ax, key, title in zip(axes,['counts','power','latent'],['Counts: reference-SD units','Power: reference-SD units','Latents: shared PCA32 units']):
        result=metrics['statistics'][key]; matrix=np.array(result['matrix'])
        im=ax.imshow(matrix,cmap=sequential,vmin=0)
        ax.set_title(title,fontsize=10); ax.set_xticks(range(19),ticks,rotation=90,fontsize=8)
        ax.set_yticks(range(19),ticks,fontsize=8)
        fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
        ci=result['excess_ci95']
        ax.set_xlabel(f"Between − within: {result['excess']:.3g}\n95% block CI [{ci[0]:.3g}, {ci[1]:.3g}]",fontsize=9)
    finish(fig,'day-distances.png','2,000 whole-block resamples within fixed days; day pairs are dependent, not 171 independent observations.\nDistances and color scales differ across representations; covariance differences are separately saved.')

    fig, axes = create('Repeated content: paired latent observations',
        f"77 cross-day prompts; 7 within-day prompts • shared PCA1–2 explains {100*metrics['PCA']['retained2']:.1f}% of latent variance",2,13,6)
    lookup={r['id']:i for i,r in enumerate(rows)}
    for ax, cross, title in zip(axes,[True,False],['Same prompt across different days','Same prompt within one day']):
        pairs=[r for r in repeated['pairs'] if r['cross_day']==cross]
        for r in pairs:
            points=latent[[lookup[r['left']],lookup[r['right']]],:2]
            ax.plot(points[:,0],points[:,1],color='#999999',linewidth=.7,alpha=.6)
            ax.scatter(*points[0],color='#2563A6',marker='o',s=15)
            ax.scatter(*points[1],color='#C86D26',marker='x',s=18)
        ax.set_title(f'{title} (n={len(pairs)} pairs)',fontsize=11)
        ax.set_xlabel('Shared latent PC1'); ax.set_ylabel('Shared latent PC2')
    limits=np.vstack([latent[lookup[r[k]],:2] for r in repeated['pairs'] for k in ['left','right']])
    for ax in axes:
        for dim, setter in enumerate([ax.set_xlim,ax.set_ylim]):
            lo,hi=limits[:,dim].min(),limits[:,dim].max(); pad=max((hi-lo)*.08,.1); setter(lo-pad,hi+pad)
    finish(fig,'repeated-prompts.png','Blue circle = earlier trial; orange cross = later trial. Display only: evidence uses all 32 dimensions.\nDifferent prompt populations and sparse within-day controls prevent a content-conditioned drift verdict.')

    fig, axes = create('Distribution shift and greedy phoneme error',
        'T12 • 19 matched training-day aggregates • frozen E017 + fixed adapter0 / global normalization',3,15,6)
    y=np.array([metrics['per']['by_day'][d]['per']*100 for d in days])
    for ax,key,title in zip(axes,['counts','power','latent'],['Counts','Power','Latent PCA32']):
        x=np.array(metrics['statistics'][key]['matrix'])[0]
        interval=np.array(metrics['statistics'][key]['reference_distance_ci95'])
        ax.hlines(y,interval[:,0],interval[:,1],color='#AAAAAA',linewidth=.8)
        ax.scatter(x,y,color='#2563A6',s=22)
        for i in range(19):
            ax.annotate(ticks[i],(x[i],y[i]),fontsize=6,xytext=(3,3),textcoords='offset points')
        rho=metrics['correlations'][key]['spearman_rho']
        ax.set_title(f'{title}: descriptive ρ={rho:.2f}',fontsize=11)
        ax.set_xlabel('Centroid distance from April 28'); ax.set_ylabel('Greedy PER (%)')
    finish(fig,'shift-versus-per.png','Horizontal intervals: conditional 95% block-bootstrap distance CI. Correlations exclude reference day (n=18).\nNo significance/causality claim; no WFST WER. This preprocessing differs from the 18.41% WER baseline.')
    write_once(output/'figure-provenance.json',sources)
    print('four figures rendered',flush=True)


def verify(output):
    p,m,days,rows=check_inputs(output)
    metrics=json.loads((output/'metrics.json').read_text())
    totals=Counter()
    missing=[]
    for r in rows:
        path=shard(output,r); meta=json.loads(path.with_suffix('.json').read_text())
        if digest(path)!=meta['sha256']:
            raise ValueError('Shard hash failure')
        with np.load(path) as a:
            e=edits(a['reference'].tolist(),a['phones'].tolist())
            if int(a['raw_bins'])!=r['bins'] or int(a['hidden_length'])!=(r['bins']-32)//4+1:
                raise ValueError('Trial/padding frame accounting failed')
            totals.update(e)
    if totals['errors']!=metrics['per']['errors'] or totals['reference_units']!=metrics['per']['reference_units']:
        raise ValueError('Independent corpus PER totals failed')
    norm=json.loads((output/'normalization.json').read_text())
    # Independently reconstruct equal-trial moments from source for three probes.
    for day in [days[0],days[13],days[-1]]:
        mat=loadmat(ROOT/m['sessions'][day]['train']['path'],variable_names=['tx1','spikePow'])
        r=next(r for r in rows if r['session']==day)
        raw=trial_raw(mat,r)
        with np.load(shard(output,r)) as a:
            assert np.array_equal(a['raw_mean'],raw.mean(0))
            assert np.array_equal(a['raw_variance'],raw.var(0))
    with np.load(output/'summaries.npz') as a:
        assert len(a['latent_mean'])==len(rows)
        recomputed=(a['latent_mean']-a['pca_mean'])@a['pca_axes'].T
        assert np.allclose(recomputed,a['latent_pca'],atol=1e-10)
    for name,item in json.loads((output/'figure-provenance.json').read_text()).items():
        assert digest(output/'figures'/name)==item['sha256']
    write_once(output/'verification.json',dict(passed=True,trials=len(rows),
        sources_hash_checked=True,per_recomputed=dict(totals),frame_accounting=True,
        raw_probes=3,shared_PCA_verified=True,final_partitions_opened=False,
        interpretation='Share with caveats; biological drift not established'))
    print('verification passed',flush=True)


def seal(output):
    if not json.loads((output/'verification.json').read_text())['passed']:
        raise ValueError('Verification required')
    paths=sorted(p for p in output.rglob('*') if p.is_file() and p.name!='seal.json')
    write_once(output/'seal.json',dict(status='complete',files={str(p.relative_to(output)):digest(p) for p in paths}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=ROOT/'research/data/manifest.json')
    parser.add_argument('--identity',type=Path,default=ROOT/'research/runs/e017-willett-selected-final-20261009-01/identity.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','pilot','collect','report','verify','seal'],required=True)
    args=parser.parse_args()
    if args.stage=='prepare': prepare(args)
    elif args.stage in ('pilot','collect'): extract(args,pilot=args.stage=='pilot')
    elif args.stage=='report': summarize(args.output); figures(args.output)
    elif args.stage=='verify': verify(args.output)
    elif args.stage=='seal': seal(args.output)


if __name__=='__main__': main()
