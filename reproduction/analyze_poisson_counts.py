"""E020: descriptive Poisson adequacy of permitted unsmoothed 20 ms counts.

No model, transcript alignment, decoder, or final-recording access. Local windows
are approximate stationarity sensitivities, NOT known phoneme-conditioned repeats.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import psutil
from scipy.io import loadmat
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.audit import digest, save_json, space_guard

WINDOWS = (10, 25, 50, 100)
SEED = 101
TAIL = 30


def ratio(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=b > 0)


def histogram(x):
    # Final category contains all counts >= TAIL, without deleting the tail.
    return np.stack([np.bincount(np.minimum(x[:, c], TAIL), minlength=TAIL+1)
                     for c in range(x.shape[1])])


def expected_hist(mu, n):
    p = poisson.pmf(np.arange(TAIL)[None, :], mu[:, None])
    return n * np.concatenate([p, poisson.sf(TAIL-1, mu)[:, None]], axis=1)


def local_stats(x, length):
    n = len(x)//length
    z = x[:n*length].reshape(n, length, x.shape[1]).astype(np.float64)
    mu = z.mean(1)
    centered = z-mu[:, None, :]
    numerator = (centered*centered).sum((0, 1))
    denominator = ((length-1)*mu).sum(0)
    # Adjacent difference: unit expectation under independent constant-rate Poisson.
    adjacent = (np.diff(z, axis=1)**2).sum((0, 1))
    ans = 2*np.sqrt(z+.375)
    ans_ss = ((ans-ans.mean(1)[:, None, :])**2).sum((0, 1))
    return np.stack([numerator, denominator, adjacent, ans_ss,
                     np.full(x.shape[1], n*(length-1)),
                     np.full(x.shape[1], n), (mu > 0).sum(0)], axis=-1)


def moments(x):
    x = x.astype(np.float64)
    return np.stack([np.full(x.shape[1], len(x)), x.sum(0), (x*x).sum(0),
                     (x == 0).sum(0)], axis=-1)


def synthetic_controls():
    rng = np.random.default_rng(SEED)
    records = {}
    for name in ('constant_poisson', 'changing_rate_poisson', 'negative_binomial'):
        if name == 'constant_poisson':
            x = rng.poisson(1., (4000, 100, 1))
        elif name == 'changing_rate_poisson':
            lam = np.r_[np.full(50, .25), np.full(50, 1.75)][None, :, None]
            x = rng.poisson(lam, (4000, 100, 1))
        else:
            x = rng.negative_binomial(1., .5, (4000, 100, 1))
        f = x.reshape(-1, 1)
        records[name] = dict(pooled_fano=float(f.var()/f.mean()),
            local_200ms_dispersion=float(ratio(*local_stats(f, 10)[:, :2].T)[0]))
    assert abs(records['constant_poisson']['pooled_fano']-1) < .03
    assert abs(records['constant_poisson']['local_200ms_dispersion']-1) < .03
    assert records['changing_rate_poisson']['pooled_fano'] > 1.45
    assert abs(records['changing_rate_poisson']['local_200ms_dispersion']-1) < .03
    assert records['negative_binomial']['local_200ms_dispersion'] > 1.8
    # At sparse rates, approximate unit variance of Anscombe is not exact.
    records['anscombe_poisson_reference'] = [dict(rate=rate,
        variance=float((2*np.sqrt(rng.poisson(rate, 200000)+.375)).var()))
        for rate in (.1, .5, 1., 2., 5.)]
    return records


def numeric_summary(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                q10=float(np.quantile(x, .1)), q90=float(np.quantile(x, .9)),
                minimum=float(x.min()), maximum=float(x.max()))


def run(args):
    out = args.output
    if (out/'seal.json').exists():
        raise RuntimeError('Sealed run; preserve unchanged')
    m = json.loads(args.manifest.read_text())
    days = m['folds']['initial_train']+m['folds']['development']
    assert len(days) == 19 and days == sorted(days)
    assert not set(days) & set(m['folds']['final'])
    prior = ROOT/'research/runs/e018-neural-drift-20261011-01/cohort.json'
    rows = [dict(r, session=d, partition='train') for d in days
            for r in m['sessions'][d]['train']['trials']]
    assert rows == json.loads(prior.read_text()) and len(rows) == 6680
    assert len({r['id'] for r in rows}) == 6680
    paths = [ROOT/m['sessions'][d]['train']['path'] for d in days]
    hashes = {str(p): digest(p) for p in paths}
    for d, p in zip(days, paths):
        assert hashes[str(p)] == m['sessions'][d]['train']['sha256']
    protocol = dict(experiment='E020', scope='19 permitted days, training partitions, count channels 0:128',
        date_local='2026-10-12', bin_ms=20, trials=6680, source_hashes=hashes,
        manifest=str(args.manifest.resolve()), manifest_sha256=digest(args.manifest),
        cohort_sha256=digest(prior), implementation_sha256=digest(__file__),
        windows_bins=list(WINDOWS), seed=SEED, bootstrap=2000,
        assumptions='Local equal-rate windows are sensitivity assumptions, not observed phoneme conditioning.',
        weighting='Bin-weighted within each day/channel; summary median gives each day/channel equal weight.',
        local_dispersion='sum within-window squared deviations / sum (window length-1)*window mean',
        adjacent_ratio='sum adjacent squared differences / (2 * local dispersion denominator)',
        TV='Total variation over counts 0..29 and >=30; fitted day-Poisson or trial-rate Poisson mixture',
        uncertainty='Shared whole-block resampling within fixed days; conditional 95% intervals, no iid-bin p-values.',
        practical_band=[.8, 1.2], band_status='Descriptive heuristic, not a validated Poisson acceptance test',
        excluded='Five final days, all public test partitions, power channels; no invalid observations cleaned.',
        chart=dict(surface='Established canonical Markdown report with static PNG',
            families=['day-channel heatmap', 'window-scale dot and interval', 'zero-rate scatter', 'Anscombe variance scatter'],
            palette=['#2563A6', '#C86D26'], grain='19 days x 128 channels, plus whole-block intervals',
            caveats='Fitted descriptive comparisons; rate variation and serial dependence remain; not model benefit'))
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text()) == protocol
    else:
        space_guard(); out.mkdir(parents=True)
        save_json(out/'protocol.json', protocol)
        save_json(out/'cohort.json', rows)
        save_json(out/'controls.json', synthetic_controls())
    all_data = []
    resources = []
    for d, p in zip(days, paths):
        space_guard()
        target = out/(d+'.npz')
        selected = [r for r in rows if r['session'] == d]
        blocks = sorted({r['block'] for r in selected})
        if target.exists():
            assert digest(target) == json.loads(target.with_suffix('.sha256.json').read_text())['sha256']
        else:
            mat = loadmat(p, variable_names=['tx1'])['tx1'].ravel()
            assert len(mat) == len(selected)
            hist = np.zeros((128, TAIL+1), dtype=np.int64)
            mix = np.zeros_like(hist, dtype=float)
            basic = np.zeros((len(blocks), 128, 4))
            local = np.zeros((len(blocks), len(WINDOWS), 128, 7))
            trial_disp = np.zeros((128, 2))
            discarded = np.zeros(len(WINDOWS), dtype=np.int64)
            trial_table = []
            max_count = 0
            for row in selected:
                raw = np.asarray(mat[row['index']])
                x = raw[:, :128]
                assert x.shape == (row['bins'], 128)
                if not np.isfinite(x).all() or (x < 0).any() or not np.equal(x, np.rint(x)).all():
                    raise ValueError('Count-valued feature validation failed; do not clean')
                x = x.astype(np.int64)
                max_count = max(max_count, int(x.max()))
                b = blocks.index(row['block'])
                basic[b] += moments(x)
                hist += histogram(x)
                mix += expected_hist(x.mean(0), len(x))
                trial_disp += local_stats(x, len(x))[:, :2]
                for j, n in enumerate(WINDOWS):
                    local[b, j] += local_stats(x, n)
                    discarded[j] += len(x) % n
                trial_table.append([row['index'], row['block'], len(x), int(x.sum()), int((x == 0).sum())])
            assert np.all(hist.sum(1) == basic[:, :, 0].sum(0))
            assert np.allclose(mix.sum(1), hist.sum(1), atol=1e-6)
            np.savez_compressed(target, basic=basic, local=local, hist=hist, mixture=mix,
                trial_disp=trial_disp, blocks=blocks, discarded=discarded,
                trial_table=trial_table, max_count=max_count)
            save_json(target.with_suffix('.sha256.json'), dict(sha256=digest(target)))
            del mat
        with np.load(target) as z:
            all_data.append({k: z[k].copy() for k in z.files})
        resources.append(dict(day=d, free_bytes=space_guard(), rss_bytes=psutil.Process().memory_info().rss))
        print('collected', d, len(selected), 'trials', flush=True)
    save_json(out/'resources.json', resources)
    analyze(out, days, all_data)


def analyze(out, days, data):
    pooled, zeros, means, tv, mix_tv, trial, local, adjacent, ans = [], [], [], [], [], [], [], [], []
    for d in data:
        b = d['basic'].sum(0); n, s, ss, z = b.T
        mu = s/n; var = ss/n-mu*mu
        ph = expected_hist(mu, n[0])/n[:, None]
        observed = d['hist']/n[:, None]
        pooled.append(ratio(var, mu)); means.append(mu); zeros.append(z/n-np.exp(-mu))
        tv.append(.5*np.abs(observed-ph).sum(1))
        mix_tv.append(.5*np.abs(observed-d['mixture']/n[:, None]).sum(1))
        trial.append(ratio(d['trial_disp'][:, 0], d['trial_disp'][:, 1]))
        l = d['local'].sum(0)
        local.append(ratio(l[:, :, 0], l[:, :, 1]))
        adjacent.append(ratio(l[:, :, 2], 2*l[:, :, 1]))
        ans.append(ratio(l[:, :, 3], l[:, :, 4]))
    values = {k: np.array(v) for k, v in dict(pooled=pooled, zero_excess=zeros, mean=means,
        TV=tv, mixture_TV=mix_tv, trial_dispersion=trial, local=local, adjacent=adjacent,
        anscombe_variance=ans).items()}
    rng = np.random.default_rng(SEED); draws = np.empty((2000, len(WINDOWS)))
    for rep in range(2000):
        samples = []
        for d in data:
            ix = rng.integers(0, len(d['blocks']), len(d['blocks']))
            l = d['local'][ix].sum(0)
            samples.append(ratio(l[:, :, 0], l[:, :, 1]))
        draws[rep] = np.nanmedian(np.stack(samples), axis=(0, 2))
    result = dict(days=len(days), trials=sum(len(d['trial_table']) for d in data),
        bins=int(sum(d['basic'][0, 0, 0]+d['basic'][1:, 0, 0].sum() for d in data)),
        count_observations=int(sum(d['basic'][:, :, 0].sum() for d in data)),
        maximum_count=int(max(d['max_count'] for d in data)),
        count_validity='All checked observations finite, nonnegative, exactly integer; none cleaned',
        zero_mean_day_channels=int(np.sum(values['mean'] == 0)),
        rate=numeric_summary(values['mean']), pooled_fano=numeric_summary(values['pooled']),
        zero_excess=numeric_summary(values['zero_excess']), day_Poisson_TV=numeric_summary(values['TV']),
        fitted_trial_mixture_TV=numeric_summary(values['mixture_TV']),
        within_trial_dispersion=numeric_summary(values['trial_dispersion']), windows={})
    for j, n in enumerate(WINDOWS):
        v = values['local'][:, j]
        result['windows'][str(n)] = dict(bin_ms=n*20, dispersion=numeric_summary(v),
            median_block_CI=np.quantile(draws[:, j], [.025, .975]).tolist(),
            fraction_in_descriptive_band=float(np.mean((v >= .8) & (v <= 1.2))),
            adjacent_difference_ratio=numeric_summary(values['adjacent'][:, j]),
            anscombe_variance=numeric_summary(values['anscombe_variance'][:, j]),
            usable_windows=int(sum(d['local'][:, j, 0, 5].sum() for d in data)),
            zero_count_channel_windows=int(sum((d['local'][:, j, :, 5]-d['local'][:, j, :, 6]).sum() for d in data)),
            remainder_bins=int(sum(d['discarded'][j] for d in data)))
    save_json(out/'metrics.json', result)
    np.savez_compressed(out/'diagnostics.npz', **values, bootstrap=draws)
    with (out/'day-channel-metrics.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['day', 'channel', 'mean_count_20ms', 'pooled_fano', 'zero_excess',
                         'TV_day', 'TV_trial_mixture']+[f'local_{n*20}ms' for n in WINDOWS])
        for i, day in enumerate(days):
            for c in range(128):
                writer.writerow([day, c, values['mean'][i,c], values['pooled'][i,c],
                    values['zero_excess'][i,c], values['TV'][i,c], values['mixture_TV'][i,c],
                    *values['local'][i,:,c]])
    save_json(out/'status.json', dict(status='collected', completed_local=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=ROOT/'research/data/manifest.json')
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
