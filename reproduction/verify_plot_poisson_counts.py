"""Independent E020 scalar spot checks, full accounting, and static evidence figure."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.stats import poisson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.audit import digest, save_json, space_guard


def verify(out):
    protocol = json.loads((out/'protocol.json').read_text())
    manifest = json.loads(Path(protocol['manifest']).read_text())
    assert digest(protocol['manifest']) == protocol['manifest_sha256']
    assert digest(ROOT/'reproduction/analyze_poisson_counts.py') == protocol['implementation_sha256']
    rows = json.loads((out/'cohort.json').read_text())
    days = manifest['folds']['initial_train']+manifest['folds']['development']
    assert not set(days) & set(manifest['folds']['final'])
    assert len(rows) == len({r['id'] for r in rows}) == 6680
    metrics = json.loads((out/'metrics.json').read_text())
    assert metrics['bins'] == sum(r['bins'] for r in rows) == 2178002
    assert metrics['count_observations'] == metrics['bins']*128
    all_local, all_fano = [], []
    examples = []
    trial_total = 0
    for i, day in enumerate(days):
        p = ROOT/manifest['sessions'][day]['train']['path']
        assert digest(p) == protocol['source_hashes'][str(p)]
        selected = [r for r in rows if r['session'] == day]
        archive = out/(day+'.npz')
        assert digest(archive) == json.loads(archive.with_suffix('.sha256.json').read_text())['sha256']
        with np.load(archive) as z:
            basic, local, table, blocks = z['basic'], z['local'], z['trial_table'], z['blocks']
            n = basic[:, :, 0].sum(0)
            s = basic[:, :, 1].sum(0); ss = basic[:, :, 2].sum(0)
            assert np.all(n == sum(r['bins'] for r in selected))
            assert np.array_equal(z['hist'].sum(1), n)
            assert np.allclose(z['mixture'].sum(1), n, atol=1e-7)
            assert np.array_equal(table[:, 0], [r['index'] for r in selected])
            assert np.array_equal(table[:, 1], [r['block'] for r in selected])
            assert np.array_equal(table[:, 2], [r['bins'] for r in selected])
            assert int(table[:, 3].sum()) == int(s.sum())
            assert int(table[:, 4].sum()) == int(basic[:, :, 3].sum())
            all_fano.extend(((ss/n-(s/n)**2)/(s/n)).tolist())
            a = local.sum(0)
            all_local.append(a[:, :, 0]/a[:, :, 1])
            for j, length in enumerate(protocol['windows_bins']):
                windows = sum(r['bins']//length for r in selected)
                assert np.all(a[j, :, 5] == windows)
                assert int(z['discarded'][j]) == sum(r['bins'] % length for r in selected)
                assert windows*length+int(z['discarded'][j]) == int(n[0])
            # Recompute each day's complete raw moments by an alternate einsum path.
            mat = loadmat(p, variable_names=['tx1'])['tx1'].ravel()
            direct_s = np.zeros(128); direct_ss = np.zeros(128); direct_z = np.zeros(128)
            for row in selected:
                x = np.asarray(mat[row['index']])[:, :128].astype(float)
                direct_s += np.einsum('tc->c', x)
                direct_ss += np.einsum('tc,tc->c', x, x)
                direct_z += np.count_nonzero(x == 0, axis=0)
            assert np.array_equal(s, direct_s) and np.array_equal(ss, direct_ss)
            assert np.array_equal(basic[:, :, 3].sum(0), direct_z)
            # Independently scalar-recompute one complete trial (all channels/scales).
            row = selected[len(selected)//2]
            x = np.asarray(mat[row['index']])[:, :128]
            expected = []
            for length in protocol['windows_bins']:
                output = np.zeros((128, 4))
                for channel in range(128):
                    for start in range(0, len(x)-length+1, length):
                        v = [int(y) for y in x[start:start+length, channel]]
                        mean = sum(v)/length
                        av = [2*math.sqrt(y+.375) for y in v]; am = sum(av)/length
                        output[channel] += [sum((y-mean)**2 for y in v),
                            (length-1)*mean, sum((v[k+1]-v[k])**2 for k in range(length-1)),
                            sum((y-am)**2 for y in av)]
                expected.append(output)
            from analyze_poisson_counts import local_stats
            for j, length in enumerate(protocol['windows_bins']):
                assert np.allclose(expected[j], local_stats(x, length)[:, :4], atol=1e-8)
            examples.append(row['id']); trial_total += len(selected)
            del mat
    assert trial_total == 6680
    assert np.isclose(np.median(all_fano), metrics['pooled_fano']['median'])
    a = np.array(all_local)
    for j, n in enumerate(protocol['windows_bins']):
        assert np.isclose(np.median(a[:, j]), metrics['windows'][str(n)]['dispersion']['median'])
    # Alternate vectorized bootstrap (same block draws) rederives all four intervals.
    rng = np.random.default_rng(101); indices = [[] for _ in days]
    for rep in range(2000):
        for i, day in enumerate(days):
            with np.load(out/(day+'.npz')) as z:
                k = len(z['blocks'])
            indices[i].append(rng.integers(0, k, k))
    bootstrap_days = []
    for i, day in enumerate(days):
        with np.load(out/(day+'.npz')) as z:
            # Counts matrix instead of selected-block summation.
            k = len(z['blocks']); counts = np.stack([np.bincount(ix, minlength=k) for ix in indices[i]])
            num = np.einsum('rb,bwc->rwc', counts, z['local'][:, :, :, 0])
            den = np.einsum('rb,bwc->rwc', counts, z['local'][:, :, :, 1])
            bootstrap_days.append(num/den)
    bs = np.median(np.stack(bootstrap_days, axis=1), axis=(1, 3))
    for j, n in enumerate(protocol['windows_bins']):
        assert np.allclose(np.quantile(bs[:, j], [.025,.975]), metrics['windows'][str(n)]['median_block_CI'])
    save_json(out/'verification.json', dict(status='pass', source_hashes=19, trials=trial_total,
        bins=metrics['bins'], raw_moments='All 278,784,256 count values independently rescanned',
        scalar_examples=examples, bootstrap='2000 alternate count-matrix draws match four intervals',
        undefined_rate_handling='No zero-mean day/channel; zero local windows retained in accounting',
        inference_limits='No conditional speech-repeat test, independence proof, WER or calibration benefit'))
    print('Independent verification passed', flush=True)


def plot(out):
    p = json.loads((out/'protocol.json').read_text()); m = json.loads((out/'metrics.json').read_text())
    v = np.load(out/'diagnostics.npz'); rows = list(csv.DictReader((out/'day-channel-metrics.csv').open()))
    # Explicitly selected worst day-Poisson fit; this is an exception illustration.
    worst = max(rows, key=lambda r: float(r['TV_day']))
    blue, orange = '#2563A6', '#C86D26'
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.8))
    fig.subplots_adjust(top=.86, bottom=.20, hspace=.52, wspace=.3)
    fig.suptitle('Poisson adequacy of recorded count features', x=.08, ha='left', fontsize=21, fontweight='bold', y=.97)
    fig.text(.08,.925,'6,680 training trials · 19 days · 128 channels · 278.8M unsmoothed 20 ms count observations',fontsize=11)
    fig.text(.08,.90,'Descriptive fitted checks; short windows approximate constant rates, not known phoneme-conditioned repeats.',fontsize=10,color='#555555')
    ax = axes[0,0]
    for key, label, color, style in [('TV','One rate per day/channel',blue,'-'),('mixture_TV','One rate per trial/channel',orange,'--')]:
        x = np.sort(v[key].ravel()); ax.plot(x, np.arange(1,len(x)+1)/len(x), label=label,color=color,linestyle=style)
    ax.set_xscale('log'); ax.set_xlim(.00005,.6); ax.set_ylim(0,1.02)
    ax.set_title('Count-histogram mismatch',loc='left',fontweight='bold')
    ax.set_xlabel('Total variation distance (log scale; smaller is closer)'); ax.set_ylabel('Fraction of 2,432 day/channel pairs')
    ax.legend(frameon=False,fontsize=9,loc='lower right'); ax.grid(axis='y',alpha=.15)
    ax = axes[0,1]
    x = np.array(p['windows_bins'])*20; med=[]; lo=[]; hi=[]; qlo=[]; qhi=[]
    for n in p['windows_bins']:
        r=m['windows'][str(n)]; med.append(r['dispersion']['median']); lo.append(r['median_block_CI'][0]); hi.append(r['median_block_CI'][1]); qlo.append(r['dispersion']['q10']); qhi.append(r['dispersion']['q90'])
    ax.fill_between(x,qlo,qhi,color=blue,alpha=.12,label='10–90% of day/channel values')
    ax.errorbar(x,med,yerr=[np.array(med)-lo,np.array(hi)-med],color=blue,marker='o',capsize=5,label='Median; conditional 95% block interval')
    ax.axhline(1,color='#444444',linestyle='--',label='Constant-rate independent Poisson expectation')
    ax.set_ylim(.65,1.2); ax.set_title('Local variance / mean',loc='left',fontweight='bold')
    ax.set_xlabel('Window duration (ms)'); ax.set_ylabel('Degrees-of-freedom corrected dispersion')
    ax.legend(frameon=False,fontsize=8,loc='lower right'); ax.grid(axis='y',alpha=.15)
    ax=axes[1,0]
    with np.load(out/(worst['day']+'.npz')) as z:
        c=int(worst['channel']); total=z['hist'][c].sum()
        observed=z['hist'][c]/total; mix=z['mixture'][c]/total
    k=np.arange(30); expected=poisson.pmf(k,float(worst['mean_count_20ms']))
    ax.plot(k,observed[:30],color=blue,marker='o',markersize=3,label='Observed counts')
    ax.plot(k,expected,color='#444444',linestyle=':',label='Fitted day Poisson')
    ax.plot(k,mix[:30],color=orange,linestyle='--',label='Fitted trial-rate mixture')
    ax.set_xlim(-.5,16); ax.set_ylim(0,.9)
    ax.set_title('Exception: July 27, channel 117 (0-based)',loc='left',fontweight='bold')
    ax.set_xlabel('Count in one 20 ms bin'); ax.set_ylabel('Probability')
    ax.legend(frameon=False,fontsize=9)
    ax.text(.97,.6,f"Selected largest day-fit mismatch\nTV: {float(worst['TV_day']):.3f} → {float(worst['TV_trial_mixture']):.3f}\nNot a representative-channel selection",transform=ax.transAxes,ha='right',fontsize=9)
    ax=axes[1,1]
    mu=np.linspace(0,2.5,300); support=np.arange(80)
    prob=poisson.pmf(support[None,:],mu[:,None]); transform=2*np.sqrt(support+.375)
    variance=prob@(transform*transform)-(prob@transform)**2
    ax.scatter(v['mean'].ravel(),v['anscombe_variance'][:,0].ravel(),s=6,alpha=.2,color=blue,label='Observed local variance; 200 ms windows')
    ax.plot(mu,variance,color=orange,linestyle='--',label='Exact variance under constant-rate Poisson')
    ax.axhline(1,color='#444444',linestyle=':')
    ax.set_xlim(0,2.5); ax.set_ylim(0,1.35)
    ax.set_title('Anscombe: sparse counts are not unit-variance',loc='left',fontweight='bold')
    ax.set_xlabel('Day/channel mean count per 20 ms'); ax.set_ylabel('Variance after 2√(count + 3/8)')
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    fig.text(.08,.07,'Whole blocks are resampled within these fixed days; bins and channels are not independent inference units.\nLocal windows omit trial-end remainders (accounted separately); all observations remain in day/trial histograms.\nPower features, final days, model training and decoding are excluded. Closer distribution fits do not establish drift tolerance.',fontsize=10,color='#555555')
    target=out/'poisson-diagnostics-v02.png'; fig.savefig(target,dpi=160); plt.close(fig)
    save_json(out/'figure-provenance-v02.json',dict(source_sha256={str(out/name):digest(out/name) for name in ['protocol.json','metrics.json','diagnostics.npz','day-channel-metrics.csv']},
        implementation_sha256=digest(__file__),image_sha256=digest(target),exception=worst,
        palette=[blue,orange],not_measured='WER, calibration reduction, actual conditional latent rates'))
    print('Saved figure',target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();space_guard();verify(args.output);plot(args.output)
