"""CPU-only repeated-sentence illustration; excludes final recordings."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter

from analyze_neural_drift import ROOT, cohort, trial_raw, KERNEL
from research.audit import digest, save_json, space_guard, normalize_text


def main(out):
    assert not out.exists(), 'Use a new immutable output directory'
    space_guard()
    manifest_path = ROOT / 'research/data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    days, rows = cohort(manifest)
    groups = defaultdict(list)
    word_days = defaultdict(set)
    for row in rows:
        groups[row['text']].append(row)
        for word in set(re.findall(r"[a-z]+(?:'[a-z]+)?", row['text'].lower())):
            word_days[word].add(row['session'])
    repeated = {text: rs for text, rs in groups.items() if len({r['session'] for r in rs}) > 1}
    assert len(repeated) == 77
    candidates = [(abs(math.log(rs[0]['bins']/rs[1]['bins'])), text, rs)
                  for text, rs in repeated.items() if len(rs) == 2]
    mismatch, prompt, selected = min(candidates, key=lambda x: (x[0], x[1]))
    selected = sorted(selected, key=lambda r: (r['session'], r['index']))
    assert not {r['session'] for r in selected} & set(manifest['folds']['final'])
    out.mkdir(parents=True)
    contract = dict(question='How do recorded features look for the same sentence on two days?',
                    takeaway='Descriptive paired illustration; not biological drift evidence or word-aligned responses',
                    family='Matrix / small-multiple heatmaps', renderer='Static Matplotlib PNG for canonical research report',
                    selection='Smallest absolute log duration ratio among exactly two-trial cross-day repeats; lexical tie-break',
                    prompt=prompt, selected=selected, eligible_trials=len(rows), eligible_days=len(days),
                    cross_day_prompts=len(repeated), max_days_per_repeated_prompt=max(len({r['session'] for r in rs}) for rs in repeated.values()),
                    channel_order='Released columns0:128, unchanged',
                    counts='20ms threshold-crossing counts; trailing9 Gaussian sigma2; divide by.02 for crossings/s',
                    power='Released power units; same trailing filter; no log or per-day normalization',
                    time='Seconds from trial start; identical x limits, no time warping or word alignment',
                    color='Shared zero-to-maximum scales, no clipping; counts linear; power symmetric-log colour norm above100 released units',
                    palette='Hard two-root cap: blue counts, gold power; identical panel ordering and labels',
                    uncertainty='One trial per day; no inferential interval; same text does not guarantee matched timing',
                    boundary='CPU only; no training, model inference, decoder, or final recording access',
                    manifest_sha256=digest(manifest_path), script_sha256=digest(__file__))
    save_json(out / 'chart-contract.json', contract)
    save_json(out / 'coverage.json', dict(
        cross_day_sentences=[dict(text=t, trials=rs, days=len({r['session'] for r in rs})) for t, rs in sorted(repeated.items())],
        word_day_coverage=sorted([dict(word=w, days=len(ds)) for w, ds in word_days.items()], key=lambda x: (-x['days'], x['word'])),
        within_day_repeated_prompts=sum(any(n > 1 for n in Counter(r['session'] for r in rs).values()) for rs in groups.values())))
    arrays = {}
    source_hashes = {}
    stats = []
    plotted = []
    for i, row in enumerate(selected):
        source = ROOT / manifest['sessions'][row['session']]['train']['path']
        source_hashes[str(source)] = digest(source)
        assert source_hashes[str(source)] == manifest['sessions'][row['session']]['train']['sha256']
        mat = loadmat(source, variable_names=['tx1', 'spikePow', 'sentenceText'])
        assert normalize_text(str(mat['sentenceText'].ravel()[row['index']].item())) == row['text']
        raw = trial_raw(mat, row)
        arrays[f'raw_{i}'] = raw
        filtered = lfilter(KERNEL, [1.], raw, axis=0)
        assert filtered.shape == raw.shape and np.isfinite(filtered).all()
        # Independent scalar check of a channel/time point, including zero startup.
        for t in (0, 8, len(raw)-1):
            for ch in (0, 127, 128, 255):
                expected = sum(KERNEL[k] * raw[t-k, ch] for k in range(9) if t >= k)
                assert np.isclose(filtered[t, ch], expected, rtol=1e-12, atol=1e-12)
        arrays[f'smoothed_{i}'] = filtered
        plotted.append((filtered[:, :128] / .02, filtered[:, 128:]))
        stats.append(dict(id=row['id'], bins=len(raw), duration_seconds=len(raw)*.02,
                          count_mean=float(raw[:, :128].mean()), count_max=float(raw[:, :128].max()),
                          power_mean=float(raw[:, 128:].mean()), power_max=float(raw[:, 128:].max())))
    np.savez_compressed(out / 'paired-features.npz', **arrays)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    cmaps = [LinearSegmentedColormap.from_list('counts', ['#f5f8fc', '#3276ad', '#153d60']),
             LinearSegmentedColormap.from_list('power', ['#fffaf0', '#c59630', '#72541b'])]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.88, top=.79, bottom=.30, hspace=.28, wspace=.13)
    maximum_time = max(r['bins'] for r in selected)*.02
    maxima = [max(pair[k].max() for pair in plotted) for k in range(2)]
    for column, row in enumerate(selected):
        for feature in range(2):
            ax = axes[feature, column]
            scale = dict(vmin=0, vmax=maxima[feature]) if feature == 0 else dict(norm=SymLogNorm(linthresh=100, vmin=0, vmax=maxima[feature]))
            image = ax.imshow(plotted[column][feature].T, origin='lower', aspect='auto',
                              extent=[0, row['bins']*.02, .5, 128.5], cmap=cmaps[feature], interpolation='nearest', **scale)
            ax.set_xlim(0, maximum_time)
            ax.set_yticks([1, 32, 64, 96, 128])
            if column == 0:
                ax.set_ylabel(('Count' if feature == 0 else 'Power') + ' channel')
            if feature == 0:
                ax.set_title(f"{row['session'][4:]} • {row['bins']*.02:.2f} s\nTrain trial {row['index']:04d} • block {row['block']}", fontsize=12)
            else:
                ax.set_xlabel('Time from trial start (s)')
            if column == 1:
                color_ax = fig.add_axes([.90, .57 if feature == 0 else .31, .016, .20])
                bar = fig.colorbar(image, cax=color_ax)
                bar.set_label('Threshold crossings/s' if feature == 0 else 'Power (released units; log colour scale)')
                if feature == 1:
                    bar.set_ticks([0, 100, 1000, 10000, 50000])
                    bar.set_ticklabels(['0', '100', '1,000', '10,000', '50,000'])
    fig.text(.08, .95, 'Same sentence, two recording days', fontsize=20, fontweight='bold', color='#20303f')
    fig.text(.08, .90, f'“{prompt}”', fontsize=16, color='#20303f')
    fig.text(.08, .85, '128 channels per feature • 20 ms source bins • one trial/day • matched display scales', fontsize=11)
    fig.text(.97, .95, '✳', ha='right', fontsize=20, color='#3276ad')
    fig.text(.08, .065, 'Display: trailing 9-bin Gaussian smoothing; first 160 ms includes zero-initialized filter startup. No per-day rescaling.\n'
             'Counts use linear colours; power uses logarithmic colours above 100 released units. No observations clipped or removed.\n'
             'Trials begin at t=0, but words and speech onset are NOT aligned; timing and recording factors remain confounds.\n'
             'Example selected for closest duration, not maximum difference. Descriptive only; no biological-drift inference.\n'
             'Coverage: 77 exact cross-day repeated sentences among 6,680 train trials / 19 days; maximum 2 days per sentence.',
             fontsize=9.5, linespacing=1.5, color='#43515e')
    figure = out / 'same-sentence-neural-activity.png'
    fig.savefig(figure, dpi=170, facecolor='white')
    plt.close(fig)
    save_json(out / 'provenance.json', dict(status='generated-needs-visual-QA', contract_sha256=digest(out/'chart-contract.json'),
              source_hashes=source_hashes, stats=stats, duration_log_mismatch=mismatch,
              shared_color_limits=[[0, float(x)] for x in maxima],
              artifacts={p.name: digest(p) for p in (figure, out/'paired-features.npz', out/'coverage.json')},
              checks='Source hashes/text, cohort/final exclusion, finite/integer features, independent scalar filter checks'))
    print(json.dumps(dict(output=str(figure), prompt=prompt, stats=stats), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
