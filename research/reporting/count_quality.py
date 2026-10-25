"""Training-only descriptive count-dispersion audit; never changes fitted models."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from research.audit import ROOT, digest, save_json, space_guard


def moments(arrays):
    n, total, squares, zeros = 0, np.zeros(128), np.zeros(128), np.zeros(128)
    trial_fano = []
    for array in arrays:
        x = np.asarray(array[:, :128], dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 128 or not len(x):
            raise ValueError('Expected nonempty bins by 128 count channels')
        if not np.isfinite(x).all() or (x < 0).any() or not np.equal(x, np.round(x)).all():
            raise ValueError('Count targets must be finite, nonnegative integers')
        mean, variance = x.mean(0), x.var(0)
        trial_fano.append(np.divide(variance, mean, out=np.full(128, np.nan), where=mean>0))
        n += len(x)
        total += x.sum(0)
        squares += (x*x).sum(0)
        zeros += (x==0).sum(0)
    if not n:
        raise ValueError('No count bins')
    mean = total/n
    variance = np.maximum(squares/n-mean*mean, 0.)
    fano = np.divide(variance, mean, out=np.full(128, np.nan), where=mean>0)
    valid = fano[np.isfinite(fano)]
    within = np.concatenate(trial_fano)
    within = within[np.isfinite(within)]
    return {'bins':n, 'trials':len(trial_fano), 'channels':128,
            'valid_pooled_channels':len(valid), 'pooled_fano_median':float(np.median(valid)) if len(valid) else None,
            'pooled_fano_p10_p90':np.quantile(valid,[.1,.9]).tolist() if len(valid) else None,
            'pooled_channels_fano_above_one':int((valid>1).sum()),
            'within_trial_channel_fano_median':float(np.median(within)) if len(within) else None,
            'within_trial_valid_cells':len(within),
            'within_trial_cells_fano_above_one_fraction':float(np.mean(within>1)) if len(within) else None,
            'mean_by_channel':mean.tolist(), 'variance_by_channel':variance.tolist(),
            'fano_by_channel':[float(v) if np.isfinite(v) else None for v in fano],
            'zero_fraction_by_channel':(zeros/n).tolist()}


def collect(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve existing audit; choose a new output path')
    manifest_path = ROOT/'research/data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    daily, sources, pooled = {}, {}, []
    for day in manifest['folds']['initial_train']:
        space_guard()
        entry = manifest['sessions'][day]['train']
        source = (ROOT/entry['path']).resolve()
        source.relative_to((ROOT/'research/data').resolve())
        if digest(source) != entry['sha256']:
            raise ValueError('Frozen raw source changed: '+str(source))
        sources[entry['path']] = entry['sha256']
        values = loadmat(source, variable_names=['tx1'])['tx1'].ravel()
        arrays = [values[r['index']][:,:128] for r in entry['trials'] if r['text']]
        daily[day] = moments(arrays)
        pooled.extend(arrays)
    result = {'manifest_sha256':digest(manifest_path), 'source_sha256':sources,
              'method_sha256':digest(Path(__file__)), 'fold':'initial_train/train only',
              'grain':'20 ms bin / threshold-crossing channel; within-trial cells are trial-channel pairs',
              'definition':'Population variance divided by mean, excluding zero-mean channels from ratios',
              'limitations':'Descriptive marginal dispersion includes task modulation, correlations and day mixtures; not a test of generator-conditioned Poisson fit or an independent-sample confidence interval.',
              'implication':'Keep the frozen Poisson working reconstruction objective; audit conditional residuals before interpreting it as a calibrated likelihood. Negative-binomial comparison belongs to a separately specified iteration.',
              'daily':daily, 'pooled':moments(pooled), 'final_neural_data_opened':False,
              'report_surface':'Canonical Markdown comparative report; exact audit table, no additional report/chart'}
    save_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    result = collect(parser.parse_args().output)
    print(json.dumps({k:v for k,v in result['pooled'].items() if not k.endswith('by_channel')},indent=2))
