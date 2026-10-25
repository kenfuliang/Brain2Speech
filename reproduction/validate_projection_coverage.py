"""Secondary E019 measurement-validity check, not a new optimized model."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.audit import digest,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    assert not (run/'seal.json').exists()
    protocol=json.loads((run/'protocol.json').read_text());rows=json.loads((run/'cohort.json').read_text())
    with np.load(Path(protocol['parent'])/'summaries.npz') as prior:
        axes=prior['pca_axes'].copy();center=prior['pca_mean'].copy()
    conditions={}
    with np.load(run/'summaries.npz') as summary:
        for key in ['G0','Gd','D0','Dd']:
            hidden=summary[key+'_latent_mean']
            z=(hidden-center)@axes.T
            total=float(hidden.var(0).sum());captured=float(z.var(0).sum())
            means=np.stack([hidden[[r['session']==day for r in rows]].mean(0) for day in protocol['days']])
            matrix=np.sqrt(np.mean((means[:,None]-means[None,:])**2,axis=-1))
            conditions[key]=dict(E018_PCA32_variance_fraction=captured/total,
                full1024_centroid_distance_matrix=matrix.tolist(),
                full1024_median_between_day_RMS=float(np.median(matrix[np.triu_indices(17,1)])))
    target=run/'projection-coverage.json';assert not target.exists()
    save_json(target,dict(purpose='Secondary measurement QA; no refitting/selection and no training',
        conditions=conditions,script_sha256=digest(Path(__file__)),
        summaries_sha256=digest(run/'summaries.npz'),
        interpretation='If projection coverage falls, PCA-only stability claims weaken; full-space shifts remain descriptive'))
    print(json.dumps({k:{f:v[f] for f in ['E018_PCA32_variance_fraction','full1024_median_between_day_RMS']}
        for k,v in conditions.items()},indent=2))


if __name__=='__main__':main()
