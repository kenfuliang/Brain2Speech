"""Final source/display/projection QA and technical notebook audit for E019."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.audit import digest,save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run
    assert not (run/'seal.json').exists()
    verification=json.loads((run/'independent-verification.json').read_text())
    assert verification['passed'] and verification['scalar_distances_recomputed']==24000
    protocol=json.loads((run/'protocol.json').read_text());m=json.loads((run/'metrics.json').read_text())
    coverage=json.loads((run/'projection-coverage.json').read_text());rows=json.loads((run/'cohort.json').read_text())
    with np.load(Path(protocol['parent'])/'summaries.npz') as old:
        axes=old['pca_axes'].copy();center=old['pca_mean'].copy()
    with np.load(run/'summaries.npz') as summary:
        for key in ['G0','Gd','D0','Dd']:
            hidden=summary[key+'_latent_mean'];z=(hidden-center)@axes.T
            retained=float(z.var(0).sum()/hidden.var(0).sum())
            assert np.isclose(retained,coverage['conditions'][key]['E018_PCA32_variance_fraction'])
            means=np.stack([hidden[[r['session']==day for r in rows]].mean(0) for day in protocol['days']])
            matrix=np.sqrt(((means[:,None]-means[None,:])**2).mean(-1))
            assert np.allclose(matrix,coverage['conditions'][key]['full1024_centroid_distance_matrix'])
    observations=[json.loads(p.read_text()) for p in (run/'resources').glob('*.json')]
    assert observations and all(p['device']=='mps' for p in observations)
    assert min(p['free_storage_bytes'] for p in observations)>=30*1024**3
    metadata=[json.loads(p.read_text()) for p in (run/'trials').rglob('*.json')]
    assert len(metadata)==18000 and all(p['device']=='mps' for p in metadata)
    notebook=json.loads((run/'inspect-evidence.executed.ipynb').read_text())
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert [c['execution_count'] for c in cells]==[1,2,3]
    assert all(c['outputs'] and all(o['output_type']=='stream' for o in c['outputs']) for c in cells)
    provenance=json.loads((run/'display-provenance.json').read_text())
    assert provenance['metrics_sha256']==digest(run/'metrics.json')
    images={}
    for name,sha in provenance['files'].items():
        path=run/'figures-v02'/name;assert digest(path)==sha
        with Image.open(path) as image:image.verify()
        with Image.open(path) as image:images[name]=image.size
    assert {'factorial-summary.png','per-by-day.png','study-architecture-v02.png'}<=set(images)
    files=[Path(__file__),ROOT/'reproduction/verify_normalization_adapters.py',
        ROOT/'reproduction/validate_projection_coverage.py',ROOT/'reproduction/draw_normalization_adapter_architecture.py',
        ROOT/'reproduction/refine_factorial_display.py',
        ROOT/'reproduction/execute_drift_notebook.py',ROOT/'research/tests/test_normalization_adapters.py']
    result=dict(passed=True,trials=6000,new_inferences=18000,all_MPS=True,
        original_parent_preserved_files=verification['parent_preserved_files'],
        projection_coverage_and_full1024_matrices_recomputed=True,plain_python_notebook_executed=True,
        figures=images,source_hashes={str(p.relative_to(ROOT)):digest(p) for p in files},
        resources=dict(observations=len(observations),
            sampled_maxima={k:max(p[k] for p in observations) for k in ['rss_bytes','mps_allocated_bytes','mps_driver_bytes']},
            minimum_observed_free_bytes=min(p['free_storage_bytes'] for p in observations),
            last_full_collection_sample_seconds=max(p['elapsed_seconds'] for p in observations if p['total']==18000),
            summed_new_trial_processing_seconds=sum(p['trial_processing_seconds'] for p in metadata),
            caveat='Sampled memory not lifetime peaks; trial timers include summaries/edit accounting; not total pipeline wall time'),
        assessment='Share with retrospective/transductive/training-partition caveats',
        no_WER_claim=True,no_biological_drift_claim=True,
        visual_QA='Final PNGs inspected; full app report rendering must be stated separately')
    assert not (run/'final-QA.json').exists()
    save_json(run/'final-QA.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
