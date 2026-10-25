"""Final E018 display/resource/source QA before immutable sealing."""
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
    verification=json.loads((run/'independent-verification-bootstrap.json').read_text())
    assert verification['passed'] and len(verification['checks'])==7
    provenance=json.loads((run/'figure-provenance-v02.json').read_text())
    metrics=json.loads((run/'metrics.json').read_text())
    assert provenance['metrics_sha256']==digest(run/'metrics.json')
    assert provenance['summaries_sha256']==digest(run/'summaries.npz')
    images={}
    for name,sha in provenance['files'].items():
        path=run/'figures-v02'/name
        assert digest(path)==sha
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:images[name]=image.size
    assert len(images)==4
    observed=[json.loads(p.read_text()) for p in (run/'resources').glob('*.json')]
    assert len(observed)==171 and all(v['device']=='mps' for v in observed)
    assert min(v['free_storage_bytes'] for v in observed)>=30*1024**3
    trials=[json.loads(p.read_text()) for p in (run/'trials').rglob('*.json')]
    assert len(trials)==6680 and all(v['device']=='mps' for v in trials)
    notebook=json.loads((run/'inspect-evidence.executed.ipynb').read_text())
    code=[v for v in notebook['cells'] if v['cell_type']=='code']
    assert [v['execution_count'] for v in code]==[1,2,3]
    assert all(v['outputs'] and all(o['output_type']=='stream' for o in v['outputs']) for v in code)
    repetitions=json.loads((run/'repeated-content.json').read_text())
    content_summary={tag:{key:float(np.median([q['distances'][key] for q in repetitions['pairs'] if q['cross_day']==cross]))
        for key in ('counts','power','latent')} for tag,cross in [('cross_day',True),('within_day',False)]}
    result=dict(passed=True,quantitative_figures=images,all_MPS=True,
        ratio_between_over_within={k:v['median_between']/v['median_within'] for k,v in metrics['statistics'].items()},
        repeated_content_descriptive_medians=content_summary,
        resources=dict(observations=len(observed),
            sampled_maxima={k:max(v[k] for v in observed) for k in ('rss_bytes','mps_allocated_bytes','mps_driver_bytes')},
            minimum_observed_free_bytes=min(v['free_storage_bytes'] for v in observed),
            last_collection_sample_elapsed_seconds=max(v['elapsed_seconds'] for v in observed if v['total']==6680),
            summed_trial_inference_seconds=sum(v['inference_seconds'] for v in trials),
            caveat='Sampled memory, not peaks; inference timers exclude raw loading/statistical analysis and preparation'),
        notebook_plain_python_execution_passed=True,
        visual_QA='Four final quantitative PNGs and study architecture inspected in tool image viewer; labels readable; full app-panel rendering not claimed',
        external_source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [
            Path(__file__),ROOT/'reproduction/verify_neural_drift.py',ROOT/'reproduction/execute_drift_notebook.py',
            ROOT/'reproduction/refine_drift_figures.py',ROOT/'reproduction/draw_neural_drift_architecture.py',
            ROOT/'research/tests/test_neural_drift.py']},
        architecture_sha256=digest(run/'figures/study-architecture.png'),
        limitations=metrics['limitations'])
    assert not (run/'final-QA.json').exists()
    save_json(run/'final-QA.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
