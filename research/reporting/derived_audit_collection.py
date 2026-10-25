"""Fresh derived-scoring observer; retains legacy audits and raw outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from research.audit import ROOT, digest, save_json
from research.reporting.audit_collection import expected_ids, process_live
from research.reporting import derived_verification
from research import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--interval',type=float,default=60.)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing evidence; use a fresh observer output')
    manifest_path = ROOT/'research/data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    identities = {str(p):digest(p) for p in [manifest_path, Path(__file__),
        Path(derived_verification.__file__), Path(metrics.__file__),
        Path(expected_ids.__code__.co_filename)]}
    save_json(args.output/'method.json',{'schema_version':2,'identities':identities,
        'pipeline_pid':args.pid,'root':str(args.root.resolve()),
        'scope':'Derived whitespace scoring; no raw neural reads, no training, no source artifact changes; development rate changes fail closed'})
    seen = {}
    try:
        while True:
            if any(digest(Path(p))!=sha for p,sha in identities.items()):
                raise ValueError('Derived audit method/data identity changed')
            for complete in sorted(args.root.glob('**/complete.json')):
                record = json.loads(complete.read_text())
                if 'ids' not in record.get('identity',{}):
                    continue
                relative = complete.parent.relative_to(args.root)
                sha = digest(complete)
                key = (str(relative),sha)
                if key in seen:
                    continue
                result = derived_verification.verify(complete.parent,expected_ids(relative,manifest))
                if relative.parts[0] in ['development','development-adaptation'] and result['rate_changed_trials']:
                    raise ValueError('Development rate changed; frozen selection requires explicit review')
                result.update(relative_source=str(relative),audited_utc=datetime.now(timezone.utc).isoformat())
                name = hashlib.sha256((str(relative)+'\n'+sha).encode()).hexdigest()
                destination = args.output/'evaluations'/f'{name}.json'
                if destination.exists():
                    raise ValueError('Preserve existing derived audit evidence')
                save_json(destination,result)
                seen[key] = (result['component_changed_trials'],result['rate_changed_trials'])
                save_json(args.output/'status.json',{'completed_evaluations_verified':len(seen),
                    'component_changed_trials':sum(v[0] for v in seen.values()),
                    'rate_changed_trials':sum(v[1] for v in seen.values()),
                    'status':'auditing-derived-completed-outputs','matrix_complete':False,
                    'scope':'Derived metrics only; corrected downstream comparison/report integration, coverage and scientific review remain required'})
            if not process_live(args.pid):
                break
            time.sleep(args.interval)
        save_json(args.output/'observer-ended.json',{'reason':'Target pipeline PID absent or reused',
            'verified_evaluations':len(seen),'matrix_complete':False})
    except Exception as error:
        save_json(args.output/'failure.json',{'error':str(error),'scientific_acceptance_allowed':False,
            'action':'Preserve artifacts and inspect; do not infer completed scientific acceptance'})
        raise


if __name__=='__main__':
    main()
