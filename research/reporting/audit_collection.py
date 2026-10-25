"""Independent audit of completed evaluations; never opens raw final neural data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from research.audit import ROOT, digest, save_json
from research.reporting import verification


def expected_ids(relative, manifest):
    parts = Path(relative).parts
    folds, sessions = manifest['folds'], manifest['sessions']
    def trial_ids(days):
        return [r['id'] for day in days for r in sessions[day]['test']['trials']]
    if parts[0]=='development':
        return trial_ids(folds['development'])
    if parts[0]=='development-adaptation':
        if parts[-1]!='evaluation' or parts[-2] not in folds['development']:
            raise ValueError('Unexpected development evaluation path')
        return trial_ids([parts[-2]])
    if parts[0]=='probes':
        return trial_ids(folds['initial_train'])
    if parts[0]=='final':
        if parts[3] not in folds['final']:
            raise ValueError('Unexpected final evaluation day')
        return trial_ids([parts[3]])
    if parts[0]=='sequential':
        if parts[-2] not in folds['final']:
            raise ValueError('Unexpected sequential update day')
        if parts[-1]=='fixed-probes':
            return trial_ids(folds['initial_train'])
        if parts[-1]=='transfer':
            return trial_ids(folds['final'])
    raise ValueError('Unexpected completed evaluator path: '+str(relative))


def scan(root, output, manifest, seen):
    for complete in sorted(root.glob('**/complete.json')):
        record = json.loads(complete.read_text())
        if 'ids' not in record.get('identity',{}):
            continue  # Training completion records are not evaluator artifacts.
        relative = complete.parent.relative_to(root)
        source_hash = digest(complete)
        key = (str(relative), source_hash)
        if key in seen:
            continue
        verified = verification.verify(complete.parent,expected_ids(relative,manifest))
        verified['relative_source'] = str(relative)
        verified['audited_utc'] = datetime.now(timezone.utc).isoformat()
        name = hashlib.sha256((str(relative)+'\n'+source_hash).encode()).hexdigest()
        destination = output/'evaluations'/f'{name}.json'
        if destination.exists():
            raise ValueError('Preserve existing audit evidence; duplicate destination')
        save_json(destination,verified)
        seen[key] = len(verified['component_corrections'])
    save_json(output/'status.json',{'completed_evaluations_verified':len(seen),
              'component_corrected_trials':sum(seen.values()),'all_audited_rates_unchanged':True,
              'status':'auditing-completed-outputs','matrix_complete':False,
              'scope':'Completed outputs only; full expected-matrix coverage and final scientific review remain separate gates'})


def process_live(pid):
    command = subprocess.run(['ps','-p',str(pid),'-o','command='],text=True,capture_output=True)
    if command.returncode not in [0,1]:
        raise RuntimeError('Process observation failed; not a terminal-process verdict')
    return ' -m research.experiment.pipeline ' in command.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--interval',type=float,default=60.)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing evidence; use a new audit output directory')
    manifest_path = ROOT/'research/data/manifest.json'
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    method_hash = digest(Path(verification.__file__))
    save_json(args.output/'method.json',{'monitor_sha256':digest(Path(__file__)),
              'verification_sha256':method_hash,'manifest_sha256':manifest_hash,
              'root':str(args.root.resolve()),'pipeline_pid':args.pid,
              'scope':'Artifact/metric audit only, no training or raw-final-data access; all original outputs preserved'})
    seen = {}
    try:
        while True:
            if digest(manifest_path)!=manifest_hash or digest(Path(verification.__file__))!=method_hash:
                raise ValueError('Audit method/data identity changed')
            scan(args.root,args.output,manifest,seen)
            if not process_live(args.pid):
                break
            time.sleep(args.interval)
        save_json(args.output/'observer-ended.json',{'reason':'Target pipeline PID absent or reused',
                  'verified_evaluations':len(seen),'matrix_complete':False})
    except Exception as error:
        save_json(args.output/'failure.json',{'error':str(error),'scientific_acceptance_allowed':False,
                  'action':'Inspect and repair this audit/metric failure before selection/result acceptance; original artifacts preserved'})
        raise


if __name__=='__main__':
    main()
