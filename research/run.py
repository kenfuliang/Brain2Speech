"""Serial, resumable initial-training stage; final days remain unopened."""
import argparse
import json
from pathlib import Path
import sys
import fcntl
import platform
import traceback

from research.audit import ROOT, digest, save_json, space_guard


def check_manifest(manifest):
    errors = []
    if manifest['causal_gate']['strict_gate'] != 'pass':
        errors.append(manifest['causal_gate']['reason'])
    folds = manifest['folds']
    ordered = folds['initial_train'] + folds['development'] + folds['final']
    if ordered != sorted(set(ordered)):
        errors.append('Chronological folds overlap or are out of order')
    if ordered != manifest['eligible_sessions']:
        errors.append('Folds do not account for every eligible session')
    for session in ordered:
        parts = manifest['sessions'][session]
        if 'train' not in parts or 'test' not in parts:
            errors.append(f'{session}: missing labeled partition')
        if max(r['block'] for r in parts['train']['trials']) >= min(
                r['block'] for r in parts['test']['trials']):
            errors.append(f'{session}: test does not follow calibration blocks')
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT / 'research/data/manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'research/runs/v1-initial')
    parser.add_argument('--stage', choices=['gate', 'smoke', 'initial-search'], default='gate')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    space_guard()
    errors = check_manifest(manifest)
    status = {'status': 'blocked' if errors else 'data-gate-passed',
              'manifest_sha256': digest(args.manifest), 'errors': errors,
              'training_started': False, 'seeds': [17, 29, 43],
              'acquisition': manifest['causal_gate']}
    if errors or args.stage == 'gate':
        save_json(args.output / 'status.json', status)
        print(json.dumps(status, indent=2))
        return 2 if errors else 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'manifest_sha256': digest(args.manifest), 'stage': args.stage,
                    'epochs': 1 if args.stage == 'smoke' else 30,
                    'selection': 'No final selection: development PER screen; fixed-WFST WER selection still required',
                    'source_hashes': {str(p.relative_to(ROOT)): digest(p)
                                      for p in sorted((ROOT/'research').glob('*.py'))}}
        evidence = args.output/'protocol.json'
        if evidence.exists() and json.loads(evidence.read_text()) != identity:
            raise ValueError('Run identity changed; use a new output directory')
        save_json(evidence, identity)
        try:
            from research.data import CausalFeatures, load_trials, prepare
            from research.train import device_for_run, fit
            import torch
            device = device_for_run()
            save_json(args.output/'environment.json', {'python': platform.python_version(),
                      'platform': platform.platform(), 'torch': torch.__version__, 'device': str(device),
                      'free_bytes_at_start': space_guard()})
            status.update(status='loading-initial-and-development', stage=args.stage, backend=str(device),
                          final_days_opened=False, remaining_stages=['fixed-WFST development selection',
                          'three-seed confirmation', 'daily adaptation', 'sequential replay', 'encoder analysis'])
            save_json(args.output/'status.json', status)
            training = load_trials(manifest, manifest['folds']['initial_train'], 'train')
            development = load_trials(manifest, manifest['folds']['development'], 'test')
            if args.stage == 'smoke':
                training, development = training[:16], development[:2]
            stats = CausalFeatures.fit(training).state()
            training, _ = prepare(training, stats)
            development, _ = prepare(development, stats)
            save_json(args.output/'sample-accounting.json', {
                'training_ids': [t['id'] for t in training],
                'development_ids': [t['id'] for t in development],
                'normalization_fit_ids': [t['id'] for t in training],
                'duplicate_policy': 'Retain prompts; mark predictions content-seen and report stratified results',
                'final_days_opened': False})
            configurations = []
            for kind in ['gru', 'supervised', 'dynamics', 'predictive']:
                for lr in ([.0003] if args.stage == 'smoke' else [.0003, .001]):
                    for auxiliary in ([.1] if args.stage == 'smoke' or kind in ['gru', 'supervised'] else [.1, 1.]):
                        configurations.append({'kind': kind, 'hidden': 256, 'seed': 17, 'lr': lr,
                            'auxiliary': auxiliary if kind in ['dynamics', 'predictive'] else 0.,
                            'epochs': identity['epochs'], 'manifest_sha256': identity['manifest_sha256']})
            for index, config in enumerate(configurations):
                name = f"{config['kind']}-s17-lr{config['lr']}-aux{config['auxiliary']}"
                status.update(status='training', training_started=True, current_run=name,
                              completed_candidates=index, total_candidates=len(configurations))
                save_json(args.output/'status.json', status)
                print(json.dumps(status), flush=True)
                fit(config, training, development, stats, args.output/name, device)
            status.update(status='smoke-complete' if args.stage == 'smoke' else 'initial-search-complete',
                          completed_candidates=len(configurations), full_comparison_complete=False)
            save_json(args.output/'status.json', status)
            return 0
        except Exception as error:
            status.update(status='paused-storage' if 'Storage guard:' in str(error) else 'failed',
                          error=str(error), full_comparison_complete=False)
            save_json(args.output/'status.json', status)
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    sys.exit(main())
