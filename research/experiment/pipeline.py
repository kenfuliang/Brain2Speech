"""Resumable development selection, seed confirmation, and final adaptation matrix."""
import argparse
import fcntl
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from research.audit import ROOT, digest, save_json, space_guard
from research.data import load_trials, prepare, prefix
from research.run import check_manifest
from research.train import adapt, fit, load_checkpoint, device_for_run
from research.experiment.evaluator import FixedDecoder, evaluate
from research.experiment.analysis import update_diagnostics, effective_rank
from research.experiment.report import collect_report
from research.experiment.coverage import audit_collection
from research.experiment.figures import build_figures

KINDS = ['gru', 'supervised', 'dynamics', 'predictive']
BUDGETS = [5, 10, 25, 50, 100, 'full']


def separation(calibration, evaluation, replay=()):
    c, e, r = [{t['id'] for t in group} for group in [calibration, evaluation, replay]]
    if c & e or r & e or c & r:
        raise ValueError('Calibration/evaluation/replay overlap')
    if any('test' in t['id'] or 'HoldOut' in t['id'] for t in replay):
        raise ValueError('Replay contains reserved evaluation/holdout trial')


class Study:
    def __init__(self, source, output, decoder):
        self.source, self.output, self.decoder = source, output, decoder
        self.manifest_path = ROOT/'research/data/manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text())
        errors = check_manifest(self.manifest)
        if errors:
            raise ValueError(errors)
        protocol = json.loads((source/'protocol.json').read_text())
        for name, expected in protocol['source_hashes'].items():
            if digest(ROOT/name) != expected:
                raise ValueError('Initial training source changed: '+name)
        if protocol['manifest_sha256'] != digest(self.manifest_path):
            raise ValueError('Initial training manifest identity mismatch')
        self.texts = {r['text'] for s in self.manifest['folds']['initial_train']
                      for r in self.manifest['sessions'][s]['train']['trials']}
        self.device = torch.device('cpu')
        self.verified_sources = set()
        self.status(stage='development-selection', final_days_opened=False)

    def status(self, **values):
        path = self.output/'status.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        save_json(path, {**previous, **values, 'full_comparison_complete': False})

    def eval(self, checkpoint, trials, stats, path, condition='normal'):
        self.status(current_evaluation=str(path))
        return evaluate(checkpoint, trials, stats, path, self.device, self.decoder, condition, self.texts)

    def load_verified(self, sessions, partition):
        for day in sessions:
            entry = self.manifest['sessions'][day][partition]
            source = (ROOT/entry['path']).resolve()
            source.relative_to((ROOT/'research/data').resolve())
            if source not in self.verified_sources:
                space_guard()
                if digest(source) != entry['sha256']:
                    raise ValueError('Frozen raw source changed: '+str(source))
                self.verified_sources.add(source)
        return load_trials(self.manifest,sessions,partition)

    def development(self, wait):
        dev = self.load_verified(self.manifest['folds']['development'], 'test')
        results = {}
        while True:
            space_guard()
            for completed in sorted(self.source.glob('*/complete.json')):
                info = json.loads(completed.read_text())
                name = completed.parent.name
                checkpoint = completed.parent/'best.pt'
                if digest(checkpoint) != info['checkpoint_sha256']:
                    raise ValueError('Initial checkpoint identity mismatch')
                if name not in results:
                    _, payload = load_checkpoint(checkpoint, self.device)
                    result = self.eval(checkpoint, dev, payload['stats'], self.output/'development'/name)
                    results[name] = {'checkpoint': str(checkpoint), 'config': info['config'],
                                     'development_wer': result['wer']['macro_day']}
                    save_json(self.output/'development-candidates.json', results)
            upstream = json.loads((self.source/'status.json').read_text())
            if upstream['status'] == 'initial-search-complete':
                if len(results) != 12:
                    raise ValueError('Initial search completed without all 12 candidate results')
                break
            if upstream['status'] in ['failed', 'paused-storage']:
                raise RuntimeError('Initial search requires attention: '+str(upstream))
            if not wait:
                self.status(stage='awaiting-initial-search', decoded_candidates=len(results))
                return None
            self.status(stage='awaiting-initial-search', decoded_candidates=len(results))
            time.sleep(15)
        selected = {kind: min([v for v in results.values() if v['config']['kind'] == kind],
                             key=lambda v: (v['development_wer'], v['config']['lr'], v['config']['auxiliary']))
                    for kind in KINDS}
        self.device = device_for_run()  # Initial GPU search has ended; no concurrent research training.
        dev_calibration = self.load_verified(self.manifest['folds']['development'], 'train')
        for kind, choice in selected.items():
            _, payload = load_checkpoint(choice['checkpoint'], self.device)
            choice['adaptation_steps'] = {}
            modes = ['adapter', 'full'] + (['unlabeled'] if kind in ['dynamics', 'predictive'] else [])
            for mode in modes:
                candidates = []
                for steps in [100, 500]:
                    scores = []
                    for day in self.manifest['folds']['development']:
                        cal = prefix([t for t in dev_calibration if t['session'] == day], 25)
                        target = [t for t in dev if t['session'] == day]
                        separation(cal, target)
                        prepared, stats = prepare(cal, payload['stats'], rolling=True)
                        config = {**choice['config'], 'adaptation_steps': steps}
                        folder = self.output/'development-adaptation'/kind/mode/str(steps)/day
                        post = adapt(choice['checkpoint'], prepared, config, stats, folder, self.device, mode)
                        scores.append(self.eval(post, target, stats, folder/'evaluation')['wer']['corpus'])
                    candidates.append((float(np.mean(scores)), steps))
                choice['adaptation_steps'][mode] = min(candidates)[1]
                choice.setdefault('adaptation_selection_scores', {})[mode] = candidates
        target = min(score for v in selected.values() if v['config']['kind'] in ['gru','supervised']
                     for mode in ['adapter', 'full']
                     for score, steps in v['adaptation_selection_scores'][mode]
                     if steps == v['adaptation_steps'][mode])
        freeze = {'manifest_sha256': digest(self.manifest_path), 'selected': selected,
                  'wer_target': target, 'target_definition': 'Strongest supervised 25-label development macro-day WER at selected steps',
                  'decoder': self.decoder.identity, 'seeds': [17,29,43], 'budgets': BUDGETS,
                  'normalization': 'Initial train only; each labeled/unlabeled calibration prefix updates moments after trials',
                  'checkpoint_selection': 'Development greedy PER within fit; fixed WFST WER across candidates',
                  'acquisition_delay_ms': 4, 'sequential_replay_fraction': .25,
                  'final_days_opened': False}
        save_json(self.output/'development-freeze.json', freeze)
        pilot = []
        for identity in self.output.glob('development-adaptation/**/post.identity.json'):
            record = json.loads(identity.read_text())
            pilot.append({'mode':record['mode'], 'kind':record['config']['kind'],
                          'steps':record['config']['adaptation_steps'],
                          'seconds':record['adaptation_seconds']})
        save_json(self.output/'resource-pilot.json', {'adaptation_pilots':pilot,
                  'free_bytes':space_guard(), 'minimum_free_bytes':30*1024**3,
                  'execution_policy':'Serial GPU training; resident WFST; pause below 30 GiB; delete nothing',
                  'warning':'Full calibration/sequential/control matrix is multi-stage; no wall-time completion promise'})
        self.status(stage='development-frozen', final_days_opened=False)
        return freeze

    def final(self, freeze):
        if freeze['manifest_sha256'] != digest(self.manifest_path) or freeze['decoder'] != self.decoder.identity:
            raise ValueError('Frozen data/decoder identity changed')
        self.device = device_for_run()
        train_raw = self.load_verified(self.manifest['folds']['initial_train'], 'train')
        dev_raw = self.load_verified(self.manifest['folds']['development'], 'test')
        bases = {}
        for kind, choice in freeze['selected'].items():
            _, initial = load_checkpoint(choice['checkpoint'], self.device)
            train, _ = prepare(train_raw, initial['stats'])
            dev, _ = prepare(dev_raw, initial['stats'])
            for seed in freeze['seeds']:
                if seed == 17:
                    bases[kind, seed] = Path(choice['checkpoint'])
                else:
                    config = {**choice['config'], 'seed': seed}
                    bases[kind, seed], _ = fit(config, train, dev, initial['stats'],
                                               self.output/'confirmation'/kind/str(seed), self.device)
        self.status(stage='final-matrix', final_days_opened=True,
                    freeze_sha256=digest(self.output/'development-freeze.json'))
        days = self.manifest['folds']['final']
        calibrations = self.load_verified(days, 'train')
        tests = self.load_verified(days, 'test')
        probes = self.load_verified(self.manifest['folds']['initial_train'], 'test')
        for (kind, seed), base in bases.items():
            choice = freeze['selected'][kind]
            _, base_payload = load_checkpoint(base, self.device)
            base_probe_folder = self.output/'probes'/kind/str(seed)
            self.eval(base, probes, base_payload['stats'], base_probe_folder)
            for day in days:
                available = [t for t in calibrations if t['session'] == day]
                target = [t for t in tests if t['session'] == day]
                root = self.output/'final'/kind/str(seed)/day
                self.eval(base, target, base_payload['stats'], root/'zero-shot')
                _, full_stats = prepare(available, base_payload['stats'], rolling=True)
                self.eval(base, target, full_stats, root/'normalization-only')
                for condition in ['shuffled', 'zero', 'dropout-0.1', 'dropout-0.25', 'dropout-0.5', 'gain-offset']:
                    self.eval(base, target, base_payload['stats'], root/'robustness'/condition, condition)
                for budget in BUDGETS:
                    cal = prefix(available, budget)
                    separation(cal, target)
                    prepared, stats = prepare(cal, base_payload['stats'], rolling=True)
                    for mode in ['adapter', 'full'] + (['unlabeled'] if kind in ['dynamics','predictive'] else []):
                        config = {**choice['config'], 'seed': seed,
                                  'adaptation_steps': choice['adaptation_steps'][mode]}
                        folder = root/mode/str(budget)
                        post = adapt(base, prepared, config, stats, folder, self.device, mode)
                        self.eval(post, target, stats, folder/'evaluation')
                        # Second equal-budget update on the same day anchors optimization-induced drift.
                        second = adapt(post, prepared, config, stats, folder/'same-day-control', self.device, mode)
                        self.eval(second, target, stats, folder/'same-day-control/evaluation')
            for budget in BUDGETS:
                for mode in ['adapter', 'full']:
                    for replay_fraction in [0., .25]:
                        parent, pool = base, list(train_raw)
                        vectors, trajectory = [], []
                        folder = self.output/'sequential'/kind/str(seed)/mode/str(budget)/str(replay_fraction)
                        # Fixed probes evaluated before and after each transition.
                        save_json(folder/'initial-probe-reference.json',
                                  {'path':str(base_probe_folder.relative_to(self.output)),
                                   'checkpoint_sha256':digest(base)})
                        for day in days:
                            _, before = load_checkpoint(parent, self.device)
                            cal = prefix([t for t in calibrations if t['session'] == day], budget)
                            target = [t for t in tests if t['session'] == day]
                            separation(cal, probes+tests, pool)
                            prepared, stats = prepare(cal, before['stats'], rolling=True)
                            replay, _ = prepare(pool, stats) if replay_fraction else ([], stats)
                            config = {**choice['config'], 'seed': seed,
                                      'adaptation_steps': choice['adaptation_steps'][mode]}
                            transition = folder/day
                            post = adapt(parent, prepared, config, stats, transition, self.device, mode, replay)
                            metrics = self.eval(post, probes, base_payload['stats'], transition/'fixed-probes')
                            self.eval(post, tests, stats, transition/'transfer')
                            _, after = load_checkpoint(post, self.device)
                            update, vector = update_diagnostics(before['model'], after['model'], vectors[-1] if vectors else None)
                            previous_embeddings = np.load((base_probe_folder if not vectors else
                                folder/days[len(vectors)-1]/'fixed-probes')/'embeddings.npy')
                            current_embeddings = np.load(transition/'fixed-probes/embeddings.npy')
                            update['fixed_probe_embedding_rms'] = float(np.sqrt(np.mean((current_embeddings-previous_embeddings)**2)))
                            update.update(day=day, parent_sha256=digest(parent), child_sha256=digest(post),
                                          historical_wer=metrics['wer']['macro_day'])
                            vectors.append(vector)
                            trajectory.append(update)
                            save_json(folder/'trajectory.json', {'transitions': trajectory, **effective_rank(vectors)})
                            pool.extend(cal)
                            parent = post
        coverage = audit_collection(self.output, self.manifest, freeze)
        if not coverage['matrix_complete']:
            raise RuntimeError('Required matrix evidence is missing or inconsistent; inspect collection-coverage.json')
        report = collect_report(self.output, freeze, self.manifest)
        report['figures_generated'] = build_figures(self.output, freeze)
        self.status(stage='matrix-collected-review-required', final_days_opened=True, report=report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'research/runs/v1-initial-accepted4ms')
    parser.add_argument('--output', type=Path, default=ROOT/'research/runs/v1-comparison')
    parser.add_argument('--stage', choices=['development', 'full'], default='development')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--require-smoke', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    decoder = None
    with (args.output/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.require_smoke and not (args.require_smoke/'complete.json').exists():
                raise ValueError('Real WFST smoke did not complete; final pipeline remains gated')
            source_identity = {'manifest_sha256':digest(ROOT/'research/data/manifest.json'),
                'pipeline_sources':{str(p.relative_to(ROOT)):digest(p)
                                    for p in sorted((ROOT/'research/experiment').glob('*.py'))},
                'initial_source':str(args.source.resolve())}
            identity_path = args.output/'execution-identity.json'
            if identity_path.exists() and json.loads(identity_path.read_text()) != source_identity:
                raise ValueError('Pipeline code/data identity changed; use a new output directory')
            save_json(identity_path, source_identity)
            decoder = FixedDecoder(args.output/'decoder.log')
            study = Study(args.source, args.output, decoder)
            freeze_path = args.output/'development-freeze.json'
            freeze = json.loads(freeze_path.read_text()) if freeze_path.exists() else study.development(args.wait)
            if freeze and args.stage == 'full':
                study.final(freeze)
        except Exception as error:
            save_json(args.output/'failure.json', {'error': str(error), 'traceback': traceback.format_exc(),
                        'status': 'paused-storage' if 'Storage guard:' in str(error) else 'failed'})
            raise
        finally:
            if decoder:
                decoder.close()


if __name__ == '__main__':
    main()
