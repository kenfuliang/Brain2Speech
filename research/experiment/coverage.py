"""Audit every required output against the approved experiment matrix."""
import json
from pathlib import Path
from research.audit import save_json, digest


def audit_transitions(root, manifest, freeze):
    """Verify preserved pre/post checkpoints against the exact approved ancestry."""
    root = Path(root)
    days = manifest['folds']['final']
    historical = [r['id'] for d in manifest['folds']['initial_train']
                  for r in manifest['sessions'][d]['train']['trials']]
    issues, checked = [], 0

    def check(folder, parent, config, mode, calibration, replay):
        nonlocal checked
        try:
            pre, post = folder/'pre.pt', folder/'post.pt'
            pre_id = json.loads((folder/'pre.identity.json').read_text())
            post_id = json.loads((folder/'post.identity.json').read_text())
            parent_hash = digest(parent)
            if pre_id['sha256'] != digest(pre) or post_id['sha256'] != digest(post):
                raise ValueError('Preserved checkpoint hash mismatch')
            if any(record['parent_sha256'] != parent_hash or record['config'] != config
                   for record in [pre_id, post_id]):
                raise ValueError('Checkpoint parent/config mismatch')
            if post_id['mode'] != mode or post_id['calibration_ids'] != calibration or post_id['replay_ids'] != replay:
                raise ValueError('Calibration/replay/mode mismatch')
            if set(calibration) & set(replay) or any('/test/' in i or 'HoldOut' in i for i in replay):
                raise ValueError('Reserved or overlapping replay trial')
            checked += 1
        except (OSError, ValueError, KeyError) as error:
            issues.append({'folder':str(folder.relative_to(root)), 'error':str(error)})
        return folder/'post.pt'

    for kind, choice in freeze['selected'].items():
        for seed in freeze['seeds']:
            base = Path(choice['checkpoint']) if seed == 17 else root/'confirmation'/kind/str(seed)/'best.pt'
            modes = ['adapter','full'] + (['unlabeled'] if kind in ['dynamics','predictive'] else [])
            for day in days:
                available = [r['id'] for r in manifest['sessions'][day]['train']['trials']]
                for budget in freeze['budgets']:
                    calibration = available if budget == 'full' else available[:budget]
                    for mode in modes:
                        config = {**choice['config'], 'seed':seed, 'adaptation_steps':choice['adaptation_steps'][mode]}
                        folder = root/'final'/kind/str(seed)/day/mode/str(budget)
                        post = check(folder, base, config, mode, calibration, [])
                        check(folder/'same-day-control', post, config, mode, calibration, [])
            for budget in freeze['budgets']:
                for mode in ['adapter','full']:
                    config = {**choice['config'], 'seed':seed, 'adaptation_steps':choice['adaptation_steps'][mode]}
                    for fraction in [0., .25]:
                        parent, pool = base, list(historical)
                        for day in days:
                            available = [r['id'] for r in manifest['sessions'][day]['train']['trials']]
                            calibration = available if budget == 'full' else available[:budget]
                            folder = root/'sequential'/kind/str(seed)/mode/str(budget)/str(fraction)/day
                            parent = check(folder, parent, config, mode, calibration, pool if fraction else [])
                            pool.extend(calibration)
    return {'verified_transitions':checked, 'issues':issues, 'ancestry_complete':not issues}


def expected_outputs(manifest, freeze):
    days = manifest['folds']['final']
    probes = [r['id'] for d in manifest['folds']['initial_train']
              for r in manifest['sessions'][d]['test']['trials']]
    tests = {d:[r['id'] for r in manifest['sessions'][d]['test']['trials']] for d in days}
    all_tests = sum(tests.values(), [])
    for kind in freeze['selected']:
        modes = ['adapter','full'] + (['unlabeled'] if kind in ['dynamics','predictive'] else [])
        for seed in freeze['seeds']:
            yield f'probes/{kind}/{seed}', probes
            for day in days:
                base = f'final/{kind}/{seed}/{day}'
                for setting in ['zero-shot','normalization-only']:
                    yield f'{base}/{setting}', tests[day]
                for condition in ['shuffled','zero','dropout-0.1','dropout-0.25','dropout-0.5','gain-offset']:
                    yield f'{base}/robustness/{condition}', tests[day]
                for budget in freeze['budgets']:
                    for mode in modes:
                        yield f'{base}/{mode}/{budget}/evaluation', tests[day]
                        yield f'{base}/{mode}/{budget}/same-day-control/evaluation', tests[day]
            for budget in freeze['budgets']:
                for mode in ['adapter','full']:
                    for replay in [0., .25]:
                        base = f'sequential/{kind}/{seed}/{mode}/{budget}/{replay}'
                        for day in days:
                            yield f'{base}/{day}/fixed-probes', probes
                            yield f'{base}/{day}/transfer', all_tests


def audit_collection(root, manifest, freeze):
    root = Path(root)
    missing, inconsistent, complete = [], [], 0
    expected = list(expected_outputs(manifest, freeze))
    for relative, ids in expected:
        folder = root/relative
        required = ['complete.json','predictions.json','metrics.json','logits.npz','embeddings.npy']
        if any(not (folder/name).is_file() for name in required):
            missing.append(relative)
            continue
        completed = json.loads((folder/'complete.json').read_text())
        identity = completed['identity']
        hashes = completed.get('artifact_sha256', {})
        if any(name not in hashes or digest(folder/name) != hashes[name] for name in required[1:]):
            inconsistent.append({'folder':relative,'error':'Evidence hashes missing or changed'})
            continue
        rows = json.loads((folder/'predictions.json').read_text())
        if identity['ids'] != ids or [r['id'] for r in rows] != ids:
            inconsistent.append({'folder':relative,'error':'Frozen sample order/accounting mismatch'})
            continue
        if any(not all(field in r for field in ['per','wer','cer']) for r in rows):
            inconsistent.append({'folder':relative,'error':'Missing edit metrics'})
            continue
        metrics = json.loads((folder/'metrics.json').read_text())
        if metrics.get('trials') != len(ids) or any(
            r[field]['errors'] != sum(r[field][component] for component in
                                     ['substitutions','deletions','insertions'])
            for r in rows for field in ['per','wer','cer']):
            inconsistent.append({'folder':relative,'error':'Edit/sample accounting mismatch'})
            continue
        complete += 1
    ancestry = audit_transitions(root, manifest, freeze)
    result = {'expected_evaluations':len(expected), 'complete_evaluations':complete,
              'missing':missing, 'inconsistent':inconsistent,
              'checkpoint_ancestry':ancestry,
              'matrix_complete':not missing and not inconsistent and ancestry['ancestry_complete'],
              'scope':'Every model/seed/day/budget/mode/replay/control/probe/transfer evaluation'}
    save_json(root/'collection-coverage.json', result)
    return result
