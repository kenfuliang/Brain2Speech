"""Paired macro-day uncertainty and encoder-update diagnostics."""
import numpy as np
import torch
from research.metrics import aggregate


def paired_effect(control, candidate, seed=101, repetitions=10000):
    left = {(r['seed'], r['session'], r['id']): r for r in control}
    right = {(r['seed'], r['session'], r['id']): r for r in candidate}
    if len(left) != len(control) or len(right) != len(candidate):
        raise ValueError('Duplicate paired seed/session/trial identities')
    if left.keys() != right.keys() or not left:
        raise ValueError('Paired predictions must share seed/session/trial identities')
    seeds = sorted({k[0] for k in left})
    days = sorted({k[1] for k in left})
    rng = np.random.default_rng(seed)
    effects, seed_effects = [], []
    for s in seeds:
        seed_effects.append(np.mean([aggregate([r for k, r in right.items() if k[:2] == (s, d)], 'wer')-
                                   aggregate([r for k, r in left.items() if k[:2] == (s, d)], 'wer') for d in days]))
    groups = {(s, d): [k for k in left if k[:2] == (s, d)] for s in seeds for d in days}
    for d in days:
        if any({k[2] for k in groups[s,d]} != {k[2] for k in groups[seeds[0],d]} for s in seeds):
            raise ValueError('Trial identities differ across seeds')
    # Same day/trial resamples across seeds retain the paired experimental design.
    for _ in range(repetitions):
        delta = []
        for d in rng.choice(days, len(days)):
            ids = [k[2] for k in groups[seeds[0], d]]
            picked = rng.choice(ids, len(ids))
            delta.append(np.mean([aggregate([right[s, d, i] for i in picked], 'wer')-
                                  aggregate([left[s, d, i] for i in picked], 'wer') for s in seeds]))
        effects.append(np.mean(delta))
    return {'candidate_minus_control_macro_wer': float(np.mean(seed_effects)),
            'ci95': np.quantile(effects, [.025, .975]).tolist(), 'seed_effects': seed_effects,
            'seed_effect_sd': float(np.std(seed_effects, ddof=1)) if len(seeds)>1 else None,
            'sessions': len(days), 'warning': 'Single participant, five final sessions; limited population inference'}


def update_diagnostics(parent, child, previous_delta=None):
    norms, absolute, pieces, layers = {}, {}, [], {}
    for name in sorted(parent):
        before = parent[name].float().reshape(-1)
        delta = child[name].float().reshape(-1)-before
        absolute[name] = float(delta.norm())
        norms[name] = float(delta.norm()/before.norm()) if before.norm()>0 else None
        layer = name.split('.')[0]
        entry = layers.setdefault(layer, [0.,0.])
        entry[0] += float(before.square().sum())
        entry[1] += float(delta.square().sum())
        pieces.append(delta)
    vector = torch.cat(pieces)
    cosine = None
    if previous_delta is not None and vector.norm()>0 and previous_delta.norm()>0:
        cosine = float(torch.dot(vector, previous_delta)/(vector.norm()*previous_delta.norm()))
    return {'normalized_parameter_updates': norms, 'update_norm': float(vector.norm()),
            'absolute_parameter_updates':absolute,
            'normalized_layer_updates':{name:(float(np.sqrt(delta/initial)) if initial>0 else None)
                                        for name,(initial,delta) in layers.items()},
            'absolute_layer_updates':{name:float(np.sqrt(delta)) for name,(_,delta) in layers.items()},
            'zero_norm_policy':'Normalized change is undefined (null) for zero parent norm; absolute change retained',
            'consecutive_update_cosine': cosine}, vector


def effective_rank(vectors):
    if not vectors:
        return {'effective_rank': 0., 'singular_values': [], 'cumulative_energy': []}
    # Small transition Gram matrix avoids parameter-dimensional SVD allocation.
    matrix = torch.stack(vectors).double()
    singular = torch.linalg.eigvalsh(matrix@matrix.T).clamp_min(0).sqrt().flip(0).numpy()
    energy = singular**2
    if energy.sum() == 0:
        return {'effective_rank': 0., 'singular_values': singular.tolist(),
                'cumulative_energy':np.zeros(len(singular)).tolist()}
    p = energy/energy.sum()
    return {'effective_rank': float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))),
            'definition': 'exp entropy of squared-singular-value energy',
            'singular_values': singular.tolist(), 'cumulative_energy': np.cumsum(p).tolist()}


def calibration_savings(control, candidate, budgets, target, labels_by_day,
                        seed=101, repetitions=2000):
    """Paired session/trial resampling of discrete, measured calibration crossings."""
    if set(control) != set(candidate) or set(control) != set(map(str,budgets)):
        raise ValueError('All tested calibration budgets are required')
    keys = {(r['seed'],r['session'],r['id']) for r in control[str(budgets[0])]}
    if not keys:
        raise ValueError('No calibration predictions')
    indexed = {}
    for name,curve in [('control',control),('candidate',candidate)]:
        for budget in budgets:
            rows = curve[str(budget)]
            index = {(r['seed'],r['session'],r['id']):r for r in rows}
            if set(index) != keys or len(index)!=len(rows):
                raise ValueError('Calibration curves require identical seed/session/trial identities')
            indexed[name,str(budget)] = index
    seeds = sorted({k[0] for k in keys}); days = sorted({k[1] for k in keys})
    ids = {d:sorted(k[2] for k in keys if k[:2]==(seeds[0],d)) for d in days}
    rng = np.random.default_rng(seed)
    ratios, failed, zero_labels = [], 0, 0
    for _ in range(repetitions):
        chosen = [(d,rng.choice(ids[d],len(ids[d]))) for d in rng.choice(days,len(days))]
        crossings = {}
        for name in ['control','candidate']:
            crossings[name] = None
            for budget in budgets:
                index = indexed[name,str(budget)]
                score = np.mean([aggregate([index[s,d,i] for i in picked],'wer')
                                 for d,picked in chosen for s in seeds])
                if score <= target:
                    crossings[name] = sum(labels_by_day[d][str(budget)] for d,_ in chosen)
                    break
        if any(v is None for v in crossings.values()):
            failed += 1
        elif any(v==0 for v in crossings.values()):
            zero_labels += 1
        else:
            ratios.append(crossings['control']/crossings['candidate'])
    interval = np.quantile(ratios,[.025,.975]).tolist() if ratios else None
    return {'control_labels_over_candidate_labels_ci95_conditional_on_both_reaching':interval,
            'both_reach_fraction':1-failed/repetitions,
            'zero_label_crossing_fraction':zero_labels/repetitions,
            'nominal_support_for_2x':failed==0 and zero_labels==0 and interval is not None and interval[0]>=2,
            'policy':'No extrapolation. Any target-unreached or zero-label crossing prevents a 2x ratio claim.',
            'target_wer':target,'bootstrap_seed':seed,'bootstrap_repetitions':repetitions}
