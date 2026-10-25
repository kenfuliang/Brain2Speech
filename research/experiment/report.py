"""Aggregate matrix outputs and paired seed/session comparisons."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from research.audit import save_json
from research.metrics import aggregate
from research.experiment.analysis import paired_effect, calibration_savings


def metric_summary(values, field, seeds, repetitions=2000):
    days = sorted({r['session'] for r in values})
    by_seed = {}
    groups = {(s,d):[r for r in values if r['seed']==s and r['session']==d] for s in seeds for d in days}
    for s in seeds:
        daily = [aggregate(groups[s,d], field) for d in days]
        by_seed[str(s)] = {'macro':float(np.mean(daily)),
                          'corpus':aggregate([r for r in values if r['seed']==s],field), 'worst_day':max(daily)}
    rng = np.random.default_rng(101)
    boot = []
    for _ in range(repetitions):
        sampled = []
        for day in rng.choice(days,len(days)):
            indexes = rng.integers(len(groups[seeds[0],day]),size=len(groups[seeds[0],day]))
            sampled.append(np.mean([aggregate([groups[s,day][i] for i in indexes],field) for s in seeds]))
        boot.append(np.mean(sampled))
    totals = {k:sum(r[field][k] for r in values) for k in values[0][field]}
    return {'by_seed':by_seed, 'mean_macro':float(np.mean([r['macro'] for r in by_seed.values()])),
            'macro_ci95':np.quantile(boot,[.025,.975]).tolist(), 'pooled_corpus':totals['errors']/totals['reference_units'],
            'seed_sd':float(np.std([r['macro'] for r in by_seed.values()],ddof=1)) if len(seeds)>1 else None,
            'edit_totals':totals,
            'edit_total_unit':'Prediction instances pooled across seeds; repeated trials are not independent participant observations',
            'uncertainty':'Shared session/trial bootstrap averaged across fixed seeds; seed SD separate'}


def collect_report(root, freeze, manifest):
    root = Path(root)
    groups = defaultdict(list)
    rows = []
    durations = {r['id']:r.get('duration_seconds',0.) for session in manifest['sessions'].values()
                 for partition in ['train','test'] for r in session[partition]['trials']}
    for prediction in sorted((root/'final').glob('**/predictions.json')):
        if not (prediction.parent/'complete.json').exists():
            continue
        parts = prediction.relative_to(root/'final').parts
        kind, seed, day = parts[:3]
        setting = '/'.join(parts[3:-1])
        values = json.loads(prediction.read_text())
        for value in values:
            value['seed'] = int(seed)
        groups[kind, setting].extend(values)
        metrics = json.loads((prediction.parent/'metrics.json').read_text())
        post_identity = prediction.parent.parent/'post.identity.json'
        adaptation = json.loads(post_identity.read_text()) if post_identity.exists() else None
        calibration_ids = adaptation['calibration_ids'] if adaptation else []
        if setting=='normalization-only':
            calibration_ids = [r['id'] for r in manifest['sessions'][day]['train']['trials']]
        row = {'kind':kind, 'seed':int(seed), 'day':day, 'setting':setting,
                     'wer':metrics['wer']['corpus'], 'per':metrics['per']['corpus'],
                     'cer':metrics['cer']['corpus'], 'trials':len(values),
                     'mean_encoder_ms':metrics['mean_encoder_time_ms_per_trial'],
                     'mean_wfst_ms':metrics['mean_wfst_ms_per_trial'],
                     'calibration_ids':calibration_ids,
                     'labeled_sentences':len(calibration_ids) if adaptation and adaptation['mode']!='unlabeled' else 0,
                     'calibration_duration_seconds':sum(durations[i] for i in calibration_ids),
                     'unlabeled_duration_seconds':sum(durations[i] for i in calibration_ids)
                         if setting=='normalization-only' or adaptation and adaptation['mode']=='unlabeled' else 0.,
                     'adaptation_seconds':adaptation['adaptation_seconds'] if adaptation else 0.,
                     'error_components':{field:{k:sum(v[field][k] for v in values) for k in values[0][field]}
                                         for field in ['wer','per','cer']}}
        rows.append(row)
    summaries, comparisons = [], []
    for (kind, setting), values in sorted(groups.items()):
        by_seed = {}
        for seed in freeze['seeds']:
            selected = [r for r in values if r['seed'] == seed]
            days = sorted({r['session'] for r in selected})
            if not days:
                continue
            daily = [aggregate([r for r in selected if r['session']==d], 'wer') for d in days]
            by_seed[str(seed)] = {'macro_wer':float(np.mean(daily)), 'corpus_wer':aggregate(selected,'wer'),
                                 'worst_day_wer':max(daily)}
        summaries.append({'kind':kind, 'setting':setting, 'by_seed':by_seed,
                          'metrics':{field:metric_summary(values,field,freeze['seeds']) for field in ['wer','per','cer']},
                          'mean_macro_wer':float(np.mean([r['macro_wer'] for r in by_seed.values()])),
                          'seed_sd':float(np.std([r['macro_wer'] for r in by_seed.values()], ddof=1))
                                    if len(by_seed)>1 else None})
        if kind in ['dynamics','predictive']:
            for control in ['gru','supervised']:
                if (control,setting) not in groups:
                    continue
                effect = paired_effect(groups[control,setting], values)
                control_mean = np.mean([aggregate([r for r in groups[control,setting]
                                   if r['seed']==s and r['session']==d], 'wer')
                                   for s in freeze['seeds'] for d in sorted({r['session'] for r in values})])
                effect.update(candidate=kind, control=control, setting=setting,
                              relative_wer_reduction=-effect['candidate_minus_control_macro_wer']/control_mean
                                if control_mean>0 else None)
                comparisons.append(effect)
    crossings = []
    for kind in freeze['selected']:
        for mode in ['adapter','full']:
            for seed in freeze['seeds']:
                zero = [r for r in rows if r['kind']==kind and r['seed']==seed and r['setting']=='zero-shot']
                options = [0] if zero and np.mean([r['wer'] for r in zero])<=freeze['wer_target'] else []
                for budget in freeze['budgets']:
                    daily = [r for r in rows if r['kind']==kind and r['seed']==seed
                             and r['setting']==f'{mode}/{budget}/evaluation']
                    if daily and np.mean([r['wer'] for r in daily]) <= freeze['wer_target']:
                        options.append(budget)
                first = options[0] if options else 'target not reached'
                matching = [r for r in rows if r['kind']==kind and r['seed']==seed
                            and r['setting']==f'{mode}/{first}/evaluation']
                crossings.append({'kind':kind,'mode':mode,'seed':seed,
                                  'first_tested_budget_reaching_target':first,
                                  'total_calibration_labels':0 if first==0 else
                                      (sum(r['labeled_sentences'] for r in matching) if matching else None),
                                  'full_budget_note':'full is day-dependent; not one fixed label count'})
    savings = []
    comparison_budgets = [0]+freeze['budgets']
    labels_by_day = {day:{str(b):(len(manifest['sessions'][day]['train']['trials']) if b=='full' else b)
                         for b in comparison_budgets} for day in manifest['folds']['final']}
    for candidate in ['dynamics','predictive']:
        for control in ['gru','supervised']:
            for mode in ['adapter','full']:
                settings = [f'{mode}/{b}/evaluation' for b in freeze['budgets']]
                if not all((kind,setting) in groups for kind in [candidate,control] for setting in settings):
                    continue
                curves = [{str(b):groups[kind,f'{mode}/{b}/evaluation'] for b in freeze['budgets']}
                          for kind in [control,candidate]]
                if not all((kind,'zero-shot') in groups for kind in [control,candidate]):
                    continue
                for curve,kind in zip(curves,[control,candidate]):
                    curve['0'] = groups[kind,'zero-shot']
                result = calibration_savings(*curves,comparison_budgets,freeze['wer_target'],labels_by_day)
                savings.append({'candidate':candidate,'control':control,'mode':mode,**result})
    save_json(root/'comparative-summary.json', {'per_day_rows':rows, 'summaries':summaries,
              'paired_comparisons':comparisons, 'calibration_crossings':crossings,'paired_calibration_savings':savings,
              'wer_target':freeze['wer_target'],
              'claim_status':'Requires strongest matched control, paired benefit and worst-day inspection; not automatically accepted'})
    return {'evaluated_day_configurations':len(rows), 'paired_comparisons':len(comparisons)}
