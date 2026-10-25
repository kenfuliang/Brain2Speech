"""Versioned derived scoring; raw/frozen evaluator artifacts are never modified."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.audit import digest, save_json
from research.metrics import aggregate, edits, greedy, summarize


def verify(folder, expected_ids=None, repetitions=2000):
    folder = Path(folder)
    completed = json.loads((folder/'complete.json').read_text())
    for name in ['predictions.json','metrics.json','logits.npz','embeddings.npy']:
        if completed.get('artifact_sha256', {}).get(name)!=digest(folder/name):
            raise ValueError('Missing or altered artifact hash: '+name)
    rows = json.loads((folder/'predictions.json').read_text())
    original = json.loads((folder/'metrics.json').read_text())
    ids = [r['id'] for r in rows]
    if not rows or len(set(ids))!=len(ids) or ids!=completed['identity']['ids'] or (expected_ids is not None and ids!=expected_ids):
        raise ValueError('Trial identity/order mismatch')
    if len(rows)!=original['trials'] or len({r['session'] for r in rows})!=original['sessions']:
        raise ValueError('Trial/session count mismatch')
    embeddings = np.load(folder/'embeddings.npy', allow_pickle=False)
    if embeddings.ndim!=2 or len(embeddings)!=len(rows) or not np.isfinite(embeddings).all():
        raise ValueError('Embedding accounting or finite-value mismatch')
    corrections, corrected = [], []
    with np.load(folder/'logits.npz', allow_pickle=False) as logits:
        if len(logits.files)!=len(rows):
            raise ValueError('Logit accounting mismatch')
        for index, row in enumerate(rows):
            x = logits[f'trial_{index:04d}']
            if x.ndim!=2 or x.shape[1]!=41 or not np.isfinite(x).all() or greedy(x)!=row['predicted_phones']:
                raise ValueError('Logit/phoneme mismatch')
            if edits(row['reference_phones'],row['predicted_phones'])!=row['per']:
                raise ValueError('Phoneme edit mismatch')
            if edits(list(row['text']),list(row['decoded_text']))!=row['cer']:
                raise ValueError('Character edit mismatch')
            legacy = edits(row['text'].split(' '),row['decoded_text'].split(' '))
            if legacy!=row['wer']:
                raise ValueError('Original word edit mismatch; not a whitespace-only correction')
            standard = edits(row['text'].split(),row['decoded_text'].split())
            if standard!=legacy:
                corrections.append({'id':row['id'],'original':legacy,'corrected':standard,
                    'rate_changed':(standard['errors'],standard['reference_units'])!=(legacy['errors'],legacy['reference_units'])})
            corrected.append({**row,'wer':standard})
    for field in ['per','wer','cer']:
        if aggregate(rows,field)!=original[field]['corpus']:
            raise ValueError('Original corpus rate mismatch: '+field)
    summary = summarize(corrected, repetitions=repetitions)
    derived = {**original, **summary}
    if 'strata' in original:
        derived['strata'] = {str(seen):summarize([r for r in corrected if r['content_seen_in_training']==seen],
            repetitions=1000) for seen in [False,True]
            if any(r['content_seen_in_training']==seen for r in corrected)}
    totals = {field:{key:sum(r[field][key] for r in corrected) for key in corrected[0][field]}
              for field in ['per','wer','cer']}
    return {'schema_version':2,'status':'verified-derived-scoring',
        'source':str(folder),'complete_sha256':digest(folder/'complete.json'),
        'method_sha256':digest(Path(__file__)),'metrics_method_sha256':digest(Path(summarize.__code__.co_filename)),
        'verified_trials':len(rows),'verified_sessions':original['sessions'],
        'corrected_edit_totals':totals,'word_corrections':corrections,
        'rate_changed_trials':sum(c['rate_changed'] for c in corrections),
        'component_changed_trials':len(corrections),
        'original_metrics':original,'derived_metrics':derived,
        'uncertainty_method':{'seed':101,'repetitions':repetitions,'strata_repetitions':1000,
            'unit':'nested session/trial bootstrap; conditional on selected checkpoint, not selection-adjusted'},
        'scope':'Standard whitespace WER including empty transcripts and repeated spaces. Raw outputs unchanged; PER/CER edit counts unchanged. Not full matrix coverage or scientific acceptance.',
        'artifact_sha256':completed['artifact_sha256']}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing derived evidence; choose a fresh output')
    result = verify(args.folder)
    save_json(args.output,result)
    print(json.dumps({k:result[k] for k in ['verified_trials','rate_changed_trials','component_changed_trials']}))
