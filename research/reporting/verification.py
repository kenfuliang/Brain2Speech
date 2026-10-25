"""Independent completed-output audit and transparent component-only corrections."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.audit import digest, save_json
from research.metrics import edits, aggregate, greedy


def word_edits(reference, hypothesis):
    # An empty transcript has zero words, not one empty-string word.
    return edits(reference.split(),hypothesis.split())


def verify(folder, expected_ids=None):
    folder = Path(folder)
    completed = json.loads((folder/'complete.json').read_text())
    rows = json.loads((folder/'predictions.json').read_text())
    metrics = json.loads((folder/'metrics.json').read_text())
    for name in ['predictions.json','metrics.json','logits.npz','embeddings.npy']:
        if completed.get('artifact_sha256',{}).get(name)!=digest(folder/name):
            raise ValueError('Missing or altered artifact hash: '+name)
    ids = [r['id'] for r in rows]
    if len(set(ids))!=len(ids) or ids!=completed['identity']['ids'] or (expected_ids is not None and ids!=expected_ids):
        raise ValueError('Trial identity/order mismatch')
    if len(rows)!=metrics['trials'] or len({r['session'] for r in rows})!=metrics['sessions']:
        raise ValueError('Trial/session count mismatch')
    corrections, verified = [], []
    with np.load(folder/'logits.npz') as logits:
        if len(logits.files)!=len(rows):
            raise ValueError('Logit accounting mismatch')
        for i,row in enumerate(rows):
            x = logits[f'trial_{i:04d}']
            if x.ndim!=2 or x.shape[1]!=41 or not np.isfinite(x).all() or greedy(x)!=row['predicted_phones']:
                raise ValueError('Logit/phoneme mismatch')
            if edits(row['reference_phones'],row['predicted_phones'])!=row['per']:
                raise ValueError('Phoneme edit mismatch')
            if edits(list(row['text']),list(row['decoded_text']))!=row['cer']:
                raise ValueError('Character edit mismatch')
            word = word_edits(row['text'],row['decoded_text'])
            if (word['errors'],word['reference_units'])!=(row['wer']['errors'],row['wer']['reference_units']):
                raise ValueError('Word rates changed under standard tokenization; selection/results require repair')
            if word!=row['wer']:
                corrections.append({'id':row['id'],'original':row['wer'],'corrected':word})
            verified.append({**row,'wer':word})
    for field in ['per','wer','cer']:
        if aggregate(verified,field)!=metrics[field]['corpus']:
            raise ValueError('Corpus rate mismatch: '+field)
    totals = {field:{key:sum(r[field][key] for r in verified) for key in verified[0][field]}
              for field in ['per','wer','cer']}
    return {'source':str(folder),'complete_sha256':digest(folder/'complete.json'),
            'method_sha256':digest(Path(__file__)),'verified_trials':len(rows),
            'verified_sessions':metrics['sessions'],'corrected_edit_totals':totals,
            'component_corrections':corrections,'all_rates_unchanged':True,
            'scope':'Corrected standard whitespace word components; original artifacts and every per-trial/corpus word error rate preserved. Original uncertainty remains conditional on the selected development checkpoint, not selection-adjusted.',
            'artifact_sha256':completed['artifact_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing audit; choose a new output path')
    result = verify(args.folder)
    save_json(args.output,result)
    print(json.dumps({'verified_trials':result['verified_trials'],
                      'component_corrections':len(result['component_corrections']),
                      'all_rates_unchanged':result['all_rates_unchanged']}))
