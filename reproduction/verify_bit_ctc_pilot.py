"""Independent saved-logit/accounting audit of E015, with conditional trial CI."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.audit import digest, save_json, normalize_text


def distance(a, b):
    costs = list(range(len(b)+1))
    for i, token in enumerate(a, 1):
        nxt = [i]
        for j, other in enumerate(b, 1):
            nxt.append(min(costs[j]+1, nxt[-1]+1, costs[j-1]+int(token != other)))
        costs = nxt
    return costs[-1]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); run = args.run.resolve()
    contract = json.loads((run/'contract.json').read_text())
    assert digest(ROOT/'reproduction/train_bit_ctc_pilot.py') == contract['runner_sha256']
    vendor = ROOT/'research/vendor/bit-f107ae9'
    assert all(digest(vendor/k) == h for k,h in contract['source_hashes'].items())
    data = json.loads((run/'data-identity.json').read_text())
    assert all(digest(ROOT/k) == h for k,h in data['sources'].items())
    trials = json.loads((run/'trials.json').read_text())
    train = [r for r in trials if r['split'] == 'train']; val = [r for r in trials if r['split'] != 'train']
    assert len(train) == 512 and len(val) == 40
    assert not ({r['id'] for r in train} & {r['id'] for r in val})
    assert not ({' '.join(r['text'].lower().split()) for r in train} &
                {' '.join(r['text'].lower().split()) for r in val})
    history = json.loads((run/'history.json').read_text())
    assert [r['step'] for r in history] == list(range(0,contract['steps']+1,50))
    for item in history:
        folder = run/f'evaluation-{item["step"]:04d}'
        rows = json.loads((folder/'predictions.json').read_text())
        assert [r['trial_id'] for r in rows] == [r['id'] for r in val]
        loss, errors = 0., 0
        with np.load(folder/'logits.npz') as archive:
            for i, (row, source) in enumerate(zip(rows,val)):
                logits = archive[f'trial_{i:04d}']
                argmax = logits.argmax(-1).tolist()
                phones = [p for j,p in enumerate(argmax) if p != 40 and (j == 0 or p != argmax[j-1])]
                n = distance([p-1 for p in source['phonemes']], phones)
                assert row['phoneme_prediction'] == phones and row['per']['errors'] == n
                errors += n
                lp = torch.from_numpy(logits[:,[40]+list(range(40))]).unsqueeze(1)
                loss += torch.nn.functional.ctc_loss(lp, torch.tensor(source['phonemes']),
                    torch.tensor([len(logits)]), torch.tensor([len(source['phonemes'])]),
                    blank=0,reduction='sum',zero_infinity=True).item()
        expected = item['validation']
        assert errors == expected['phone_errors']
        assert abs(loss/expected['reference_phones'] - expected['ctc_per_reference_phone']) < 1e-5
        assert digest(run/f'checkpoint-{item["step"]:04d}.pt') == item['checkpoint_sha256']
    selected = json.loads((run/'selection.json').read_text())
    assert selected['step'] == min(history, key=lambda h:h['validation']['per'])['step']
    runtime = json.loads((run/'verification.json').read_text())
    assert runtime['passed'] and runtime['vendor_hashes_unchanged'] and runtime['changed_tensors']
    assert runtime['finite_nonzero_gradient_updates'] == contract['steps']
    audit = {'passed': True, 'saved_logit_ctc_per_inventory_and_selection': True,
             'source_and_checkpoint_hashes': True, 'train_validation_prompt_disjoint': True,
             'limitations': 'One seed and historically selected one-day validation; not independent test. Gradient/reload checks use runner evidence.'}
    if (run/'decoded-metrics.json').exists():
        rows = json.loads((run/'decoded-predictions.json').read_text())
        metrics = json.loads((run/'decoded-metrics.json').read_text())
        for field in ('wer','cer'):
            for row in rows:
                ref, hyp = normalize_text(row['reference']), normalize_text(row['transcript'])
                a,b = (ref.split(),hyp.split()) if field=='wer' else (ref,hyp)
                assert distance(a,b) == row[field]['errors']
                assert len(a) == row[field]['reference_units']
            assert sum(r[field]['errors'] for r in rows) == metrics[field]['errors']
        rng = np.random.default_rng(101); rates = []
        for _ in range(2000):
            sampled = [rows[i] for i in rng.integers(0,len(rows),len(rows))]
            rates.append(sum(r['wer']['errors'] for r in sampled)/sum(r['wer']['reference_units'] for r in sampled))
        audit.update(independent_word_character_accounting=True,
                     conditional_trial_bootstrap_wer_ci95=np.quantile(rates,[.025,.975]).tolist(),
                     uncertainty='Trial-only interval conditional on selected checkpoint/cohort; not selection-adjusted, session-level or seed dispersion')
    save_json(run/'independent-verification.json',audit)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
