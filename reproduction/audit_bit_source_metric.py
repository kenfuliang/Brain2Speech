"""Fresh-process E016 selected-checkpoint replay with padded source-PER audit.

Run serially after Willett training; no optimization, reselection or new split.
"""
import argparse
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import torch

from export_willett_final import require_idle
from train_bit_full import forward, upstream, VENDOR
from utils.config import DictConfig
from research.audit import digest, save_json, space_guard
from verify_bit_full import distance


def source_prediction(sequence):
    # Independent reproduction of source defect: drop blanks before collapsing.
    return [token for token, _ in groupby(token for token in sequence if token != 0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_idle(); space_guard()
    assert not args.output.exists()
    run = args.run.resolve()
    contract = json.loads((run/'contract.json').read_text())
    assert contract['epochs'] == 800
    assert digest(run/'prepared.pt') == json.loads((run/'cache-identity.json').read_text())['sha256']
    for name, expected in contract['source_hashes'].items():
        assert digest(VENDOR/name) == expected
    assert digest(Path(__file__).parent/'train_bit_full.py') == contract['runner_sha256']
    assert digest(Path(__file__).parent/'train_bit_ctc_pilot.py') == contract['helper_sha256']
    chosen = json.loads((run/'best-per.json').read_text())
    assert digest(run/chosen['checkpoint']) == chosen['sha256']
    final = json.loads((run/'final-metrics.json').read_text())
    assert final['selected'] == chosen
    assert json.loads((run/'final-audit-20261009-01.json').read_text())['passed']
    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    cfg = DictConfig(json.loads((run/'effective-config.json').read_text()))
    data = torch.load(run/'prepared.pt', map_location='cpu', weights_only=False)
    state = torch.load(run/chosen['checkpoint'], map_location='cpu', weights_only=False)
    net = upstream.NDT(cfg.model, method_name='ctc', features='all', vocab_size=41,
                       blank_id=0, zero_infinity=True).float().to('mps')
    net.load_state_dict(state['model']); net.eval()
    args.output.mkdir(parents=True)
    results = {}
    for partition, name in (('val', 'validation'), ('test', 'test')):
        records = data['records'][partition]
        archive, rows, batches = {}, [], []
        with np.load(run/f'selected-{name}'/'logits.npz') as old:
            for start in range(0, len(records), 32):
                group = records[start:start+32]
                with torch.no_grad():
                    _, logp, lengths = forward(net, group, data['arrays'], 'mps')
                batch_errors = batch_phones = 0
                for offset, (record, values, length) in enumerate(zip(group, logp.numpy(), lengths.tolist())):
                    index = start+offset
                    assert np.isfinite(values).all()
                    native = values[:length, list(range(1, 41))+[0]]
                    np.testing.assert_array_equal(native, old[f'trial_{index:04d}'])
                    prediction = source_prediction(values.argmax(-1).tolist())
                    count = distance(record['phonemes'], prediction)
                    batch_errors += count; batch_phones += len(record['phonemes'])
                    archive[f'trial_{index:04d}'] = values
                    rows.append(dict(trial_id=record['id'], batch_start=start, valid_bins=length,
                        reference_phones=record['phonemes'], source_prediction=prediction,
                        source_errors=count, reference_phones_count=len(record['phonemes'])))
                batches.append(dict(start=start, errors=batch_errors, phones=batch_phones,
                                    per=batch_errors/batch_phones))
        source_per = float(np.mean([batch['per'] for batch in batches]))
        assert source_per == final[name]['source_compatible_per']
        np.savez_compressed(args.output/f'{name}-padded-source-logits.npz', **archive)
        save_json(args.output/f'{name}-rows.json', rows)
        results[name] = dict(source_compatible_per=source_per, batches=batches, trials=len(rows),
                            fresh_trimmed_logits_exact=True)
    save_json(args.output/'verification.json', dict(passed=True, selected=chosen, results=results,
        cache_sha256=digest(run/'prepared.pt'), auditor_sha256=digest(Path(__file__)),
        artifact_sha256={path.name: digest(path) for path in args.output.iterdir() if path.is_file()},
        scope='Fresh selected-checkpoint reload, exact trimmed logit replay and independent padded/broken-collapse/batch-mean PER',
        limitations=['Not interrupted optimizer/data-sampler replay',
                     'Source-compatible defects preserved diagnostically; not standard corpus PER',
                     'Paper/source split and recipe differences remain']))
    print('passed-fresh-selected-checkpoint-source-metric-audit')


if __name__ == '__main__':
    main()
