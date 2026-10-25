"""CPU comparison of isolated E016 reference and fresh-process restart states."""
import argparse
import json
import math
from pathlib import Path

import torch
from bit_resume_probe import exact
from research.audit import digest, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--restarted', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    states, seals = [], []
    for folder in (args.reference, args.restarted):
        status = json.loads((folder/'status.json').read_text())
        assert status['status'] == 'probe-complete-comparison-required' and status['production_unchanged']
        path = folder/'final.pt'
        seal = json.loads(path.with_suffix('.json').read_text())
        assert digest(path) == seal['sha256']
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert {'model', 'optimizer', 'scheduler', 'sampler_rng', 'torch_rng', 'mps_rng',
                'batches', 'cursor', 'steps', 'journal'} <= state.keys()
        assert state['steps'] == 219727 and state['cursor'] == 2
        assert len(state['journal']) == 2
        assert sorted(i for group in state['batches'] for i in group) == list(range(8800))
        for index, row in enumerate(state['journal']):
            assert row['batch'] == index and len(row['ids']) == len(state['batches'][index])
            assert len(row['lengths']) == len(row['ids']) and all(n > 0 for n in row['lengths'])
            assert math.isfinite(row['loss']) and math.isfinite(row['lr'])
            assert len(row['logp_sha256']) == 64
        assert all(torch.isfinite(value).all() for value in state['model'].values())
        assert state['optimizer']['state'] and state['optimizer']['param_groups']
        for slots in state['optimizer']['state'].values():
            for value in slots.values():
                if isinstance(value, torch.Tensor):
                    assert torch.isfinite(value).all()
        assert digest(path) == seal['sha256']
        states.append(state); seals.append(seal)
    assert seals[0]['parent_sha256'] == seals[1]['parent_sha256']
    assert seals[0]['probe_sha256'] == seals[1]['probe_sha256'] == digest(Path(__file__).parent/'bit_resume_probe.py')
    exact(states[0], states[1])
    save_json(args.output, dict(passed=True, reference=str(args.reference.resolve()),
        restarted=str(args.restarted.resolve()), seals=seals, auditor_sha256=digest(Path(__file__)),
        exact=['model', 'optimizer', 'scheduler', 'sampler_rng', 'torch_rng', 'mps_rng',
               'batch_order', 'logit_hashes', 'losses', 'learning_rates', 'steps'],
        scope='Two isolated epoch800 updates from saved epoch799; one fresh-process restart',
        limitations=['Does not retroactively execute a full production interruption',
                     'No production budget extension, checkpoint reselection or new performance claim']))
    print('passed-exact-BIT-fresh-process-resume')


if __name__ == '__main__':
    main()
