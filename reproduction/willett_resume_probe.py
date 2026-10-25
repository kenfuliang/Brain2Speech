"""Guarded, isolated short GPU continuation of the sealed E008 parent."""
import argparse
import json
import sys
from pathlib import Path

import willett_fail_stop as safety


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--last-attempt', type=int, required=True)
    parser.add_argument('--boundary', type=int)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    safety.require_replay_evidence(contract)
    assert safety.recipe.sha(Path(safety.__file__)) == contract['safety_wrapper_sha256']
    assert not args.output.exists(), 'Never overwrite a probe'
    parent = Path(contract['parent']).resolve()
    assert parent not in args.output.resolve().parents
    assert args.last_attempt in (2957, 2958), 'Only the frozen two-update gate is permitted'
    if args.boundary is not None:
        assert args.boundary == 2958
        assert safety.recipe.sha(args.checkpoint) == contract['parent_checkpoint_sha256']
    else:
        assert args.checkpoint.name == 'resume-boundary-2958.pt'
    identity = json.loads((parent / 'run-identity.json').read_text())
    for relative, expected in identity['source_hashes'].items():
        assert safety.recipe.sha(safety.recipe.ROOT / relative) == expected
    safety.require_serial_resources()
    safety.install_guards()
    sys.argv = [sys.argv[0], '--output', str(args.output), '--seed', '17',
                '--resume-probe', str(args.checkpoint), '--last-attempt', str(args.last_attempt)]
    if args.boundary is not None:
        sys.argv += ['--resume-boundary', str(args.boundary)]
    safety.recipe.main()
    assert safety.recipe.sha(parent / 'latest.pt') == contract['parent_checkpoint_sha256']


if __name__ == '__main__':
    main()
