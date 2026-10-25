"""Launch only the prepared continuation with sealed current-state resume proof."""
import argparse
import json
import sys
from pathlib import Path

import willett_fail_stop as safety


def verify_gate(contract, gate_path):
    assert safety.recipe.sha(gate_path) == contract['guarded_gpu_resume_gate_sha256']
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'passed-recipe-and-fresh-resume'
    assert gate['next_attempt'] == 2959
    assert gate['parent_unchanged'] and gate['model_and_optimizer_exact'] and gate['data_stream_and_rng_exact']
    assert gate['parent_checkpoint_sha256'] == contract['parent_checkpoint_sha256']
    for path, expected in gate['artifact_sha256'].items():
        assert safety.recipe.sha(Path(path)) == expected, 'Resume-gate artifact changed: ' + path
    for relative, expected in gate['source_hashes'].items():
        assert safety.recipe.sha(safety.recipe.ROOT / relative) == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.output / 'continuation-contract.json').read_text())
    verify_gate(contract, args.gate)
    # First full launch must use the independent, unmodified prepared child.
    # A later interrupted-run resume needs a separately sealed current-state check.
    for relative, expected in contract['parent_artifact_sha256'].items():
        assert safety.recipe.sha(args.output / relative) == expected, 'Prepared child artifact changed'
    identity = json.loads((args.output / 'run-identity.json').read_text())
    for relative, expected in identity['tfrecord_hashes'].items():
        assert safety.recipe.sha(safety.recipe.ROOT / relative) == expected, 'Training/evaluation data changed'
    safety.require_replay_evidence(contract)
    safety.require_serial_resources()
    sys.argv = [sys.argv[0], '--output', str(args.output), '--seed', '17', '--resume',
                '--last-attempt', '10000', '--require-gate', str(args.gate)]
    safety.main()


if __name__ == '__main__':
    main()
