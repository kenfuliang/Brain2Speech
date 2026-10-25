"""Sealed E017/E000 600-trial paired WER comparison, never checkpoint selection."""
import argparse
import json
from pathlib import Path

from train_pytorch_recipe import ROOT, sha
from paired_reproduction_metrics import paired_wer


def align_released(candidate, released):
    assert len(candidate) == len(released['true_transcripts']) == len(released['decoded_transcripts']) == 600
    baseline = []
    for day in range(4, 19):
        group = candidate[(day-4)*40:(day-3)*40]
        assert [row['day'] for row in group] == [day]*40
        assert [row['trial_index'] for row in group] == list(range(40))
    for row, reference, transcript in zip(candidate, released['true_transcripts'], released['decoded_transcripts']):
        assert row['text'] == reference, 'Released-reference order/content mismatch'
        assert row['trial_id'] == f"{row['session']}/test/{row['trial_index']:04d}"
        baseline.append(dict(trial_id=row['trial_id'], session=row['session'], reference=reference, transcript=transcript))
    return baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    verification = json.loads((args.run/'verification.json').read_text())
    assert verification['passed'] and verification['owned_decoder_exited']
    for name, expected in verification['scored_artifact_sha256'].items():
        assert sha(args.run/name) == expected
    assert json.loads((args.run/'status.json').read_text())['status'] == 'complete'
    rows = json.loads((args.run/'decoded-predictions.json').read_text())
    baseline_path = ROOT/'reproduction/results/released-checkpoint-3gram.json'
    released = json.loads(baseline_path.read_text())
    baseline = align_released(rows, released)
    candidate = [dict(trial_id=row['trial_id'], session=row['session'], reference=row['text'],
                      transcript=row['decoded_text']) for row in rows]
    legacy = paired_wer(candidate, baseline, tokenization='legacy-literal-space')
    standard = paired_wer(candidate, baseline, tokenization='standard-whitespace')
    assert abs(legacy['baseline_wer']-released['measured']['wer'][0]) < 1e-12
    result = dict(primary=legacy, secondary=standard,
        baseline_sha256=sha(baseline_path), verification_sha256=sha(args.run/'verification.json'),
        comparator_sha256=sha(Path(__file__)),
        trial_identity_scope='Released artifact lacks explicit IDs; exact600-reference order bound to19-session source evaluator,day4..18,40 per day; duplicate text alone is not an identity match',
        limitations=['Conditional checkpoint comparison; public selection/evaluation overlap',
                     'MPS/TensorFlow stochastic differences remain; numerical agreement is not exact protocol matching'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(legacy_wer=legacy['candidate_wer'], gap=legacy['paired_gap_candidate_minus_baseline'],
                         ci95=legacy['paired_gap_ci95'], verdict=legacy['verdict'])))


if __name__ == '__main__':
    main()
