"""Read-only goal evidence audit; writes a new snapshot, never trains/selects."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.audit import digest, save_json
from research.metrics import edits, summarize
from paired_reproduction_metrics import paired_wer
from compare_willett_final import align_released
from export_willett_final import require_idle
from train_bit_full import effective_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    require_idle()
    seals = {}

    def read(path):
        seals[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    def sealed(folder, mapping):
        for name, expected in mapping.items():
            path = folder/name
            assert digest(path) == expected, str(path)
            seals[str(path.relative_to(ROOT))] = expected

    bit = ROOT/'research/runs/e016-bit-full-s42-20261009-01'
    wil = ROOT/'research/runs/e017-willett-continuation-s17-20261009-01'
    final = ROOT/'research/runs/e017-willett-selected-final-20261009-01'
    ba = read(bit/'completion-audit-20261009-02.json')
    wa = read(wil/'completion-audit-20261009-02.json')
    assert ba['passed'] and (ba['epoch'], ba['steps']) == (800, 220000)
    assert wa['passed'] and wa['complete_frozen_budget'] and wa['continuation_attempts_verified'] == 7044
    assert all((bit/f'epoch-{i:04d}.pt').is_file() for i in range(1, 801))
    assert read(bit/'effective-config.json') == effective_config(800)
    gate = read(ROOT/'research/runs/willett-current-resume-gate-20261009-02.json')
    assert gate['status'] == 'passed-recipe-and-fresh-resume'
    sealed(ROOT, gate['source_hashes'])
    for name, expected in gate['artifact_sha256'].items():
        assert digest(Path(name)) == expected
        seals[str(Path(name).relative_to(ROOT))] = expected
    data = read(bit/'data-identity.json')
    sealed(ROOT, data['source_files'])
    assert digest(ROOT/'research/data/manifest.json') == data['manifest_sha256']
    assert data['counts'] == dict(train=8800, val=440, test=440)
    assert digest(bit/'prepared.pt') == read(bit/'cache-identity.json')['sha256']
    source_dir = ROOT/'research/runs/e016-bit-source-metric-20261009-01'
    source = read(source_dir/'verification.json')
    assert source['passed']
    sealed(source_dir, source['artifact_sha256'])
    fm = read(bit/'final-metrics.json')
    for name in ('validation', 'test'):
        assert source['results'][name]['fresh_trimmed_logits_exact']
        assert source['results'][name]['source_compatible_per'] == fm[name]['source_compatible_per']
    restart = read(ROOT/'research/runs/e016-bit-resume-verification-20261009-01.json')
    assert restart['passed'] and len(restart['exact']) == 11
    for folder, seal in zip((restart['reference'], restart['restarted']), restart['seals']):
        assert digest(Path(folder)/'final.pt') == seal['sha256']
    identity = read(final/'identity.json')
    sealed(final, identity['artifacts_sha256'])
    assert digest(Path(identity['checkpoint'])) == identity['checkpoint_sha256']
    assert read(final/'export-verification.json')['passed']
    verification = read(final/'verification.json')
    assert verification['passed'] and verification['owned_decoder_exited']
    sealed(final, verification['scored_artifact_sha256'])
    assert read(final/'status.json')['status'] == 'complete'
    rows = read(final/'decoded-predictions.json')
    for row in rows:
        assert edits(row['text'].split(' '), row['decoded_text'].split(' ')) == row['wer']
        assert edits(list(row['text']), list(row['decoded_text'])) == row['cer']
    metrics = read(final/'metrics.json')
    recomputed = summarize(rows, repetitions=2000)
    for key in ('per', 'wer', 'cer', 'by_day'):
        assert recomputed[key] == metrics[key]
    released = read(ROOT/'reproduction/results/released-checkpoint-3gram.json')
    baseline = align_released(rows, released)
    candidate = [dict(trial_id=r['trial_id'], session=r['session'], reference=r['text'],
                      transcript=r['decoded_text']) for r in rows]
    paired = read(ROOT/'research/runs/e017-willett-paired-comparison-20261009-01.json')
    for key, policy in (('primary', 'legacy-literal-space'), ('secondary', 'standard-whitespace')):
        assert paired_wer(candidate, baseline, tokenization=policy) == paired[key]
    bdecode = bit/'fixed3gram-20261009-01'
    bv = read(bdecode/'verification.json')
    assert bv['passed'] and bv['owned_decoder_exited']
    sealed(bit, read(bdecode/'contract.json')['export_sha256'])
    for name in ('validation', 'test'):
        br = read(bdecode/f'{name}-predictions.json')
        bm = read(bdecode/f'{name}-metrics.json')
        for r in br:
            assert edits(r['reference'].split(), r['transcript'].split()) == r['wer']
            assert edits(list(r['reference']), list(r['transcript'])) == r['cer']
        total = summarize(br, repetitions=2000)
        for key in ('per', 'wer', 'cer', 'by_day'):
            assert total[key] == bm[key]
    preservation = read(ROOT/'research/runs/e015-bit-ctc-s17-20261008-01/v1-preservation.json')
    assert len(preservation['files']) == 1543
    sealed(ROOT/'research/runs/v1-comparison', preservation['files'])
    charts = read(ROOT/'docs/assets/reproduction-learning-20261009-01/provenance.json')
    sealed(ROOT, charts['sources_sha256'])
    journals = [ROOT/'reproduction/results/retraining/pytorch-retrain-s17-20261007-01/training.jsonl']
    journals += sorted(wil.glob('training-resume-*.jsonl'))
    batches = []
    for path in journals:
        seals[str(path.relative_to(ROOT))] = digest(path)
        batches.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    assert [r['attempt'] for r in batches] == list(range(10001))
    assert all(math.isfinite(r['seconds']) and r['seconds'] >= 0 for r in batches)
    runtime = dict(full_logged_batch_seconds=sum(r['seconds'] for r in batches),
        continuation_logged_batch_seconds=sum(r['seconds'] for r in batches[2957:]),
        scope='Summed per-update timers, excluding stream startup/replay, validation, checkpoint I/O; not complete wall time',
        bit_wall_seconds=read(bit/'status.json')['elapsed_seconds'])
    save_json(args.output, dict(passed=True, timestamp=datetime.now(timezone.utc).isoformat(),
        scope='Fresh CPU budget/source/data/hash/scoring/uncertainty and preservation audit; prior actual reload/restart proofs sealed, not rerun here',
        v1_preserved_files_verified=1543, bit_archives_present=800, runtime=runtime,
        claims=dict(willett='inconclusive paired agreement', bit_scratch='inconclusive unmatched',
                    bit_all='blocked prerequisites', bit_cascade='blocked prerequisites', online='blocked offline'),
        evidence_sha256=seals, auditor_sha256=digest(Path(__file__)),
        limitations=['No claim of exact paper reproduction, seed dispersion, true peak memory or full rendered-report QA',
                     'Source-access search is bounded; unidentified artifacts may exist elsewhere']))
    print(json.dumps(dict(passed=True, runtime=runtime, evidence_count=len(seals))))


if __name__ == '__main__':
    main()
