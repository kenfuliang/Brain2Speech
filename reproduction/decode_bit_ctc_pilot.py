"""Authorized E015 decoder handover; seal v1, decode, verify, then resume.

No writes to frozen code or completed outputs. Interrupted evaluations recompute.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.audit import digest, save_json, normalize_text, space_guard
from research.metrics import edits
from research.experiment.evaluator import FixedDecoder


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def running(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    return result.stdout.strip()


def independent_distance(a, b):
    previous = list(range(len(b)+1))
    for i, token in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(token != other)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--v1-pid', type=int, required=True)
    parser.add_argument('--v1-container', required=True)
    parser.add_argument('--recover-after-pause', action='store_true',
                        help='Recover only a recorded pausing stage after old processes have exited')
    args = parser.parse_args()
    run = args.run.resolve(); v1 = ROOT / 'research/runs/v1-comparison'
    if args.recover_after_pause:
        preserved = json.loads((run/'handover.json').read_text())
        assert preserved['status'] == 'pausing'
        assert preserved['controller_pid'] == args.v1_pid and preserved['container'] == args.v1_container
        assert not running(args.v1_pid)
        assert not command('docker', '--context', 'colima-speechbci', 'ps', '-q')
    else:
        assert not (run/'handover.json').exists(), 'Use preserved handover for recovery; never repeat blindly'
        assert 'research.experiment.pipeline --stage full' in running(args.v1_pid)
        container = json.loads(command('docker', '--context', 'colima-speechbci', 'inspect', args.v1_container))[0]
        assert container['State']['Running'] and '/worker/wfst_worker.py' in container['Config']['Cmd']
    selected = json.loads((run/'selection.json').read_text())
    assert digest(run/selected['checkpoint']) == selected['checkpoint_sha256']
    assert json.loads((run/'verification.json').read_text())['passed']
    evaluation = run / f'evaluation-{selected["step"]:04d}'
    records = json.loads((evaluation/'predictions.json').read_text())
    with np.load(evaluation/'logits.npz') as archive:
        assert len(archive.files) == len(records) == 40
        for i, row in enumerate(records):
            x = archive[f'trial_{i:04d}']; assert x.shape[1] == 41 and np.isfinite(x).all()
            assert np.allclose(np.exp(x).sum(1), 1., atol=1e-5)
            labels = next(r['phonemes'] for r in json.loads((run/'trials.json').read_text()) if r['id'] == row['trial_id'])
            assert independent_distance([p-1 for p in labels], row['phoneme_prediction']) == row['per']['errors']
    resume = [str(ROOT/'.venv-v1/bin/python'), '-u', '-m', 'research.experiment.pipeline',
              '--stage', 'full', '--wait', '--require-smoke', str(ROOT/'research/runs/v1-wfst-smoke'),
              '--output', str(v1)]
    handover = {'authorization': 'User: Pause v1 for pilot WER, then resume (2026-10-08)',
                'status': 'pausing', 'controller_pid': args.v1_pid, 'container': args.v1_container,
                'interrupted_state': json.loads((v1/'status.json').read_text()), 'resume_command': resume,
                'decoder_definition': 'Existing E000 fixed3gram, no OPT rescoring'}
    if args.recover_after_pause:
        handover = preserved
        handover['shutdown_recovery'] = 'Docker stop reported no exit event; subsequent inspect confirmed container absent, old controller/child absent; no competing container'
    else:
        save_json(run/'handover.json', handover)
        os.kill(args.v1_pid, signal.SIGTERM)
        for _ in range(100):
            if not running(args.v1_pid):
                break
            time.sleep(.1)
        assert not running(args.v1_pid), 'Controller did not exit; stop here'
        subprocess.run(['docker', '--context', 'colima-speechbci', 'stop', '-t', '10', args.v1_container], check=True)
    remaining = command('docker', '--context', 'colima-speechbci', 'ps', '-q')
    assert not remaining, 'Another container owns resources; refuse competing decoder'
    # Seal after the controller exits; preserve all completed records and each referenced artifact.
    sealed = {}
    for completion in sorted(v1.rglob('complete.json')):
        sealed[str(completion.relative_to(v1))] = digest(completion)
        value = json.loads(completion.read_text())
        for name, sha in value.get('artifact_sha256', {}).items():
            path = completion.parent/name
            assert digest(path) == sha, f'Existing completed artifact mismatch: {path}'
            sealed[str(path.relative_to(v1))] = sha
    identity = json.loads((v1/'execution-identity.json').read_text())
    assert digest(ROOT/'research/data/manifest.json') == identity['manifest_sha256']
    assert all(digest(ROOT/k) == sha for k,sha in identity['pipeline_sources'].items())
    save_json(run/'v1-preservation.json', {'files': sealed, 'execution_identity': identity})
    handover.update(status='v1_paused_decoder_released', sealed_files=len(sealed))
    save_json(run/'handover.json', handover)
    print(json.dumps({'status': handover['status'], 'sealed_files': len(sealed)}), flush=True)
    space_guard()
    decoder = FixedDecoder(run/'decoder.log')
    try:
        save_json(run/'decoder-identity.json', decoder.identity)
        response = decoder.decode(evaluation)
        assert len(response['transcripts']) == len(records)
        for row, text, ms in zip(records, response['transcripts'], response['decode_ms']):
            reference = normalize_text(row['reference']); text = normalize_text(text)
            row.update(transcript=text, wer=edits(reference.split(), text.split()),
                       cer=edits(reference, text), decode_ms=ms)
            assert row['wer']['errors'] == independent_distance(reference.split(), text.split())
            assert row['cer']['errors'] == independent_distance(reference, text)
        save_json(run/'decoded-predictions.json', records)
        result = {}
        for metric in ('per', 'wer', 'cer'):
            counts = {k: sum(r[metric][k] for r in records) for k in records[0][metric]}
            counts['rate'] = counts['errors']/counts['reference_units']
            assert counts['errors'] == sum(counts[k] for k in ('substitutions','deletions','insertions'))
            result[metric] = counts
        result.update(selected_update=selected['step'], trials=40,
                      decoder='fixed3gram, not BIT5gram/OPT', decode_ms_total=sum(response['decode_ms']),
                      evidence='Historical selected validation feasibility pilot, not independent test')
        save_json(run/'decoded-metrics.json', result)
        print(json.dumps(result), flush=True)
    finally:
        decoder.close()
    # A decoder handover must release memory before resuming the large resident v1 graph.
    for _ in range(100):
        remaining = command('docker', '--context', 'colima-speechbci', 'ps', '-q')
        if not remaining:
            break
        time.sleep(.1)
    assert not remaining, 'Pilot container remains; do not start a competing v1 decoder'
    assert all(digest(v1/k) == sha for k,sha in sealed.items())
    assert json.loads((v1/'execution-identity.json').read_text()) == identity
    assert all(digest(ROOT/k) == sha for k,sha in identity['pipeline_sources'].items())
    save_json(run/'decode-verification.json', {'passed': True, 'independent_edit_accounting': True,
                                               'sealed_v1_files_unchanged': len(sealed),
                                               'pilot_decoder_exited': True})
    with (run/'v1-resume.log').open('a') as log:
        resumed = subprocess.Popen(resume, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    time.sleep(1)
    assert resumed.poll() is None, 'v1 restart exited; inspect resume log'
    handover.update(status='pilot_verified_v1_resumed', resumed_controller_pid=resumed.pid)
    save_json(run/'handover.json', handover)
    status = json.loads((run/'status.json').read_text()); status['status'] = 'complete_pilot'
    save_json(run/'status.json', status)


if __name__ == '__main__':
    main()
