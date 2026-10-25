"""Verify source archive, audit raw binned features, and freeze chronological folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tarfile

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
ACQUISITION_EVIDENCE = {
    'source': 'Willett 2023 Supplementary Methods section 1.4, Neural recording and processing',
    'source_url': 'https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-023-06377-x/MediaObjects/41586_2023_6377_MOESM1_ESM.pdf',
    'digital_filter': '300 Hz high-pass applied non-causally with 4 ms delay',
    'lookahead_ms': 4,
    'raw_voltage_available_in_release': False,
    'strict_gate': 'pass',
    'accepted_protocol': '4 ms acquisition delay accepted by user; downstream causal relative to available bins',
    'reason': 'Zero neural lookahead is not claimed; released-bin inference includes the documented 4 ms acquisition delay.',
    'possible_revision': 'Explicitly accept 4 ms acquisition delay, timestamp feature availability accordingly, and report bounded-delay causality instead of zero neural lookahead.'
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def space_guard(path=ROOT):
    free = shutil.disk_usage(path).free
    if free < 30 * 1024**3:
        raise RuntimeError('Storage guard: less than 30 GiB free; collection paused')
    return free


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def normalize_text(text):
    return re.sub(r"[^a-zA-Z\- ']", '', text).replace('--', '').lower().strip()


def texts_from_mat(value):
    if value.dtype.kind in 'US' and value.ndim == 2:
        return [''.join(row).strip() for row in value]
    return [''.join(np.asarray(x).astype(str).ravel()).strip() for x in value.ravel()]


def verify_extract(archive, output):
    expected = next(x for x in json.loads((ROOT / 'reproduction/artifacts.json').read_text())
                    ['dataset']['files'] if x['name'] == archive.name)
    if archive.stat().st_size != expected['size'] or digest(archive) != expected['sha256']:
        raise ValueError('Competition source archive integrity check failed')
    print('source SHA-256 verified', flush=True)
    marker = output / '.competition-source-verified.json'
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior.get('sha256') == expected['sha256'] and all(
                (output / item['path']).is_file() and (output / item['path']).stat().st_size == item['size']
                for item in prior['files']):
            return expected['sha256']
    extracted = []
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                raise ValueError(f'Unsafe archive member: {member.name}')
            if not member.isfile() or not member.name.endswith('.mat'):
                continue
            destination = output / path
            extracted.append({'path': str(path), 'size': member.size})
            if destination.exists():
                if destination.stat().st_size != member.size:
                    raise ValueError(f'Incomplete prior extraction: {destination}')
                continue
            space_guard()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, destination.with_suffix('.partial').open('wb') as target:
                shutil.copyfileobj(incoming, target)
            destination.with_suffix('.partial').replace(destination)
    save_json(marker, {'sha256': expected['sha256'], 'files': extracted})
    return expected['sha256']


def audit(data, tfrecords):
    sessions = {}
    prompts = {}
    for partition in ('train', 'test', 'competitionHoldOut'):
        for path in sorted((data / partition).glob('*.mat')):
            mat = loadmat(path)
            texts = texts_from_mat(mat['sentenceText'])
            blocks = mat['blockIdx'].ravel().astype(int)
            tx = mat['tx1'].ravel()
            power = mat['spikePow'].ravel()
            if not (len(tx) == len(power) == len(texts) == len(blocks)):
                raise ValueError(f'Trial count mismatch: {path}')
            rows = []
            for index, (counts, p, text, block) in enumerate(zip(tx, power, texts, blocks)):
                counts, p = counts[:, :128], p[:, :128]
                if counts.shape != p.shape or counts.shape[1] != 128 or not len(counts):
                    raise ValueError(f'Invalid feature shape: {path}:{index}')
                if not (np.isfinite(counts).all() and np.isfinite(p).all()):
                    raise ValueError(f'Nonfinite raw features: {path}:{index}')
                if np.min(counts) < 0 or not np.allclose(counts, np.rint(counts), atol=1e-5):
                    raise ValueError(f'Threshold crossing features are not counts: {path}:{index}')
                if np.min(p) < 0:
                    raise ValueError(f'Negative raw power: {path}:{index}')
                normalized = normalize_text(text)
                uid = f'{path.stem}/{partition}/{index:04d}'
                row = {'id': uid, 'index': index, 'block': int(block),
                       'bins': len(counts), 'duration_seconds': len(counts) * .02,
                       'text': normalized if partition != 'competitionHoldOut' else None}
                rows.append(row)
                if normalized and partition != 'competitionHoldOut':
                    prompts.setdefault(normalized, []).append(uid)
            entry = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                     'trials': rows, 'trial_count': len(rows),
                     'blocks_monotonic': bool(np.all(np.diff(blocks) >= 0)),
                     'duration_seconds': sum(r['duration_seconds'] for r in rows)}
            if not entry['blocks_monotonic']:
                raise ValueError(f'Raw trials not ordered by block: {path}')
            sessions.setdefault(path.stem, {})[partition] = entry
            print(path.stem, partition, len(rows), flush=True)
    eligible = sorted(s for s, parts in sessions.items() if 'train' in parts and 'test' in parts)
    n = len(eligible)
    if n < 5:
        raise ValueError('Too few eligible sessions for chronological research')
    a, b = math.floor(.6*n), math.floor(.8*n)
    folds = {'initial_train': eligible[:a], 'development': eligible[a:b], 'final': eligible[b:]}
    tf = sorted(p.name for p in tfrecords.iterdir() if p.is_dir())
    return {'schema_version': 1, 'sessions': sessions, 'eligible_sessions': eligible,
            'folds': folds, 'tfrecord_sessions': tf,
            'tfrecord_only_sessions': sorted(set(tf) - set(eligible)),
            'raw_only_sessions': sorted(set(eligible) - set(tf)),
            'duplicate_prompts': {k: v for k, v in prompts.items() if len(v) > 1},
            'feature_boundary': 'released unsmoothed, unnormalized 20 ms tx1 counts and spikePow',
            'causal_gate': {**ACQUISITION_EVIDENCE, 'downstream_recovery': 'pass',
                            'end_to_end_causal_claim': 'bounded-delay stream; 4 ms acquisition plus bin/output scheduling and compute'},
            'silent_period_coverage': 'competition snippets contain go periods only; preparation requires sentences archive; between-block rest not recorded',
            'test_policy': 'original test partition held out; train-prefix adaptation; competitionHoldOut excluded',
            'data_audit_only': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'research/data')
    args = parser.parse_args()
    source_hash = verify_extract(ROOT / 'artifacts/dryad/competitionData.tar.gz', args.output)
    report = audit(args.output / 'competitionData', ROOT / 'workspace/derived/tfRecords')
    report['source_archive_sha256'] = source_hash
    save_json(args.output / 'manifest.json', report)
    print(json.dumps({'eligible': report['eligible_sessions'], 'folds': report['folds'],
                      'causal_gate': report['causal_gate']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
