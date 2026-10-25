"""Checkpoint → causal features → logits → fixed WFST → auditable metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from research.audit import ROOT, digest, save_json, space_guard
from research.data import prepare, load_trials
from research.metrics import edits, summarize
from research.train import load_checkpoint, predict
from research.experiment.streaming import stream

IMAGE = 'speechbci-repro:522a9605'
IMAGE_ID = 'sha256:d7dad9120179787a8e2f949801fce1f9f2a360b170699e4724311fa4a8141432'
SCORING = {'acoustic_scale': .8, 'beam': 18, 'nbest': 1, 'blank_penalty': float(np.log(2)),
           'phone_order': [40, 39]+list(range(39)), 'rescore': False}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class FixedDecoder:
    def __init__(self, log):
        space_guard()
        inspected = subprocess.check_output(['docker', '--context', 'colima-speechbci',
                    'image', 'inspect', IMAGE, '--format', '{{.Id}}'], text=True).strip()
        if inspected != IMAGE_ID:
            raise ValueError('E000 image identity mismatch')
        self.identity = {'image_id': inspected, 'scoring': SCORING,
            'worker_sha256': digest(ROOT/'research/experiment/wfst_worker.py'),
            'lm_files': {n: digest(ROOT/'workspace/languageModel'/n) for n in ['TLG.fst', 'words.txt']}}
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        self.log = Path(log).open('a')
        self.process = subprocess.Popen(['docker', '--context', 'colima-speechbci', 'run', '--rm', '-i',
            '--platform', 'linux/amd64', '-v', f'{ROOT}/workspace:/data:ro',
            '-v', f'{ROOT}/research/runs:/runs', '-v', f'{ROOT}/research/experiment:/worker:ro',
            '--entrypoint', 'python', IMAGE, '-u', '/worker/wfst_worker.py'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        if not self.read().get('ready'):
            raise RuntimeError('Decoder did not initialize')

    def read(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('WFST worker exited; inspect decoder log')
            if line.startswith('WFST_JSON '):
                response = json.loads(line[len('WFST_JSON '):])
                if 'error' in response:
                    raise RuntimeError(response['error'])
                return response
            self.log.write(line)

    def decode(self, output):
        relative = Path(output).resolve().relative_to(ROOT/'research/runs')
        self.process.stdin.write(json.dumps({'folder': '/runs/'+str(relative)})+'\n')
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if getattr(self, 'process', None):
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
            self.log.close()


def evaluate(checkpoint, trials, stats, output, device, decoder, condition='normal', training_texts=()):
    output = Path(output)
    identity = {'checkpoint_sha256': digest(checkpoint), 'ids': [t['id'] for t in trials],
                'stats_sha256': json_hash(stats), 'condition': condition, 'backend': str(device),
                'references_sha256': json_hash([(t['text'], t['labels'].tolist()) for t in trials]),
                'content_training_sha256': json_hash(sorted(training_texts)), 'decoder': decoder.identity}
    if (output/'complete.json').exists():
        completed = json.loads((output/'complete.json').read_text())
        if completed['identity'] != identity:
            raise ValueError('Evaluation identity mismatch')
        for name, expected in completed.get('artifact_sha256', {}).items():
            if not (output/name).exists() or digest(output/name) != expected:
                raise ValueError('Completed evaluation artifact changed: '+name)
        return json.loads((output/'metrics.json').read_text())
    space_guard()
    model, _ = load_checkpoint(checkpoint, device)
    prepared, _ = prepare(trials, stats)
    native = predict(model, prepared, device, output, condition, training_texts)
    # Fixed first-trial prefix makes streaming latency collection bounded and repeatable.
    _, streaming = stream(model, prepared[0]['x'][:min(128,len(prepared[0]['x']))], device)
    decoded = decoder.decode(output)
    rows = json.loads((output/'predictions.json').read_text())
    if len(decoded['transcripts']) != len(rows):
        raise ValueError('Decoded transcript accounting mismatch')
    for row, text, latency in zip(rows, decoded['transcripts'], decoded['decode_ms']):
        row.update(decoded_text=text, wer=edits(row['text'].split(' '), text.split(' ')),
                   cer=edits(list(row['text']), list(text)), wfst_ms=latency)
    metrics = {**native, **summarize(rows, repetitions=2000)}
    metrics['streaming_encoder'] = streaming
    metrics['mean_wfst_ms_per_trial'] = float(np.mean(decoded['decode_ms']))
    metrics['wfst_timing_scope'] = 'x86 QEMU; separate from native encoder, not streaming endpoint latency'
    metrics['strata'] = {str(seen): summarize([r for r in rows if r['content_seen_in_training'] == seen],
                        repetitions=1000) for seen in [False, True]
                        if any(r['content_seen_in_training'] == seen for r in rows)}
    save_json(output/'predictions.json', rows)
    save_json(output/'metrics.json', metrics)
    save_json(output/'complete.json', {'identity': identity,
              'artifact_sha256':{name:digest(output/name) for name in
                                ['predictions.json','metrics.json','logits.npz','embeddings.npy']}})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((ROOT/'research/data/manifest.json').read_text())
    _, payload = load_checkpoint(args.checkpoint, torch.device('cpu'))
    trials = load_trials(manifest, manifest['folds']['development'], 'test')
    if args.limit:
        trials = trials[:args.limit]
    texts = {r['text'] for s in manifest['folds']['initial_train']
             for r in manifest['sessions'][s]['train']['trials']}
    decoder = FixedDecoder(args.output/'decoder.log')
    try:
        print(json.dumps(evaluate(args.checkpoint, trials, payload['stats'], args.output,
                                 torch.device('cpu'), decoder, training_texts=texts), indent=2))
    finally:
        decoder.close()


if __name__ == '__main__':
    main()
