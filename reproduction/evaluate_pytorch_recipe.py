"""Read-only checkpoint evaluation using the published preprocessing and fixed WFST.

Export first; decoding is a separate explicit action so the resident v1 worker is
never interrupted or duplicated implicitly. No optimizer or training is invoked.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from train_pytorch_recipe import ROOT, RecipeData, sha, tensors, greedy, write
from pytorch_baseline import PublishedGRU

sys.path.insert(0, str(ROOT))
from research.metrics import edits, summarize


def export(checkpoint, output):
    if output.exists():
        raise ValueError('Use a new export directory; preserve existing artifacts')
    if __import__('shutil').disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError('Storage guard: below 30 GiB')
    output.mkdir(parents=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = PublishedGRU().to('mps')
    model.load_state_dict(state['model'])
    model.eval()
    service = RecipeData(output, state['seed'])
    rows, logits = [], {}
    try:
        initial = service.call('initial')
        configuration = json.loads(str(initial['configuration']))
        sessions = configuration['dataset']['sessions']
        for index in range(19):
            data = service.call('validation', index=index)
            day = int(data['day'])
            if day < 4:
                continue
            x, target, lengths, labels = tensors(data)
            with torch.no_grad():
                y = model(x, day).cpu()
                losses = torch.nn.functional.ctc_loss(
                    y.log_softmax(-1).transpose(0, 1), target, lengths, labels,
                    blank=40, reduction='none')
            if not torch.isfinite(y).all() or not torch.isfinite(losses).all():
                raise ValueError('Nonfinite evaluation output')
            for i in range(len(x)):
                native = y[i, :int(lengths[i])].numpy()
                logits['trial_%04d' % len(rows)] = native
                codes = data['transcription'][i]
                end = int(np.flatnonzero(codes == 0)[0])
                text = ''.join(chr(int(c)) for c in codes[:end])
                rows.append(dict(day=day, session=sessions[day], trial_index=i,
                                 text=text, prediction_loss=float(losses[i]),
                                 per=edits(target[i, :int(labels[i])].tolist(),
                                           greedy(y[i], int(lengths[i])))))
            write(output/'status.json', dict(status='exporting', trials=len(rows)))
            print(json.dumps(dict(day=day, trials=len(rows))), flush=True)
    finally:
        service.close()
    assert len(rows) == 600
    assert {d: sum(r['day'] == d for r in rows) for d in range(4, 19)} == {d: 40 for d in range(4, 19)}
    np.savez_compressed(output/'logits.npz', **logits)
    write(output/'predictions.json', rows)
    write(output/'identity.json', dict(
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha(checkpoint),
        next_attempt=state['next_attempt'], seed=state['seed'],
        configuration=configuration, trials=600, evaluation_days=list(range(4, 19)),
        preprocessing='original released public-test preprocessing; no augmentation',
        source_sha256={name: sha(ROOT/'reproduction'/name) for name in
                       ['evaluate_pytorch_recipe.py', 'pytorch_baseline.py', 'tf_recipe_service.py']},
        artifacts_sha256={name: sha(output/name) for name in ['logits.npz', 'predictions.json']}))
    write(output/'metrics-before-lm.json', dict(
        **summarize(rows), mean_prediction_loss=float(np.mean([r['prediction_loss'] for r in rows])),
        loss_definition='mean unnormalized CTC loss per sentence, evaluation mode, no augmentation'))
    write(output/'status.json', dict(status='awaiting-decoder', trials=600, training_restarted=False))


def decode(output):
    # This reuses the verified worker/scoring without modifying its frozen source.
    from research.experiment.evaluator import FixedDecoder
    subprocess = __import__('subprocess')
    active = subprocess.check_output(['docker', '--context', 'colima-speechbci', 'ps',
                                     '--filter', 'ancestor=speechbci-repro:522a9605', '--format', '{{.ID}}'], text=True)
    if active.strip():
        raise RuntimeError('Resident WFST is in use; wait or explicitly pause its owner first')
    identity = json.loads((output/'identity.json').read_text())
    for name, expected in identity['artifacts_sha256'].items():
        if sha(output/name) != expected:
            raise ValueError('Export artifact changed: '+name)
    write(output/'status.json', dict(status='loading-decoder', trials=600, training_restarted=False))
    decoder = FixedDecoder(output/'decoder.log')
    try:
        write(output/'status.json', dict(status='decoding', trials=600, training_restarted=False))
        decoded = decoder.decode(output)
        rows = json.loads((output/'predictions.json').read_text())
        assert len(decoded['transcripts']) == len(decoded['decode_ms']) == len(rows) == 600
        for row, text, latency in zip(rows, decoded['transcripts'], decoded['decode_ms']):
            row.update(decoded_text=text, wer=edits(row['text'].split(' '), text.split(' ')),
                       cer=edits(list(row['text']), list(text)), wfst_ms=latency)
        write(output/'decoded-predictions.json', rows)
        metrics = summarize(rows, repetitions=2000)
        metrics.update(decoder=decoder.identity,
                       mean_prediction_loss=float(np.mean([r['prediction_loss'] for r in rows])),
                       checkpoint_sha256=identity['checkpoint_sha256'])
        write(output/'metrics.json', metrics)
        write(output/'status.json', dict(status='complete', trials=600, training_restarted=False))
        print(json.dumps(metrics, indent=2), flush=True)
    finally:
        decoder.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decode-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.decode_only:
        try:
            decode(args.output)
        except Exception as error:
            write(args.output/'status.json', dict(status='failed-decoding', error=str(error), training_restarted=False))
            raise
    else:
        if args.checkpoint is None:
            parser.error('--checkpoint is required for export')
        export(args.checkpoint, args.output)
