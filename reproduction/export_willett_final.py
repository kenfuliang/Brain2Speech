"""Selected E017 checkpoint export with raw phone references and reload proof.

No training or decoder launch; run only after the complete frozen budget.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import train_pytorch_recipe as recipe
import willett_fail_stop as safety


def require_idle():
    safety.require_serial_resources()
    for line in subprocess.check_output(['ps', '-ax', '-o', 'pid=,command='], text=True).splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        if any(name in fields[1] for name in ('launch_willett_verified.py', 'willett_resume_probe.py')):
            # Ignore inventory/launcher shells: only an actual Python process counts.
            import shlex
            if Path(shlex.split(fields[1])[0]).name.lower().startswith(('python', 'pypy')):
                raise RuntimeError('Willett trainer/probe is still live')


def checked_phones(target, length):
    values = target[:length].tolist()
    assert values and all(isinstance(v, int) and 0 <= v < 40 for v in values)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_idle()
    assert not args.output.exists(), 'Preserve every existing export'
    completion = json.loads((args.run / 'completion.json').read_text())
    assert completion['status'] == 'training-complete-evaluation-required' and completion['attempts'] == 10001
    identity = json.loads((args.run / 'run-identity.json').read_text())
    for mapping in ('source_hashes', 'tfrecord_hashes'):
        for relative, expected in identity[mapping].items():
            assert recipe.sha(recipe.ROOT / relative) == expected
    selected = json.loads((args.run / 'selected.json').read_text())
    checkpoint = args.run / selected['checkpoint']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['next_attempt'] == selected['attempt'] + 1
    assert state['best_per'] == selected['per']
    assert all(torch.isfinite(value).all() for value in state['model'].values())
    torch.set_num_threads(8)
    model = recipe.PublishedGRU().to('mps')
    model.load_state_dict(state['model'])
    model.eval()
    args.output.mkdir(parents=True)
    service = recipe.RecipeData(args.output, state['seed'])
    selection_rows, rows, logits, timings = [], [], {}, []
    from research.metrics import edits, summarize
    try:
        initial = service.call('initial')
        config = json.loads(str(initial['configuration']))
        sessions = config['dataset']['sessions']
        first = service.call('validation', index=0)
        assert int(first['batches']) == 19
        for index in range(19):
            data = first if index == 0 else service.call('validation', index=index)
            day = int(data['day'])
            assert day == index
            x, target, lengths, labels = recipe.tensors(data)
            torch.mps.synchronize()
            started = time.monotonic()
            with torch.no_grad():
                y = model(x, day).cpu()
            torch.mps.synchronize()
            timings.append(dict(day=day, trials=len(x), seconds=time.monotonic()-started))
            if index == 0:
                reload_model = recipe.PublishedGRU().to('mps')
                reload_model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
                reload_model.eval()
                with torch.no_grad():
                    np.testing.assert_array_equal(y.numpy(), reload_model(x, day).cpu().numpy())
                del reload_model
            losses = torch.nn.functional.ctc_loss(y.log_softmax(-1).transpose(0, 1), target,
                lengths, labels, blank=40, reduction='none')
            assert torch.isfinite(y).all() and torch.isfinite(losses).all()
            for trial in range(len(x)):
                phones = checked_phones(target[trial], int(labels[trial]))
                prediction = recipe.greedy(y[trial], int(lengths[trial]))
                codes = data['transcription'][trial]
                end = int(np.flatnonzero(codes == 0)[0])
                text = ''.join(chr(int(c)) for c in codes[:end])
                row = dict(day=day, session=sessions[day], trial_index=trial,
                    trial_id=f'{sessions[day]}/test/{trial:04d}', text=text,
                    reference_phones=phones, predicted_phones=prediction,
                    output_bins=int(lengths[trial]), source_batch_sha256=recipe.data_hash(data),
                    prediction_loss=float(losses[trial]), per=edits(phones, prediction))
                selection_rows.append(row)
                if day >= 4:
                    logits[f'trial_{len(rows):04d}'] = y[trial, :int(lengths[trial])].numpy()
                    rows.append(row)
            recipe.write(args.output/'status.json', dict(status='exporting', selection_trials=len(selection_rows), trials=len(rows)))
    finally:
        service.close()
    assert len(selection_rows) == 680 and len(rows) == 600
    for day in range(19):
        assert sum(row['day'] == day for row in selection_rows) == (20 if day < 4 else 40)
    selection_per = sum(r['per']['errors'] for r in selection_rows)/sum(r['per']['reference_units'] for r in selection_rows)
    assert selection_per == selected['per'], 'Fresh selected-checkpoint PER differs'
    np.savez_compressed(args.output/'logits.npz', **logits)
    recipe.write(args.output/'predictions.json', rows)
    recipe.write(args.output/'selection-predictions.json', selection_rows)
    recipe.write(args.output/'metrics-before-lm.json', dict(**summarize(rows, repetitions=2000),
        mean_prediction_loss=float(np.mean([r['prediction_loss'] for r in rows])),
        loss_definition='Mean unnormalized CTC per sentence; no augmentation',
        selection_per=selection_per, encoder_batch_timings=timings,
        timing_scope='Native batched encoder plus device-to-host logits; not end-to-end/streaming latency',
        parameter_count=sum(p.numel() for p in model.parameters())))
    recipe.write(args.output/'identity.json', dict(checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=recipe.sha(checkpoint), next_attempt=state['next_attempt'], seed=state['seed'],
        selected=selected, full_budget_attempts=10001, trials=600, selection_trials=680,
        configuration=config, evaluation_days=list(range(4, 19)),
        fresh_model_reload_logits_exact=True, selected_per_exact=True,
        trial_identity_scope='Source ordered single validation batch per day; labels/text and transformed batch hash retained',
        preprocessing='Original offline released test preprocessing; centered smoothing; no augmentation',
        source_sha256={'export_willett_final.py': recipe.sha(Path(__file__))},
        artifacts_sha256={name: recipe.sha(args.output/name) for name in
            ('logits.npz', 'predictions.json', 'selection-predictions.json', 'metrics-before-lm.json')}))
    recipe.write(args.output/'status.json', dict(status='awaiting-decoder', trials=600))


if __name__ == '__main__':
    main()
