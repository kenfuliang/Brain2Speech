"""Preserved-state E012 continuation with validation-WER checkpoint selection.

Never mutate E012-01 or frozen v1. A selected validation score is not test WER.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

from causal_neural_llm import CausalTextEncoder, causal_check, dataset
from bit_feasibility import load_language, frozen_digest, MODEL, REVISION
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit import ROOT, digest, save_json, space_guard
from research.metrics import edits


class StabilityMonitor:
    """Predeclared sustained validation-loss deterioration, not batch jitter."""
    def __init__(self, baseline, ratio=1.25, patience=5):
        self.best = baseline
        self.ratio = ratio
        self.patience = patience
        self.bad = 0

    def observe(self, value):
        if not np.isfinite(value):
            return 'nonfinite_validation_loss'
        self.best = min(self.best, value)
        self.bad = self.bad+1 if value > self.best*self.ratio else 0
        return 'sustained_validation_deterioration' if self.bad >= self.patience else None


def better(candidate, incumbent):
    # Fewer word errors, then lower token loss; exact ties retain earlier step.
    return (candidate['wer']['errors'], candidate['loss']['real']) < (
        incumbent['wer']['errors'], incumbent['loss']['real'])


def run(parent, output, max_step=10000, max_hours=4):
    space_guard(); torch.manual_seed(17); torch.set_num_threads(4)
    if not torch.backends.mps.is_available():
        raise RuntimeError('Validated MPS backend unavailable')
    old = json.loads((parent/'identity.json').read_text())
    assert digest(Path(__file__).with_name('causal_neural_llm.py')) == old['code_sha256']
    assert digest(MODEL/'model.safetensors') == old['model_sha256']
    state = torch.load(parent/'checkpoint-40.pt', map_location='cpu', weights_only=False)
    assert state['step'] == 40
    trials = dataset(output)
    for name in ['trials.json', 'normalization.json', 'data-identity.json']:
        assert json.loads((output/name).read_text()) == json.loads((parent/name).read_text())
    train, validation = trials[:64], trials[64:]
    model, tokenizer, dim = load_language()
    model.load_adapter(parent/'lora-step-40', adapter_name='resume', is_trainable=True)
    model.set_adapter('resume')
    encoder = CausalTextEncoder(dim).to('mps')
    encoder.load_state_dict(state['encoder'])
    parameters = list(encoder.parameters())+[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in parameters) == old['trainable_parameters']
    opt = torch.optim.AdamW(parameters, lr=old['lr'])
    opt.load_state_dict(state['optimizer'])
    for p in parameters:
        assert opt.state[p]['exp_avg'].shape == p.shape
        assert opt.state[p]['exp_avg'].device == p.device
    rng = np.random.default_rng(); rng.bit_generator.state = state['rng']
    frozen_before = frozen_digest(model)
    prompt = torch.tensor(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Decode the preceding neural activity into an English sentence.'}],
        tokenize=True, add_generation_prompt=True), device='mps')
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append('signal_requested_stop'))
    signal.signal(signal.SIGINT, lambda *_: stop_requested.append('signal_requested_stop'))

    def prefix(row, condition='real', source=None):
        x = (source or row)['x'].to('mps').unsqueeze(0)
        if condition == 'zero':
            x = torch.zeros_like(x)
        neural = encoder(x)
        if condition == 'language-only':
            neural = neural[:, :0]
        return torch.cat([neural, model.get_input_embeddings()(prompt)[None]], 1)

    def loss(row, condition='real', source=None):
        embeds = prefix(row, condition, source)
        ids = torch.tensor(tokenizer.encode(row['text'], add_special_tokens=False)+
                           [tokenizer.eos_token_id], device='mps')[None]
        return model(inputs_embeds=torch.cat([embeds, model.get_input_embeddings()(ids)], 1),
                     labels=torch.cat([torch.full(embeds.shape[:2], -100, device='mps',
                                                  dtype=torch.long), ids], 1), use_cache=False).loss

    def losses():
        result = {}
        with torch.no_grad():
            for condition in ['real', 'zero', 'shuffled', 'language-only']:
                values, counts = [], []
                for i, row in enumerate(validation):
                    source = validation[(i+1) % 16] if condition == 'shuffled' else None
                    values.append(float(loss(row, condition, source)))
                    counts.append(len(tokenizer.encode(row['text'], add_special_tokens=False))+1)
                result[condition] = float(np.average(values, weights=counts))
            result['train_fixed_probe'] = float(np.mean([float(loss(r)) for r in train[:4]]))
        return result

    def generate(condition):
        predictions = []
        with torch.no_grad():
            for i, row in enumerate(validation):
                source = validation[(i+1) % 16] if condition == 'shuffled' else None
                embeds = prefix(row, condition, source)
                ids = model.generate(inputs_embeds=embeds,
                    attention_mask=torch.ones(embeds.shape[:2], device='mps', dtype=torch.long),
                    max_new_tokens=40, do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
                prediction = tokenizer.decode(ids[0], skip_special_tokens=True)
                predictions.append(dict(id=row['id'], reference=row['text'], prediction=prediction,
                    wer=edits(row['text'].lower().split(), prediction.lower().split())))
        account = {k: sum(r['wer'][k] for r in predictions)
                   for k in ['errors', 'substitutions', 'deletions', 'insertions', 'reference_units']}
        account['rate'] = account['errors']/account['reference_units']
        account['unique_predictions'] = len({r['prediction'] for r in predictions})
        return predictions, account

    def checkpoint(step):
        folder = output/f'step-{step:06d}'; folder.mkdir(exist_ok=False)
        model.save_pretrained(folder/'lora', save_embedding_layers=False)
        # Only the active adapter is needed for ancestry/reload.
        torch.save(dict(encoder=encoder.state_dict(), optimizer=opt.state_dict(),
                        rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), step=step),
                   folder/'checkpoint.pt')
        return folder

    def evaluate(step, folder):
        model.eval(); encoder.eval()
        save_json(output/'status.json', dict(status='evaluating', step=step))
        values = losses(); predictions, account = generate('real')
        record = dict(step=step, loss=values, wer=account,
                      checkpoint=str(folder), checkpoint_sha256=digest(folder/'checkpoint.pt'))
        save_json(folder/'validation-predictions.json', predictions)
        save_json(folder/'evaluation.json', record)
        print(json.dumps({'evaluation': record}), flush=True)
        return record

    start = time.monotonic()
    config = dict(parent=str(parent), parent_checkpoint_sha256=digest(parent/'checkpoint-40.pt'),
        parent_adapter_sha256=digest(parent/'lora-step-40/adapter_model.safetensors'),
        parent_step=40, max_absolute_step=max_step, max_wall_hours=max_hours,
        evaluation_interval=100, selection='minimum validation word edits; then token loss; then earlier step',
        stability=dict(nonfinite='stop immediately', validation_ratio=1.25, consecutive_checks=5,
                       interpretation='sustained generalization deterioration, not proof of numerical instability'),
        storage_floor_gib=30, lr=old['lr'], schedule='unchanged constant',
        device='mps', model_revision=REVISION, source_sha256=digest(Path(__file__)),
        original_identity=old, optimizer_restored=True, sampler_restored=True,
        limitation='No exact bitwise continuation claim: original step-40 checkpoint did not save Torch RNG; current training has no stochastic forward layers. Validation is reused for monitoring/selection, not untouched test.')
    save_json(output/'protocol.json', config)
    initial_folder = checkpoint(40)
    best = evaluate(40, initial_folder)
    reference = json.loads((parent/'final.json').read_text())
    assert abs(best['loss']['train_fixed_probe']-reference['train_fixed_probe']) < 1e-5
    assert abs(best['loss']['real']-reference['real']) < 1e-5
    assert best['wer']['errors'] == 165 and best['wer']['reference_units'] == 135
    save_json(output/'resume-validation.json', dict(passed=True, parent_probe_and_validation_loss_replayed=True,
                                                  parent_word_edits_replayed=True))
    save_json(output/'best.json', best)
    monitor = StabilityMonitor(best['loss']['real'])
    step = 40; reason = None
    with (output/'training.jsonl').open('x') as stream, (output/'evaluations.jsonl').open('x') as ev:
        ev.write(json.dumps(best)+'\n'); ev.flush()
        while step < max_step:
            if stop_requested or (output/'STOP_REQUESTED').exists():
                reason = stop_requested[0] if stop_requested else 'user_requested_stop'; break
            if time.monotonic()-start > max_hours*3600:
                reason = 'wall_time_review_boundary'; break
            try:
                space_guard()
            except RuntimeError:
                reason = 'storage_floor'; break
            model.train(); encoder.train(); opt.zero_grad(set_to_none=True)
            batch_values = []
            for index in rng.choice(64, 2, replace=False):
                value = loss(train[int(index)])
                if not torch.isfinite(value):
                    reason = 'nonfinite_training_loss'; break
                (value/2).backward(); batch_values.append(float(value.detach()))
            if reason:
                break
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                reason = 'invalid_gradient'; break
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
            opt.step(); torch.mps.synchronize(); step += 1
            row = dict(step=step, loss=float(np.mean(batch_values)), gradient_norm=norm,
                       elapsed_seconds=time.monotonic()-start,
                       mps_driver_bytes=torch.mps.driver_allocated_memory())
            stream.write(json.dumps(row)+'\n'); stream.flush()
            save_json(output/'status.json', dict(status='training', best_step=best['step'],
                                                best_validation_wer=best['wer']['rate'], **row))
            if step % 20 == 0:
                print(json.dumps(row), flush=True)
            if step % 100 == 0:
                folder = checkpoint(step); record = evaluate(step, folder)
                ev.write(json.dumps(record)+'\n'); ev.flush()
                if better(record, best):
                    best = record; save_json(output/'best.json', best)
                reason = monitor.observe(record['loss']['real'])
                save_json(output/'stability.json', dict(best_loss=monitor.best,
                                                       consecutive_bad_checks=monitor.bad, reason=reason))
                if reason:
                    break
    reason = reason or 'step_budget_review_boundary'
    # A resource/nonfinite stop must not perform additional large evaluation.
    if reason not in ['storage_floor', 'nonfinite_training_loss', 'invalid_gradient']:
        final_folder = output/f'step-{step:06d}'
        if not final_folder.exists():
            final_folder = checkpoint(step)
            record = evaluate(step, final_folder)
            if better(record, best):
                best = record; save_json(output/'best.json', best)
        assert frozen_digest(model) == frozen_before
        saved = torch.load(Path(best['checkpoint'])/'checkpoint.pt', map_location='cpu', weights_only=False)
        encoder.load_state_dict(saved['encoder'])
        adapter = Path(best['checkpoint'])/'lora/resume'
        model.load_adapter(adapter, adapter_name='best-review'); model.set_adapter('best-review')
        model.eval(); encoder.eval()
        reloaded = losses()
        assert abs(reloaded['real']-best['loss']['real']) < 1e-5
        controls = {}
        for condition in ['real', 'zero', 'shuffled', 'language-only']:
            predictions, account = generate(condition)
            save_json(output/f'best-{condition}-predictions.json', predictions)
            controls[condition] = account
        save_json(output/'best-review.json', dict(best=best, loss=reloaded, controls=controls,
             reload_consistent=True, frozen_weights_unchanged=True,
             encoder_prefix_max_error=causal_check(encoder, 'mps'),
             limitation='Best validation WER, not independent test; one seed, 16 sentences, one day'))
    save_json(output/'status.json', dict(status='stopped', reason=reason, step=step,
        best_step=best['step'], best_validation_wer=best['wer']['rate'],
        elapsed_seconds=time.monotonic()-start))
    print(json.dumps(json.loads((output/'status.json').read_text())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-step', type=int, default=10000)
    parser.add_argument('--max-hours', type=float, default=4)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'status.json', dict(status='initializing'))
    try:
        run(args.parent.resolve(), args.output.resolve(), args.max_step, args.max_hours)
    except Exception as error:
        save_json(args.output/'status.json', dict(status='failed', error=str(error)))
        raise
