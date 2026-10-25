"""E014 matched bridge-only ablation; entire neural encoder and LLM frozen."""
import argparse
import json
from pathlib import Path
import sys
import time
import shutil

import numpy as np
import torch

from bit_feasibility import load_language, frozen_digest, MODEL, REVISION, ENCODER, ENCODER_HASH
from continue_causal_neural_llm import StabilityMonitor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit import digest, save_json, space_guard
from research.metrics import edits


def run(output, steps=1800, parent=None):
    space_guard(); torch.manual_seed(17); torch.set_num_threads(4)
    assert parent is not None
    assert torch.backends.mps.is_available()
    parent_protocol = json.loads((parent/'training-protocol.json').read_text())
    assert digest(MODEL/'model.safetensors') == parent_protocol['model_sha256']
    for name in ['features.npz', 'trials.json', 'data-identity.json', 'encoder-gate.json',
                 'phoneme-predictions.json', 'normalization.json', 'causality.json',
                 'uncertainty-protocol.json', 'environment.json']:
        shutil.copyfile(parent/name, output/name)
    origin = torch.load(parent/'step-000000/checkpoint.pt', map_location='cpu', weights_only=False)
    assert origin['step'] == 0
    data = json.loads((output/'data-identity.json').read_text())
    assert json.loads((output/'encoder-gate.json').read_text())['passed']
    assert digest(output/'features.npz') == data['features_sha256']
    assert digest(output/'trials.json') == data['trials_sha256']
    rows = json.loads((output/'trials.json').read_text())
    train = [r for r in rows if r['split']=='train']
    validation = [r for r in rows if r['split']=='validation']
    assert len(train)==512 and len(validation)==40
    with np.load(output/'features.npz') as source:
        features = {key: torch.tensor(source[key]) for key in source.files}
    model, tokenizer, dim = load_language()
    model = model.unload()  # Remove initial zero-output LoRA without merging.
    model.requires_grad_(False)
    assert all(not p.requires_grad for p in model.parameters())
    assert not any('lora_' in name for name, _ in model.named_parameters())
    bridge = torch.nn.Sequential(torch.nn.LayerNorm(1024), torch.nn.Linear(1024, 512),
                                torch.nn.GELU(), torch.nn.Linear(512, dim)).to('mps')
    bridge.load_state_dict(origin['bridge'])
    torch.set_rng_state(origin['torch_rng'])
    prompt = torch.tensor(tokenizer.apply_chat_template(
        [{'role':'user', 'content':'Decode the preceding neural activity into an English sentence.'}],
        tokenize=True, add_generation_prompt=True), device='mps')
    parameters = list(bridge.parameters())
    initial_parameters = [p.detach().cpu().clone() for p in parameters]
    opt = torch.optim.AdamW(parameters, lr=.0003)
    frozen_before = frozen_digest(model)
    rng = np.random.default_rng(17)

    def prefix(row, condition='real', source=None):
        neural = features[(source or row)['id']].to('mps')[None]
        if condition == 'zero':
            neural = torch.zeros_like(neural)
        embeds = bridge(neural)
        if condition == 'language-only':
            embeds = embeds[:, :0]
        return torch.cat([embeds, model.get_input_embeddings()(prompt)[None]], 1)

    def loss(row, condition='real', source=None):
        embeds = prefix(row, condition, source)
        ids = torch.tensor(tokenizer.encode(row['text'], add_special_tokens=False)+
                           [tokenizer.eos_token_id], device='mps')[None]
        return model(inputs_embeds=torch.cat([embeds, model.get_input_embeddings()(ids)], 1),
                     labels=torch.cat([torch.full(embeds.shape[:2], -100, device='mps', dtype=torch.long), ids], 1),
                     use_cache=False).loss

    def losses():
        model.eval(); bridge.eval(); result = {}
        with torch.no_grad():
            for condition in ['real','zero','shuffled','language-only']:
                values, counts = [], []
                for i, row in enumerate(validation):
                    source = validation[(i+1)%40] if condition=='shuffled' else None
                    values.append(float(loss(row, condition, source)))
                    counts.append(len(tokenizer.encode(row['text'], add_special_tokens=False))+1)
                result[condition] = float(np.average(values, weights=counts))
            result['train_fixed_probe'] = float(np.mean([float(loss(r)) for r in train[:4]]))
        return result

    def generate(condition='real'):
        model.eval(); bridge.eval(); rows = []
        with torch.no_grad():
            for i, row in enumerate(validation):
                source = validation[(i+1)%40] if condition=='shuffled' else None
                embeds = prefix(row, condition, source)
                ids = model.generate(inputs_embeds=embeds,
                    attention_mask=torch.ones(embeds.shape[:2], device='mps', dtype=torch.long),
                    do_sample=False, max_new_tokens=40, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
                prediction = tokenizer.decode(ids[0], skip_special_tokens=True)
                rows.append(dict(id=row['id'], reference=row['text'], prediction=prediction,
                    wer=edits(row['text'].lower().split(), prediction.lower().split())))
        account = {k:sum(r['wer'][k] for r in rows) for k in
                   ['errors','reference_units','substitutions','deletions','insertions']}
        account['rate'] = account['errors']/account['reference_units']
        account['unique_predictions'] = len({r['prediction'] for r in rows})
        return rows, account

    def checkpoint(step):
        folder = output/f'step-{step:06d}'; folder.mkdir(exist_ok=False)
        torch.save(dict(bridge=bridge.state_dict(), optimizer=opt.state_dict(),
                        rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), step=step), folder/'checkpoint.pt')
        return folder

    def evaluate(step, folder):
        save_json(output/'status.json', dict(status='evaluating', step=step))
        values = losses(); predictions, account = generate()
        record = dict(step=step, loss=values, wer=account,
            checkpoint=str(folder), checkpoint_sha256=digest(folder/'checkpoint.pt'))
        save_json(folder/'predictions.json', predictions); save_json(folder/'evaluation.json', record)
        print(json.dumps({'evaluation':record}), flush=True)
        return record

    save_json(output/'training-protocol.json', dict(steps=steps, parent=str(parent),
        parent_initial_checkpoint_sha256=digest(parent/'step-000000/checkpoint.pt'),
        bridge_initialization='exact E013 step-zero tensors', lora=False,
        language_trainable_parameters=0, plateau_checks=10, seed=17, lr=.0003, optimizer='AdamW',
        schedule='constant', batch_size=4, gradient_accumulation=4, gradient_clip=1.,
        device='mps', dtype='float32', evaluation_interval=100, model_revision=REVISION,
        model_sha256=digest(MODEL/'model.safetensors'), source_sha256=digest(Path(__file__)),
        trainable_parameters=sum(p.numel() for p in parameters), encoder_frozen=True,
        selection='minimum validation word edits, then lower validation loss, then earlier step',
        stop='1800-step matched-budget boundary; nonfinite loss/gradient; storage <30 GiB; five checks with validation loss >1.25x best; ten checks without fewer validation word edits',
        loss='mean per-trial target-token CE in training; token-weighted validation CE',
        generation='greedy max_new_tokens=40; no WFST; offline after full neural trial',
        limitation='Historical public-validation exposure in E008; different data/encoder vs E012; not independent test or exact BIT reproduction'))
    started = time.monotonic(); best = evaluate(0, checkpoint(0))
    expected = json.loads((parent/'initial.json').read_text())
    assert abs(best['loss']['real']-expected['loss']['real']) < 1e-5
    assert best['wer']['errors'] == expected['wer']['errors']
    save_json(output/'initial-replay.json', dict(passed=True, loss_and_word_edits_replayed=True))
    no_wer_improvement = 0
    save_json(output/'initial.json', best); save_json(output/'best.json', best)
    monitor = StabilityMonitor(best['loss']['real']); reason=None; step=0
    with (output/'training.jsonl').open('x') as log:
        for step in range(1, steps+1):
            space_guard(); model.eval(); bridge.train(); opt.zero_grad(set_to_none=True)
            values = []
            for index in rng.choice(len(train), 4, replace=False):
                value = loss(train[int(index)])
                if not torch.isfinite(value):
                    raise RuntimeError('Nonfinite training loss')
                (value/4).backward(); values.append(float(value.detach()))
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            assert all(p.grad is None for p in model.parameters())
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
            opt.step(); torch.mps.synchronize()
            row = dict(step=step, loss=float(np.mean(values)), gradient_norm=norm,
                elapsed_seconds=time.monotonic()-started, mps_driver_bytes=torch.mps.driver_allocated_memory())
            log.write(json.dumps(row)+'\n'); log.flush()
            save_json(output/'status.json', dict(status='training', best_step=best['step'],
                best_validation_wer=best['wer']['rate'], **row))
            if step%20==0:
                print(json.dumps(row), flush=True)
            if step%100==0 or step==steps:
                current=evaluate(step, checkpoint(step))
                no_wer_improvement = 0 if current['wer']['errors'] < best['wer']['errors'] else no_wer_improvement+1
                if (current['wer']['errors'],current['loss']['real']) < (best['wer']['errors'],best['loss']['real']):
                    best=current; save_json(output/'best.json', best)
                reason=monitor.observe(current['loss']['real'])
                if reason is None and no_wer_improvement >= 10:
                    reason='validation_wer_plateau'
                save_json(output/'stability.json', dict(best_validation_loss=monitor.best,
                    consecutive_loss_deterioration_checks=monitor.bad,
                    checks_without_wer_improvement=no_wer_improvement, reason=reason))
                if reason:
                    break
    changed=sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(initial_parameters,parameters))
    assert changed>0 and frozen_digest(model)==frozen_before
    last=json.loads((output/f'step-{step:06d}/evaluation.json').read_text())
    stored=torch.load(Path(best['checkpoint'])/'checkpoint.pt', map_location='cpu', weights_only=False)
    bridge.load_state_dict(stored['bridge'])
    reloaded=losses(); assert abs(reloaded['real']-best['loss']['real'])<1e-5
    controls={}
    for condition in ['real','zero','shuffled','language-only']:
        predictions,account=generate(condition)
        save_json(output/f'best-{condition}-predictions.json', predictions); controls[condition]=account
    assert digest(ENCODER)==ENCODER_HASH
    assert digest(output/'features.npz')==data['features_sha256']
    result=dict(initial=json.loads((output/'initial.json').read_text()), last=last, best=best,
        best_loss=reloaded, controls=controls, steps_completed=step, changed_tensors=changed,
        frozen_language_weights_unchanged=True, language_trainable_parameters=0,
        frozen_language_digest=frozen_before, lora_removed=True, frozen_neural_checkpoint_unchanged=True,
        reload_consistent=True, elapsed_seconds=time.monotonic()-started,
        stop_reason=reason or 'matched_step_budget', limitation='Exploratory selected validation, not untouched test; one seed')
    save_json(output/'metrics.json', result)
    save_json(output/'status.json', dict(status='complete', step=step, reason=result['stop_reason'],
        best_step=best['step'], best_validation_wer=best['wer']['rate']))
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1800)
    parser.add_argument('--parent',type=Path,required=True); args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert not (args.output/'training-protocol.json').exists(), 'Preserve prior training'
    try:
        run(args.output.resolve(), args.steps, args.parent.resolve())
    except Exception as error:
        save_json(args.output/'status.json',dict(status='failed',error=str(error))); raise
