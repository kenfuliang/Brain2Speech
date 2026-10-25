"""E012: joint causal neural encoder / text-prefix / LoRA feasibility.

No WFST, no public-test selection, no exact BIT or LFADS reproduction claim.
The encoder is causal; sentence generation is offline after the trial ends.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit import ROOT, digest, save_json, space_guard
from research.data import CausalFeatures
from research.metrics import edits
from bit_feasibility import load_language, frozen_digest, MODEL, REVISION


class CausalTextEncoder(torch.nn.Module):
    """Unidirectional inference/generator, deterministic 32-D inferred inputs.

    Text-only first pilot: no stochastic posterior, reconstruction or KL loss.
    Fixed non-overlapping 20-bin trailing averages emit every 400 ms.
    """
    def __init__(self, language_dim):
        super().__init__()
        self.readin = torch.nn.Linear(256, 128)
        self.inference = torch.nn.GRU(128, 128, batch_first=True)
        self.inputs = torch.nn.Linear(128, 32)
        self.generator = torch.nn.GRU(32, 128, batch_first=True)
        self.factors = torch.nn.Linear(128, 64)
        self.bridge = torch.nn.Sequential(torch.nn.LayerNorm(64),
                                          torch.nn.Linear(64, language_dim))

    def forward(self, x):
        inferred, _ = self.inference(torch.tanh(self.readin(x)))
        generated, _ = self.generator(self.inputs(inferred))
        factors = self.factors(generated)
        n = factors.shape[1] // 20
        if not n:
            raise ValueError('Trial shorter than one complete neural window')
        pooled = factors[:, :n*20].reshape(x.shape[0], n, 20, 64).mean(2)
        return self.bridge(pooled)


def causal_check(encoder, device):
    encoder.eval()
    x = torch.randn(1, 100, 256, device=device)
    changed = x.clone(); changed[:, 60:] += 10
    with torch.no_grad():
        a, b, c = encoder(x), encoder(changed), encoder(x[:, :60])
    error = max(float((a[:, :3]-b[:, :3]).abs().max()),
                float((a[:, :3]-c).abs().max()))
    if error > 1e-5:
        raise AssertionError(f'Prefix invariance failed: {error}')
    return error


def dataset(output):
    manifest_path = ROOT/'research/data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    day = manifest['folds']['initial_train'][0]
    entry = manifest['sessions'][day]['train']
    # Entire pilot uses the labeled training partition of one initial day only.
    seen, rows = set(), []
    for row in entry['trials']:
        prompt = ' '.join(row['text'].lower().split())
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        rows.append({**row, 'session': day,
                     'split': 'train' if len(rows) < 64 else 'validation'})
        if len(rows) == 80:
            break
    assert len(rows) == 80
    mat = loadmat(ROOT/entry['path'], variable_names=['tx1', 'spikePow'])
    trials = []
    for row in rows:
        i = row['index']
        raw = np.concatenate([mat['tx1'].ravel()[i][:, :128],
                              mat['spikePow'].ravel()[i][:, :128]], 1).astype(np.float32)
        assert np.isfinite(raw).all() and (raw[:, 128:] >= 0).all()
        trials.append({**row, 'raw': raw})
    stats = CausalFeatures.fit(trials[:64])
    for trial in trials:
        trial['x'] = torch.tensor(stats.process(trial['raw']))
    # Check preprocessing independently of inference.
    probe = trials[0]['raw']; cut = len(probe)//2
    changed = probe.copy(); changed[cut:] += 1
    assert np.array_equal(stats.process(probe)[:cut], stats.process(changed)[:cut])
    assert np.array_equal(stats.process(probe)[:cut], stats.process(probe[:cut]))
    save_json(output/'trials.json', rows)
    save_json(output/'normalization.json', stats.state())
    save_json(output/'data-identity.json', {
        'manifest_sha256': digest(manifest_path), 'source_sha256': digest(ROOT/entry['path']),
        'trials_sha256': digest(output/'trials.json'), 'session': day,
        'train': 64, 'validation': 16, 'public_test_used': False,
        'final_days_used': False, 'prompt_overlap': 0, 'preprocessing_prefix_invariant': True,
        'normalization': '64 training trials only; fixed at validation',
        'acquisition_delay_ms': 4, 'bin_ms': 20, 'token_interval_ms': 400,
        'partial_final_window': 'discarded (<20 bins)',
        'limitations': 'one participant/day; small chronological within-training validation; no untouched benchmark or foundation-model claim'})
    return trials


def run(output, steps):
    space_guard(); torch.manual_seed(17); torch.set_num_threads(4)
    if not torch.backends.mps.is_available():
        raise RuntimeError('This pilot requires the validated Apple GPU backend')
    device = 'mps'
    trials = dataset(output)
    model, tokenizer, dim = load_language()
    encoder = CausalTextEncoder(dim).to(device)
    causal_error = causal_check(encoder, device)
    prompt = torch.tensor(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Decode the preceding neural activity into an English sentence.'}],
        tokenize=True, add_generation_prompt=True), device=device)
    train, validation = trials[:64], trials[64:]
    parameters = list(encoder.parameters())+[p for p in model.parameters() if p.requires_grad]
    before = [p.detach().cpu().clone() for p in parameters]
    frozen_before = frozen_digest(model)
    opt = torch.optim.AdamW(parameters, lr=.0003)

    def prefix(row, condition='real', source=None):
        x = (source or row)['x'].to(device).unsqueeze(0)
        if condition == 'zero':
            x = torch.zeros_like(x)
        neural = encoder(x)
        if condition == 'language-only':
            neural = neural[:, :0]
        return torch.cat([neural, model.get_input_embeddings()(prompt)[None]], 1)

    def loss(row, condition='real', source=None):
        embeds = prefix(row, condition, source)
        ids = torch.tensor(tokenizer.encode(row['text'], add_special_tokens=False)+
                           [tokenizer.eos_token_id], device=device)[None]
        return model(inputs_embeds=torch.cat([embeds, model.get_input_embeddings()(ids)], 1),
                     labels=torch.cat([torch.full(embeds.shape[:2], -100, device=device,
                                                  dtype=torch.long), ids], 1), use_cache=False).loss

    def evaluate():
        model.eval(); encoder.eval()
        with torch.no_grad():
            result = {}
            for condition in ['real', 'zero', 'shuffled', 'language-only']:
                values, counts = [], []
                for i, row in enumerate(validation):
                    source = validation[(i+1) % len(validation)] if condition == 'shuffled' else None
                    values.append(float(loss(row, condition, source)))
                    counts.append(len(tokenizer.encode(row['text'], add_special_tokens=False))+1)
                result[condition] = float(np.average(values, weights=counts))
            result['train_fixed_probe'] = float(np.mean([float(loss(r)) for r in train[:4]]))
        return result

    identity = dict(seed=17, steps=steps, batch_size=2, gradient_accumulation=2,
                    optimizer='AdamW', lr=.0003, schedule='constant', gradient_clip=1.,
                    device=device, dtype='float32', encoder='random-initialized causal deterministic inference/generator GRUs',
                    model_revision=REVISION, model_sha256=digest(MODEL/'model.safetensors'),
                    code_sha256=digest(Path(__file__)), trainable_parameters=sum(p.numel() for p in parameters),
                    total_parameters=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in encoder.parameters()),
                    encoder_prefix_max_error=causal_error, checkpoint_selection='fixed final step; no validation tuning',
                    loss='target-token cross entropy; per-trial mean for training, token-weighted validation',
                    generation='greedy max_new_tokens=40; offline after full trial; not streaming text')
    save_json(output/'identity.json', identity)
    initial = evaluate(); save_json(output/'initial.json', initial)
    print(json.dumps({'initial': initial}), flush=True)
    rng = np.random.default_rng(17); started = time.monotonic()
    with (output/'training.jsonl').open('x') as log:
        for step in range(steps):
            space_guard(); model.train(); encoder.train(); opt.zero_grad(set_to_none=True)
            indices = rng.choice(64, 2, replace=False); losses = []
            for index in indices:
                value = loss(train[int(index)])
                assert torch.isfinite(value)
                (value/2).backward(); losses.append(float(value.detach()))
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
            opt.step(); torch.mps.synchronize()
            row = dict(step=step+1, loss=float(np.mean(losses)), gradient_norm=norm,
                       elapsed_seconds=time.monotonic()-started)
            if (step+1) % 10 == 0:
                model.eval(); encoder.eval()
                with torch.no_grad():
                    row['train_fixed_probe'] = float(np.mean([float(loss(r)) for r in train[:4]]))
                model.save_pretrained(output/f'lora-step-{step+1}', save_embedding_layers=False)
                torch.save({'encoder': encoder.state_dict(), 'optimizer': opt.state_dict(),
                            'rng': rng.bit_generator.state, 'step': step+1}, output/f'checkpoint-{step+1}.pt')
            log.write(json.dumps(row)+'\n'); log.flush()
            save_json(output/'status.json', {'status': 'training', **row}); print(json.dumps(row), flush=True)
    final = evaluate(); save_json(output/'final.json', final)
    model.save_pretrained(output/'lora-final', save_embedding_layers=False)
    torch.save(encoder.state_dict(), output/'encoder-final.pt')
    assert frozen_digest(model) == frozen_before
    probe = final['train_fixed_probe']
    model.load_adapter(output/'lora-final', adapter_name='reload-check'); model.set_adapter('reload-check')
    encoder.load_state_dict(torch.load(output/'encoder-final.pt', weights_only=True))
    model.eval(); encoder.eval()
    with torch.no_grad():
        reloaded = float(np.mean([float(loss(r)) for r in train[:4]]))
    assert abs(probe-reloaded) < 1e-5
    outputs = []; scores = {}
    with torch.no_grad():
        for condition in ['real', 'zero', 'shuffled', 'language-only']:
            errors = units = 0
            for i, row in enumerate(validation):
                source = validation[(i+1) % 16] if condition == 'shuffled' else None
                embeds = prefix(row, condition, source)
                ids = model.generate(inputs_embeds=embeds,
                                     attention_mask=torch.ones(embeds.shape[:2], device=device, dtype=torch.long),
                                     max_new_tokens=40, do_sample=False,
                                     use_cache=True, pad_token_id=tokenizer.eos_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
                prediction = tokenizer.decode(ids[0], skip_special_tokens=True)
                account = edits(row['text'].lower().split(), prediction.lower().split())
                errors += account['errors']; units += account['reference_units']
                outputs.append(dict(id=row['id'], condition=condition, reference=row['text'],
                                    prediction=prediction, wer=account))
            scores[condition] = dict(errors=errors, reference_words=units, wer=errors/units)
            save_json(output/'generated-transcripts.json', outputs)
    result = dict(initial=initial, final=final, generation=scores,
                  training_loss_decreased=final['train_fixed_probe'] < initial['train_fixed_probe'],
                  validation_loss_decreased=final['real'] < initial['real'],
                  changed_tensors=sum(not torch.equal(a, p.detach().cpu()) for a, p in zip(before, parameters)),
                  frozen_llm_weights_unchanged=True, reload_consistent=True,
                  encoder_prefix_max_error=causal_check(encoder, device),
                  wall_seconds=time.monotonic()-started, mps_driver_bytes=torch.mps.driver_allocated_memory(),
                  limitation='One seed/day feasibility, not neural grounding, BIT reproduction, good WER or streaming text; controls share trained model, not separately trained baselines')
    save_json(output/'metrics.json', result)
    save_json(output/'status.json', {'status': 'complete', 'steps': steps})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=40)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'status.json', {'status': 'initializing'})
    try:
        run(args.output, args.steps)
    except Exception as error:
        save_json(args.output/'status.json', {'status': 'failed', 'error': str(error)})
        raise
