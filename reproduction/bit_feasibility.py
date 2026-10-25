"""Small frozen-GRU -> neural prefix -> Aero language-backbone feasibility pilot.

Not exact BIT: no transformer pretraining, waveform frontend or contrastive loss.
Only projector and LoRA train. Never touch frozen v1 sources or training outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from train_pytorch_recipe import ROOT, RecipeData, sha, tensors, write
from pytorch_baseline import PublishedGRU

REVISION = '2efba6e9de291fa30e11075849631ae410b8485a'
MODEL = ROOT/'workspace/models/Aero-1-Audio-2efba6e'
ENCODER = ROOT/'reproduction/results/retraining/pytorch-retrain-s17-20261007-01/latest.pt'
ENCODER_HASH = '3110a3c1e1996fc94c70c96a6fd180fe83cdd59a1c58a0240169bc1f583831d7'


def guard():
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError('Storage guard: below 30 GiB; preserve artifacts')


def text(row):
    end = int(np.flatnonzero(row == 0)[0])
    return ''.join(chr(int(c)) for c in row[:end])


def normal(s):
    return ' '.join(s.lower().split())


def prepare(output):
    guard()
    assert sha(ENCODER) == ENCODER_HASH
    state = torch.load(ENCODER, map_location='cpu', weights_only=False)
    encoder = PublishedGRU().to('mps').eval()
    encoder.load_state_dict(state['model'])
    captured = []
    hook = encoder.layers[-1].register_forward_hook(lambda module, args, result: captured.append(result[0]))
    service = RecipeData(output, 17)
    records, features = [], {}

    def extract(data, split, allowed):
        captured.clear()
        x, _, lengths, _ = tensors(data)
        with torch.no_grad():
            encoder(x, 4)
        for i in allowed:
            hidden = captured[0][i, :int(lengths[i])]
            # Frozen feature compression only: variable-length adaptive pooling
            # is not supported on MPS. Explicit CPU operation, not silent fallback.
            pooled = torch.nn.functional.adaptive_avg_pool1d(hidden.cpu().T.unsqueeze(0), 32)[0].T
            assert torch.isfinite(pooled).all()
            key = 'trial_%04d' % len(records)
            features[key] = pooled.cpu().numpy()
            records.append(dict(key=key, split=split, text=text(data['transcription'][i]),
                                day=4, batch_trial_index=int(i)))

    try:
        service.call('initial')
        validation = service.call('validation', index=4)
        assert int(validation['day']) == 4
        # Reserve evaluation prompts first; training excludes all public-test prompts.
        forbidden = {normal(text(row)) for row in validation['transcription']}
        seen = set()
        for _ in range(8):
            data = service.call('train', day=4)
            allowed = []
            for i, row in enumerate(data['transcription']):
                value = normal(text(row))
                if value not in forbidden and value not in seen and len(records)+len(allowed) < 64:
                    seen.add(value)
                    allowed.append(i)
            extract(data, 'train', allowed)
            if len(records) == 64:
                break
        assert len(records) == 64, 'Insufficient unique disjoint train prompts'
        extract(validation, 'evaluation', list(range(16)))
        assert len(records) == 80
        assert not ({normal(r['text']) for r in records if r['split']=='train'} &
                    {normal(r['text']) for r in records if r['split']=='evaluation'})
    finally:
        hook.remove()
        service.close()
    np.savez_compressed(output/'features.npz', **features)
    write(output/'trials.json', records)
    write(output/'data-identity.json', dict(encoder_sha256=ENCODER_HASH,
         feature_sha256=sha(output/'features.npz'), trials_sha256=sha(output/'trials.json'),
         train=64, evaluation=16, session='t12.2022.05.24', neural_tokens=32, hidden_dim=1024,
         preprocessing='Original released TF pipeline; training features drawn once with original augmentation; evaluation unaugmented',
         limitation='Encoder previously monitored public-test PER; not untouched encoder holdout'))


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_language():
    from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer
    from safetensors import safe_open
    from peft import get_peft_model, LoraConfig
    config = json.loads((MODEL/'config.json').read_text())
    cfg = Qwen2Config(**config['text_config'])
    cfg._attn_implementation = 'eager'
    # Load only the exact audio-trained language backbone; no remote Python execution.
    # Qwen rotary buffers are not checkpoint parameters. CPU construction keeps
    # those initialized; meta construction leaves unmaterialized buffers.
    from transformers.modeling_utils import no_init_weights
    with no_init_weights():
        model = Qwen2ForCausalLM(cfg)
    with safe_open(MODEL/'model.safetensors', framework='pt', device='cpu') as f:
        arrays = {key.removeprefix('language_model.'): f.get_tensor(key)
                  for key in f.keys() if key.startswith('language_model.')}
    if 'lm_head.weight' not in arrays and cfg.tie_word_embeddings:
        arrays['lm_head.weight'] = arrays['model.embed_tokens.weight']
    model.load_state_dict(arrays, strict=True, assign=True)
    del arrays
    model.tie_weights()
    model = model.to(device='mps', dtype=torch.float32)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
                                           target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    return model, tokenizer, cfg.hidden_size


def run(output, steps, features_from=None):
    guard()
    torch.manual_seed(17)
    torch.set_num_threads(4)
    if features_from:
        old = json.loads((features_from/'data-identity.json').read_text())
        assert sha(features_from/'features.npz') == old['feature_sha256']
        assert sha(features_from/'trials.json') == old['trials_sha256']
        for name in ['features.npz','trials.json','data-identity.json']:
            shutil.copyfile(features_from/name,output/name)
    if not (output/'features.npz').exists():
        prepare(output)
    rows = json.loads((output/'trials.json').read_text())
    identity = json.loads((output/'data-identity.json').read_text())
    assert sha(output/'features.npz') == identity['feature_sha256']
    assert sha(output/'trials.json') == identity['trials_sha256']
    with np.load(output/'features.npz') as file:
        features = {k: torch.tensor(file[k], device='mps') for k in file.files}
    torch.manual_seed(17)  # Independent of whether frozen features were reused.
    model, tokenizer, dim = load_language()
    projector = torch.nn.Sequential(torch.nn.LayerNorm(1024), torch.nn.Linear(1024,512),
                                    torch.nn.GELU(), torch.nn.Linear(512,dim)).to('mps')
    prompt = tokenizer.apply_chat_template([{'role':'user','content':'Decode the preceding neural activity into an English sentence.'}],
                                           tokenize=True, add_generation_prompt=True)
    prompt = torch.tensor(prompt, device='mps')
    train = [r for r in rows if r['split']=='train']
    evaluation = [r for r in rows if r['split']=='evaluation']
    frozen_before = frozen_digest(model)
    trainable = list(projector.parameters())+[p for p in model.parameters() if p.requires_grad]
    before = [p.detach().cpu().clone() for p in trainable]
    opt = torch.optim.AdamW(trainable, lr=0.0003)

    def prefix(batch, condition):
        neural = torch.stack([features[r['key']] for r in batch])
        if condition == 'zero':
            neural = torch.zeros_like(neural)
        elif condition == 'shuffled':
            neural = neural.roll(1, 0)
        embeds = projector(neural)
        if condition == 'language-only':
            embeds = embeds[:, :0]
        words = model.get_input_embeddings()(prompt).unsqueeze(0).expand(len(batch),-1,-1)
        return torch.cat([embeds, words], 1)

    def loss(batch, condition='real'):
        embeds = prefix(batch, condition)
        targets = [tokenizer.encode(r['text'], add_special_tokens=False)+[tokenizer.eos_token_id] for r in batch]
        length = max(map(len, targets))
        ids = torch.full((len(batch),length),tokenizer.eos_token_id,device='mps',dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for i, values in enumerate(targets):
            ids[i,:len(values)] = torch.tensor(values, device='mps')
            labels[i,:len(values)] = ids[i,:len(values)]
            mask[i,:len(values)] = 1
        return model(inputs_embeds=torch.cat([embeds,model.get_input_embeddings()(ids)],1),
                     attention_mask=torch.cat([torch.ones(embeds.shape[:2],device='mps',dtype=torch.long),mask],1),
                     labels=torch.cat([torch.full(embeds.shape[:2],-100,device='mps',dtype=torch.long),labels],1),
                     use_cache=False).loss

    def evaluate():
        model.eval(); projector.eval()
        with torch.no_grad():
            result = {c: float(np.mean([float(loss(evaluation[i:i+4],c)) for i in range(0,16,4)]))
                      for c in ['real','zero','shuffled','language-only']}
            result['train_diagnostic'] = float(loss(train[:4]))
        return result

    initial = evaluate()
    write(output/'initial-evaluation.json', initial)
    write(output/'run-identity.json', dict(data=identity, model_revision=REVISION, model_sha256=sha(MODEL/'model.safetensors'),
          source_sha256=sha(Path(__file__)), steps=steps, seed=17, lr=0.0003,
          scope='Aero audio-trained language backbone, text-space neural prefix; audio tower bypassed',
          frozen_digest=frozen_before, trainable_parameters=sum(p.numel() for p in trainable),
          device='mps', dtype='float32', text_loss='masked target-token mean cross entropy'))
    rng = np.random.default_rng(17)
    with (output/'training.jsonl').open('x') as stream:
        for step in range(steps):
            guard(); model.train(); projector.train()
            batch = [train[int(i)] for i in rng.choice(len(train),4,replace=False)]
            opt.zero_grad(set_to_none=True)
            start = time.monotonic()
            value = loss(batch)
            assert torch.isfinite(value), 'Nonfinite training loss'
            value.backward()
            for p in trainable:
                assert p.grad is not None and torch.isfinite(p.grad).all(), 'Invalid gradient'
            norm = torch.nn.utils.clip_grad_norm_(trainable,1.)
            opt.step(); torch.mps.synchronize()
            row = dict(step=step+1,loss=float(value.detach()),gradient_norm=float(norm),seconds=time.monotonic()-start)
            if (step+1) % 10 == 0:
                model.eval(); projector.eval()
                with torch.no_grad():
                    row['train_diagnostic'] = float(loss(train[:4]))
            stream.write(json.dumps(row)+'\n'); stream.flush()
            write(output/'status.json',dict(status='training',**row))
            print(json.dumps(row),flush=True)
    final = evaluate()
    frozen_after = frozen_digest(model)
    assert frozen_before == frozen_after
    changed = sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,trainable))
    assert changed > 0
    model.save_pretrained(output/'lora',save_embedding_layers=False)
    torch.save(projector.state_dict(),output/'projector.pt')
    # Check adapter/projector reload in place preserves the same fixed-probe loss.
    probe = final['train_diagnostic']
    model.load_adapter(output/'lora',adapter_name='reload-check')
    model.set_adapter('reload-check')
    projector.load_state_dict(torch.load(output/'projector.pt',weights_only=True))
    model.eval(); projector.eval()
    with torch.no_grad():
        reloaded = float(loss(train[:4]))
    assert abs(probe-reloaded) <= 1e-5
    outputs = []
    with torch.no_grad():
        for c in ['real','zero','shuffled','language-only']:
            batch = evaluation[:4]
            embeds = prefix(batch,c)
            generated = model.generate(inputs_embeds=embeds,
                attention_mask=torch.ones(embeds.shape[:2],device='mps',dtype=torch.long),
                max_new_tokens=40,do_sample=False,use_cache=True,
                pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
            outputs.append(dict(condition=c,references=[r['text'] for r in batch],
                                predictions=tokenizer.batch_decode(generated,skip_special_tokens=True)))
    write(output/'generated-transcripts.json',outputs)
    result = dict(initial=initial,final=final,steps=steps,changed_trainable_tensors=changed,
                  frozen_weights_unchanged=True,reload_loss_consistent=True,
                  training_diagnostic_decreased=final['train_diagnostic'] < initial['train_diagnostic'],
                  heldout_loss_improved=final['real'] < initial['real'],
                  neural_dependence_not_proven_by_loss_decline=True,
                  mps_driver_bytes=torch.mps.driver_allocated_memory())
    write(output/'metrics.json',result)
    write(output/'status.json',dict(status='complete',steps=steps))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--features-from',type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        run(args.output,args.steps,args.features_from)
    except Exception as error:
        write(args.output/'status.json',dict(status='failed',error=str(error)))
        raise
