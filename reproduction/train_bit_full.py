"""E016: full public T12, author effective NDT recipe, paper-length 800 epochs.

Unmodified vendor; explicit Mac port and source/paper discrepancies. No WFST
resident graph needed for training. Validation-only dual selection, final test once.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from train_bit_ctc_pilot import ROOT, VENDOR, batch, upstream
from research.audit import digest, save_json, space_guard
from research.metrics import edits
from utils.config import ConfigBuilder
import registry
from data.willett_2023_text.prepare_data import load_data, create_phonemes_ctc_labels


def effective_config(epochs):
    # ConfigBuilder mutates registry dictionaries: restore them after each build.
    original = copy.deepcopy((registry.dataset_defaults, registry.embedder_defaults))
    cwd = Path.cwd()
    try:
        os.chdir(VENDOR)
        cfg, _ = ConfigBuilder(SimpleNamespace(training_mode='train_from_scratch', encoder='ndt',
            task='phoneme', dataset='willett_2023_text', features='all', ft_ckpt='none',
            ds_config='none', kwargs={'training.num_epochs': str(epochs)})).build()
    finally:
        os.chdir(cwd)
        registry.dataset_defaults.clear(); registry.dataset_defaults.update(original[0])
        registry.embedder_defaults.clear(); registry.embedder_defaults.update(original[1])
    return cfg


def collapse(sequence, source=False):
    result, last = [], -1
    for p in sequence:
        if p != last and p != 0:
            result.append(p)
        if not source or p != 0:
            last = p
    return result


def prepare(run, config):
    manifest_path = ROOT/'research/data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = {entry['path']: digest(ROOT/entry['path']) for d in manifest['sessions'].values()
               for key,entry in d.items() if key in ('train','test')}
    dataset = load_data(str(ROOT/'research/data/competitionData'), **config.data.dataset)
    dataset.pop('holdout')  # Hidden labels are never targets or evaluated.
    dataset = create_phonemes_ctc_labels(dataset, str(VENDOR/'vocab.json'))
    original = {}
    for split in ('train','test'):
        original[split] = [dict(r, session=day, partition=split) for day in sorted(manifest['sessions'])
                           for r in manifest['sessions'][day][split]['trials']]
    perm = np.random.RandomState(42).permutation(len(original['test']))
    row_sources = {'train': original['train'], 'val': [original['test'][i] for i in perm[:len(perm)//2]],
                   'test': [original['test'][i] for i in perm[len(perm)//2:]]}
    records, arrays = {}, {}
    infeasible = []
    for split in ('train','val','test'):
        assert len(dataset[split]) == len(row_sources[split])
        records[split] = []
        for source, prepared in zip(row_sources[split], dataset[split]):
            assert tuple(source['session'].split('.')[1:]) == tuple(prepared['day'])
            x = prepared['spikes']; y = prepared['phonemes_idx'].tolist()
            assert np.isfinite(x).all() and x.shape[1] == 256
            assert len(x) <= config.model.encoder.embedder.max_F
            assert y and all(1 <= p <= 40 for p in y)
            required = len(y) + sum(a == b for a,b in zip(y,y[1:]))
            if math.ceil(len(x)/5) < required:
                infeasible.append(source['id'])
            records[split].append(dict(source, split=split, text=prepared['sentence'], phonemes=y))
            arrays[source['id']] = x
    assert len(records['train']) == 8800 and len(records['val']) == len(records['test']) == 440
    assert not ({r['id'] for r in records['val']} & {r['id'] for r in records['test']})
    save_json(run/'trials.json', records)
    save_json(run/'data-identity.json', {'source_files': sources, 'manifest_sha256': digest(manifest_path),
        'counts': {k:len(v) for k,v in records.items()}, 'ctc_infeasible_ids': infeasible,
        'max_bins': max(len(x) for x in arrays.values()),
        'normalization': 'Direct original load_data: file/partition block moments; train-only day moments; source float32 arithmetic; split_val RandomState42',
        'holdout': 'Source loader reads unlabeled neural data for its day map; discarded before label creation; no SSL or holdout scoring'})
    assert not infeasible, 'CTC feasibility failed; no silent exclusion'
    torch.save({'records':records, 'arrays':arrays}, run/'prepared.pt')
    save_json(run/'cache-identity.json', {'sha256':digest(run/'prepared.pt')})
    return records, arrays


def forward(net, rows, arrays, device):
    x, mask, ts, ds = batch(rows, arrays, device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        h, mask, _, _, _ = net.encoder(x, mask, ts, ds, None, net.training)
        logits = net.decoder(h, ds)
    lengths = mask.sum(1).cpu()
    logp = logits.float().cpu()
    target = torch.tensor([p for r in rows for p in r['phonemes']], dtype=torch.long)
    target_lengths = torch.tensor([len(r['phonemes']) for r in rows])
    loss = torch.nn.functional.ctc_loss(logp.transpose(0,1), target, lengths, target_lengths,
                                        blank=0, reduction='sum', zero_infinity=True)
    return loss, logp, lengths


@torch.no_grad()
def evaluate(net, rows, arrays, device, export=None):
    net.eval(); errors = phones = 0; total_loss = 0.; source_rates = []; predictions = []; archive = {}
    for start in range(0,len(rows),32):
        group = rows[start:start+32]
        loss, logp, lengths = forward(net, group, arrays, device)
        total_loss += loss.item(); batch_errors = batch_phones = 0
        for row, lp, length in zip(group,logp.numpy(),lengths.tolist()):
            pred = collapse(lp[:length].argmax(-1).tolist())
            counts = edits(row['phonemes'],pred)
            errors += counts['errors']; phones += len(row['phonemes'])
            batch_errors += edits(row['phonemes'],collapse(lp.argmax(-1).tolist(), source=True))['errors']
            batch_phones += len(row['phonemes'])
            if export:
                archive[f'trial_{len(predictions):04d}'] = lp[:length,list(range(1,41))+[0]]
                predictions.append({'trial_id':row['id'], 'reference':row['text'],
                                     'session':row['session'], 'per':counts,
                                     'phoneme_prediction':[p-1 for p in pred]})
        source_rates.append(batch_errors/batch_phones)
    metrics = {'per':errors/phones, 'phone_errors':errors, 'reference_phones':phones,
               'ctc_per_reference_phone':total_loss/phones, 'source_compatible_per':float(np.mean(source_rates)),
               'trials':len(rows)}
    if export:
        export.mkdir(exist_ok=False)
        save_json(export/'metrics.json',metrics); save_json(export/'predictions.json',predictions)
        np.savez_compressed(export/'logits.npz',**archive)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=800)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(); run = args.run.resolve()
    torch.set_num_threads(4)
    config = effective_config(args.epochs)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    space_guard()
    if args.resume:
        contract = json.loads((run/'contract.json').read_text())
        assert contract['epochs'] == args.epochs and contract['runner_sha256'] == digest(__file__)
        assert digest(run/'prepared.pt') == json.loads((run/'cache-identity.json').read_text())['sha256']
        data = torch.load(run/'prepared.pt',weights_only=False,map_location='cpu')
        records,arrays = data['records'],data['arrays']
    else:
        run.mkdir(parents=True,exist_ok=False)
        source_hashes = {str(p.relative_to(VENDOR)):digest(p) for p in VENDOR.rglob('*')
                         if p.is_file() and p.suffix in ('.py','.yaml','.json')}
        contract = {'experiment':'E016','epochs':args.epochs,'seed':42,'batch':32,
            'source_commit':'f107ae988d782609b78d1d7488490cca9fb7a103','source_hashes':source_hashes,
            'runner_sha256':digest(__file__),'helper_sha256':digest(ROOT/'reproduction/train_bit_ctc_pilot.py'),
            'device':device,'torch':torch.__version__,
            'recipe':'Author effective CLI architecture/data/optimizer/sampler; paper800epoch length, not YAML3000',
            'port':'Native MPS BF16 autocast, FP32 weights/buffers, differentiable FP32 CPU CTC, no gradient clipping',
            'source_discrepancies':'Paper masking0.5 vs CLI maskingOFF; paper800 vs YAML3000; sourcePER blank/padding/batch defects',
            'selection':'Separate correct corpus-PER and source-compatible-PER best checkpoints; validation only',
            'target':{'published_scratch_validation_per':0.1726,'exploratory_similarity_band_absolute':0.01,
                       'qualification':'Not an exact reproduced performance claim: selected source hyperparameters/split/metric correspondence unresolved; historical public labels exposed'},
            'stop':'800 complete epochs unless nonfinite loss/gradient or free storage<30GiB; no PER plateau early stop',
            'evaluation':'Validation440 each epoch; test440 only after training; fixed3gram WER after training separately, published5gramOPT route not implemented',
            'full_BIT_All':'Not this run; 367h pretraining and source checkpoints not available locally',
            'frozen_v1':'No frozen scientific code/checkpoint/split mutation; resource pause awaiting user choice'}
        save_json(run/'contract.json',contract); save_json(run/'effective-config.json',config)
        save_json(run/'status.json',{'status':'preparing'})
        records,arrays = prepare(run,config)
    if args.prepare_only:
        save_json(run/'status.json',{'status':'prepared_training_not_started'})
        return
    assert all(digest(VENDOR/k)==sha for k,sha in contract['source_hashes'].items())
    torch.manual_seed(42); np.random.seed(42)
    net = upstream.NDT(config.model,method_name='ctc',features='all',vocab_size=41,
                       blank_id=0,zero_infinity=True).float().to(device)
    optimizer = torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-5,eps=1e-8)
    n_batches = math.ceil(len(records['train'])/32)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer,total_steps=args.epochs*n_batches,
                                                   max_lr=5e-4,pct_start=0.,div_factor=25)
    sampler = np.random.default_rng(42)
    history=[]; start_epoch=1; best={'per':float('inf'),'source_compatible_per':float('inf')}
    elapsed_before=0.; steps=0; started=time.monotonic()
    if (run/'latest.json').exists():
        assert args.resume
        latest=json.loads((run/'latest.json').read_text()); path=run/latest['checkpoint']
        assert digest(path)==latest['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);schedule.load_state_dict(state['scheduler'])
        sampler.bit_generator.state=state['sampler_rng'];torch.set_rng_state(state['torch_rng'])
        if device=='mps':torch.mps.set_rng_state(state['mps_rng'])
        history=state['history'];best=state['best'];start_epoch=state['epoch']+1
        elapsed_before=state['elapsed'];steps=state['steps']
    else:
        initial=evaluate(net,records['val'],arrays,device)
        save_json(run/'initial-validation.json',initial)
    try:
        for epoch in range(start_epoch,args.epochs+1):
            epoch_start=time.monotonic();space_guard();net.train()
            indices=np.arange(len(records['train']));sampler.shuffle(indices)
            batches=[indices[i:i+32].tolist() for i in range(0,len(indices),32)]
            sampler.shuffle(batches)  # Match original GroupBatchSampler, including batch-order shuffle.
            total_loss=0.;phones=0;max_driver=0
            for b in batches:
                space_guard();optimizer.zero_grad(set_to_none=True)
                rows=[records['train'][i] for i in b]
                loss,_,_=forward(net,rows,arrays,device)
                if not torch.isfinite(loss).item():raise RuntimeError('Nonfinite CTC loss')
                loss.backward()
                if not all(torch.isfinite(p.grad).all().item() for p in net.parameters() if p.grad is not None):
                    raise RuntimeError('Nonfinite gradient')
                optimizer.step();schedule.step();steps+=1
                total_loss+=loss.item();phones+=sum(len(r['phonemes']) for r in rows)
                if device=='mps':max_driver=max(max_driver,torch.mps.driver_allocated_memory())
            val=evaluate(net,records['val'],arrays,device)
            row={'epoch':epoch,'step':steps,'train_ctc_per_reference_phone':total_loss/phones,
                 'validation':val,'lr':optimizer.param_groups[0]['lr'],
                 'epoch_seconds':time.monotonic()-epoch_start,'sampled_driver_max_bytes':max_driver}
            history.append(row)
            improved=[key for key in best if val[key]<best[key]]
            for key in improved:best[key]=val[key]
            checkpoint=run/f'epoch-{epoch:04d}.pt'
            state={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':schedule.state_dict(),
                   'sampler_rng':sampler.bit_generator.state,'torch_rng':torch.get_rng_state(),
                   'mps_rng':torch.mps.get_rng_state() if device=='mps' else None,
                   'epoch':epoch,'steps':steps,'history':history,'best':best,
                   'elapsed':elapsed_before+time.monotonic()-started}
            torch.save(state,checkpoint);sha=digest(checkpoint)
            save_json(run/'latest.json',{'checkpoint':checkpoint.name,'sha256':sha,'epoch':epoch})
            for key in improved:
                save_json(run/f'best-{key}.json',{'epoch':epoch,'metric':val[key],'checkpoint':checkpoint.name,'sha256':sha})
            save_json(run/'history.json',history)
            save_json(run/'status.json',{'status':'training','epoch':epoch,'target_epochs':args.epochs,'step':steps,
                                        'validation':val,'best':best,'elapsed_seconds':state['elapsed']})
            print(json.dumps(row),flush=True)
        chosen=json.loads((run/'best-per.json').read_text())
        assert digest(run/chosen['checkpoint'])==chosen['sha256']
        state=torch.load(run/chosen['checkpoint'],map_location='cpu',weights_only=False)
        net.load_state_dict(state['model'])
        replay=evaluate(net,records['val'],arrays,device,run/'selected-validation')
        assert abs(replay['per']-chosen['metric'])<1e-12
        test=evaluate(net,records['test'],arrays,device,run/'selected-test')
        save_json(run/'final-metrics.json',{'selected':chosen,'validation':replay,'test':test,
                                          'parameters':sum(p.numel() for p in net.parameters())})
        save_json(run/'status.json',{'status':'training_complete_wer_pending','epochs':args.epochs,'steps':steps,
                                    'elapsed_seconds':elapsed_before+time.monotonic()-started})
    except Exception as error:
        save_json(run/'failure.json',{'error':str(error),'last_completed_epoch':history[-1]['epoch'] if history else 0,
                                     'resume':'Last sealed full epoch; interrupted epoch recomputed',
                                     'status':'paused-storage' if 'Storage guard' in str(error) else 'failed'})
        raise


if __name__=='__main__':
    main()
