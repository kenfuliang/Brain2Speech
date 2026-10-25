"""Independent artifact accounting for the completed E009 pilot, not model quality proof."""
import argparse
import json
from pathlib import Path
import numpy as np
from importlib.metadata import version
from train_pytorch_recipe import sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args = p.parse_args()
    run = args.run
    identity = json.loads((run/'run-identity.json').read_text())
    metrics = json.loads((run/'metrics.json').read_text())
    status = json.loads((run/'status.json').read_text())
    data = json.loads((run/'trials.json').read_text())
    history = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
    assert status['status'] == 'complete'
    assert len(history) == metrics['steps'] == identity['steps'] == status['steps']
    assert [r['step'] for r in history] == list(range(1,len(history)+1))
    assert np.isfinite([[r['loss'],r['gradient_norm'],r['seconds']] for r in history]).all()
    assert identity['model_sha256'] == 'f481632a9ced2621706a60d1b58dda5c03307c3bf9e963214561f97e837ff1ef'
    assert len({r['key'] for r in data}) == len(data) == 80
    train = [r for r in data if r['split']=='train']
    test = [r for r in data if r['split']=='evaluation']
    assert len(train) == 64 and len(test) == 16
    norm = lambda r: ' '.join(r['text'].lower().split())
    assert not set(map(norm,train)) & set(map(norm,test))
    assert sha(run/'features.npz') == identity['data']['feature_sha256']
    assert sha(run/'trials.json') == identity['data']['trials_sha256']
    with np.load(run/'features.npz') as features:
        assert set(features.files) == {r['key'] for r in data}
        assert all(features[k].shape == (32,1024) and np.isfinite(features[k]).all() for k in features.files)
    assert metrics['training_diagnostic_decreased'] == (metrics['final']['train_diagnostic'] < metrics['initial']['train_diagnostic'])
    assert metrics['heldout_loss_improved'] == (metrics['final']['real'] < metrics['initial']['real'])
    generated = json.loads((run/'generated-transcripts.json').read_text())
    assert {r['condition'] for r in generated} == {'real','zero','shuffled','language-only'}
    assert all(len(r['predictions']) == len(r['references']) == 4 for r in generated)
    assert metrics['changed_trainable_tensors'] > 0
    assert metrics['frozen_weights_unchanged'] and metrics['reload_loss_consistent']
    result = dict(status='passed-artifact-accounting', trials=80, train=64, evaluation=16,
                  environment={name:version(name) for name in ['torch','transformers','peft','accelerate','safetensors']},
                  steps=len(history), model_weights_match_pinned_upstream_sha256=True,
                  finite_history_and_features=True, prompt_separation=True,
                  mean_update_seconds=float(np.mean([r['seconds'] for r in history])),
                  update_seconds_scope='recorded forward/backward/optimizer/synchronization; excludes setup and diagnostic evaluation',
                  artifacts_sha256={name:sha(run/name) for name in
                                    ['metrics.json','training.jsonl','trials.json','features.npz','generated-transcripts.json','projector.pt']},
                  caveat='Checks recompute artifact accounting; frozen/reload checks were executed inside trainer; no neural-grounding benefit established by this audit')
    write(run/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
