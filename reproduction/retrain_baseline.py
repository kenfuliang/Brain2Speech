#!/usr/bin/env python3
"""Original TensorFlow baseline, from scratch; isolated from causal v1.

Run in the pinned replay image with this file and the repository mounted read-only.
Smoke measures one genuine batch without full validation, not reproduction success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('/data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh directory; no existing evidence is overwritten.')
    if shutil.disk_usage(args.output.parent).free < 30 * 1024**3:
        raise RuntimeError('Less than 30 GiB free; do not start collection.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import tensorflow as tf
    from omegaconf import OmegaConf
    from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

    recipe = args.workspace / 'derived/rnns/baselineRelease/args.yaml'
    config = OmegaConf.load(recipe)
    assert config.model.nLayers == 5 and config.model.nUnits == 1024
    assert config.model.stack_kwargs.kernel_size == 32
    assert config.batchSize == 64 and config.nBatchesToTrain == 10000
    assert len(config.dataset.sessions) == 19
    config.outputDir = str(args.output)
    config.loadDir = None
    config.loadCheckpointIdx = None
    config.mode = 'train'
    config.seed = args.seed
    config.dataset.dataDir = [str(args.workspace / 'derived/tfRecords')] * 19
    for session in config.dataset.sessions:
        for split in ('train', 'test'):
            if not (args.workspace / 'derived/tfRecords' / session / split).is_dir():
                raise FileNotFoundError(f'{session}/{split}')
    args.output.mkdir()
    source = Path(__import__('neuralDecoder').__file__).parent
    identity = {
        'experiment': 'E008-original-baseline-retraining',
        'mode': 'one-batch-smoke' if args.smoke else 'full-retraining',
        'seed': args.seed, 'tensorflow': tf.__version__,
        'recipe_sha256': sha(recipe),
        'original_code_sha256': {
            str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*.py'))},
        'runner_sha256': sha(Path(__file__)),
        'initialization': 'random; no released weights loaded',
        'configuration': OmegaConf.to_container(config, resolve=True),
        'deviations': ['Explicit fixed seed instead of timestamp seed',
                       'Linux amd64 CPU emulation instead of NVIDIA GPU',
                       'Keep all newly generated best checkpoints rather than pruning to ten'],
        'selection': 'Original greedy phoneme error on public test, every 50 batches',
        'evaluation_target': '600 public test trials from configured days with index >=4; 18.8% 3-gram WER',
        'limitations': ['Offline artifact reproduction; not causal v1 or live Nature results',
                        'Smoke omits validation and cannot establish WER reproduction'],
    }
    (args.output / 'run-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    started = time.monotonic()
    decoder = NeuralSequenceDecoder(config)
    decoder.ckptManager = tf.train.CheckpointManager(
        decoder.checkpoint, str(args.output), max_to_keep=None)
    if args.smoke:
        # Exact batch selector used by the authors; no architecture or batch reduction.
        index = int(np.argwhere(np.random.multinomial(1, config.dataset.datasetProbability))[0][0])
        step_started = time.monotonic()
        result = decoder._trainStep(tf.constant(index, tf.int32),
                                   tf.constant(config.dataset.datasetToLayerMap[index], tf.int32))
        loss = float(result['predictionLoss'] + result['regularizationLoss'])
        if not np.isfinite(loss):
            raise RuntimeError('Nonfinite smoke loss')
        measured = {'status': 'smoke-passed', 'loss': loss,
                    'grad_norm': float(result['gradNorm']),
                    'first_batch_seconds_including_trace': time.monotonic()-step_started,
                    'total_seconds': time.monotonic()-started,
                    'backend_parameters': decoder.model.count_params(),
                    'wer': None, 'not_a_reproduction_result': True}
    else:
        decoder.train()
        measured = {'status': 'training-complete-evaluation-required',
                    'total_seconds': time.monotonic()-started,
                    'selected_checkpoint': tf.train.latest_checkpoint(str(args.output)),
                    'wer': None}
    (args.output / 'completion.json').write_text(json.dumps(measured, indent=2)+'\n')
    print(json.dumps(measured), flush=True)


if __name__ == '__main__':
    main()
