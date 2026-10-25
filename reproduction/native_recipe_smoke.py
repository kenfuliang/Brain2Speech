#!/usr/bin/env python3
"""Real-data original-model smoke on native TF/Metal, never full training.

Retains original code, model, batch, objective and session selection. Timing is
diagnostic (concurrent v1 load); not a controlled speedup or reproduction result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    root = Path(__file__).absolute().parent.parent
    args.output = args.output.absolute()
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(args.output.parent).free < 30*1024**3:
        raise RuntimeError('Below 30 GiB free-storage guard')
    sys.path.insert(0, str(root / 'NeuralDecoder'))
    import numpy as np
    import tensorflow as tf
    from omegaconf import OmegaConf
    from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
    assert tf.__version__ == '2.7.0'
    assert tf.config.list_physical_devices('GPU'), 'GPU required'
    recipe = root / 'workspace/derived/rnns/baselineRelease/args.yaml'
    config = OmegaConf.load(recipe)
    config.outputDir = str(args.output)
    config.loadDir = None
    config.loadCheckpointIdx = None
    config.mode = 'train'
    config.seed = 17
    config.dataset.dataDir = [str(root/'workspace/derived/tfRecords')]*19
    args.output.mkdir()
    identity = {'scope':'Original-recipe real-data training smoke only',
                'tensorflow':tf.__version__, 'architecture':platform.machine(),
                'macos':platform.mac_ver()[0], 'recipe_sha256':hashlib.sha256(recipe.read_bytes()).hexdigest(),
                'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'configuration':OmegaConf.to_container(config,resolve=True),
                'deviations':['Explicit seed 17; original random seed unavailable',
                              'Native ARM64 TF-macos/Metal rather than NVIDIA CUDA'],
                'limitations':['Soft placement permits CPU fallback; individual original-model op placement not certified',
                               'Concurrent v1 load: diagnostic timing only, not controlled acceleration benchmark',
                               'No full training, test selection, transcript decoding or retrained WER']}
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    start = time.monotonic()
    decoder = NeuralSequenceDecoder(config)
    assert decoder.model.count_params() == 53551145
    assert len(decoder.inputLayers) == 19
    assert decoder.model.initStates.trainable
    assert any(v is decoder.model.initStates for v in decoder.trainableVariables)
    observations = []
    # Match the authors' session sampler; use their own _trainStep unchanged.
    for batch in range(args.batches):
        index = int(np.argwhere(np.random.multinomial(1,config.dataset.datasetProbability))[0][0])
        t = time.monotonic()
        result = decoder._trainStep(tf.constant(index,tf.int32),
                                   tf.constant(config.dataset.datasetToLayerMap[index],tf.int32))
        loss = float(result['predictionLoss']+result['regularizationLoss'])
        grad = float(result['gradNorm'])
        if not np.isfinite(loss) or not np.isfinite(grad):
            raise RuntimeError('Nonfinite loss or gradient')
        row = {'batch':batch,'session':config.dataset.sessions[index],
               'seconds_including_trace_if_first':time.monotonic()-t,
               'loss':loss,'grad_norm':grad,'optimizer_iterations':int(decoder.optimizer.iterations.numpy())}
        observations.append(row)
        (args.output/f'batch-{batch:02d}.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    # Save/reload original variables and verify inference on fixed synthetic probe.
    probe = tf.zeros((1,40,256))
    expected = decoder.model(probe,training=False).numpy()
    saved = decoder.checkpoint.save(str(args.output/'smoke-checkpoint'))
    original_state = decoder.model.initStates.numpy().copy()
    decoder.model.initStates.assign_add(tf.ones_like(decoder.model.initStates))
    decoder.checkpoint.restore(saved).assert_existing_objects_matched()
    np.testing.assert_array_equal(decoder.model.initStates.numpy(),original_state)
    np.testing.assert_allclose(decoder.model(probe,training=False).numpy(),expected,rtol=1e-5,atol=1e-6)
    completion = {'status':'passed-real-data-training-and-reload-smoke',
                  'batches':observations,'seconds':time.monotonic()-start,
                  'backend_parameters':decoder.model.count_params(),
                  'input_transform_parameters':sum(l.count_params() for l in decoder.inputLayers),
                  'checkpoint':saved,'checkpoint_reload_passed':True,
                  'gpu_runtime_probe_passed':True,'full_gpu_operation_coverage_verified':False,
                  'cross_device_parity_verified':False,'wer':None,'full_retraining_started':False}
    (args.output/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(completion),flush=True)


if __name__ == '__main__':
    main()
