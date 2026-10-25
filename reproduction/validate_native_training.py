#!/usr/bin/env python3
"""CPU/Metal parity and profiled pilot; does not launch full retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

# Prespecified before reading CPU/GPU differences. Float32, deterministic probe.
TOLERANCES = {'logits': (1e-4, 1e-5), 'loss': (1e-4, 1e-5),
              'gradients': (1e-3, 1e-5), 'updated_weights': (1e-4, 1e-5)}


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('cpu-reference', 'gpu-parity', 'gpu-pilot'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    root = Path(__file__).absolute().parent.parent
    out = args.output.absolute()
    if out.exists():
        raise FileExistsError(out)
    if shutil.disk_usage(out.parent).free < 30*1024**3:
        raise RuntimeError('Below 30 GiB storage guard')
    out.mkdir()
    sys.path.insert(0, str(root/'NeuralDecoder'))
    import numpy as np
    import tensorflow as tf
    from omegaconf import OmegaConf
    from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder, gaussSmooth
    from neuralDecoder.datasets.speechDataset import SpeechDataset
    if args.mode == 'cpu-reference':
        tf.config.set_visible_devices([], 'GPU')
    else:
        assert tf.config.list_physical_devices('GPU'), 'GPU required'
    recipe = root/'workspace/derived/rnns/baselineRelease/args.yaml'
    config = OmegaConf.load(recipe)
    config.outputDir = str(out)
    config.loadDir = None
    config.loadCheckpointIdx = None
    config.mode = 'train'
    config.seed = 17
    config.dataset.dataDir = [str(root/'workspace/derived/tfRecords')]*19
    write(out/'identity.json', {'mode':args.mode,'tensorflow':tf.__version__,
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'recipe_sha256':hashlib.sha256(recipe.read_bytes()).hexdigest(),
          'configuration':OmegaConf.to_container(config,resolve=True),
          'tolerances_rtol_atol':TOLERANCES,
          'scope':'Diagnostic parity/pilot only; no full training or final WER',
          'parity_design':'Same initial checkpoint and two fixed initial-day TRAIN trials; dropout/noise disabled only for deterministic parity probe',
          'timing_limitation':'Concurrent v1 and host load; pilot timings not controlled speedup'} )
    decoder = NeuralSequenceDecoder(config)
    if args.mode == 'gpu-parity':
        assert args.reference is not None
        decoder.checkpoint.restore(str(args.reference/'initial-1')).assert_existing_objects_matched()
    elif args.mode == 'cpu-reference':
        decoder.checkpoint.save(str(out/'initial'))

    if args.mode != 'gpu-pilot':
        # Original parser, deterministic first two initial training trials.
        ds = SpeechDataset(str(root/'workspace/derived/tfRecords'/config.dataset.sessions[0]/'train'),
                           256,40,500,1024)
        data = next(iter(ds.build(2,isTraining=False)))
        x = data['inputFeatures']
        if config.smoothInputs:
            x = gaussSmooth(x,config.smoothKernelSD)
        # Same original normalization setting: normLayer=False (identity).
        x = x*config.inputScale
        variables = decoder.model.trainable_variables + decoder.inputLayers[0].trainable_variables
        with tf.GradientTape() as tape:
            predictions = decoder.model(decoder.inputLayers[0](x,training=False),training=False)
            labels = tf.cast(tf.sparse.from_dense(data['seqClassIDs']),tf.int32)
            labels = tf.SparseTensor(labels.indices,labels.values-1,[2,500])
            ctc = tf.reduce_mean(tf.compat.v1.nn.ctc_loss_v2(labels,predictions,None,
                     decoder.model.getSubsampledTimeSteps(data['nTimeSteps']),
                     logits_time_major=False,blank_index=-1))
            loss = ctc+tf.math.add_n(decoder.model.losses)+tf.math.add_n(decoder.inputLayers[0].losses)
        gradients = tape.gradient(loss,variables)
        assert all(g is not None for g in gradients), 'Missing gradient'
        assert any(v is decoder.model.initStates for v in variables), 'Learned state not trained'
        assert all(bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g in gradients)
        # Preserve original optimizer and clipping; all values materialized synchronously.
        arrays = {'logits':predictions.numpy(),'loss':np.asarray(float(loss.numpy()))}
        for i,g in enumerate(gradients):
            arrays[f'gradient_{i:03d}'] = g.numpy()
        clipped,norm = tf.clip_by_global_norm(gradients,config.gradClipValue)
        decoder.optimizer.apply_gradients(zip(clipped,variables))
        for i,v in enumerate(variables):
            arrays[f'weight_{i:03d}'] = v.numpy()
        np.savez(out/'values.npz',**arrays)
        result = {'status':'cpu-reference-recorded','loss':float(loss.numpy()),
                  'gradient_norm':float(norm.numpy()),'logits_device':predictions.device,
                  'ctc_loss_device':ctc.device,'input_lengths':data['nTimeSteps'].numpy().tolist(),
                  'label_lengths':data['nSeqElements'].numpy().tolist(),
                  'variable_names':[v.name for v in variables],
                  'data_features_sha256':hashlib.sha256(data['inputFeatures'].numpy().tobytes()).hexdigest(),
                  'cpu_gpu_training_dropout_parity_verified':False}
        if args.mode == 'gpu-parity':
            comparisons = []
            with np.load(args.reference/'values.npz') as reference:
                assert set(reference.files)==set(arrays)
                for name,value in arrays.items():
                    family = 'gradients' if name.startswith('gradient_') else 'updated_weights' if name.startswith('weight_') else name
                    baseline = reference[name]
                    rtol,atol = TOLERANCES[family]
                    passed = bool(np.allclose(value,baseline,rtol=rtol,atol=atol))
                    comparisons.append({'name':name,'passed':passed,'max_abs_difference':float(np.max(np.abs(value-baseline)))})
            cpu = json.loads((args.reference/'completion.json').read_text())
            assert result['variable_names']==cpu['variable_names']
            assert result['data_features_sha256']==cpu['data_features_sha256']
            result.update(status='passed-deterministic-parity' if all(c['passed'] for c in comparisons) else 'failed-parity',comparisons=comparisons)
        write(out/'completion.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('variable_names','comparisons')}),flush=True)
        return 0 if result['status']!='failed-parity' else 1

    # Unchanged original training function, stochastic objectives/augmentation enabled.
    rows = []
    for batch in range(3):
        if shutil.disk_usage(out).free < 30*1024**3:
            raise RuntimeError('Below storage guard')
        index = int(np.argwhere(np.random.multinomial(1,config.dataset.datasetProbability))[0][0])
        if batch == 2:
            tf.profiler.experimental.start(str(out/'profile'))
        start = time.monotonic()
        try:
            values = decoder._trainStep(tf.constant(index,tf.int32),tf.constant(index,tf.int32))
            loss = float(values['predictionLoss']+values['regularizationLoss'])
            grad = float(values['gradNorm'])
            assert np.isfinite(loss) and np.isfinite(grad)
            row = {'batch':batch,'session':config.dataset.sessions[index],
                   'seconds':time.monotonic()-start,'loss':loss,'gradient_norm':grad,
                   'optimizer_iterations':int(decoder.optimizer.iterations.numpy()),
                   'logits_device':values['logits'].device,'profiled':batch==2}
        finally:
            if batch == 2:
                tf.profiler.experimental.stop()
        rows.append(row)
        write(out/f'batch-{batch}.json',row)
        print(json.dumps(row),flush=True)
    # Summarize actual profiler planes/events; GPU absence is not proof of CPU execution.
    from tensorflow.core.profiler.protobuf import xplane_pb2
    planes = []
    for file in (out/'profile').rglob('*.xplane.pb'):
        space = xplane_pb2.XSpace()
        space.ParseFromString(file.read_bytes())
        for plane in space.planes:
            events = []
            for line in plane.lines:
                for event in line.events:
                    name = plane.event_metadata[event.metadata_id].name
                    events.append({'name':name,'duration_ps':event.duration_ps})
            planes.append({'file':str(file),'plane':plane.name,'event_count':len(events),
                           'matrix_or_ctc_events':[e for e in events if any(s in e['name'].lower() for s in ('matmul','ctc','gru'))][:100]})
    write(out/'completion.json',{'status':'pilot-complete-placement-review-required','batches':rows,
          'profiler_planes':planes,'full_retraining_started':False,'wer':None,
          'placement_limitation':'Returned tensor devices and profiler coverage are not exhaustive kernel residency proof'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
