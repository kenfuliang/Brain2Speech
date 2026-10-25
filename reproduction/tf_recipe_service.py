"""Original TF CPU initialization/data pipeline, binary RPC for native Torch training.

No model training runs here. stdout is length-framed NPZ; diagnostics go to stderr.
"""
import argparse
import io
import json
from pathlib import Path
import struct
import sys

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,required=True); a=p.parse_args()
    wire=sys.stdout.buffer
    sys.stdout=sys.stderr
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    root=Path(__file__).absolute().parent.parent
    sys.path.insert(0,str(root/'NeuralDecoder'))
    from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(root/'workspace/derived/rnns/baselineRelease/args.yaml')
    cfg.outputDir=str(a.output.absolute()); cfg.seed=a.seed
    cfg.dataset.dataDir=[str(root/'workspace/derived/tfRecords')]*19
    cfg.loadDir=None; cfg.loadCheckpointIdx=None
    decoder=NeuralSequenceDecoder(cfg)
    # Use the same switch-case data graph as the original training function.
    @tf.function
    def train_data(day):
        return tf.switch_case(day,decoder.trainDatasetSelector)
    validation=None
    for line in sys.stdin:
        request=json.loads(line)
        command=request['command']
        if command=='close': break
        if command=='initial':
            vars=decoder.model.trainable_variables+decoder.inputLayers[0].trainable_variables
            arrays={f'initial_{i:03d}':v.numpy() for i,v in enumerate(vars)}
            arrays['configuration']=np.asarray(json.dumps(OmegaConf.to_container(cfg,resolve=True)))
            arrays['tensorflow']=np.asarray(tf.__version__)
            arrays['gru_implementation']=np.asarray(decoder.model.rnnLayers[0].get_config()['implementation'])
        elif command=='train':
            data=train_data(tf.constant(request['day'],tf.int32))
            arrays={k:v.numpy() for k,v in data.items()}
        elif command=='validation':
            if validation is None:
                validation=[]
                for day,ds in enumerate(decoder.tfValDatasets):
                    for raw in ds:
                        data=decoder._datasetLayerTransform(raw,decoder.normLayers[day],0,0,0,0,0)
                        validation.append((day,{k:v.numpy() for k,v in data.items()}))
            day,arrays=validation[request['index']]
            arrays=dict(arrays,day=np.asarray(day),batches=np.asarray(len(validation)))
        else: raise ValueError(command)
        payload=io.BytesIO(); np.savez(payload,**arrays)
        value=payload.getvalue()
        wire.write(struct.pack('!Q',len(value))); wire.write(value); wire.flush()

if __name__=='__main__': main()
