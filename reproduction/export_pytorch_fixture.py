"""Export independent TF CPU weights/data and an Adam update for port validation."""
import argparse
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=2)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    root=Path(__file__).absolute().parent.parent
    sys.path.insert(0,str(root/'NeuralDecoder'))
    from neuralDecoder.datasets.speechDataset import SpeechDataset
    from neuralDecoder.neuralSequenceDecoder import gaussSmooth
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(root/'workspace/derived/rnns/baselineRelease/args.yaml')
    d=next(iter(SpeechDataset(str(root/'workspace/derived/tfRecords'/cfg.dataset.sessions[0]/'train'),256,40,500,1024).build(a.batch,isTraining=False)))
    arrays={'raw':d['inputFeatures'].numpy(), 'input':gaussSmooth(d['inputFeatures'],cfg.smoothKernelSD).numpy()*cfg.inputScale,
            'labels':d['seqClassIDs'].numpy(),'lengths':d['nTimeSteps'].numpy(),'label_lengths':d['nSeqElements'].numpy()}
    reader=tf.train.load_checkpoint(str(root/'reproduction/results/retraining/native-validation-20261007-cpu-01/initial-1'))
    keys=[]
    for i in range(5):
        keys.extend(f'net/rnnLayers/{i}/cell/{n}/.ATTRIBUTES/VARIABLE_VALUE' for n in ('kernel','recurrent_kernel','bias'))
    keys.extend(f'net/{n}/.ATTRIBUTES/VARIABLE_VALUE' for n in ('dense/kernel','dense/bias','initStates'))
    keys.extend(f'inputLayer_0/layer_with_weights-0/{n}/.ATTRIBUTES/VARIABLE_VALUE' for n in ('kernel','bias'))
    for i,k in enumerate(keys): arrays[f'initial_{i:03d}']=reader.get_tensor(k)
    v=tf.Variable([1.,-2.],dtype=tf.float32)
    opt=tf.keras.optimizers.Adam(.02,epsilon=.1)
    opt.apply_gradients([(tf.constant([.3,-.7]),v)])
    arrays['adam_expected']=v.numpy()
    np.savez(a.output,**arrays)

if __name__=='__main__': main()
