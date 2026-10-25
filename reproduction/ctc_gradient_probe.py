"""Minimized CTC-only CPU/Metal derivative check on frozen parity logits."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir()
    root=Path(__file__).absolute().parent.parent
    sys.path.insert(0,str(root/'NeuralDecoder'))
    import numpy as np
    import tensorflow as tf
    from neuralDecoder.datasets.speechDataset import SpeechDataset
    ds=SpeechDataset(str(root/'workspace/derived/tfRecords/t12.2022.04.28/train'),256,40,500,1024)
    data=next(iter(ds.build(2,isTraining=False)))
    with np.load(args.reference/'values.npz') as values:
        logits=values['logits'].copy()
    gradients=[]
    losses=[]
    for device in ('CPU','GPU'):
        with tf.device(f'/{device}:0'):
            x=tf.Variable(logits)
            labels=tf.cast(tf.sparse.from_dense(data['seqClassIDs']),tf.int32)
            labels=tf.SparseTensor(labels.indices,labels.values-1,[2,500])
            lengths=tf.cast((data['nTimeSteps']-32)/4+1,tf.int32)
            with tf.GradientTape() as tape:
                loss=tf.reduce_mean(tf.compat.v1.nn.ctc_loss_v2(labels,x,None,lengths,
                       logits_time_major=False,blank_index=-1))
            grad=tape.gradient(loss,x)
        gradients.append(grad.numpy())
        losses.append(float(loss.numpy()))
        np.save(args.output/f'{device.lower()}-gradient.npy',grad.numpy())
    passed=bool(np.allclose(gradients[0],gradients[1],rtol=1e-3,atol=1e-5))
    result={'status':'passed-ctc-gradient' if passed else 'failed-ctc-gradient',
            'loss_cpu_gpu':losses,'gradient_norm_cpu_gpu':[float(np.linalg.norm(g)) for g in gradients],
            'max_abs_gradient_difference':float(np.max(np.abs(gradients[0]-gradients[1]))),
            'lengths':lengths.numpy().tolist(),'rtol':1e-3,'atol':1e-5,
            'scope':'CTC-only diagnostic, not full-model parity'}
    (args.output/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
