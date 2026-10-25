"""Small deterministic recurrent derivative differential, without CTC/data loader."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir()
    import numpy as np
    import tensorflow as tf
    rng=np.random.RandomState(17)
    data=rng.randn(2,8,3).astype('float32')*.2
    rows=[]
    for units in (4,32,1024):
        weights=None
        results=[]
        for device in ('CPU','GPU'):
            with tf.device(f'/{device}:0'):
                tf.random.set_seed(17)
                model=tf.keras.layers.GRU(units,return_sequences=True,reset_after=True)
                model(tf.constant(data))
                if weights is None:
                    weights=model.get_weights()
                else:
                    model.set_weights(weights)
                with tf.GradientTape() as tape:
                    y=model(tf.constant(data),training=False)
                    loss=tf.reduce_sum(y*y)
                grads=tape.gradient(loss,model.trainable_variables)
            results.append({'y':y.numpy(),'loss':float(loss.numpy()),'grad':[g.numpy() for g in grads]})
        comparisons=[{'variable':i,'passed':bool(np.allclose(a,b,rtol=1e-3,atol=1e-5)),
                       'max_abs_difference':float(np.max(np.abs(a-b)))}
                      for i,(a,b) in enumerate(zip(results[0]['grad'],results[1]['grad']))]
        row={'units':units,'forward_passed':bool(np.allclose(results[0]['y'],results[1]['y'],rtol=1e-4,atol=1e-5)),
             'loss_cpu_gpu':[r['loss'] for r in results], 'gradients':comparisons}
        rows.append(row)
        print(json.dumps(row),flush=True)
    passed=all(r['forward_passed'] and all(c['passed'] for c in r['gradients']) for r in rows)
    (args.output/'completion.json').write_text(json.dumps({'status':'passed-small-gru-gradients' if passed else 'failed-small-gru-gradients','rows':rows},indent=2)+'\n')
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
