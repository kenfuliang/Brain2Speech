"""Four-unit GRU derivative repro, with numerical derivative and optional optimizer toggle."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--disable-meta-optimizer',action='store_true')
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir()
    import numpy as np
    import tensorflow as tf
    if args.disable_meta_optimizer:
        tf.config.optimizer.set_experimental_options({'disable_meta_optimizer':True})
    data=np.random.RandomState(17).randn(2,8,3).astype('float32')*.2
    models=[]
    gradients=[]
    outputs=[]
    weights=None
    for device in ('CPU','GPU'):
        with tf.device(f'/{device}:0'):
            tf.random.set_seed(17)
            model=tf.keras.layers.GRU(4,return_sequences=True,reset_after=True)
            model(tf.constant(data))
            if weights is None:
                weights=[a.copy() for a in model.get_weights()]
            model.set_weights(weights)
            with tf.GradientTape() as tape:
                y=model(tf.constant(data),training=False)
                loss=tf.reduce_sum(y*y)
            grads=tape.gradient(loss,model.trainable_variables)
        gradients.append([g.numpy().copy() for g in grads])
        outputs.append(y.numpy().copy())
        models.append(model)
    # Finite differences on the largest disagreeing bias component, chosen deterministically.
    index=np.unravel_index(np.argmax(np.abs(gradients[0][2]-gradients[1][2])),gradients[0][2].shape)
    eps=1e-3
    finite=[]
    for device,model in zip(('CPU','GPU'),models):
        values=[]
        for sign in (-1,1):
            changed=[a.copy() for a in weights]
            changed[2][index]+=sign*eps
            with tf.device(f'/{device}:0'):
                model.set_weights(changed)
                y=model(tf.constant(data),training=False)
                values.append(float(tf.reduce_sum(y*y).numpy()))
        finite.append((values[1]-values[0])/(2*eps))
    cpu=float(gradients[0][2][index])
    gpu=float(gradients[1][2][index])
    result={'status':'derivative-diagnosis-recorded','disable_meta_optimizer':args.disable_meta_optimizer,
            'bias_index':[int(i) for i in index],'analytic_cpu_gpu':[cpu,gpu],
            'finite_difference_cpu_gpu':finite,'finite_difference_epsilon':eps,
            'forward_max_abs_difference':float(np.max(np.abs(outputs[0]-outputs[1]))),
            'gradient_max_abs_difference':float(np.max(np.abs(gradients[0][2]-gradients[1][2]))),
            'cpu_matches_finite_difference':abs(cpu-finite[0])<1e-3,
            'gpu_matches_finite_difference':abs(gpu-finite[1])<1e-3}
    (args.output/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if result['cpu_matches_finite_difference'] and result['gpu_matches_finite_difference'] else 1


if __name__=='__main__':
    raise SystemExit(main())
