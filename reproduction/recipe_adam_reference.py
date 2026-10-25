"""Independent original-framework optimizer fixture with changing day adapters."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    variables=[tf.Variable([1.,-2.]),tf.Variable([3.,-4.])]
    opt=tf.keras.optimizers.Adam(.02,epsilon=.1)
    rows=[]
    for gradients in (([.3,-.7],[0.,0.]),([0.,0.],[.2,-.4]),([.6,.1],[0.,0.])):
        opt.apply_gradients(zip([tf.constant(g) for g in gradients],variables))
        rows.append([v.numpy().tolist() for v in variables])
    a.output.write_text(json.dumps({'weights':rows,'gradients':[[[.3,-.7],[0.,0.]],[[0.,0.],[.2,-.4]],[[.6,.1],[0.,0.]]]},indent=2)+'\n')

if __name__=='__main__': main()
