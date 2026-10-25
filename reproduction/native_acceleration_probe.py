#!/usr/bin/env python3
"""Isolated, fail-closed native runtime probes; never launches full training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

PROBES = {
    'import': '''import tensorflow as tf, json
print(json.dumps({"tensorflow":tf.__version__, "gpus":[d.name for d in tf.config.list_physical_devices("GPU")]}))
assert tf.config.list_physical_devices("GPU"), "No GPU discovered"
''',
    'gpu-gradient': '''import tensorflow as tf, json
tf.config.set_soft_device_placement(False)
with tf.device("/GPU:0"):
    x=tf.Variable(tf.ones((16,16)))
    with tf.GradientTape() as tape:
        y=tf.reduce_mean(tf.matmul(x,x))
    g=tape.gradient(y,x)
assert "GPU:0" in y.device and g is not None
assert bool(tf.reduce_all(tf.math.is_finite(g)).numpy())
print(json.dumps({"output_device":y.device,"gradient_device":g.device,"loss":float(y.numpy())}))
''',
}


def interpreter_path(path):
    # Resolving the venv symlink selects base Python and loses its packages.
    return str(path.absolute())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(args.output.parent).free < 30*1024**3:
        raise RuntimeError('Below 30 GiB storage guard')
    args.output.mkdir()
    identity = {'python':interpreter_path(args.python), 'architecture':platform.machine(),
                'macos':platform.mac_ver()[0], 'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope':'Import and GPU gradient checks only; no model training or reproduction claim'}
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='1')
    stages = []
    for name, source in PROBES.items():
        start = time.monotonic()
        result = subprocess.run([interpreter_path(args.python),'-c',source],env=env,
                                capture_output=True,text=True,timeout=120)
        (args.output/f'{name}.stdout.log').write_text(result.stdout)
        (args.output/f'{name}.stderr.log').write_text(result.stderr)
        stages.append({'stage':name,'exit_code':result.returncode,'seconds':time.monotonic()-start})
        print(json.dumps(stages[-1]),flush=True)
        if result.returncode:
            break
    packages = subprocess.run([interpreter_path(args.python),'-m','pip','freeze'],capture_output=True,text=True)
    # uv-created environments need not contain pip; capture distributions directly.
    if packages.returncode:
        packages = subprocess.run([interpreter_path(args.python),'-c',
            'import importlib.metadata as m; print("\\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions())))'],capture_output=True,text=True)
    (args.output/'packages.txt').write_text(packages.stdout)
    (args.output/'completion.json').write_text(json.dumps({'status':'passed-runtime-only' if len(stages)==len(PROBES) and not stages[-1]['exit_code'] else 'failed', 'stages':stages,'training_passed':False},indent=2)+'\n')
    return 0 if len(stages)==len(PROBES) and not stages[-1]['exit_code'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
