"""Detached fixed-decoder evaluation; no training or automatic v1 restart."""
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    log = output/'decode-launch.log'
    record = output/'decode-launch.json'
    if record.exists() or log.exists():
        raise FileExistsError('Preserve existing launch artifacts')
    command = [str(root/'.venv-v1/bin/python'), '-u',
               str(root/'reproduction/evaluate_pytorch_recipe.py'),
               '--decode-only', '--output', str(output)]
    with log.open('xb') as stream:
        proc = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    identity = dict(pid=proc.pid, command=command, log=str(log),
                    launched_at=datetime.now(timezone.utc).isoformat(),
                    scope='Stopped E008 checkpoint; 600 public-test trials; fixed 3-gram',
                    training_restarted=False, automatically_resume_v1=False)
    record.write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
