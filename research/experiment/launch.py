"""Queue completion after the active real decoder smoke; never duplicate graph loads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from research.audit import ROOT, save_json, space_guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-pid', type=int, required=True)
    parser.add_argument('--smoke-output', type=Path, default=ROOT/'research/runs/v1-wfst-smoke')
    parser.add_argument('--output', type=Path, default=ROOT/'research/runs/v1-comparison')
    args = parser.parse_args()
    while True:
        space_guard()
        command = subprocess.run(['ps','-p',str(args.smoke_pid),'-o','command='],
                                 text=True,capture_output=True).stdout.strip()
        if not command or 'research.experiment.evaluator' not in command:
            break
        save_json(args.output/'queue-status.json', {'status':'awaiting-real-wfst-smoke',
                  'queue_pid':os.getpid(), 'smoke_pid':args.smoke_pid,
                  'next_stage':'development selection, then frozen final matrix',
                  'final_days_opened':False})
        time.sleep(15)
    complete = args.smoke_output/'complete.json'
    if not complete.exists():
        save_json(args.output/'queue-status.json', {'status':'blocked-smoke-failed',
                  'reason':'No completed real WFST smoke; no downstream experiment launched'})
        raise RuntimeError('Real WFST smoke failed; inspect its decoder log')
    save_json(args.output/'queue-status.json', {'status':'dispatching-completion-pipeline',
              'queue_pid':os.getpid(), 'smoke_identity':json.loads(complete.read_text())['identity']})
    os.execv(sys.executable, [sys.executable,'-u','-m','research.experiment.pipeline',
             '--stage','full','--wait','--require-smoke',str(args.smoke_output),
             '--output',str(args.output)])


if __name__ == '__main__':
    main()
