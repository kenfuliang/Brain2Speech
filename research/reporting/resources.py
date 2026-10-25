"""External resource sampling; never imports or changes the frozen training code."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from research.audit import digest, save_json


def native_processes(pids):
    result = subprocess.run(['ps','-p',','.join(map(str,pids)),
                             '-o','pid=,rss=,%cpu=,command='],text=True,capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        pid, rss, cpu, command = line.strip().split(maxsplit=3)
        if ' -m research.run ' not in command and ' -m research.experiment.pipeline ' not in command:
            continue  # A reused PID is not our experiment.
        rows.append({'pid':int(pid),'rss_bytes':int(rss)*1024,'cpu_percent':float(cpu),
                     'module':'research.experiment.pipeline' if 'research.experiment.pipeline' in command else 'research.run'})
    return rows


def decoder_memory(container):
    result = subprocess.run(['docker','--context','colima-speechbci','exec',container,'sh','-c',
                             'cat /sys/fs/cgroup/memory.current; cat /sys/fs/cgroup/memory.peak; cat /sys/fs/cgroup/memory.stat'],
                             text=True,capture_output=True,timeout=15)
    if result.returncode:
        return {'container':container,'available':False,'reason':'Worker exited or observation unavailable; not a research failure verdict'}
    lines = result.stdout.splitlines()
    stats = dict(line.split() for line in lines[2:])
    return {'container':container,'available':True,'current_bytes':int(lines[0]),
            'lifetime_peak_bytes':int(lines[1]),
            'anon_bytes':int(stats['anon']),'file_cache_bytes':int(stats['file']),
            'kernel_bytes':int(stats.get('kernel',0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pids',type=int,nargs='+',required=True)
    parser.add_argument('--container',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=60.)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing resource evidence; choose a new output directory')
    args.output.mkdir(parents=True)
    save_json(args.output/'method.json',{'method_sha256':digest(Path(__file__)),
              'pids':args.pids,'container':args.container,'interval_seconds':args.interval,
              'scope':'macOS process RSS and CPU; Linux decoder cgroup memory including cache and kernel. No summed host/unified-memory or native GPU-only peak claim.',
              'limitations':'Sampling starts after dispatch. Native RSS maxima are observed lower bounds, not whole-run peaks. Decoder cgroup peak covers its lifetime. An unavailable observation is not proof of process termination.'})
    index = 0
    while True:
        rows = native_processes(args.pids)
        try:
            decoder = decoder_memory(args.container)
        except (subprocess.TimeoutExpired, ValueError) as error:
            decoder = {'container':args.container,'available':False,'reason':type(error).__name__}
        save_json(args.output/f'sample-{index:06d}.json',
                  {'utc':datetime.now(timezone.utc).isoformat(),'native':rows,'decoder':decoder})
        index += 1
        if not rows:
            print(json.dumps({'status':'native-targets-ended','samples':index}),flush=True)
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
