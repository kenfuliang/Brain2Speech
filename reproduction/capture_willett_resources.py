"""Read-only live-process resource/provenance snapshot; never initializes GPU."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    status = json.loads((args.run/'status.json').read_text())
    assert status['status'] == 'training' and status['pid'] == args.pid
    pids = [args.pid]+[int(value) for value in subprocess.check_output(
        ['pgrep', '-P', str(args.pid)], text=True).split()]
    processes = []
    for pid in pids:
        command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True).strip()
        assert ('launch_willett_verified.py' if pid == args.pid else 'tf_recipe_service.py') in command
        fields = subprocess.check_output(['ps', '-p', str(pid), '-o', 'rss=,pcpu=,etime='], text=True).split()
        processes.append(dict(pid=pid, command=command, rss_bytes=int(fields[0])*1024,
            cpu_percent=float(fields[1]), elapsed=fields[2],
            started_local=subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart='], text=True).strip()))
    result = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
        run=str(args.run.resolve()), last_observed_attempt=status['last_batch']['attempt'],
        best_validation_per=status['best_validation_per'], processes=processes,
        free_storage_bytes=shutil.disk_usage(args.run).free,
        physical_memory_bytes=int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
        chip=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        os=platform.platform(), torch_distribution_version=version('torch'),
        limitations=['Instantaneous per-process RSS/CPU, not peak memory',
                     'RSS does not measure MPS allocator or unique unified-memory usage; do not sum as system footprint',
                     'ps elapsed includes replay, validation and checkpoint I/O; batch timings omit those phases'])
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
