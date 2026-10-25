"""Read-only decoder resource sample; never starts or stops a workload."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from research.audit import digest, save_json


def call(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    command = call('ps', '-p', str(args.pid), '-o', 'command=')
    assert 'decode_willett_final.py' in command and str(args.run) in command
    info = json.loads(call('docker', '--context', 'colima-speechbci', 'inspect', args.container))[0]
    assert info['State']['Running'] and info['Config']['Image'] == 'speechbci-repro:522a9605'
    stats = json.loads(call('docker', '--context', 'colima-speechbci', 'stats', '--no-stream',
                          '--format', '{{json .}}', args.container))
    save_json(args.output, dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
        run=str(args.run.resolve()), pid=args.pid, command=command,
        host_ps=call('ps', '-p', str(args.pid), '-o', 'rss=,pcpu=,etime=,lstart='),
        container_id=info['Id'], image_id=info['Image'], container_started=info['State']['StartedAt'],
        docker_context='colima-speechbci', docker_stats_display=stats,
        decoder_log_tail=(args.run/'decoder.log').read_text().splitlines()[-8:],
        free_storage_bytes=shutil.disk_usage(args.run).free, collector_sha256=digest(Path(__file__)),
        limitations=['One instantaneous sample; not peak memory or total graph-load time',
                     'Docker stats are rounded display values inside the emulated VM, not host unified-memory footprint',
                     'Host ps RSS is only the Python controller; graph memory is in the container',
                     'No sentence latency or WER inferred from loading state']))
    print('captured-live-decoder-resource-sample')


if __name__ == '__main__':
    main()
