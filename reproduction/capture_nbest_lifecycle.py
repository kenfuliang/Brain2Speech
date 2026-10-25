"""Capture preserved native probes; no decoder launch or training."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ['docker', '--context', 'colima-speechbci']
NAMES = {
    'unsafe': 'nbest-lifecycle-acyclic-red-20261007',
    'fresh': 'nbest-lifecycle-acyclic-fresh-20261007',
    'retained': 'nbest-lifecycle-acyclic-retained-20261007',
    'fixed_pair': 'nbest-lifecycle-acyclic-pair-20261007',
}


def main():
    observations = {}
    for kind, name in NAMES.items():
        info = json.loads(subprocess.check_output(DOCKER + ['inspect', name], text=True))[0]
        if info['State']['Running']:
            raise RuntimeError('Cannot seal live probe')
        log = subprocess.check_output(DOCKER + ['logs', name], stderr=subprocess.STDOUT, text=True)
        rows = [json.loads(line) for line in log.splitlines() if line.startswith('{"requested":')]
        observations[kind] = dict(container=name, state=info['State'], image_id=info['Image'],
                                  command=info['Config']['Cmd'], log=log, rows=rows)
    assert observations['unsafe']['state']['ExitCode'] != 0
    assert observations['unsafe']['rows'][0]['count'] == 0
    for kind in ('fresh', 'retained', 'fixed_pair'):
        value = observations[kind]
        assert value['state']['ExitCode'] == 0 and not value['state']['OOMKilled']
        assert [r['requested'] for r in value['rows']][:2] == [1, 50]
        assert [r['count'] for r in value['rows']][:2] == [1, 2]
        assert all(r['candidates'][0] == value['rows'][0]['candidates'][0] for r in value['rows'])
    assert len(observations['fixed_pair']['rows']) == 4
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in (
        'reproduction/probe_nbest_lifecycle.py', 'reproduction/fixtures/nbest-acyclic.fst.txt',
        'reproduction/nbest_fixed_decoder.py', 'reproduction/nbest_worker_fixed_options.py',
        'reproduction/capture_nbest_lifecycle.py')}
    out = ROOT / 'research/runs/nbest-lifecycle-diagnostic-20261007-01'
    out.mkdir(parents=True, exist_ok=True)
    evidence = dict(scope='synthetic-native-lifecycle-only', observations=observations,
                    current_source_sha256=hashes,
                    earlier_acyclic_probe_sha256='f4292839a4d84fcd35513f04e4bd6b7425cefffc81e120ff1e3e03603d262dc9',
                    source_note='Unsafe/fresh/retained used the earlier version; fixed_pair used the current version. Logs and command identity retained, earlier source hash observed before edits.',
                    rejected_fixture='Released T.fst failed constructor-fixed N50 too; its determinization error is not accepted as a reproduction of the full-graph bug.',
                    conclusion='Option-lifetime hazard supported by source references and matched tiny native intervention; fixed-pair workaround passes synthetic seam only.',
                    full_graph_parity=False, oracle_wer=None, v1_restarted=False,
                    limitation='No sanitizer trace; exact full-graph heap-error origin and full parity remain unverified.')
    with (out / 'evidence.json').open('x') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'captured': True, 'native_fixed_pair_passed': True,
                      'full_graph_parity': False, 'path': str(out / 'evidence.json')}))


if __name__ == '__main__':
    main()
