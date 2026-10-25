"""Prospectively frozen fixed-option retry; preparation never launches decoding."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import selectors
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from research.reporting.nbest_metrics import summarize_candidates
IMAGE='speechbci-repro:522a9605'
IMAGE_ID='sha256:d7dad9120179787a8e2f949801fce1f9f2a360b170699e4724311fa4a8141432'
SOURCE=ROOT/'research/runs/e008-stopped-wer-20261007-01'
WORKER=ROOT/'reproduction/nbest_worker_fixed_options.py'
HELPER=ROOT/'reproduction/nbest_fixed_decoder.py'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def save(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def validate_trial(value,index,expected_text):
    if value['index']!=index: raise ValueError('Wrong trial index')
    single=value['outputs']['1']['candidates'];many=value['outputs']['50']['candidates']
    if len(single)!=1 or not 1<=len(many)<=50: raise ValueError('Invalid returned candidate counts')
    if single[0]['text']!=expected_text or many[0]['text']!=single[0]['text']:
        raise ValueError('Top-1 agreement failed')
    summarize_candidates([dict(id=str(index),session='technical',text='probe',candidates=many)],50)
    return many


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--memory-gib',type=int,choices=(48,64),default=48)
    a=p.parse_args()
    out=a.output.resolve()
    if shutil.disk_usage(ROOT).free<31*1024**3: raise RuntimeError('30 GiB floor + 1 GiB output reserve')
    command=['docker','--context','colima-speechbci']
    vm_memory=int(subprocess.check_output(command+['info','--format','{{.MemTotal}}'],text=True))
    if vm_memory<(a.memory_gib+8)*1024**3:
        raise RuntimeError('VM RAM must leave at least 8 GiB outside decoder limit')
    memory=str(a.memory_gib)+'g'
    if subprocess.check_output(command+['ps','--filter','ancestor='+IMAGE,'--format','{{.ID}}'],text=True).strip():
        raise RuntimeError('Decoder image already in use; do not interrupt its owner')
    image=subprocess.check_output(command+['image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()
    if image!=IMAGE_ID: raise ValueError('Image identity changed')
    decoder_identity=json.loads((SOURCE/'metrics.json').read_text())['decoder']
    for name,expected in decoder_identity['lm_files'].items():
        if sha(ROOT/'workspace/languageModel'/name)!=expected: raise ValueError('Decoder graph/dictionary changed')
    identity=json.loads((SOURCE/'identity.json').read_text())
    for name,expected in identity['artifacts_sha256'].items():
        if sha(SOURCE/name)!=expected: raise ValueError('Original export changed: '+name)
    original=json.loads((SOURCE/'decoded-predictions.json').read_text())
    if len(original)!=600: raise ValueError('Require 600 original trials')
    indices=[0,40] if a.smoke else list(range(600))
    protocol={'date':'2026-10-07','scope':'technical-smoke' if a.smoke else 'exposed-public-test-diagnostic',
        'indices':indices,'input_sha256':identity['artifacts_sha256'],'checkpoint_sha256':identity['checkpoint_sha256'],
        'original_decoded_sha256':sha(SOURCE/'decoded-predictions.json'),
        'worker_sha256':sha(WORKER),'controller_sha256':sha(Path(__file__)),
        'helper_sha256':sha(HELPER),
        'native_option_lifecycle':'Two constructor-fixed decoders share one DecodeResource; no SetOpt',
        'accounting_sha256':sha(ROOT/'research/reporting/nbest_metrics.py'),
        'image_id':image,'decoder':decoder_identity,
        'scoring':{'acoustic_scale':.8,'beam':18,'lattice_beam':8,'max_active':7000,'min_active':200,
                   'blank_skip_threshold':1.,'length_penalty':0.,'nbest_extract':50,'curves':[1,10,50],
                   'rescore':False,'phone_order':[40,39]+list(range(39))},
        'agreement':'Separate N=1 must exactly match saved E008 text; N=50 first candidate must exactly match N=1',
        'acceptance':'Technical accounting only; no learned-benefit or source-performance claim',
        'data_access':'Already-monitored 600 public tests; targets only used by host diagnostic scorer',
        'resource_budget':{'container_memory':memory,'cpus':4,'wall_seconds':14400,'reserve_bytes':1024**3},
        'training_restarted':False,'v1_restarted':False}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=protocol: raise ValueError('Resume protocol mismatch')
    else: save(out/'protocol.json',protocol)
    uncertainty={'seed':101,'repetitions':2000,
        'method':'Paired session-then-trial resampling of top1 and oracle errors with shared draws',
        'minimum_sessions':3,'small_session_rule':'No interval below three sessions',
        'diagnostic_only':True,'module_sha256':sha(ROOT/'research/reporting/nbest_uncertainty.py'),
        'scope':'Prespecified diagnostic uncertainty; separate from frozen native extraction protocol',
        'source':'Frozen before candidate collection in this run'}
    if (out/'uncertainty-plan.json').exists():
        if json.loads((out/'uncertainty-plan.json').read_text())!=uncertainty:
            raise ValueError('Uncertainty resume protocol mismatch')
    else: save(out/'uncertainty-plan.json',uncertainty)
    if a.prepare_only:
        print(json.dumps({'prepared':True,'launched':False,'protocol_sha256':sha(out/'protocol.json')}))
        return
    authorization_path=out/'authorization.json'
    if not authorization_path.exists():
        raise RuntimeError('Explicit user retry authorization required; prepare-only is permitted')
    authorization=json.loads(authorization_path.read_text())
    if authorization.get('approved') is not True or authorization.get('protocol_sha256')!=sha(out/'protocol.json'):
        raise ValueError('Authorization must approve this exact protocol hash')
    completed=[]
    for i in indices:
        if (out/('trial-%04d.json'%i)).exists(): completed.append(i)
    pending=[i for i in indices if i not in completed]
    if (out/'complete.json').exists():
        sealed=json.loads((out/'complete.json').read_text())
        for name,expected in sealed['sha256'].items():
            if sha(out/name)!=expected: raise ValueError('Completed artifact changed')
        for i,expected in sealed['trial_sha256'].items():
            if sha(out/('trial-%04d.json'%int(i)))!=expected: raise ValueError('Completed trial changed')
    container='wer-nbest-'+str(time.time_ns())
    cmd=command+['run','--name',container,'--memory',memory,'--cpus','4','--platform','linux/amd64',
        '-v',str(ROOT/'workspace')+':/data:ro','-v',str(SOURCE)+':/input:ro',
        '-v',str(WORKER)+':/worker.py:ro','-v',str(HELPER)+':/nbest_fixed_decoder.py:ro','--entrypoint','python',image,'-u','/worker.py',
        '--logits','/input/logits.npz','--indices',json.dumps(pending)]
    if pending:
        # Preserve container and log; no automatic deletion, failed launch never silently restarts.
        with (out/('worker-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.log')).open('x') as log:
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log,text=True)
            save(out/('launch-'+str(proc.pid)+'.json'),dict(pid=proc.pid,container=container,command=cmd))
            started=time.monotonic();selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
            try:
                while True:
                    if time.monotonic()-started>14400 or shutil.disk_usage(ROOT).free<30*1024**3:
                        raise RuntimeError('Resource/wall-time guard; preserve partial outputs')
                    if not selector.select(timeout=1):
                        if proc.poll() is not None: break
                        continue
                    line=proc.stdout.readline()
                    if not line: break
                    log.write(line);log.flush()
                    if not line.startswith('NBEST_JSON '): continue
                    value=json.loads(line[len('NBEST_JSON '):])
                    if value.get('ready'): print('Decoder ready',flush=True);continue
                    i=value['index']
                    if i not in pending: raise ValueError('Unexpected trial')
                    try: many=validate_trial(value,i,original[i]['decoded_text'])
                    except Exception:
                        save(out/('mismatch-%04d.json'%i),value);raise
                    save(out/('trial-%04d.json'%i),value)
                    print(json.dumps({'completed_index':i,'candidates':len(many)}),flush=True)
                if proc.wait(timeout=30)!=0: raise RuntimeError('Native worker failed; preserve log/container')
            except BaseException:
                if proc.poll() is None:
                    subprocess.run(command+['stop','--time','20',container],timeout=40,check=False)
                    proc.wait(timeout=30)
                raise
            finally: selector.close()
    rows=[]
    for i in indices:
        value=json.loads((out/('trial-%04d.json'%i)).read_text())
        validate_trial(value,i,original[i]['decoded_text'])
        rows.append(dict(id=original[i]['session']+'/test/%04d'%original[i]['trial_index'],
                         session=original[i]['session'],text=original[i]['text'],
                         candidates=value['outputs']['50']['candidates']))
    if not (out/'metrics.json').exists():
        save(out/'metrics.json',dict(curves={str(n):summarize_candidates(rows,n) for n in (1,10,50)},
             scope=protocol['scope'],top1_agreement_passed=True,performance_improvement_claim=False))
    if not (out/'complete.json').exists():
        save(out/'complete.json',dict(trials=len(rows),sha256={name:sha(out/name) for name in ['protocol.json','metrics.json']},
                                    trial_sha256={str(i):sha(out/('trial-%04d.json'%i)) for i in indices}))
    print(json.dumps({'complete':True,'trials':len(rows),'scope':protocol['scope']}),flush=True)


if __name__=='__main__': main()
