"""Serial fixed3gram evaluation of E016's selected validation and test logits.

Not BIT5gram/OPT reproduction. Requires a completed independently audited run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.audit import digest,save_json,space_guard
from research.experiment.evaluator import FixedDecoder,IMAGE
from research.metrics import edits,summarize
from verify_bit_full import distance


def docker(*args):
    return subprocess.check_output(['docker','--context','colima-speechbci',*args],text=True).strip()


def worker_ids():
    return docker('ps','--filter','ancestor='+IMAGE,'--format','{{.ID}}').split()


def require_stopped_training(run):
    pid=json.loads((run/'launch.json').read_text())['pid']
    result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    assert not ('train_bit_full.py' in result.stdout), 'BIT trainer still live; wait, never restart it'
    processes=subprocess.check_output(['ps','-ax','-o','command='],text=True).splitlines()
    assert not any('research.experiment.pipeline --stage' in p or
                   ('train_pytorch_recipe.py' in p and '--output' in p) or
                   ('willett_fail_stop.py' in p and '--output' in p) for p in processes), 'Competing research workload'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve()
    require_stopped_training(run);space_guard()
    assert not worker_ids(),'Resident decoder already owned; do not start another'
    assert json.loads((run/'status.json').read_text())['status']=='training_complete_wer_pending'
    audit=json.loads(args.audit.read_text())
    assert audit['passed'] and audit['epoch']==800 and audit['steps']==220000
    assert audit['validation']['passed'] and audit['test']['passed']
    chosen=json.loads((run/'best-per.json').read_text());assert digest(run/chosen['checkpoint'])==chosen['sha256']
    sealed={}
    for split in ('validation','test'):
        for name in ('logits.npz','predictions.json','metrics.json'):
            path=run/f'selected-{split}'/name;sealed[str(path.relative_to(run))]=digest(path)
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'contract.json',{'run':str(run),'selected_checkpoint':chosen,'audit_sha256':digest(args.audit),
        'export_sha256':sealed,'runner_sha256':digest(Path(__file__)),'mode':'fixed3gram diagnostic, not BIT5gram+OPT',
        'references':'Original prepared lowercase punctuation-stripped sentences; whitespace word tokens; characters include spaces',
        'selection':'Frozen standard validation-PER-selected checkpoint; no WER reselection',
        'uncertainty':'2000 session-then-trial bootstrap repetitions,seed101; conditional on selection; one seed/historical public-label exposure'})
    save_json(out/'status.json',{'status':'loading-decoder'})
    decoder=FixedDecoder(out/'decoder.log')
    owned=worker_ids();assert len(owned)==1
    save_json(out/'decoder-identity.json',dict(decoder.identity,container=owned[0]))
    try:
        for split in ('validation','test'):
            folder=run/f'selected-{split}'
            rows=json.loads((folder/'predictions.json').read_text());assert len(rows)==440
            save_json(out/'status.json',{'status':'decoding','split':split,'trials':440})
            print(json.dumps({'status':'decoding','split':split}),flush=True)
            response=decoder.decode(folder)
            assert len(response['transcripts'])==len(response['decode_ms'])==len(rows)
            for row,transcript,ms in zip(rows,response['transcripts'],response['decode_ms']):
                reference=row['reference'];hypothesis=transcript.strip()
                row.update(transcript=hypothesis,wer=edits(reference.split(),hypothesis.split()),
                           cer=edits(reference,hypothesis),decode_ms=ms)
                assert distance(reference.split(),hypothesis.split())==row['wer']['errors']
                assert distance(reference,hypothesis)==row['cer']['errors']
            save_json(out/f'{split}-predictions.json',rows)
            result=summarize(rows,repetitions=2000)
            result['error_components']={metric:{key:sum(r[metric][key] for r in rows)
                for key in ('errors','reference_units','substitutions','deletions','insertions')}
                for metric in ('per','wer','cer')}
            result.update(decode_ms_total=sum(response['decode_ms']),selected_checkpoint_sha256=chosen['sha256'],
                          claim='Fixed3gram public diagnostic; no headline BIT-All reproduction')
            save_json(out/f'{split}-metrics.json',result);print(json.dumps(result),flush=True)
    finally:
        decoder.close()
    # Large graph teardown may outlast the client close timeout. Verify the
    # exact owned container rather than failing from a single late observation.
    for attempt in range(120):
        if owned[0] not in worker_ids():break
        if attempt%3==0:print(json.dumps({'status':'waiting-owned-decoder-exit','container':owned[0]}),flush=True)
        time.sleep(5)
    assert owned[0] not in worker_ids(),'Decoder cleanup incomplete; no next workload may start'
    assert all(digest(run/name)==sha for name,sha in sealed.items())
    save_json(out/'verification.json',{'passed':True,'independent_word_character_accounting':True,
                                      'sealed_exports_unchanged':True,'owned_decoder_exited':True})
    save_json(out/'status.json',{'status':'complete','validation_trials':440,'test_trials':440,
                               'v1':'Remains paused under BIT/Willett-first goal'})


if __name__=='__main__':main()
