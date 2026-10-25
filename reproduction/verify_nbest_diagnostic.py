"""Independent scalar edit-distance validation of sealed candidate diagnostics."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def distance(a,b):
    grid=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):grid[i][0]=i
    for j in range(len(b)+1):grid[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            grid[i][j]=min(grid[i-1][j]+1,grid[i][j-1]+1,grid[i-1][j-1]+(a[i-1]!=b[j-1]))
    return grid[-1][-1]


def components(a,b):
    """Independent full-grid/backtrace; ties prefer substitution, deletion, insertion."""
    grid=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):grid[i][0]=i
    for j in range(len(b)+1):grid[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            grid[i][j]=min(grid[i-1][j]+1,grid[i][j-1]+1,grid[i-1][j-1]+(a[i-1]!=b[j-1]))
    i,j=len(a),len(b);s=d=k=0
    while i or j:
        if i and j and a[i-1]==b[j-1]:i-=1;j-=1
        elif i and j and grid[i][j]==grid[i-1][j-1]+1:s+=1;i-=1;j-=1
        elif i and grid[i][j]==grid[i-1][j]+1:d+=1;i-=1
        else:k+=1;j-=1
    return dict(errors=grid[-1][-1],substitutions=s,deletions=d,insertions=k,reference_units=len(a))


def verify(run,source):
    protocol=json.loads((run/'protocol.json').read_text())
    complete=json.loads((run/'complete.json').read_text())
    metrics=json.loads((run/'metrics.json').read_text())
    original=json.loads((source/'decoded-predictions.json').read_text())
    assert sha(source/'decoded-predictions.json')==protocol['original_decoded_sha256']
    for name,expected in protocol['input_sha256'].items():assert sha(source/name)==expected
    for name,expected in complete['sha256'].items():assert sha(run/name)==expected
    assert complete['trials']==len(protocol['indices'])
    results={}
    for n in (1,10,50):
        totals={'top1_errors':0,'oracle_errors':0,'reference_words':0};by_day={}
        curve=metrics['curves'][str(n)]
        assert curve['trials']==len(protocol['indices'])
        for position,i in enumerate(protocol['indices']):
            file=run/('trial-%04d.json'%i)
            assert sha(file)==complete['trial_sha256'][str(i)]
            record=json.loads(file.read_text());assert record['index']==i
            one=record['outputs']['1']['candidates'];candidates=record['outputs']['50']['candidates'][:n]
            for candidate in one+candidates:
                assert isinstance(candidate['text'],str)
                assert all(math.isfinite(candidate[key]) for key in ('ac_score','lm_score'))
            assert one[0]['text']==original[i]['decoded_text']==candidates[0]['text']
            reference=original[i]['text'].split()
            counts=[distance(reference,c['text'].split()) for c in candidates]
            minimum=min(counts);rank=counts.index(minimum)+1
            audit=curve['trial_accounting'][position]
            assert audit['id']==original[i]['session']+'/test/%04d'%original[i]['trial_index']
            assert audit['session']==original[i]['session']
            assert audit['top1']['errors']==counts[0] and audit['oracle']['errors']==minimum
            assert audit['top1']==components(reference,candidates[0]['text'].split())
            assert audit['oracle']==components(reference,candidates[rank-1]['text'].split())
            for field in ('top1','oracle'):
                assert audit[field]['reference_units']==len(reference)
                assert audit[field]['errors']==sum(audit[field][key] for key in ('substitutions','deletions','insertions'))
            assert audit['oracle_rank']==rank and audit['returned']==len(candidates)
            assert audit['unique']==len({' '.join(c['text'].split()) for c in candidates})
            day=by_day.setdefault(original[i]['session'],dict(top1_errors=0,oracle_errors=0,reference_words=0))
            for target in (totals,day):
                target['top1_errors']+=counts[0];target['oracle_errors']+=minimum;target['reference_words']+=len(reference)
        for target in [totals,*by_day.values()]:
            target['top1_wer']=target['top1_errors']/target['reference_words']
            target['oracle_wer']=target['oracle_errors']/target['reference_words']
            target['oracle_gap']=(target['top1_errors']-target['oracle_errors'])/target['reference_words']
        assert curve['corpus']==totals and curve['by_day']==by_day
        results[str(n)]=totals
    assert results['1']['oracle_errors']>=results['10']['oracle_errors']>=results['50']['oracle_errors']
    return dict(status='passed-independent-oracle-accounting',trials=complete['trials'],scope=protocol['scope'],
                curves=results,metrics_sha256=sha(run/'metrics.json'),
                caveat='Oracle is reference-selected diagnostic, never deployed WER; not a model-improvement claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();result=verify(a.run,a.source)
    with (a.run/'verification.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
