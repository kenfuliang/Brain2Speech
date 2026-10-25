"""Post-hoc accounting of saved E009 examples, not full 16-trial evaluation."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from research.metrics import edits
from verify_nbest_diagnostic import distance, components


def score(examples,trials):
    expected=[r for r in trials if r['split']=='evaluation'][:4]
    if len(expected)!=4 or len(examples)!=4:
        raise ValueError('Require four saved examples for each of four conditions')
    result={}
    for group in examples:
        condition=group['condition']
        if condition in result or condition not in ('real','zero','shuffled','language-only'):
            raise ValueError('Duplicate or unknown condition')
        if group['references']!=[r['text'] for r in expected] or len(group['predictions'])!=4:
            raise ValueError('Saved first-four identity/order mismatch')
        rows=[]
        for trial,text in zip(expected,group['predictions']):
            if not isinstance(text,str):raise ValueError('Non-text prediction')
            reference=trial['text']
            if not reference.split():raise ValueError('Empty reference')
            row=dict(key=trial['key'],reference=reference,prediction=text)
            for field,a,b in [('wer',reference.split(),text.split()),('cer',list(reference),list(text))]:
                row[field]=edits(a,b)
                assert row[field]['errors']==distance(a,b)
                assert row[field]==components(a,b)
            rows.append(row)
        totals={field:{k:sum(r[field][k] for r in rows) for k in rows[0][field]}
                for field in ('wer','cer')}
        result[condition]=dict(trials=4,totals=totals,
            rates={field:totals[field]['errors']/totals[field]['reference_units'] for field in totals},
            rows=rows)
    return result


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    source=ROOT/'reproduction/results/bit-feasibility/pilot-20261007-01'
    output=ROOT/'research/runs/e009-saved-generation-accounting-20261007-01'
    if output.exists():raise ValueError('Preserve prior accounting; use a new version')
    verified=json.loads((source/'verification.json').read_text())
    names=('generated-transcripts.json','trials.json')
    for name in names:
        if sha(source/name)!=verified['artifacts_sha256'][name]:raise ValueError('Original artifact changed')
    protocol=dict(inputs={name:sha(source/name) for name in names},code_sha256=sha(Path(__file__)),
        independent_verifier_sha256=sha(ROOT/'reproduction/verify_nbest_diagnostic.py'),
        scope='Post-hoc descriptive accounting of existing first-four generation examples, not full evaluation',
        normalization='Standard whitespace words; case/punctuation unchanged; raw characters',
        selection='Original trainer generated evaluation[:4] in every condition; no selection by result',
        controls='Inference ablations of same trained model, not separately trained equal-budget controls',
        uncertainty='Not estimated: four sentences, one session, one seed',
        per='Not applicable: no phoneme sequence emitted by direct text generation',
        training_restarted=False,generation_repeated=False)
    output.mkdir()
    with (output/'protocol.json').open('x') as stream:json.dump(protocol,stream,indent=2)
    result=score(json.loads((source/names[0]).read_text()),json.loads((source/names[1]).read_text()))
    with (output/'metrics.json').open('x') as stream:json.dump(result,stream,indent=2)
    with (output/'verification.json').open('x') as stream:json.dump(dict(
        status='passed-independent-edit-components-and-source-identities',
        metrics_sha256=sha(output/'metrics.json'),protocol_sha256=sha(output/'protocol.json'),
        full_evaluation_wer_measured=False,neural_grounding_established=False),stream,indent=2)
    print(json.dumps({condition:{k:v for k,v in group.items() if k!='rows'} for condition,group in result.items()},indent=2))
