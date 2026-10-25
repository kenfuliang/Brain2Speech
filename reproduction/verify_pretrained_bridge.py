"""Independent E013 phone/word accounting, split and checkpoint selection audit."""
import json
from pathlib import Path
import sys
import numpy as np

from verify_causal_neural_llm import distance
from bit_feasibility import ENCODER, ENCODER_HASH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit import digest, save_json


def verify(path):
    read=lambda name: json.loads((path/name).read_text())
    data=read('data-identity.json'); trials=read('trials.json')
    assert digest(ENCODER)==ENCODER_HASH==data['encoder_sha256']
    assert digest(path/'features.npz')==data['features_sha256']
    assert digest(path/'trials.json')==data['trials_sha256']
    training=[r for r in trials if r['split']=='train']; validation=[r for r in trials if r['split']=='validation']
    assert len(training)==512 and len(validation)==40
    norm=lambda s: ' '.join(s.lower().split())
    assert not {norm(r['text']) for r in training} & {norm(r['text']) for r in validation}
    assert len({r['id'] for r in trials})==552
    phone_rows=read('phoneme-predictions.json')
    assert {r['id'] for r in phone_rows}=={r['id'] for r in validation}
    errors=units=0
    for row in phone_rows:
        error=distance(row['reference'],row['prediction'])
        assert error==row['per']['errors']
        errors+=error; units+=len(row['reference'])
    gate=read('encoder-gate.json')
    assert errors==gate['errors'] and units==gate['reference_phones']
    assert abs(errors/units-gate['per'])<1e-12
    assert gate['passed']==(errors/units<=gate['usability_threshold'])
    references={r['id']:r['text'] for r in validation}
    def account(rows):
        assert len(rows)==40 and {r['id'] for r in rows}==set(references)
        errors=units=char_errors=chars=0
        for row in rows:
            assert row['reference']==references[row['id']]
            a,b=norm(row['reference']),norm(row['prediction'])
            error=distance(a.split(),b.split()); n=len(a.split())
            assert error==row['wer']['errors'] and n==row['wer']['reference_units']
            assert error==sum(row['wer'][k] for k in ['substitutions','deletions','insertions'])
            errors+=error; units+=n; char_errors+=distance(a,b); chars+=len(a)
        return dict(errors=errors,reference_words=units,wer=errors/units,
                    character_errors=char_errors,reference_characters=chars,cer=char_errors/chars)
    checked=[]
    for file in sorted(path.glob('step-*/evaluation.json')):
        row=json.loads(file.read_text()); assert digest(file.parent/'checkpoint.pt')==row['checkpoint_sha256']
        measured=account(json.loads((file.parent/'predictions.json').read_text()))
        assert measured['errors']==row['wer']['errors']
        assert abs(measured['wer']-row['wer']['rate'])<1e-12
        checked.append(row)
    best=read('best.json')
    assert best==min(checked,key=lambda row:(row['wer']['errors'],row['loss']['real'],row['step']))
    result=dict(passed=True,causal_per=gate['per'],verified_evaluations=len(checked),
        best_step=best['step'],best_validation_wer=best['wer']['rate'],
        historical_encoder_validation_exposure=True,independent_test=False,
        component_alignment_independently_verified=False)
    if (path/'metrics.json').exists():
        metrics=read('metrics.json'); assert metrics['best']==best
        controls={}
        prompts={norm(r['text']) for r in training}
        for condition, recorded in metrics['controls'].items():
            rows=read(f'best-{condition}-predictions.json'); measured=account(rows)
            assert measured['errors']==recorded['errors']
            assert abs(measured['wer']-recorded['rate'])<1e-12
            measured['unique_generations']=len({r['prediction'] for r in rows})
            measured['training_prompt_matches']=sum(norm(r['prediction']) in prompts for r in rows)
            controls[condition]=measured
        result['selected_controls']=controls
        protocol=read('uncertainty-protocol.json'); rng=np.random.default_rng(protocol['seed'])
        real=read('best-real-predictions.json')
        units=np.array([r['wer']['reference_units'] for r in real])
        real_errors=np.array([r['wer']['errors'] for r in real])
        indices=rng.integers(0,40,size=(protocol['repetitions'],40))
        contrasts={}
        for condition in protocol['controls']:
            rows=read(f'best-{condition}-predictions.json')
            assert [r['id'] for r in rows]==[r['id'] for r in real]
            differences=np.array([r['wer']['errors'] for r in rows])-real_errors
            samples=differences[indices].sum(1)/units[indices].sum(1)
            contrasts[condition]=dict(control_minus_real=float(differences.sum()/units.sum()),
                conditional_ci95=np.quantile(samples,[.025,.975]).tolist())
        result['exploratory_paired_contrasts']=contrasts
        result['uncertainty_limit']=protocol['limits']
    return result


if __name__=='__main__':
    path=Path(sys.argv[1]); result=verify(path)
    if len(sys.argv)>2:
        output=Path(sys.argv[2]); assert not output.exists(); save_json(output,result)
    print(json.dumps(result,indent=2))
