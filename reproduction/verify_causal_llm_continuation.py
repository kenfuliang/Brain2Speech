"""Read-only independent accounting of selected continuation validation WER."""
import json
from pathlib import Path
import sys

from verify_causal_neural_llm import distance
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.audit import digest, save_json


def verify(path):
    protocol = json.loads((path/'protocol.json').read_text())
    assert digest(Path(__file__).with_name('continue_causal_neural_llm.py')) == protocol['source_sha256']
    parent = Path(protocol['parent'])
    assert digest(parent/'checkpoint-40.pt') == protocol['parent_checkpoint_sha256']
    assert digest(parent/'lora-step-40/adapter_model.safetensors') == protocol['parent_adapter_sha256']
    assert json.loads((path/'resume-validation.json').read_text())['passed']
    trials = json.loads((path/'trials.json').read_text())
    assert trials == json.loads((parent/'trials.json').read_text())
    lookup = {r['id']: r['text'] for r in trials if r['split'] == 'validation'}

    def account(rows):
        assert len(rows) == 16 and {r['id'] for r in rows} == set(lookup)
        errors = units = 0
        for row in rows:
            assert row['reference'] == lookup[row['id']]
            reference = row['reference'].lower().split()
            computed = distance(reference, row['prediction'].lower().split())
            assert row['wer']['errors'] == computed
            assert row['wer']['reference_units'] == len(reference)
            assert computed == sum(row['wer'][key] for key in ['substitutions', 'deletions', 'insertions'])
            errors += computed; units += len(reference)
        return errors, units

    checked = []
    for file in sorted(path.glob('step-*/evaluation.json')):
        row = json.loads(file.read_text())
        assert digest(file.parent/'checkpoint.pt') == row['checkpoint_sha256']
        errors, units = account(json.loads((file.parent/'validation-predictions.json').read_text()))
        assert errors == row['wer']['errors'] and units == row['wer']['reference_units']
        assert abs(errors/units-row['wer']['rate']) < 1e-12
        checked.append(row)
    selected = min(checked, key=lambda row: (row['wer']['errors'], row['loss']['real'], row['step']))
    best = json.loads((path/'best.json').read_text())
    assert selected == best
    status = json.loads((path/'status.json').read_text())
    if status.get('reason') == 'sustained_validation_deterioration':
        lowest, bad = float('inf'), 0
        for row in sorted(checked, key=lambda item: item['step']):
            value = row['loss']['real']; lowest = min(lowest, value)
            bad = bad+1 if value > lowest*protocol['stability']['validation_ratio'] else 0
        assert bad >= protocol['stability']['consecutive_checks']
    review = path/'best-review.json'
    character_metrics = {}
    if review.exists():
        recorded = json.loads(review.read_text())
        assert recorded['best'] == best
        for condition, row in recorded['controls'].items():
            predictions = json.loads((path/f'best-{condition}-predictions.json').read_text())
            errors, units = account(predictions)
            assert errors == row['errors'] and units == row['reference_units']
            assert abs(errors/units-row['rate']) < 1e-12
            char_errors = char_units = 0
            for prediction in predictions:
                reference = ' '.join(prediction['reference'].lower().split())
                hypothesis = ' '.join(prediction['prediction'].lower().split())
                char_errors += distance(reference, hypothesis); char_units += len(reference)
            character_metrics[condition] = dict(errors=char_errors, reference_characters=char_units,
                cer=char_errors/char_units, normalization='lowercase, collapsed whitespace, spaces included')
    return dict(passed=True, verified_evaluations=len(checked), best_step=best['step'],
                best_validation_wer=best['wer']['rate'], best_word_errors=best['wer']['errors'],
                reference_words=best['wer']['reference_units'], review_available=review.exists(),
                stop_reason=status.get('reason'),
                selected_checkpoint_cer=character_metrics,
                independently_recomputed_scalar_edits=True, independent_test=False,
                component_backtracking_independently_verified=False)


if __name__ == '__main__':
    path = Path(sys.argv[1]); result = verify(path)
    if len(sys.argv) > 2:
        output = Path(sys.argv[2]); assert not output.exists(), 'Preserve existing verification'
        save_json(output, result)
    print(json.dumps(result, indent=2))
