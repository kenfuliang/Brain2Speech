"""Post-collection corrected inputs for the existing summary/figure readers.

The destination is a derived analytical view, not a decoder replay or checkpoint
collection. Never run the raw coverage auditor against this view.
"""
import argparse
import json
from pathlib import Path
from research.audit import digest, save_json
from research.experiment.coverage import expected_outputs
from research.reporting.derived_inputs import DerivedEvaluationLoader


def export_inputs(root, audit_root, output, manifest, freeze):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve existing derived view; choose a fresh output')
    if output==root or root in output.parents:
        raise ValueError('Derived analytical view must be outside raw collection')
    loader = DerivedEvaluationLoader(audit_root)
    expected = list(expected_outputs(manifest,freeze))
    missing = [relative for relative,_ in expected if (root/relative).resolve() not in loader.records]
    if missing:
        raise ValueError(f'Incomplete derived matrix: {len(missing)} required audited evaluations missing')
    # Validate before any destination is created. All source outputs are sealed.
    for relative,ids in expected:
        loader.load(root/relative,ids)
    provenance = []
    for relative,ids in expected:
        rows,metrics,evidence = loader.load(root/relative,ids)
        destination = output/relative
        save_json(destination/'predictions.json',rows)
        save_json(destination/'metrics.json',metrics)
        save_json(destination/'complete.json',{'derived_analytical_view':True,
            'not_a_raw_evaluator_completion':True,**evidence})
        provenance.append({'relative':relative,**evidence})
        identity = (root/relative).parent/'post.identity.json'
        if identity.exists():
            save_json(destination.parent/'post.identity.json',json.loads(identity.read_text()))
    for path in sorted((root/'sequential').glob('**/trajectory.json')):
        original = json.loads(path.read_text())
        derived = {**original,'transitions':[],'derived_scoring':'standard-whitespace-v2',
            'original_trajectory_sha256':digest(path)}
        for transition in original['transitions']:
            _,metrics,evidence = loader.load(path.parent/transition['day']/'fixed-probes')
            derived['transitions'].append({**transition,'historical_wer':metrics['wer']['macro_day'],
                'historical_scoring_provenance':evidence})
        relative = path.relative_to(root)
        save_json(output/relative,derived)
        reference = path.parent/'initial-probe-reference.json'
        save_json(output/relative.parent/reference.name,json.loads(reference.read_text()))
    save_json(output/'derived-inputs-manifest.json',{'status':'complete-derived-input-export',
        'raw_collection':str(root),'audit_root':str(Path(audit_root).resolve()),
        'export_method_sha256':digest(Path(__file__)),'expected_evaluations':len(expected),
        'evaluations':provenance,'scoring':'standard-whitespace-v2',
        'limitations':'Not checkpoint ancestry validation, scientific acceptance, or rendered figure QA. Raw inputs preserved; derived view contains no replay checkpoints/logits.'})
    return len(expected)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--audit-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = json.loads(Path('research/data/manifest.json').read_text())
    freeze = json.loads((args.root/'development-freeze.json').read_text())
    print(export_inputs(args.root,args.audit_root,args.output,manifest,freeze))
