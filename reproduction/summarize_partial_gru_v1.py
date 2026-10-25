"""Descriptive all-completed seed-17 GRU final-day control, not v1 acceptance."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from research.audit import digest,save_json
from research.experiment.coverage import expected_outputs
from research.reporting.derived_inputs import DerivedEvaluationLoader
from research.metrics import summarize


def group_paths(manifest,freeze):
    groups={}
    for relative,ids in expected_outputs(manifest,freeze):
        parts=relative.split('/')
        if parts[:3]==['final','gru','17']:
            groups.setdefault('/'.join(parts[4:]),[]).append((relative,ids))
    if len(groups)!=32 or any(len(paths)!=5 for paths in groups.values()):
        raise ValueError('Require the complete frozen 32-condition/five-day GRU control')
    return groups


if __name__=='__main__':
    root=ROOT/'research/runs/v1-comparison'
    output=ROOT/'research/runs/v1-partial-gru-s17-summary-20261007-01'
    if output.exists():raise ValueError('Preserve prior summaries')
    manifest_path=ROOT/'research/data/manifest.json';freeze_path=root/'development-freeze.json'
    manifest=json.loads(manifest_path.read_text());freeze=json.loads(freeze_path.read_text())
    groups=group_paths(manifest,freeze)
    loader=DerivedEvaluationLoader(ROOT/'research/runs/v1-comparison-derived-audit-20261007-01')
    # Refuse partial groups; no outcome-dependent choice of complete days.
    for paths in groups.values():
        for relative,ids in paths:loader.load(root/relative,ids)
    output.mkdir()
    save_json(output/'protocol.json',dict(manifest_sha256=digest(manifest_path),
        freeze_sha256=digest(freeze_path),code_sha256=digest(Path(__file__)),
        selection='All 32 prespecified conditions, all five final days; GRU seed17 only',
        seed=17,bootstrap_seed=101,bootstrap_repetitions=2000,
        scope='Post-hoc descriptive summary of existing outputs; not architecture comparison or completed v1',
        exposure='Final days previously opened; no tuning/retraining performed',
        uncertainty='Session-then-trial bootstrap conditional on one fixed seed/checkpoint; five-day and one-participant limitations',
        adaptation_steps=freeze['selected']['gru']['adaptation_steps'],
        source_checkpoint=freeze['selected']['gru']['checkpoint'],
        training_restarted=False,decoder_restarted=False))
    result={}
    for condition,paths in groups.items():
        rows=[];provenance=[]
        for relative,ids in paths:
            r,_,p=loader.load(root/relative,ids);rows.extend(r);provenance.append(dict(relative=relative,**p))
        totals={f:{k:sum(r[f][k] for r in rows) for k in rows[0][f]} for f in ('wer','per','cer')}
        result[condition]=dict(metrics=summarize(rows,seed=101,repetitions=2000),
            edit_totals=totals,source_audits=provenance)
        print(condition,round(result[condition]['metrics']['wer']['corpus']*100,4),flush=True)
    save_json(output/'metrics.json',result)
    save_json(output/'verification.json',dict(status='passed-derived-input-identities-and-complete-condition-coverage',
        conditions=len(result),source_evaluations=160,unique_test_trials=200,
        metrics_sha256=digest(output/'metrics.json'),protocol_sha256=digest(output/'protocol.json'),
        limitations='Totals use previously verified corrected edits; this is not an independent full-matrix or new-bootstrap implementation audit',
        v1_complete=False,lfads_benefit_established=False,three_seed_result=False))
