"""Fail-closed corrected inputs for offline paired and forgetting analyses."""
import json
from pathlib import Path
from research.audit import digest
from research.metrics import edits, aggregate, summarize
from research.reporting import derived_verification


class DerivedEvaluationLoader:
    def __init__(self, audit_root):
        self.audit_root = Path(audit_root)
        self.records = {}
        for path in sorted((self.audit_root/'evaluations').glob('*.json')):
            audit = json.loads(path.read_text())
            source = Path(audit['source']).resolve()
            if source in self.records:
                raise ValueError('Ambiguous derived evaluation versions')
            self.records[source] = (path,audit)

    def load(self, source, expected_ids=None):
        source = Path(source).resolve()
        if source not in self.records:
            raise ValueError('Missing derived audit; raw metrics cannot substitute')
        path,audit = self.records[source]
        if audit.get('schema_version')!=2 or audit.get('status')!='verified-derived-scoring':
            raise ValueError('Unrecognized derived scoring audit')
        if audit['method_sha256']!=digest(Path(derived_verification.__file__)) or audit['metrics_method_sha256']!=digest(Path(summarize.__code__.co_filename)):
            raise ValueError('Derived method identity changed')
        if audit['complete_sha256']!=digest(source/'complete.json'):
            raise ValueError('Completed evaluation identity changed')
        for name,sha in audit['artifact_sha256'].items():
            if digest(source/name)!=sha:
                raise ValueError('Source artifact hash changed: '+name)
        rows = json.loads((source/'predictions.json').read_text())
        if expected_ids is not None and [r['id'] for r in rows]!=expected_ids:
            raise ValueError('Derived input trial identity mismatch')
        corrected = [{**r,'wer':edits(r['text'].split(),r['decoded_text'].split())} for r in rows]
        if len(corrected)!=audit['verified_trials']:
            raise ValueError('Derived input trial count mismatch')
        for field in ['wer','per','cer']:
            totals = {k:sum(r[field][k] for r in corrected) for k in corrected[0][field]}
            if totals!=audit['corrected_edit_totals'][field] or aggregate(corrected,field)!=audit['derived_metrics'][field]['corpus']:
                raise ValueError('Derived input edits/metrics disagree with audit')
        return corrected,audit['derived_metrics'],{
            'audit_path':str(path.resolve()),'audit_sha256':digest(path),
            'source_complete_sha256':audit['complete_sha256'],
            'scoring':'standard-whitespace-v2','rate_changed_trials':audit['rate_changed_trials']}
