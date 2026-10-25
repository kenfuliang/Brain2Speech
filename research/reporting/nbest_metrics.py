"""Diagnostic-only N-best accounting; references must never select deployed output.

Input rows contain id, session, text and an ordered candidates list. Candidates
contain text, ac_score and lm_score. Order is the decoder's ranking, not a score
recomputed here. Outputs are descriptive, not an improvement significance test.
"""
import math
from research.metrics import edits


def summarize_candidates(rows, n):
    if not rows or not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('Require nonempty rows and positive integer N')
    ids = set()
    scored = []
    for row in rows:
        if row['id'] in ids:
            raise ValueError('Duplicate trial identity')
        ids.add(row['id'])
        reference = row['text'].split()
        if not reference:
            raise ValueError('Empty reference denominator')
        candidates = row['candidates'][:n]
        if not candidates:
            raise ValueError('Missing candidates; do not manufacture an empty hypothesis')
        for candidate in candidates:
            if not isinstance(candidate['text'], str) or not all(
                    math.isfinite(candidate[key]) for key in ('ac_score', 'lm_score')):
                raise ValueError('Invalid candidate or score')
        counts = [edits(reference, candidate['text'].split()) for candidate in candidates]
        oracle_index = min(range(len(counts)), key=lambda i: counts[i]['errors'])
        top, oracle = counts[0], counts[oracle_index]
        scored.append(dict(id=row['id'], session=row['session'], top1=top,
                           oracle=oracle, oracle_rank=oracle_index+1,
                           returned=len(candidates),
                           unique=len({' '.join(c['text'].split()) for c in candidates})))
    def rates(group):
        units = sum(r['top1']['reference_units'] for r in group)
        top = sum(r['top1']['errors'] for r in group)
        oracle = sum(r['oracle']['errors'] for r in group)
        return dict(reference_words=units, top1_errors=top, oracle_errors=oracle,
                    top1_wer=top/units, oracle_wer=oracle/units,
                    oracle_gap=(top-oracle)/units)
    by_day = {day: rates([r for r in scored if r['session'] == day])
              for day in sorted({r['session'] for r in scored})}
    return dict(requested_n=n, trials=len(scored), sessions=len(by_day),
                corpus=rates(scored), by_day=by_day, trial_accounting=scored,
                diagnostic_only=True,
                uncertainty='Not estimated; descriptive candidate diagnostic, not a supported improvement',
                normalization='Whitespace word tokenization; no case/punctuation changes',
                caveat='Reference-selected oracle is not deployable; paths may duplicate transcripts; missing candidates fail closed')
