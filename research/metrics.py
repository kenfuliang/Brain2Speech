"""Edit accounting and paired resampling; fractions in machine-readable output."""
import numpy as np


def edits(reference, hypothesis):
    # State stores (total, substitutions, deletions, insertions), deterministic tie order.
    previous = [(j, 0, 0, j) for j in range(len(hypothesis)+1)]
    for i, r in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, h in enumerate(hypothesis, 1):
            if r == h:
                current.append(previous[j-1])
                continue
            t, s, d, k = previous[j-1]
            sub = (t+1, s+1, d, k)
            t, s, d, k = previous[j]
            delete = (t+1, s, d+1, k)
            t, s, d, k = current[j-1]
            insert = (t+1, s, d, k+1)
            current.append(min((sub, delete, insert), key=lambda v: v[0]))
        previous = current
    total, sub, delete, insert = previous[-1]
    return {'errors': total, 'substitutions': sub, 'deletions': delete,
            'insertions': insert, 'reference_units': len(reference)}


def greedy(logits):
    sequence = np.argmax(logits, axis=-1).tolist()
    return [p for i, p in enumerate(sequence) if p != 40 and (i == 0 or p != sequence[i-1])]


def aggregate(rows, field):
    errors = sum(r[field]['errors'] for r in rows)
    units = sum(r[field]['reference_units'] for r in rows)
    return errors/units if units else None


def summarize(rows, seed=101, repetitions=1000):
    days = sorted({r['session'] for r in rows})
    groups = {d: [r for r in rows if r['session'] == d] for d in days}
    per_day = {d: {f: aggregate(g, f) for f in ('per', 'wer', 'cer') if f in g[0]}
               for d, g in groups.items()}
    result = {'trials': len(rows), 'sessions': len(days), 'by_day': per_day,
              'uncertainty_warning': 'single participant; few sessions limit population inference'}
    rng = np.random.default_rng(seed)
    for field in ('per', 'wer', 'cer'):
        if not rows or field not in rows[0]:
            continue
        values = []
        for _ in range(repetitions):
            sample = []
            for day in rng.choice(days, len(days)):
                g = groups[day]
                sample.extend(g[i] for i in rng.integers(0, len(g), len(g)))
            values.append(aggregate(sample, field))
        result[field] = {'corpus': aggregate(rows, field),
                         'macro_day': float(np.mean([x[field] for x in per_day.values()])),
                         'worst_day': float(max(x[field] for x in per_day.values())),
                         'ci95': np.quantile(values, [.025, .975]).tolist()}
    return result
