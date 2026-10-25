"""Chronological trial loader and past-only feature processing."""
from functools import lru_cache
import re
import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
from research.audit import ROOT, normalize_text

PHONES = 'AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH SIL'.split()
BLANK = 40


@lru_cache(maxsize=1)
def g2p():
    from g2p_en import G2p
    return G2p()


@lru_cache(maxsize=20000)
def labels(text):
    tokens = []
    for p in g2p()(normalize_text(text)):
        if p == ' ':
            tokens.append('SIL')
        else:
            p = re.sub('[0-9]', '', p)
            if re.match('[A-Z]+', p):
                if p not in PHONES:
                    raise ValueError(f'Unknown phoneme: {p}')
                tokens.append(p)
    tokens.append('SIL')
    return np.array([PHONES.index(p) for p in tokens], dtype=np.int64)


def load_trials(manifest, sessions, partition):
    out = []
    for session in sessions:
        entry = manifest['sessions'][session][partition]
        mat = loadmat(ROOT / entry['path'], variable_names=['tx1', 'spikePow'])
        counts, power = mat['tx1'].ravel(), mat['spikePow'].ravel()
        for row in entry['trials']:
            if not row['text']:
                continue
            i = row['index']
            raw = np.concatenate([counts[i][:, :128], power[i][:, :128]], axis=1).astype(np.float32)
            out.append({**row, 'session': session, 'raw': raw, 'labels': labels(row['text'])})
    return out


class CausalFeatures:
    """No look-ahead relative to supplied 20 ms bins; moments updated after each trial."""
    def __init__(self, mean, variance, samples, rolling=False):
        self.mean = np.asarray(mean, dtype=np.float64).copy()
        self.variance = np.asarray(variance, dtype=np.float64).copy()
        self.samples = int(samples)
        self.rolling = rolling

    @staticmethod
    def transform(raw):
        x = np.asarray(raw, dtype=np.float32).copy()
        x[:, 128:] = np.log1p(x[:, 128:])
        return x

    @classmethod
    def fit(cls, trials):
        n, total, squares = 0, np.zeros(256), np.zeros(256)
        for trial in trials:
            x = cls.transform(trial['raw']).astype(np.float64)
            n += len(x)
            total += x.sum(0)
            squares += (x*x).sum(0)
        mean = total/n
        return cls(mean, np.maximum(squares/n-mean*mean, 1e-4), n)

    def state(self):
        return {'mean': self.mean.tolist(), 'variance': self.variance.tolist(),
                'samples': self.samples, 'rolling': self.rolling}

    def process(self, raw, update=False):
        x = self.transform(raw)
        standardized = (x-self.mean)/np.sqrt(np.maximum(self.variance, 1e-4))
        # Trailing Gaussian, sigma 2 bins, truncation 4 sigma. Reset at trial start.
        kernel = np.exp(-.5*(np.arange(9)/2)**2)
        kernel /= kernel.sum()
        output = lfilter(kernel, [1.], standardized, axis=0).astype(np.float32)
        if update:
            if self.rolling:
                self.mean = .95*self.mean + .05*x.mean(0)
                self.variance = .95*self.variance + .05*x.var(0)
            else:
                m, v, k = x.mean(0), x.var(0), len(x)
                delta = m-self.mean
                total = self.samples+k
                self.variance = (self.samples*self.variance+k*v+delta*delta*self.samples*k/total)/total
                self.mean += delta*k/total
                self.samples = total
        return output


def prepare(trials, stats, rolling=False):
    state = CausalFeatures(**stats)
    state.rolling = rolling
    return [{**t, 'x': state.process(t['raw'], update=rolling)} for t in trials], state.state()


def prefix(trials, budget):
    return trials if budget == 'full' else trials[:int(budget)]
