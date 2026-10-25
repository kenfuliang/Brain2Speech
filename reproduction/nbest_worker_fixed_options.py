"""Candidate replacement diagnostic; no targets and no mutable native options.

This is not the frozen v1 worker or the original failed E010 worker. A new
protocol/identity and full-graph parity check are required before acceptance.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from neuralDecoder.utils import lmDecoderUtils as lm
from nbest_fixed_decoder import fixed_decoder_pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logits', required=True)
    p.add_argument('--indices', required=True)
    args = p.parse_args()
    model = Path('/data/languageModel')
    # Same optional rescoring-resource policy as the released Python builder.
    rescore = model / 'G_no_prune.fst'
    resource = lm.lm_decoder.DecodeResource(
        str(model / 'TLG.fst'), str(model / 'G.fst') if rescore.exists() else '',
        str(rescore) if rescore.exists() else '', str(model / 'words.txt'), '')
    decoders = fixed_decoder_pair(lm.lm_decoder, resource)
    print('NBEST_JSON ' + json.dumps({'ready': True}), flush=True)
    with np.load(args.logits, allow_pickle=False) as archive:
        for index in json.loads(args.indices):
            x = archive['trial_%04d' % index]
            if x.ndim != 2 or x.shape[1] != 41 or not np.isfinite(x).all():
                raise ValueError('Expected finite T x 41 logits')
            x = np.ascontiguousarray(x[:, [40, 39] + list(range(39))], dtype=np.float32)
            outputs = {}
            for n, decoder in decoders.items():
                start = time.perf_counter()
                candidates = lm.lm_decode(decoder, x, returnNBest=True,
                                          rescore=False, blankPenalty=np.log(2))
                outputs[str(n)] = {
                    'candidates': [dict(text=t.strip(), ac_score=float(ac), lm_score=float(ls))
                                   for t, ac, ls in candidates],
                    'decode_ms': (time.perf_counter() - start) * 1000}
            print('NBEST_JSON ' + json.dumps({'index': index, 'outputs': outputs},
                                             allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
