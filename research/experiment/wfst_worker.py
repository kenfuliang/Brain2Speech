"""Persistent fixed-decoder worker, executed only inside the E000 image."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from neuralDecoder.utils import lmDecoderUtils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm', default='/data/languageModel')
    args = parser.parse_args()
    decoder = lmDecoderUtils.build_lm_decoder(args.lm, acoustic_scale=.8, nbest=1, beam=18)
    print('WFST_JSON '+json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        try:
            folder = Path(json.loads(line)['folder'])
            rows = json.loads((folder/'predictions.json').read_text())
            transcripts, milliseconds = [], []
            with np.load(folder/'logits.npz') as logits:
                if len(logits.files) != len(rows):
                    raise ValueError('Logit/trial accounting mismatch')
                for i in range(len(rows)):
                    x = logits['trial_%04d' % i]
                    if x.ndim != 2 or x.shape[1] != 41 or not np.isfinite(x).all():
                        raise ValueError('Expected finite T x 41 phoneme logits')
                    # Native: 39 CMU phones, SIL, blank. Kaldi: blank, SIL, phones.
                    x = np.ascontiguousarray(x[:, [40, 39]+list(range(39))], dtype=np.float32)
                    start = time.perf_counter()
                    transcripts.append(lmDecoderUtils.lm_decode(decoder, x, blankPenalty=np.log(2)).strip())
                    milliseconds.append((time.perf_counter()-start)*1000)
            print('WFST_JSON '+json.dumps({'transcripts': transcripts, 'decode_ms': milliseconds}), flush=True)
        except Exception as error:
            print('WFST_JSON '+json.dumps({'error': str(error)}), flush=True)


if __name__ == '__main__':
    main()
