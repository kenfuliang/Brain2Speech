"""Tiny native lifecycle regression probe; not a language-model WER run.

Use a compiled three-frame acyclic fixture, not the 28 GB word graph, to
exercise the same option-update/decode/reset seam. Strings are not meaningful.
"""
import argparse
import json
import numpy as np
from neuralDecoder.utils import lmDecoderUtils as lm
from nbest_fixed_decoder import fixed_decoder_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['temporary', 'retained', 'fixed1', 'fixed50', 'fresh', 'pair'], default='temporary')
    parser.add_argument('--graph', default='/tmp/nbest-acyclic.fst')
    args = parser.parse_args()
    resource = lm.lm_decoder.DecodeResource(
        args.graph, '', '', '/data/languageModel/words.txt', '')
    initial_n = 1 if args.mode == 'fixed1' else 50
    options = lm.lm_decoder.DecodeOptions(7000, 200, 18., 8., .8, 1., 0., initial_n)
    decoder = lm.lm_decoder.BrainSpeechDecoder(resource, options)
    retained = [options] if args.mode == 'retained' else []
    # Match build_lm_decoder: only the native decoder owns its initial options.
    del options
    pair = fixed_decoder_pair(lm.lm_decoder, resource) if args.mode == 'pair' else None
    # Pinned synthetic three-frame neural input: blank, phone, blank.
    logits = np.full((3, 41), -10., dtype=np.float32)
    logits[0, 0] = 10.; logits[1, 2] = 3.; logits[1, 3] = 2.; logits[2, 0] = 10.
    sequence = (initial_n, initial_n) if args.mode.startswith('fixed') else (1, 50, 1, 50)
    first_text = None
    for n in sequence:
        if args.mode == 'pair':
            decoder = pair[n]
        elif args.mode == 'fresh':
            options = lm.lm_decoder.DecodeOptions(7000, 200, 18., 8., .8, 1., 0., n)
            decoder = lm.lm_decoder.BrainSpeechDecoder(resource, options)
        elif not args.mode.startswith('fixed'):
            opt = lm.lm_decoder.DecodeOptions(7000, 200, 18., 8., .8, 1., 0., n)
            if args.mode == 'retained':
                retained.append(opt)
            decoder.SetOpt(opt)
        candidates = lm.lm_decode(decoder, logits, returnNBest=True,
                                  rescore=False, blankPenalty=np.log(2))
        print(json.dumps({'requested': n, 'count': len(candidates),
                          'candidates': candidates}), flush=True)
        if not 1 <= len(candidates) <= n:
            raise AssertionError('Native candidate count exceeds requested N')
        if first_text is None:
            first_text = candidates[0][0]
        elif first_text != candidates[0][0]:
            raise AssertionError('Native first candidate changed across modes/repeats')


if __name__ == '__main__':
    main()
