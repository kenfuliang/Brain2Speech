"""Separate diagnostic worker: sees phoneme logits, never target transcripts."""
import argparse
import json
import time
import numpy as np
from neuralDecoder.utils import lmDecoderUtils as lm


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--logits',required=True)
    p.add_argument('--indices',required=True)
    a=p.parse_args()
    decoder=lm.build_lm_decoder('/data/languageModel',acoustic_scale=.8,beam=18,nbest=50)
    print('NBEST_JSON '+json.dumps({'ready':True}),flush=True)
    with np.load(a.logits,allow_pickle=False) as archive:
        for index in json.loads(a.indices):
            x=archive['trial_%04d'%index]
            if x.ndim!=2 or x.shape[1]!=41 or not np.isfinite(x).all():
                raise ValueError('Expected finite T x 41 logits')
            x=np.ascontiguousarray(x[:,[40,39]+list(range(39))],dtype=np.float32)
            outputs={}
            for n in (1,50):
                decoder.SetOpt(lm.lm_decoder.DecodeOptions(7000,200,18.,8.,.8,1.,0.,n))
                start=time.perf_counter()
                candidates=lm.lm_decode(decoder,x,returnNBest=True,rescore=False,blankPenalty=np.log(2))
                outputs[str(n)]={'candidates':[dict(text=t.strip(),ac_score=float(ac),lm_score=float(ls))
                                               for t,ac,ls in candidates],
                                 'decode_ms':(time.perf_counter()-start)*1000}
            print('NBEST_JSON '+json.dumps({'index':index,'outputs':outputs},allow_nan=False),flush=True)


if __name__=='__main__': main()
