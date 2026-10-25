"""Avoid SetOpt lifetime hazards; two fixed searchers share one graph resource."""


def fixed_decoder_pair(binding, resource):
    return {
        n: binding.BrainSpeechDecoder(
            resource, binding.DecodeOptions(7000, 200, 18., 8., .8, 1., 0., n))
        for n in (1, 50)
    }
