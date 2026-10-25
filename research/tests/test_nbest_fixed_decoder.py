import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'reproduction'))
from nbest_fixed_decoder import fixed_decoder_pair


def test_fixed_pair_shares_resource_and_never_mutates_options():
    class Binding:
        @staticmethod
        def DecodeOptions(*args):
            return args

        class BrainSpeechDecoder:
            def __init__(self, resource, options):
                self.resource = resource
                self.options = options

            def SetOpt(self, options):
                raise AssertionError('Forbidden unsafe option mutation')

    resource = object()
    decoders = fixed_decoder_pair(Binding, resource)
    assert list(decoders) == [1, 50]
    assert all(d.resource is resource for d in decoders.values())
    assert decoders[1] is not decoders[50]
    for n, decoder in decoders.items():
        assert decoder.options == (7000, 200, 18., 8., .8, 1., 0., n)
