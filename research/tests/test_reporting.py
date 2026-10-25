import pytest
from research.reporting.collect import replay_metrics


def test_replay_diagnostics_and_identity_accounting():
    replay = {'true_transcripts': ['a b'], 'decoded_transcripts': ['a c'],
              'measured': {'trials': 1, 'wer': [.5, .5, .5]}}
    result = replay_metrics(replay)
    assert result['metrics']['wer']['errors'] == 1
    assert result['metrics']['cer']['rate'] == 1/3
    assert result['metrics']['wer']['substitutions'] == 1
    replay['measured']['wer'][0] = .2
    with pytest.raises(ValueError, match='disagrees'):
        replay_metrics(replay)
