import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reproduction'))
from decode_willett_final import score_text


def test_source_and_standard_empty_hypothesis_separate():
    legacy, _ = score_text('hello world', '', 'legacy-literal-space')
    standard, _ = score_text('hello world', '', 'standard-whitespace')
    assert legacy['errors'] == standard['errors'] == 2
    assert legacy['substitutions'] == 1 and standard['substitutions'] == 0


def test_repeated_spaces_not_silently_normalized():
    legacy, _ = score_text('hello world', 'hello  world', 'legacy-literal-space')
    standard, _ = score_text('hello world', 'hello  world', 'standard-whitespace')
    assert legacy['errors'] == 1 and standard['errors'] == 0


def test_invalid_policy_rejected():
    with pytest.raises(AssertionError):
        score_text('hello', 'hello', 'unknown')
