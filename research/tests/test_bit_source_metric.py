import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reproduction'))
from audit_bit_source_metric import source_prediction


def test_source_defect_preserved():
    assert source_prediction([1, 0, 1]) == [1]


def test_different_phone_and_padding():
    assert source_prediction([1, 1, 0, 2, 2, 0, 0]) == [1, 2]


def test_all_blank():
    assert source_prediction([0, 0]) == []
