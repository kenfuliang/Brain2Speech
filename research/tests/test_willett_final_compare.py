import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reproduction'))
from compare_willett_final import align_released


def fixture():
    rows = [dict(day=day, trial_index=trial, session=f'day{day}',
                 trial_id=f'day{day}/test/{trial:04d}', text=f'ref{day}-{trial}')
            for day in range(4, 19) for trial in range(40)]
    released = dict(true_transcripts=[r['text'] for r in rows], decoded_transcripts=['word']*600)
    return rows, released


def test_reference_alignment():
    rows, released = fixture()
    assert align_released(rows, released)[0]['trial_id'] == 'day4/test/0000'


def test_order_rejected():
    rows, released = fixture()
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(AssertionError):
        align_released(rows, released)


def test_reference_mismatch_rejected():
    rows, released = fixture()
    released['true_transcripts'][0] = 'wrong'
    with pytest.raises(AssertionError):
        align_released(rows, released)
