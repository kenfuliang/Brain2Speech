import pytest
from research.reporting.nbest_uncertainty import paired_oracle_interval


def rows(equal=False):
    return [dict(id=str(i),session='day'+str(i),top1=dict(errors=1,reference_units=2),
                 oracle=dict(errors=1 if equal else 0,reference_units=2)) for i in range(3)]


def test_pairs_preserve_zero_gap_and_reproducibility():
    result=paired_oracle_interval(rows(True),repetitions=100)
    assert result==paired_oracle_interval(rows(True),repetitions=100)
    assert result['ci95']['oracle_gap']==[0.,0.]
    assert result['point']['oracle_gap']==0


def test_small_session_smoke_has_no_interval():
    result=paired_oracle_interval(rows()[:2])
    assert result['ci95'] is None and result['diagnostic_only']


def test_nonnegative_oracle_gap():
    result=paired_oracle_interval(rows(),repetitions=100)
    assert result['point']['oracle_gap']==.5
    assert result['ci95']['oracle_gap']==[.5,.5]


def test_fail_closed_denominators_duplicates_and_error_order():
    r=rows();r[0]['oracle']['reference_units']=3
    with pytest.raises(ValueError):paired_oracle_interval(r)
    with pytest.raises(ValueError):paired_oracle_interval([rows()[0],rows()[0]])
    r=rows();r[0]['oracle']['errors']=2
    with pytest.raises(ValueError):paired_oracle_interval(r)
