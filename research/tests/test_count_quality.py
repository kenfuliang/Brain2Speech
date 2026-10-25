import numpy as np
import pytest
from research.reporting.count_quality import moments


def test_count_dispersion_and_undefined_zero_channels():
    counts = np.zeros((4,128))
    counts[:,0] = [0,0,0,4]
    result = moments([counts])
    assert result['bins']==4 and result['trials']==1
    assert result['fano_by_channel'][0]==3.
    assert result['fano_by_channel'][1] is None
    assert result['valid_pooled_channels']==1
    assert result['within_trial_valid_cells']==1
    silent = moments([np.zeros((4,128))])
    assert silent['valid_pooled_channels']==0 and silent['pooled_fano_median'] is None


@pytest.mark.parametrize('value',[-1.,.5,np.nan])
def test_invalid_count_targets_rejected(value):
    with pytest.raises(ValueError,match='nonnegative integers'):
        moments([np.full((4,128),value)])
