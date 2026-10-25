import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reproduction'))
from pretrained_causal_bridge import normalized, trailing_pool


def test_normalization_and_filter_prefix():
    rng = np.random.default_rng(17); raw = rng.normal(size=(100, 256))
    mean, variance = np.zeros(256), np.ones(256)
    other = raw.copy(); other[60:] += 20
    np.testing.assert_array_equal(normalized(raw, mean, variance)[:60],
                                  normalized(other, mean, variance)[:60])
    np.testing.assert_array_equal(normalized(raw, mean, variance)[:60],
                                  normalized(raw[:60], mean, variance))


def test_fixed_trailing_pool():
    x = torch.arange(12.).reshape(12, 1)
    assert trailing_pool(x).tolist() == [[2.], [7.]]
    assert torch.equal(trailing_pool(x[:10]), trailing_pool(x))
