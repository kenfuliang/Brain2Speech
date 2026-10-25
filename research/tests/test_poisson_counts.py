import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[2]/'reproduction/analyze_poisson_counts.py'
spec = importlib.util.spec_from_file_location('poisson_counts', path)
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def test_exact_moments_and_window_accounting():
    x = np.array([[0, 1], [2, 1], [1, 1], [1, 1], [9, 1]])
    v = p.local_stats(x, 2)
    assert np.allclose(v[:, 0], [2., 0.])
    assert np.allclose(v[:, 1], [2., 2.])
    assert np.allclose(v[:, 2], [4., 0.])
    assert np.allclose(v[:, 4:6], [[2, 2], [2, 2]])
    assert np.array_equal(p.histogram(x).sum(1), [5, 5])


def test_poisson_control_and_rate_mixture_distinction():
    controls = p.synthetic_controls()
    assert controls['changing_rate_poisson']['pooled_fano'] > 1.45
    assert abs(controls['changing_rate_poisson']['local_200ms_dispersion']-1) < .03


def test_histogram_preserves_tail_and_zero_rates():
    h = p.histogram(np.array([[0, 0], [100, 0]]))
    assert h[0, -1] == 1 and h[0, 0] == 1 and h[1, 0] == 2
    assert np.allclose(p.expected_hist(np.array([0., 2.]), 100).sum(1), 100)
    assert np.isnan(p.ratio(0., 0.))


def test_zero_counts_not_dropped_from_transform_variance():
    s = p.local_stats(np.zeros((25, 2), dtype=int), 10)
    assert np.array_equal(s[:, 4], [18, 18])
    assert np.array_equal(s[:, 5], [2, 2])
    assert np.array_equal(s[:, 6], [0, 0])
    assert np.allclose(s[:, 3], 0)
