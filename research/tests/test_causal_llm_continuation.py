import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reproduction'))
from continue_causal_neural_llm import StabilityMonitor, better


def test_sustained_deterioration_not_single_jitter():
    monitor = StabilityMonitor(4., patience=3)
    assert monitor.observe(5.1) is None
    assert monitor.observe(4.) is None
    assert monitor.bad == 0
    assert monitor.observe(3.) is None
    assert monitor.observe(4.) is None
    assert monitor.observe(4.) is None
    assert monitor.observe(4.) == 'sustained_validation_deterioration'


def test_nonfinite_stops():
    assert StabilityMonitor(4.).observe(float('nan')) == 'nonfinite_validation_loss'


def test_selection_uses_wer_before_loss():
    record = lambda edits, loss: {'wer': {'errors': edits}, 'loss': {'real': loss}}
    assert better(record(100, 5.), record(110, 2.))
    assert not better(record(110, 2.), record(100, 5.))
    assert better(record(100, 2.), record(100, 3.))
    assert not better(record(100, 2.), record(100, 2.))
