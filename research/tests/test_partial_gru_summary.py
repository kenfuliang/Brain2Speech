import importlib.util
from pathlib import Path
from unittest.mock import patch
import pytest
path=Path(__file__).resolve().parents[2]/'reproduction/summarize_partial_gru_v1.py'
spec=importlib.util.spec_from_file_location('partial_gru',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_frozen_condition_coverage_excludes_other_models_and_seeds():
    paths=[(f'final/gru/17/day{d}/condition{c}',[str(d)]) for c in range(32) for d in range(5)]
    paths.append(('final/dynamics/17/day0/zero-shot',['x']))
    paths.append(('final/gru/29/day0/zero-shot',['x']))
    with patch.object(module,'expected_outputs',return_value=paths):
        assert len(module.group_paths({},{}))==32


def test_incomplete_control_fails_instead_of_selecting_available_days():
    with patch.object(module,'expected_outputs',return_value=[('final/gru/17/day0/zero-shot',['x'])]):
        with pytest.raises(ValueError):module.group_paths({},{})
