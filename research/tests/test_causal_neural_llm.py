import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reproduction'))
from causal_neural_llm import CausalTextEncoder, causal_check


def test_encoder_is_prefix_invariant():
    torch.manual_seed(17)
    encoder = CausalTextEncoder(48)
    assert causal_check(encoder, 'cpu') < 1e-5
    assert encoder(torch.randn(2, 99, 256)).shape == (2, 4, 48)


def test_all_encoder_parameters_receive_finite_gradients():
    encoder = CausalTextEncoder(48)
    encoder(torch.randn(2, 60, 256)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in encoder.parameters())
