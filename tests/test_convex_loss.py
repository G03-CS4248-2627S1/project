"""Check the actual joint Reader objective and gradients for the convex-weight trial."""
from types import SimpleNamespace

import pytest
import torch

from squadqa import heads as H
from squadqa.config import load_config


@pytest.mark.parametrize("normalized", [False, True])
def test_reader_coefficients_and_gradients(normalized):
    cfg = load_config("configs/experiments/row1-convex.yaml")
    cfg["normalize_loss_weights"] = normalized
    encoder = torch.nn.Identity()
    encoder.config = SimpleNamespace(hidden_size=4)
    model = H.Reader(encoder, cfg)
    span = torch.tensor([[[0.1, 0.4], [float("-inf"), 0.3]]], requires_grad=True)
    out = {"span": span}
    model.forward = lambda batch: out
    batch = {"content": torch.tensor([[True, True]]), "core_start": torch.tensor([0]),
             "core_end": torch.tensor([1]), "gold_start": torch.tensor([0]),
             "gold_end": torch.tensor([1]), "example": torch.tensor([0]),
             "first_word": torch.tensor([0]), "word_mask": torch.ones(1, 2, dtype=torch.bool)}
    group = H.group_nll(out, batch, 1)
    soft, _ = H.f1_target_nll(out, batch, 1, 0.3, 0.0, "group", batch["content"])
    risk = H.risk_loss(out, batch, 1, 1.0, batch["content"])
    expected = 0.5 * group + 0.25 * soft + 0.25 * risk if normalized else (2 * group + soft + risk) / 3
    actual = model.loss(batch, 1)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), span, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), span)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_normalization_rejects_heads_with_other_objectives():
    cfg = load_config("configs/experiments/row1-convex.yaml")
    cfg["span_head"] = "compound"
    with pytest.raises(ValueError, match="joint head"):
        H.check_head_config(cfg)
