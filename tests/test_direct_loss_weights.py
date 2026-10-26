"""Direct convex mixtures, including true zero-group-loss and pure-risk endpoints."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from squadqa import heads as H
from squadqa.config import load_config


CASES = [(1, 0, 0), (.75, 0, .25), (.5, 0, .5), (.1, 0, .9), (.01, 0, .99),
         (0, 0, 1), (0, 1, 0), (0, .75, .25), (0, .5, .5), (0, .25, .75)]


@pytest.mark.parametrize("weights", CASES)
def test_direct_mixture_values_gradients_and_skipped_terms(weights, monkeypatch):
    cfg = load_config("configs/experiments/row1-ce-risk.yaml")
    cfg["loss_weights"] = dict(zip(("group", "f1_soft", "risk"), weights))
    cfg["risk_em_weight"] = 1.0 if weights[2] else 0.0
    encoder = torch.nn.Identity()
    encoder.config = SimpleNamespace(hidden_size=4)
    model = H.Reader(encoder, cfg)
    span = torch.tensor([[[.1, .4], [float("-inf"), .3]]], requires_grad=True)
    out = {"span": span}
    model.forward = lambda batch: out
    batch = {"content": torch.tensor([[True, True]]), "core_start": torch.tensor([0]),
             "core_end": torch.tensor([1]), "gold_start": torch.tensor([0]),
             "gold_end": torch.tensor([1]), "example": torch.tensor([0]),
             "first_word": torch.tensor([0]), "word_mask": torch.ones(1, 2, dtype=torch.bool)}
    terms = [H.group_nll(out, batch, 1),
             H.f1_target_nll(out, batch, 1, .3, 0, "group", batch["content"])[0],
             H.risk_loss(out, batch, 1, 1, batch["content"])]
    expected = sum(w * term for w, term in zip(weights, terms))

    def forbidden(*args, **kwargs):
        raise AssertionError("A zero-weight loss was evaluated")

    for weight, name in zip(weights, ("interpolated_group_nll", "f1_target_nll", "risk_loss")):
        if not weight:
            monkeypatch.setattr(H, name, forbidden)
    actual = model.loss(batch, 1)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), span, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), span)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()


@pytest.mark.parametrize("weights", [
    {"group": .5, "f1_soft": .5},
    {"group": -.1, "f1_soft": .5, "risk": .6},
    {"group": .5, "f1_soft": .5, "risk": .1},
    {"group": float("nan"), "f1_soft": 0, "risk": 1},
    {"group": True, "f1_soft": 0, "risk": 0},
])
def test_invalid_direct_weights_raise(weights):
    cfg = load_config("configs/experiments/row1-ce-risk.yaml")
    cfg["loss_weights"] = weights
    with pytest.raises(ValueError, match="loss_weights"):
        H.check_head_config(cfg)


@pytest.mark.parametrize("key,value", [("f1_target_weight", .2), ("risk_weight", .2),
                                      ("normalize_loss_weights", True)])
def test_direct_weights_reject_ambiguous_legacy_settings(key, value):
    cfg = load_config("configs/experiments/row1-ce-risk.yaml")
    cfg[key] = value
    with pytest.raises(ValueError, match="cannot be combined"):
        H.check_head_config(cfg)


def test_prepared_ce_risk_sweep_resolves_without_f1_soft_targets():
    plan = json.loads(Path("sandbox/heshi/ce_risk_trials.json").read_text())
    assert len(plan["trials"]) == 8
    for trial in plan["trials"]:
        args = trial["train_arguments"]
        overrides = [args[i + 1] for i, arg in enumerate(args) if arg == "--set"]
        cfg = load_config(plan["config"], overrides)
        H.check_head_config(cfg)
        assert cfg["loss_weights"] == trial["loss_weights"]
        assert cfg["loss_weights"]["f1_soft"] == cfg["f1_target_weight"] == 0
        assert cfg["risk_weight"] == 0 and not cfg["normalize_loss_weights"]
