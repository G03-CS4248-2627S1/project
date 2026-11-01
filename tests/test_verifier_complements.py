"""Hand-computed losses/gradients and independent DeBERTa preflight for verifier ablations."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from squadqa import verifier as V
from squadqa import train_verifier as TV
from squadqa.config import load_config


def test_gold_group_loss_accepts_both_variants_with_correct_gradients():
    scores = torch.tensor([0., math.log(2), math.log(3)], dtype=torch.float64, requires_grad=True)
    loss = V.training_loss(scores, torch.zeros(3, dtype=torch.long), torch.tensor([1., 0., 0.]), 1,
                           {"loss": "gold_group"}, torch.tensor([True, True, False]))
    assert loss.item() == pytest.approx(math.log(2))       # correct mass = (1+2)/(1+2+3)
    grad = torch.autograd.grad(loss.sum(), scores)[0]
    torch.testing.assert_close(grad, torch.tensor([-1/6, -1/3, 1/2], dtype=torch.float64))


def test_gold_group_handles_variable_list_lengths_and_all_correct():
    scores = torch.tensor([.2, -.3, .5, .4, 1.], requires_grad=True)
    q = torch.tensor([0, 0, 1, 1, 1])
    correct = torch.tensor([True, False, True, True, True])
    loss = V.training_loss(scores, q, torch.tensor([1., 0., 1., 0., 0.]), 2,
                           {"loss": "gold_group"}, correct)
    assert loss[0].item() == pytest.approx(math.log1p(math.exp(-.5)), abs=1e-6)
    assert loss[1].item() == pytest.approx(0, abs=1e-6)
    assert torch.isfinite(torch.autograd.grad(loss.sum(), scores)[0]).all()


def test_margin_uses_highest_scoring_wrong_answer_not_correct_twin():
    scores = torch.tensor([2., 2.9, 2.5], dtype=torch.float64, requires_grad=True)
    q, target = torch.zeros(3, dtype=torch.long), torch.tensor([1., 0., 0.], dtype=torch.float64)
    correct = torch.tensor([True, True, False])
    settings = {"loss": "ce_margin", "ranking_weight": .1, "ranking_margin": 1.}
    loss = V.training_loss(scores, q, target, 1, settings, correct)
    ce = torch.logsumexp(scores, 0) - scores[0]
    torch.testing.assert_close(loss[0], .9 * ce + .1 * 1.5)
    want = .9 * (scores.softmax(0) - target) + torch.tensor([-.1, 0., .1], dtype=torch.float64)
    torch.testing.assert_close(torch.autograd.grad(loss.sum(), scores)[0], want)


def test_margin_is_zero_without_eligible_negatives():
    scores = torch.tensor([.2, .4, .5], requires_grad=True)
    q, target = torch.tensor([0, 0, 1]), torch.tensor([1., 0., 1.])
    loss = V.training_loss(scores, q, target, 2,
                           {"loss": "ce_margin", "ranking_weight": .1, "ranking_margin": 1.},
                           torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(loss, .9 * V.listwise_loss(scores, q, target, 2))
    assert torch.isfinite(torch.autograd.grad(loss.sum(), scores)[0]).all()


def test_margin_is_zero_when_correct_score_exceeds_margin():
    scores = torch.tensor([3., 0.], requires_grad=True)
    q, target = torch.zeros(2, dtype=torch.long), torch.tensor([1., 0.])
    loss = V.training_loss(scores, q, target, 1,
                           {"loss": "ce_margin", "ranking_weight": .1, "ranking_margin": 1.},
                           torch.tensor([True, False]))
    torch.testing.assert_close(loss, .9 * V.listwise_loss(scores, q, target, 1))


def test_old_configuration_keeps_exact_original_ce():
    scores = torch.tensor([.3, .7, -.2], requires_grad=True)
    q, target = torch.zeros(3, dtype=torch.long), torch.tensor([1., 0., 0.])
    torch.testing.assert_close(V.training_loss(scores, q, target, 1, {}), V.listwise_loss(scores, q, target, 1))


@pytest.mark.parametrize("objective", ["gold_group", "ce_margin"])
def test_new_losses_refuse_questions_without_correct_candidates(objective):
    with pytest.raises(ValueError, match="no EM-1"):
        V.training_loss(torch.tensor([0., 1.]), torch.tensor([0, 1]), torch.ones(2), 2,
                        {"loss": objective}, torch.tensor([True, False]))


@pytest.mark.parametrize("entry", [
    {"spans": [[0, 0]], "gold": 0},
    {"spans": [[0, 0]], "gold": 0, "em": [None]},
    {"spans": [[0, 0]], "gold": 0, "em": [0]},
    {"spans": [[0, 0]], "gold": None, "em": [1]},
])
def test_training_mask_rejects_invalid_labels(entry):
    with pytest.raises(ValueError):
        TV.question_correct(entry)


def test_training_mask_preserves_normalized_em_twins():
    assert TV.question_correct({"spans": [[0, 1], [1, 1], [3, 3]], "gold": 0,
                                "em": [1., 1., 0.]}).tolist() == [True, True, False]


@pytest.mark.parametrize("override", ["verifier.loss=unknown", "verifier.ranking_weight=0.1",
    "verifier.loss=ce_margin", "verifier.ranking_weight=-0.1", "verifier.ranking_margin=-1",
    "verifier.ranking_weight=1.1"])
def test_invalid_loss_settings_raise(override):
    with pytest.raises(ValueError):
        V.check_loss_config(load_config("configs/experiments/verifier-separate.yaml", [override]))


@pytest.mark.parametrize("override", ["f1_target_weight=0.1", "verifier.list_unit=group"])
def test_new_objectives_do_not_silently_change_targets_or_lists(override):
    with pytest.raises(ValueError):
        V.check_loss_config(load_config("configs/experiments/verifier-gold-group.yaml", [override]))


@pytest.mark.parametrize("name", ["verifier-separate", "verifier-gold-group", "verifier-ce-margin",
                                 "verifier-deberta-independent"])
def test_experiment_configs_resolve(name):
    cfg = load_config(f"configs/experiments/{name}.yaml")
    V.check_loss_config(cfg)
    assert cfg["verifier"]["list_unit"] == "span" and cfg["f1_target_weight"] == 0
    assert cfg["verifier"]["train"]["lr"] == 3e-5


def test_separate_deberta_loader_requests_pinned_fp32_eager(monkeypatch):
    from squadqa import train_reader as T
    got = {}
    monkeypatch.setattr(T, "load_encoder", lambda name, **kw: got.update(name=name, **kw))
    V.load_separate_encoder("microsoft/deberta-v3-large")
    assert got == {"name": "microsoft/deberta-v3-large", "attn_implementation": "eager"}


def test_tiny_independent_deberta_scores_markers_and_trains_all_weights():
    transformers = pytest.importorskip("transformers")
    cfg = load_config("configs/experiments/verifier-deberta-independent.yaml")
    cfg["head_dim"], cfg["train"]["head_dropout"] = 8, 0.
    encoder = transformers.AutoModel.from_config(transformers.DebertaV2Config(
        vocab_size=32, hidden_size=12, num_hidden_layers=1, num_attention_heads=3,
        intermediate_size=24, max_position_embeddings=64, relative_attention=True,
        position_buckets=16, pos_att_type=["p2c", "c2p"], type_vocab_size=0),
        attn_implementation="eager")
    encoder.gradient_checkpointing_enable()
    model = V.SeparateVerifier(encoder, 20, 21, cfg).train()
    batch = V.collate_marked([V.Encoded(np.array([1, 4, 20, 5, 21, 2]), 2, 4),
                              V.Encoded(np.array([1, 4, 20, 6, 7, 21, 2]), 2, 5)], 0)
    scores = model(batch)
    loss = V.listwise_loss(scores, torch.zeros(2, dtype=torch.long), torch.tensor([1., 0.]), 1)
    loss.sum().backward()
    assert all(p.requires_grad for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None for p in model.encoder.parameters())
