"""Compatibility at the legacy-additive/current-share merge boundary."""
from types import SimpleNamespace

import pytest
import torch

from squadqa.config import load_config
from squadqa import heads as H
from squadqa import predict as P


@pytest.mark.parametrize('share,legacy,normalize', [(.25, None, False), (1., None, False),
                                                   (0., .3, False), (0., .3, True)])
def test_merge_preserves_objective_and_gradient_scales(share, legacy, normalize):
    cfg = load_config('configs/experiments/row1-joint.yaml')
    cfg.update(risk_share=share, risk_weight=legacy, normalize_loss_weights=normalize,
               risk_em_weight=1.)
    encoder = torch.nn.Identity()
    encoder.config = SimpleNamespace(hidden_size=4)
    model = H.Reader(encoder, cfg)
    span = torch.tensor([[[.1, .4], [float('-inf'), .3]]], requires_grad=True)
    out = {'span': span}
    model.forward = lambda batch: out
    batch = {'content': torch.tensor([[True, True]]), 'core_start': torch.tensor([0]),
             'core_end': torch.tensor([1]), 'gold_start': torch.tensor([0]),
             'gold_end': torch.tensor([1]), 'example': torch.tensor([0]),
             'first_word': torch.tensor([0]), 'word_mask': torch.ones(1, 2, dtype=torch.bool)}
    ce = H.group_nll(out, batch, 1)
    risk = H.risk_loss(out, batch, 1, 1., batch['content'])
    expected = (1 - share) * ce + share * risk if legacy is None else (ce + legacy * risk)
    if normalize:
        expected = expected / (1 + legacy)
    actual = model.loss(batch, 1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), span, retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(), span)[0])


def test_merge_rejects_ambiguous_old_and_new_risk_parameters():
    cfg = load_config('configs/experiments/row1-joint.yaml')
    cfg.update(risk_share=.25, risk_weight=.3)
    with pytest.raises(ValueError, match='cannot be combined'):
        H.check_head_config(cfg)


@pytest.mark.parametrize('legacy_line', ['risk_weight: null', 'risk_weight: 0.3333333333333333\nnormalize_loss_weights: true'])
def test_new_and_normalized_old_saved_configs_load(tmp_path, monkeypatch, legacy_line):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'config.yaml').write_text('span_head: joint\nrisk_share: 0.0\n' + legacy_line + '\n')
    monkeypatch.setattr(P, 'RESULTS', tmp_path)
    cfg = P.run_config('run')
    assert 'risk_weight' not in cfg
    assert cfg['risk_share'] == pytest.approx(.25 if 'normalize_loss' in legacy_line else 0.)
    assert not cfg['normalize_loss_weights']
