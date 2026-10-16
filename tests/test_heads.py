"""The span heads of milestone 1b (spec §4.1-4.5): the joint, compound and conditional objectives
of Fajcik et al. (2021), the length bias and position-encoding options, and the no-answer option."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from squadqa import data as D
from squadqa import evaluate as E
from squadqa import heads as H
from squadqa import predict as P
from squadqa import train_reader as T
from squadqa.config import load_config

INF = float("-inf")


def inputs(W=2, n=6, d=8, seed=0):
    """Random encoder states for W windows of n words; window 1 has only 4 words, and window w
    starts at passage word 3w."""
    g = torch.Generator().manual_seed(seed)
    x_first, x_last = torch.randn(W, n, d, generator=g), torch.randn(W, n, d, generator=g)
    x_cls = torch.randn(W, d, generator=g)
    mask = torch.ones(W, n, dtype=torch.bool)
    if W > 1:
        mask[1, 4:] = False
    positions = torch.arange(n)[None, :] + 3 * torch.arange(W)[:, None]
    return x_first, x_last, x_cls, mask, positions


def valid(mask, w, i, j, max_len=3):
    return bool(mask[w, i] and mask[w, j]) and 0 <= j - i < max_len


# The joint head (spec §4.1).

PAIR_TERMS = [("full", "none"), ("full", "additive"), ("full", "rotary"), ("dot", "none"), ("dot", "additive"),
              ("dot", "rotary")]


@pytest.mark.parametrize("form, pos", PAIR_TERMS)
@pytest.mark.parametrize("length_bias", [False, True])
def test_joint_head_starts_as_the_standard_head(length_bias, form, pos):
    """Spec §4.5 equivalence test: with the pair term and b at zero, the same scores and argmax as
    the standard head, for every pair term and position option; the null too."""
    torch.manual_seed(0)
    std = H.StandardSpanHead(8, 4, 3, 0.0)
    joint = H.JointSpanHead(8, 4, 3, 0.0, length_bias=length_bias, pos=pos, form=form, rank=4)
    missing = joint.load_state_dict(std.state_dict(), strict=False).missing_keys     # same MLPs, u and v
    pair = {"W"} if form == "full" else {"A.weight", "B.weight"}
    assert set(missing) == pair | ({"len_bias"} if length_bias else set())
    x_first, x_last, x_cls, mask, pos = inputs()
    with torch.no_grad():
        std_out = std(x_first, x_last, mask, x_cls)
        s_std = std.span_scores(std_out, mask)
        out = joint(x_first, x_last, mask, x_cls, pos)
    assert torch.equal(torch.isinf(out["span"]), torch.isinf(s_std))
    assert torch.allclose(out["span"][torch.isfinite(s_std)], s_std[torch.isfinite(s_std)], atol=1e-6)
    assert out["span"].flatten(1).argmax(1).tolist() == s_std.flatten(1).argmax(1).tolist()
    assert torch.allclose(joint.null_score(out), std.null_score(std_out), atol=1e-6)


def test_joint_score_formula_with_position_encodings_and_length_bias():
    torch.manual_seed(1)
    head = H.JointSpanHead(8, 4, 3, 0.0, length_bias=True, pos="additive")
    with torch.no_grad():
        head.W.normal_()
        head.len_bias.copy_(torch.tensor([0.5, -1.0, 2.0]))
        x_first, x_last, x_cls, mask, pos = inputs()
        out = head(x_first, x_last, mask, x_cls, pos)
        hs, he, pe = head.start(x_first), head.end(x_last), H.sinusoidal(pos, 4)
        for w in range(2):
            for i in range(6):
                for j in range(6):
                    if not valid(mask, w, i, j):
                        assert out["span"][w, i, j] == INF
                        continue
                    expected = ((hs[w, i] + pe[w, i]) @ head.W @ (he[w, j] + pe[w, j])    # positions: bilinear only
                                + head.u(hs[w, i]) + head.v(he[w, j]) + head.len_bias[j - i])
                    assert float(out["span"][w, i, j]) == pytest.approx(float(expected), abs=1e-5)
        cs, ce = head.start(x_cls), head.end(x_cls)                                       # (CLS, CLS), no extras
        null = ((cs @ head.W) * ce).sum(-1) + head.u(cs).squeeze(-1) + head.v(ce).squeeze(-1)
    assert torch.allclose(out["null"], null, atol=1e-5)


def test_position_encodings_change_scores_once_W_is_learned():
    torch.manual_seed(2)
    plain = H.JointSpanHead(8, 4, 3, 0.0, pos="none")
    with_pe = H.JointSpanHead(8, 4, 3, 0.0, pos="additive")
    with torch.no_grad():
        plain.W.normal_()
    with_pe.load_state_dict(plain.state_dict())
    x_first, x_last, _, mask, pos = inputs()
    with torch.no_grad():
        a, b = plain(x_first, x_last, mask)["span"], with_pe(x_first, x_last, mask, positions=pos)["span"]
    ok = torch.isfinite(a)
    assert not torch.allclose(a[ok], b[ok])
    with pytest.raises(ValueError, match="positions"):
        with_pe(x_first, x_last, mask)


def test_sinusoidal_products_depend_only_on_the_distance():
    pe = H.sinusoidal(torch.arange(60), 16)
    assert pe.shape == (60, 16) and pe[0].tolist() == [0.0, 1.0] * 8
    for k in (0, 1, 7):
        dots = (pe[:60 - k] * pe[k:]).sum(-1)
        assert torch.allclose(dots, dots[0].expand_as(dots), atol=1e-4)


def test_rotary_products_depend_only_on_the_distance():
    g = torch.Generator().manual_seed(6)
    q, k = torch.randn(8, generator=g), torch.randn(8, generator=g)
    for i, j in [(0, 5), (3, 8), (40, 45), (7, 2)]:
        a = (H.rotary(q[None], torch.tensor([i])) * H.rotary(k[None], torch.tensor([j]))).sum()
        b = (q * H.rotary(k[None], torch.tensor([j - i]))[0]).sum()
        assert float(a) == pytest.approx(float(b), abs=1e-5)


def test_dot_rotary_scores_ignore_where_the_window_starts():
    """GlobalPointer's property: the rotary pair term depends on j - i, so moving every word by the
    same offset changes nothing; additive encodings also carry the absolute position."""
    torch.manual_seed(7)
    rot = H.JointSpanHead(8, 4, 3, 0.0, pos="rotary", form="dot", rank=4)
    add = H.JointSpanHead(8, 4, 3, 0.0, pos="additive", form="dot", rank=4)
    with torch.no_grad():
        rot.B.weight.normal_()
        add.load_state_dict(rot.state_dict())
        x_first, x_last, _, mask, pos = inputs()
        a, b = rot(x_first, x_last, mask, positions=pos)["span"], rot(x_first, x_last, mask, positions=pos + 17)["span"]
        ok = torch.isfinite(a)
        assert torch.allclose(a[ok], b[ok], atol=1e-4)
        c, d = add(x_first, x_last, mask, positions=pos)["span"], add(x_first, x_last, mask, positions=pos + 17)["span"]
        assert not torch.allclose(c[ok], d[ok], atol=1e-4)
        hs, he = rot.start(x_first), rot.end(x_last)
        q, k = H.rotary(rot.A(hs), pos), H.rotary(rot.B(he), pos)
        expected = q @ k.transpose(1, 2) + rot.u(hs) + rot.v(he).transpose(1, 2)        # b is still zero
    assert torch.allclose(a[ok], expected[ok], atol=1e-5)


def test_full_rotary_rotates_h_before_W_and_also_sees_absolute_position():
    """With the full W, (R_i h_i)' W (R_j h_j): a general W doesn't commute with the rotations, so
    unlike the dot form the scores change when every word moves by the same offset."""
    torch.manual_seed(8)
    head = H.JointSpanHead(8, 4, 3, 0.0, pos="rotary", form="full")
    with torch.no_grad():
        head.W.normal_()
        x_first, x_last, _, mask, pos = inputs()
        a, b = head(x_first, x_last, mask, positions=pos)["span"], head(x_first, x_last, mask, positions=pos + 17)["span"]
        hs, he = head.start(x_first), head.end(x_last)
        expected = (H.rotary(hs, pos) @ head.W) @ H.rotary(he, pos).transpose(1, 2) + head.u(hs) + head.v(he).transpose(1, 2)
    ok = torch.isfinite(a)
    assert torch.allclose(a[ok], expected[ok], atol=1e-5)
    assert not torch.allclose(a[ok], b[ok], atol=1e-4)


# Losses (spec §4.3, §3.4).

def test_joint_nll_is_one_softmax_over_every_window_span_and_null():
    span = torch.full((2, 3, 3), INF)
    span[0, 0, 0], span[0, 0, 1], span[0, 1, 1] = 1.0, 2.0, 0.5         # window 0: gold (0, 1)
    span[1, 0, 0], span[1, 0, 1] = 3.0, -1.0                             # window 1: the same span is its (0, 0)
    batch = {"gold_start": torch.tensor([0, 0]), "gold_end": torch.tensor([1, 0]), "example": torch.tensor([0, 0])}
    gold = torch.logsumexp(torch.tensor([2.0, 3.0]), 0)
    spans = torch.tensor([1.0, 2.0, 0.5, 3.0, -1.0])
    nll = H.joint_nll({"span": span, "null": torch.tensor([0.2, -0.4])}, batch, 1)
    assert float(nll) == pytest.approx(float(torch.logsumexp(torch.cat([spans, torch.tensor([0.2, -0.4])]), 0) - gold))
    assert float(H.joint_nll({"span": span}, batch, 1)) == pytest.approx(float(torch.logsumexp(spans, 0) - gold))


def test_independent_nll_puts_the_null_in_both_softmaxes():
    out = {"start": torch.tensor([[1.0, 0.0, INF]]), "end": torch.tensor([[0.5, 2.0, INF]]),
           "start_null": torch.tensor([0.3]), "end_null": torch.tensor([-1.0])}
    batch = {"gold_start": torch.tensor([0]), "gold_end": torch.tensor([1]), "example": torch.tensor([0])}
    expected = (-torch.log_softmax(torch.tensor([1.0, 0.0, 0.3]), 0)[0]
                - torch.log_softmax(torch.tensor([0.5, 2.0, -1.0]), 0)[1])
    assert float(H.independent_nll(out, batch, 1)) == pytest.approx(float(expected))


# The conditional head.

def test_conditional_scores_are_start_plus_log_p_of_the_end_given_the_start():
    torch.manual_seed(3)
    head = H.ConditionalSpanHead(8, 4, 3, 0.0)
    x_first, x_last, x_cls, mask, pos = inputs()
    with torch.no_grad():
        out = head(x_first, x_last, mask, x_cls, pos)
        s = head.span_scores(out, mask)
        for w in range(2):
            for i in range(6):
                ends = [j for j in range(6) if valid(mask, w, i, j)]
                assert all(s[w, i, j] == INF for j in range(6) if j not in ends)
                if ends:                                                     # Fajcik et al.'s w.tanh(A h_j + B h_i + c)
                    logits = torch.stack([head.w(torch.tanh(head.end_proj(x_last[w, j]) + head.start_proj(x_first[w, i])))[0]
                                          for j in ends])
                    assert torch.allclose(s[w, i, ends], out["start"][w, i] + logits.log_softmax(0), atol=1e-5)
        p = torch.cat([s[0].flatten(), head.null_score(out)[:1]]).softmax(0)          # spans and null of window 0
        p_start = torch.cat([out["start"][0], out["start_null"][:1]]).softmax(0)
    assert torch.allclose(p[:-1].view(6, 6).sum(1), p_start[:-1], atol=1e-6)        # p(i, j) = p(i) p(j | i)
    assert float(p[-1]) == pytest.approx(float(p_start[-1]), abs=1e-6)


def test_conditional_loss_is_the_nll_of_the_decoded_distribution():
    torch.manual_seed(4)
    head = H.ConditionalSpanHead(8, 4, 3, 0.0)
    x_first, x_last, x_cls, mask, pos = inputs(W=1)
    batch = {"gold_start": torch.tensor([2]), "gold_end": torch.tensor([3]), "example": torch.tensor([0]),
             "word_mask": mask}
    with torch.no_grad():
        out = head(x_first, x_last, mask, x_cls, pos)
        loss = head.loss(out, batch, 1)
        expected = -torch.cat([head.span_scores(out, mask)[0].flatten(), head.null_score(out)]).log_softmax(0)[2 * 6 + 3]
    assert float(loss) == pytest.approx(float(expected), abs=1e-5)


def test_conditional_loss_uses_only_windows_holding_the_answer():
    torch.manual_seed(5)
    head = H.ConditionalSpanHead(8, 4, 3, 0.0)
    x_first, x_last, x_cls, mask, pos = inputs()
    batch = {"gold_start": torch.tensor([-1, 1]), "gold_end": torch.tensor([-1, 2]), "example": torch.tensor([0, 0]),
             "word_mask": mask}
    loss = head.loss(head(x_first, x_last, mask, x_cls, pos), batch, 1)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)


# Config.

@pytest.mark.parametrize("overrides, error", [
    (["span_head=pointer"], ValueError),
    (["span_head=standard", "bilinear_pos=additive"], ValueError),
    (["span_head=conditional", "bilinear_pos=additive"], ValueError),
    (["span_head=standard", "bilinear_form=dot"], ValueError),
    (["bilinear_form=cubic"], ValueError),
    (["bilinear_pos=learned"], ValueError),
    (["f1_target_weight=0.3"], NotImplementedError),
    (["risk_weight=0.1"], NotImplementedError),
    (["risk_weight=0.1", "risk_em_weight=1.0"], NotImplementedError),
    (["risk_em_weight=1.0"], ValueError),                       # an EM weight for a risk loss that is off
    (["risk_weight=0.1", "risk_em_weight=-1.0"], ValueError),
    (["target_unit=answer"], ValueError),
    (["span_head=joint", "target_unit=group"], NotImplementedError),
    (["span_head=compound", "target_unit=group", "f1_target_weight=0.3"], NotImplementedError),
    (["span_head=joint", "target_unit=group", "f1_target=positional"], ValueError),   # no variants to pool
    (["span_head=standard", "target_unit=group"], ValueError),                        # no span softmax
    (["W_init=random"], NotImplementedError),
])
def test_settings_that_are_not_built_or_do_not_apply_raise(overrides, error):
    with pytest.raises(error):
        H.check_head_config(load_config(None, overrides))


def test_old_run_configs_get_the_settings_they_were_trained_with():
    old = {k: v for k, v in load_config(None, ["span_head=standard"]).items() if k not in P.ADDED_KEYS}
    cfg = {**P.ADDED_KEYS, **old}
    H.check_head_config(cfg)
    assert cfg["no_answer"] is False and cfg["f1_target_weight"] == 0.0


# Each objective end to end on a tiny encoder.

HEAD_CONFIGS = {
    "standard": ["span_head=standard"],
    "conditional": ["span_head=conditional"],
    "joint": ["span_head=joint", "length_bias=false", "bilinear_pos=additive"],
    "compound": ["span_head=compound", "bilinear_pos=additive"],
    "compound-rotary": ["span_head=compound", "bilinear_pos=rotary"],
}


def tiny_encoder(seed=0):
    torch.manual_seed(seed)
    config = transformers.DebertaV2Config(
        vocab_size=128100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=512, relative_attention=True, position_buckets=256, pos_att_type=["p2c", "c2p"],
        type_vocab_size=0, position_biased_input=False, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    return transformers.AutoModel.from_config(config)


@pytest.fixture(scope="module")
def tok():
    try:
        return D.load_tokenizer(load_config()["model"])
    except (OSError, RuntimeError) as e:
        pytest.skip(f"tokenizer not available: {e}")


@pytest.fixture(scope="module")
def small_train(tok):
    return D.featurize(D.load_split("train")[:24], tok, load_config(), training=True)


@pytest.mark.parametrize("head", list(HEAD_CONFIGS))
def test_every_objective_trains_with_the_no_answer_option(head, small_train, tok, tmp_path):
    cfg = load_config(None, HEAD_CONFIGS[head] + [
        "no_answer=true", "train.head_dropout=0.0", "train.lr=1e-3", "train.effective_batch=4", "train.epochs=10",
        "train.warmup_frac=0.0"])
    model = H.Reader(tiny_encoder(), cfg)
    out = T.train(model, small_train, cfg, 0, tok.pad_token_id, tmp_path / "ck.pt", per_device=2, device="cpu",
                  log=lambda m: None)
    assert np.isfinite(out["losses"]).all()
    assert np.mean(out["losses"][-6:]) < 0.8 * np.mean(out["losses"][:6])


@pytest.mark.parametrize("head", list(HEAD_CONFIGS))
def test_predict_with_the_no_answer_option(head, tok, tmp_path):
    val = D.load_split("validation")
    questions = [q for q in val if q.title == val[0].title]
    dataset = [a for a in E.load_eval_set("validation") if a["title"] == val[0].title]
    cfg = load_config(None, HEAD_CONFIGS[head] + ["no_answer=true"])
    out = P.predict_split(H.Reader(tiny_encoder(), cfg), tok, cfg, tmp_path, "cpu", "validation", questions, dataset,
                          temperature=1.0)
    for rule in cfg["decode_rules"]:
        preds = E.load_predictions(tmp_path / f"predictions_validation_{rule}.json")
        assert set(preds) == {q.id for q in questions} and all(a.strip() for a in preds.values())   # never "no answer"
    summary = out["no_answer_validation"]
    assert summary["questions"] == len(questions) and 0 < summary["mean_p"] < 1


def test_calibration_keeps_the_null_in_the_softmax():
    from types import SimpleNamespace

    from squadqa.decode import SpanScores, log_softmax, temperature_nll
    rng = np.random.default_rng(0)
    spans, examples = [], []
    for _ in range(200):
        s = SpanScores(np.arange(6), np.arange(6), (3 * rng.normal(size=6)).astype(np.float32), null=float(rng.normal()))
        g = int(rng.choice(6, p=np.exp(log_softmax(s.scores, 1.5))))      # gold drawn at T = 1.5
        spans.append(s)
        examples.append(SimpleNamespace(gold=(g, g)))
    cal = P.calibrate(SimpleNamespace(examples=examples), spans)
    items = [(s.with_null(), ex.gold[0]) for ex, s in zip(examples, spans)]
    assert cal["nll_at_1"] == pytest.approx(temperature_nll(items, 1.0))
    assert cal["nll"] == pytest.approx(temperature_nll(items, cal["temperature"]))


def test_no_null_without_the_option(tok):
    cfg = load_config(None, ["span_head=compound"])
    rows = D.featurize(D.load_split("validation")[:3], tok, cfg, training=False)
    batch = D.collate(rows.batch([0, 1, 2]), tok.pad_token_id)
    model = H.Reader(tiny_encoder(), cfg).eval()
    with torch.no_grad():
        spans, null = model.span_scores(batch)
    assert null is None and spans.shape[0] == len(batch["input_ids"])
