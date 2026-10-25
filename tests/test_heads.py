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

PAIR_TERMS = [
    ("full", "none"), ("full", "additive"), ("full", "rotary"),
    ("dot", "none"), ("dot", "additive"), ("dot", "rotary"),
    ("diag", "none"), ("diag", "additive"), ("mlp", "none"), ("mlp", "additive"),
]
PAIR_PARAMS = {"full": {"W"}, "dot": {"A.weight", "B.weight"}, "diag": {"w"},
               "mlp": {"A.weight", "A.bias", "B.weight", "norm.weight", "norm.bias", "w"}}


@pytest.mark.parametrize("form, pos", PAIR_TERMS)
@pytest.mark.parametrize("length_bias", [False, True])
def test_joint_head_starts_as_the_standard_head(length_bias, form, pos):
    """Spec §4.5 equivalence test: with the pair term and b at zero, the same scores and argmax as
    the standard head, for every pair term and position option; the null too."""
    torch.manual_seed(0)
    std = H.StandardSpanHead(8, 4, 3, 0.0)
    joint = H.JointSpanHead(8, 4, 3, 0.0, length_bias=length_bias, pos=pos, form=form, rank=4)
    missing = joint.load_state_dict(std.state_dict(), strict=False).missing_keys     # same MLPs, u and v
    assert set(missing) == PAIR_PARAMS[form] | ({"len_bias"} if length_bias else set())
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


@pytest.mark.parametrize("form", ["diag", "mlp"])
def test_diag_and_mlp_pair_terms_by_hand(form):
    """diag: w'(h_i * h_j); mlp: w' LN(relu(A h_i + B h_j + c)) (Fajcik et al.'s f_MLP), on every
    valid span and on the null, with positions added before the pair term."""
    torch.manual_seed(9)
    head = H.JointSpanHead(8, 4, 3, 0.0, length_bias=False, pos="additive", form=form, rank=5)
    with torch.no_grad():
        head.w.normal_()
        x_first, x_last, x_cls, mask, pos = inputs()
        out = head(x_first, x_last, mask, x_cls, pos)
        hs, he, pe = head.start(x_first), head.end(x_last), H.sinusoidal(pos, 4)
        bs, be = hs + pe, he + pe

        def pair(a, b):
            if form == "diag":
                return (head.w * a * b).sum()
            return head.w @ torch.nn.functional.layer_norm(torch.relu(head.A(a) + head.B(b)), (5,),
                                                           head.norm.weight, head.norm.bias)
        for w in range(2):
            for i in range(6):
                for j in range(6):
                    if not valid(mask, w, i, j):
                        assert out["span"][w, i, j] == INF
                        continue
                    expected = pair(bs[w, i], be[w, j]) + head.u(hs[w, i]) + head.v(he[w, j])
                    assert float(out["span"][w, i, j]) == pytest.approx(float(expected), abs=1e-5)
        cs, ce = head.start(x_cls), head.end(x_cls)
        null = torch.stack([pair(cs[w], ce[w]) for w in range(2)]) + head.u(cs).squeeze(-1) + head.v(ce).squeeze(-1)
    assert torch.allclose(out["null"], null, atol=1e-5)


@pytest.mark.parametrize("form", ["diag", "mlp"])
def test_rotary_needs_the_full_or_dot_pair_term(form):
    with pytest.raises(ValueError, match="pair term"):
        H.JointSpanHead(8, 4, 3, 0.0, pos="rotary", form=form)


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


# target_unit group: -log p(G*), the gold's group of boundary variants.

# Passage words: the Eiffel Tower . is tall; content words: Eiffel Tower is tall. Gold "the Eiffel Tower" = (0, 2),
# content core (1, 2). Window 0 holds words 0-3; window 1 holds words 1-4, so it cuts off the gold's "the".
PASSAGE_CONTENT = [False, True, True, False, True, True]


def group_batch():
    content = torch.tensor([PASSAGE_CONTENT[0:4], PASSAGE_CONTENT[1:5]])
    return {"content": content, "core_start": torch.tensor([1, 0]), "core_end": torch.tensor([2, 1]),
            "gold_start": torch.tensor([0, -1]), "gold_end": torch.tensor([2, -1]), "example": torch.tensor([0, 0])}


def group_scores(seed=0):
    g = torch.Generator().manual_seed(seed)
    return H.valid_spans(torch.randn(2, 4, 4, generator=g), torch.ones(2, 4, dtype=torch.bool), 3)


def test_gold_group_finds_every_boundary_variant_in_every_window():
    from squadqa.decode import group_members
    member = H.gold_group(group_scores(), group_batch())
    found = {(w, i, j) for w, i, j in torch.nonzero(member).tolist()}
    # Window 0: "the Eiffel Tower", "Eiffel Tower", "Eiffel Tower ." ("the Eiffel Tower ." is 4 words, over the limit).
    # Window 1, which does not hold the whole gold: "Eiffel Tower" and "Eiffel Tower ." again.
    assert found == {(0, 0, 2), (0, 1, 2), (0, 1, 3), (1, 0, 1), (1, 0, 2)}
    starts, ends = zip(*[(i, j) for i in range(6) for j in range(i, min(i + 3, 6))])
    flags = np.array(PASSAGE_CONTENT)
    by_numpy = {(starts[k], ends[k]) for k in group_members(flags, np.array(starts), np.array(ends), (0, 2))}
    assert by_numpy == {(i + w, j + w) for w, i, j in found}          # window w starts at passage word w


def test_group_nll_sums_the_group_over_every_window_and_keeps_the_null():
    span, batch = group_scores(), group_batch()
    null = torch.tensor([0.3, -0.2])
    nll = H.group_nll({"span": span, "null": null}, batch, 1)
    members = span[H.gold_group(span, batch)]
    everything = torch.cat([span[torch.isfinite(span)], null])
    assert float(nll) == pytest.approx(float(torch.logsumexp(everything, 0) - torch.logsumexp(members, 0)), abs=1e-6)
    assert float(nll) < float(H.joint_nll({"span": span, "null": null}, batch, 1))     # the group holds the gold


def test_group_nll_is_joint_nll_when_the_gold_has_no_variant():
    span, batch = group_scores(1), group_batch()
    batch.update(content=torch.ones(2, 4, dtype=torch.bool), core_start=torch.tensor([0, -1]),
                 core_end=torch.tensor([2, -1]))                       # every word a content word: no variants
    out = {"span": span}
    assert float(H.group_nll(out, batch, 1)) == pytest.approx(float(H.joint_nll(out, batch, 1)), abs=1e-6)


def test_interpolated_group_loss_partially_credits_dropped_preposition():
    # Gold "to Paris"; official normalization counts "to", expanded normalization ignores it.
    span = torch.tensor([[[0.1, 0.4], [INF, 0.3]]], requires_grad=True)
    out = {"span": span}
    batch = {"content": torch.tensor([[False, True]]), "core_start": torch.tensor([1]),
             "core_end": torch.tensor([1]), "official_content": torch.tensor([[True, True]]),
             "official_core_start": torch.tensor([0]), "official_core_end": torch.tensor([1]),
             "gold_start": torch.tensor([0]), "gold_end": torch.tensor([1]), "example": torch.tensor([0])}
    official = H.interpolated_group_nll(out, batch, 1, 0.0)
    expanded = H.interpolated_group_nll(out, batch, 1, 1.0)
    mixed = H.interpolated_group_nll(out, batch, 1, 0.5)
    assert float(expanded) < float(mixed) < float(official)
    assert float(mixed) == pytest.approx(float((official + expanded) / 2), abs=1e-6)
    grad = lambda loss: torch.autograd.grad(loss.sum(), span, retain_graph=True)[0][0, 1, 1]
    assert float(grad(mixed)) == pytest.approx(float((grad(official) + grad(expanded)) / 2), abs=1e-6)


def test_a_gold_without_content_words_is_its_own_group():
    span = H.valid_spans(torch.zeros(1, 3, 3), torch.ones(1, 3, dtype=torch.bool), 3)
    batch = {"content": torch.tensor([[False, False, True]]), "core_start": torch.tensor([-1]),
             "core_end": torch.tensor([-1]), "gold_start": torch.tensor([0]), "gold_end": torch.tensor([0]),
             "example": torch.tensor([0])}
    assert torch.nonzero(H.gold_group(span, batch)).tolist() == [[0, 0, 0]]


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
    (["span_head=joint", "bilinear_form=diag", "bilinear_pos=rotary"], ValueError),   # rotary: full and dot only
    (["span_head=compound", "bilinear_form=mlp", "bilinear_pos=rotary"], ValueError),
    (["bilinear_pos=learned"], ValueError),
    (["span_head=standard", "f1_target_weight=0.3"], ValueError),                     # 1a needs a span softmax
    (["span_head=conditional", "risk_weight=0.1"], ValueError),
    (["f1_target_weight=1.5"], ValueError),
    (["f1_target_weight=0.5", "f1_target_temperature=0"], ValueError),
    (["f1_target_floor=-0.1"], ValueError),
    (["risk_weight=-1.0"], ValueError),
    (["risk_em_weight=1.0"], ValueError),                       # an EM weight for a risk loss that is off
    (["risk_weight=0.1", "risk_em_weight=-1.0"], ValueError),
    (["target_unit=answer"], ValueError),
    (["span_head=joint", "target_unit=group", "loss_utility=positional"], ValueError),   # no variants to pool
    (["span_head=standard", "target_unit=group"], ValueError),                           # no span softmax
    (["span_head=joint", "target_unit=group", "expanded_group_weight=-0.1"], ValueError),
    (["span_head=joint", "target_unit=group", "expanded_group_weight=1.1"], ValueError),
    (["span_head=joint", "expanded_group_weight=0.5"], ValueError),
    (["W_init=random"], NotImplementedError),
])
def test_settings_that_are_not_built_or_do_not_apply_raise(overrides, error):
    with pytest.raises(error):
        H.check_head_config(load_config(None, overrides))


@pytest.mark.parametrize("overrides", [
    ["f1_target_weight=0.3"], ["risk_weight=0.1", "risk_em_weight=1.0"],
    ["span_head=joint", "target_unit=group", "f1_target_weight=0.5", "risk_weight=1.0", "risk_em_weight=1.0"],
])
def test_the_1a_losses_are_accepted_on_span_softmax_heads(overrides):
    H.check_head_config(load_config(None, overrides))


# The F1-weighted target and the risk loss against brute force over passage spans (spec §4.3).

def loss_batch():
    """group_batch with what the 1a losses also read: window 1 starts at passage word 1."""
    return {**group_batch(), "first_word": torch.tensor([0, 1]), "word_mask": torch.ones(2, 4, dtype=torch.bool)}


def passage_entries(span):
    """[(passage start, passage end, score)] of every valid window span; window w starts at word w."""
    return [(i + w, j + w, float(span[w, i, j])) for w in range(2) for i in range(4) for j in range(4)
            if torch.isfinite(span[w, i, j])]


def brute_force_units(span, null, unit):
    """{unit: (p(unit), F1, EM)} over distinct passage spans or groups, with each copy's probability."""
    from squadqa.decode import group_keys, span_f1_em
    flags = np.array(PASSAGE_CONTENT)
    entries = passage_entries(span)
    lse = torch.logsumexp(torch.tensor([s for _, _, s in entries] + [float(x) for x in null]), 0)
    units: dict = {}
    for a, b, s in entries:
        key = group_keys(flags, np.array([a]), np.array([b]))[0] if unit == "group" else (a, b)
        f1, em = span_f1_em(flags, a, b, 0, 2)
        p = float(torch.exp(s - lse))
        old = units.get(key, (0.0, float(f1), float(em)))
        assert unit == "group" or old[1:] == (float(f1), float(em))
        units[key] = (old[0] + p, float(f1), float(em))
    return units, float(lse)


def test_gold_f1_em_matches_decode_on_passage_coordinates():
    from squadqa.decode import span_f1_em
    span, batch = group_scores(), loss_batch()
    f1, em = H.gold_f1_em(span, batch, 1, batch["content"])
    for w in range(2):
        for i in range(4):
            for j in range(4):
                if torch.isfinite(span[w, i, j]):
                    want_f1, want_em = span_f1_em(np.array(PASSAGE_CONTENT), i + w, j + w, 0, 2)
                    assert float(f1[w, i, j]) == pytest.approx(float(want_f1)) and float(em[w, i, j]) == want_em


@pytest.mark.parametrize("unit", ["span", "group"])
@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_f1_target_nll_counts_each_unit_once_and_sums_its_copies(unit, floor):
    span, batch, null = group_scores(), loss_batch(), torch.tensor([0.3, -0.2])
    tau = 0.3
    loss, expected = H.f1_target_nll({"span": span, "null": null}, batch, 1, tau, floor, unit, batch["content"])
    units, _ = brute_force_units(span, null, unit)
    credited = {k: v for k, v in units.items() if v[1] > 0 and v[1] >= floor}
    w = {k: np.exp(-(1 - f1) / tau) for k, (_, f1, _) in credited.items()}
    z = sum(w.values())
    want = -sum(w[k] / z * np.log(p) for k, (p, _, _) in credited.items())
    assert float(loss[0]) == pytest.approx(want, rel=1e-5)
    assert float(expected[0]) == pytest.approx(sum(w[k] / z * f1 for k, (_, f1, _) in credited.items()), rel=1e-5)


def test_group_units_pool_the_boundary_variants():
    span, batch = group_scores(), loss_batch()
    by_span, _ = brute_force_units(span, [], "span")
    by_group, _ = brute_force_units(span, [], "group")
    assert len(by_group) < len(by_span)                         # variants of one answer share a unit
    loss_span = H.f1_target_nll({"span": span}, batch, 1, 0.3, 0.0, "span", batch["content"])[0]
    loss_group = H.f1_target_nll({"span": span}, batch, 1, 0.3, 0.0, "group", batch["content"])[0]
    assert not torch.allclose(loss_span, loss_group)


@pytest.mark.parametrize("em_weight", [0.0, 1.0])
def test_risk_loss_is_the_expected_span_loss_and_the_null_is_wrong(em_weight):
    span, batch, null = group_scores(1), loss_batch(), torch.tensor([0.3, -0.2])
    risk = H.risk_loss({"span": span, "null": null}, batch, 1, em_weight, batch["content"])
    units, lse = brute_force_units(span, null, "span")
    want = sum(p * ((1 - f1) + em_weight * (1 - em)) for p, f1, em in units.values())
    want += sum(float(torch.exp(x - lse)) for x in null) * (1 + em_weight)
    assert float(risk[0]) == pytest.approx(want, rel=1e-5)


def test_the_reader_mixes_the_parts_as_the_spec_says(tok):
    torch.manual_seed(0)
    cfg = load_config(None, ["span_head=joint", "no_answer=true", "target_unit=group", "f1_target_weight=0.5",
                             "f1_target_temperature=0.3", "f1_target_floor=0.5", "risk_weight=2.0",
                             "risk_em_weight=1.0", "train.head_dropout=0.0"])
    model = H.Reader(tiny_encoder(), cfg).eval()
    batch = D.collate(D.featurize(D.load_split("train")[:3], tok, cfg, training=True).batch([0, 1, 2]), tok.pad_token_id)
    with torch.no_grad():
        out = model(batch)
        em_part = H.group_nll(out, batch, 3)
        f1_part, _ = H.f1_target_nll(out, batch, 3, 0.3, 0.5, "group", batch["content"])
        risk = H.risk_loss(out, batch, 3, 1.0, batch["content"])
        assert torch.allclose(model.loss(batch, 3), 0.5 * em_part + 0.5 * f1_part + 2.0 * risk, atol=1e-5)
    assert 0 < model.stats["t_f1_expected_f1"] <= 1


def test_old_run_configs_get_the_settings_they_were_trained_with():
    old = {k: v for k, v in load_config(None, ["span_head=standard"]).items() if k not in P.ADDED_KEYS}
    cfg = {**P.ADDED_KEYS, **old}
    H.check_head_config(cfg)
    assert cfg["no_answer"] is False and cfg["f1_target_weight"] == 0.0 and cfg["window_layout"] == "left"


def test_old_run_configs_get_renamed_keys(tmp_path, monkeypatch):
    run = tmp_path / "old-run"
    run.mkdir()
    (run / "config.yaml").write_text("span_head: joint\nf1_target: positional\n", encoding="utf-8")
    monkeypatch.setattr(P, "RESULTS", tmp_path)
    cfg = P.run_config("old-run")
    assert cfg["loss_utility"] == "positional" and "f1_target" not in cfg


def test_every_committed_run_config_has_every_current_key():
    runs = [d.name for d in sorted(P.RESULTS.iterdir()) if (d / "config.yaml").exists()]
    if not runs:
        pytest.skip("no committed runs")
    base = load_config(None, [])
    for run in runs:
        cfg = P.run_config(run)
        assert set(base) <= set(cfg) and set(base["train"]) <= set(cfg["train"]), run
        H.check_head_config(cfg)


# Each objective end to end on a tiny encoder.

HEAD_CONFIGS = {
    "standard": ["span_head=standard"],
    "conditional": ["span_head=conditional"],
    "joint": ["span_head=joint", "length_bias=false", "bilinear_pos=additive"],
    "joint-diag": ["span_head=joint", "length_bias=false", "bilinear_form=diag"],
    "joint-mlp": ["span_head=joint", "length_bias=false", "bilinear_form=mlp", "bilinear_pos=additive"],
    "compound": ["span_head=compound", "bilinear_pos=additive"],
    "compound-rotary": ["span_head=compound", "bilinear_pos=rotary"],
    "joint-group": ["span_head=joint", "length_bias=false", "target_unit=group"],
    "joint-soft-group": ["span_head=joint", "length_bias=false", "target_unit=group", "f1_target_weight=0.5",
                         "f1_target_temperature=0.3", "f1_target_floor=0.5"],
    "joint-risk-group": ["span_head=joint", "length_bias=false", "target_unit=group", "risk_weight=1.0",
                         "risk_em_weight=1.0"],
    "compound-soft-risk": ["span_head=compound", "f1_target_weight=0.3", "risk_weight=0.3"],
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


def test_calibration_under_group_fits_the_gold_groups_probability():
    from types import SimpleNamespace

    from squadqa.decode import SpanScores, group_members, log_softmax, temperature_nll
    text = "Visit the Eiffel Tower . Big Ben"                 # words: Visit the Eiffel Tower . Big Ben
    words = D.word_spans(text)
    starts, ends = np.array([1, 2, 2, 5, 0]), np.array([3, 3, 4, 6, 0])    # 3 variants of "Eiffel Tower"; Big Ben; Visit
    rng = np.random.default_rng(0)
    spans, examples = [], []
    for _ in range(200):
        s = SpanScores(starts, ends, (3 * rng.normal(size=5)).astype(np.float32))
        k = int(rng.choice(5, p=np.exp(log_softmax(s.scores, 1.5))))
        spans.append(s)
        examples.append(SimpleNamespace(gold=(int(starts[k]), int(ends[k])), question=SimpleNamespace(context=text),
                                        context=SimpleNamespace(words=words)))
    cal = P.calibrate(SimpleNamespace(examples=examples), spans, "group")
    flags = np.array([True, False, True, True, False, True, True])
    assert group_members(flags, starts, ends, (2, 3)).tolist() == [0, 1, 2]
    items = [(s.with_null(), group_members(flags, starts, ends, ex.gold)) for ex, s in zip(examples, spans)]
    assert cal["unit"] == "group" and cal["questions"] == 200
    assert cal["nll"] == pytest.approx(temperature_nll(items, cal["temperature"]))
    span_cal = P.calibrate(SimpleNamespace(examples=examples), spans)
    assert span_cal["unit"] == "span" and cal["nll_at_1"] < span_cal["nll_at_1"]     # the group holds the gold


def test_windows_carry_content_flags_and_the_gold_core(small_train):
    from squadqa.decode import content_flags
    batch = D.collate(small_train.batch(list(range(len(small_train.examples)))), 0)
    rows = small_train.batch(list(range(len(small_train.examples))))
    for r, (b, w) in enumerate(rows):
        ex = small_train.examples[b]
        flags = content_flags(ex.question.context, ex.context.words)
        assert batch["content"][r, :w.n_words].tolist() == flags[w.first_word:w.first_word + w.n_words].tolist()
        if w.gold is not None:                             # the core lies inside the gold, on content words
            cs, ce = int(batch["core_start"][r]), int(batch["core_end"][r])
            gold = flags[w.first_word + w.gold[0]:w.first_word + w.gold[1] + 1]
            if cs < 0:
                assert not gold.any()
            else:
                assert w.gold[0] <= cs <= ce <= w.gold[1] and flags[w.first_word + cs] and flags[w.first_word + ce]


def test_mixed_group_features_keep_both_gold_cores(tok):
    cfg = load_config("configs/experiments/row1-boundary-mix.yaml")
    q = D.Question("q", "t", "Travel to Paris", "Where did they travel?", (("to Paris", 7),))
    feats = D.featurize([q], tok, cfg, training=True)
    batch = D.collate(feats.batch([0]), tok.pad_token_id)
    assert feats.examples[0].gold == (1, 2)
    assert batch["content"][0, :3].tolist() == [True, False, True]
    assert batch["official_content"][0, :3].tolist() == [True, True, True]
    assert (int(batch["core_start"][0]), int(batch["core_end"][0])) == (2, 2)
    assert (int(batch["official_core_start"][0]), int(batch["official_core_end"][0])) == (1, 2)


def test_no_null_without_the_option(tok):
    cfg = load_config(None, ["span_head=compound"])
    rows = D.featurize(D.load_split("validation")[:3], tok, cfg, training=False)
    batch = D.collate(rows.batch([0, 1, 2]), tok.pad_token_id)
    model = H.Reader(tiny_encoder(), cfg).eval()
    with torch.no_grad():
        spans, null = model.span_scores(batch)
    assert null is None and spans.shape[0] == len(batch["input_ids"])
