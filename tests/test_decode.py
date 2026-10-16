import numpy as np
import pytest

from squadqa import decode as Dc
from squadqa.data import word_spans


def words_of(text):
    return word_spans(text)


def flags_of(text):
    return Dc.content_flags(text, word_spans(text))


def test_content_flags():
    assert flags_of("The Eiffel Tower , an icon of theory .").tolist() == [
        False, True, True, False, False, True, True, True, False]


# Content-word F1 and EM.

EIFFEL = "Visit the Eiffel Tower ."          # words: Visit the Eiffel Tower .


def test_boundary_variant_is_an_exact_match():
    f = flags_of(EIFFEL)
    f1, em = Dc.span_f1_em(f, 1, 3, 2, 3)              # "the Eiffel Tower" vs "Eiffel Tower"
    assert (f1, em) == (1.0, 1.0)
    f1, em = Dc.span_f1_em(f, 2, 4, 2, 3)              # trailing "."
    assert (f1, em) == (1.0, 1.0)
    f1, em = Dc.span_f1_em(np.ones_like(f), 1, 3, 2, 3)  # positional: 0.8, as in spec §4.3
    assert f1 == pytest.approx(0.8) and em == 0.0


def test_same_text_elsewhere_gets_nothing():
    f = flags_of("the cat and the cat")
    assert Dc.span_f1_em(f, 1, 1, 4, 4) == (0.0, 0.0)
    assert Dc.span_f1_em(f, 0, 1, 3, 4) == (0.0, 0.0)


def test_spans_without_content_words_match_only_themselves():
    f = flags_of("the cat and the cat")
    assert Dc.span_f1_em(f, 0, 0, 0, 0) == (1.0, 1.0)
    assert Dc.span_f1_em(f, 0, 0, 3, 3) == (0.0, 0.0)   # "the" elsewhere: a different position
    assert Dc.span_f1_em(f, 0, 0, 1, 1) == (0.0, 0.0)


def brute_force(flags, a, b):
    ca = [w for w in range(a[0], a[1] + 1) if flags[w]]
    cb = [w for w in range(b[0], b[1] + 1) if flags[w]]
    if not ca or not cb:
        same = float(not ca and not cb and a == b)
        return same, same
    ov = len(set(ca) & set(cb))
    if ov == 0:
        return 0.0, 0.0
    p, r = ov / len(ca), ov / len(cb)
    return 2 * p * r / (p + r), float(ca == cb)


def test_f1_em_match_brute_force_and_broadcast():
    rng = np.random.default_rng(0)
    flags = rng.random(40) < 0.7
    spans = [tuple(sorted(rng.integers(0, 40, 2))) for _ in range(60)]
    s = np.array([a for a, _ in spans])
    e = np.array([b for _, b in spans])
    f1, em = Dc.span_f1_em(flags, s[:, None], e[:, None], s[None, :], e[None, :])
    for x, a in enumerate(spans):
        for y, b in enumerate(spans):
            bf1, bem = brute_force(flags, a, b)
            assert f1[x, y] == pytest.approx(bf1) and em[x, y] == bem


def test_group_keys():
    f = flags_of("Visit the Eiffel Tower . The cat")
    keys = Dc.group_keys(f, np.array([1, 2, 2, 4, 6, 1]), np.array([3, 3, 4, 5, 6, 1]))
    assert keys == [(2, 3), (2, 3), (2, 3), (-1, 4, 5), (6, 6), (-1, 1, 1)]


# One softmax per question.

def window(scores, first_word, first_tok, ctx_start, ctx_end):
    first_tok = np.asarray(first_tok)
    return np.asarray(scores, dtype=np.float32), first_word, first_tok, first_tok, ctx_start, ctx_end


def test_combine_takes_the_most_central_window():
    inf = -np.inf
    # Window A holds words 0-3 at tokens 5-8 (passage tokens 5..8); window B holds words 2-5 at tokens 5-8.
    a = np.full((4, 4), inf); a[2, 3] = 1.0; a[0, 0] = 0.5; a[3, 3] = 0.2
    b = np.full((4, 4), inf); b[0, 1] = 2.0; b[1, 1] = 0.7; b[3, 3] = 0.1
    spans = Dc.combine_windows([window(a, 0, [5, 6, 7, 8], 5, 8), window(b, 2, [5, 6, 7, 8], 5, 8)], 30)
    got = {(int(i), int(j)): float(s) for i, j, s in zip(spans.starts, spans.ends, spans.scores)}
    # (2, 3): in A it has 2 tokens left and 0 right (min 0); in B 0 left and 2 right (min 0): tie -> A.
    # (3, 3): A gives min(3, 0) = 0; B (local 1) gives min(1, 2) = 1 -> B's 0.7.
    assert got == pytest.approx({(0, 0): 0.5, (2, 3): 1.0, (3, 3): 0.7, (5, 5): 0.1})


def test_combine_sums_the_window_nulls():
    a = np.full((2, 2), -np.inf); a[0, 0] = 1.0
    b = np.full((2, 2), -np.inf); b[1, 1] = 2.0
    spans = Dc.combine_windows([window(a, 0, [3, 4], 3, 4), window(b, 5, [3, 4], 3, 4)], 30, [0.5, -1.0])
    assert spans.null == pytest.approx(float(np.log(np.exp(0.5) + np.exp(-1.0))))
    assert Dc.combine_windows([window(a, 0, [3, 4], 3, 4)], 30).null is None


def test_null_joins_the_softmax_but_is_never_a_candidate():
    spans = Dc.SpanScores(np.array([0, 1]), np.array([0, 1]), np.array([1.0, 2.0], dtype=np.float32), null=0.5)
    z = np.array([1.0, 2.0, 0.5])
    logp, null = Dc.span_logp(spans, 2.0)
    assert np.exp(logp) == pytest.approx(np.exp(z[:2] / 2) / np.exp(z / 2).sum())
    assert np.exp(null) == pytest.approx(np.exp(0.25) / np.exp(z / 2).sum())
    c = Dc.top_k(spans, 5)
    assert c.starts.tolist() == [1, 0] and np.exp(c.logp).sum() == pytest.approx(1 - np.exp(0.5) / np.exp(z).sum())
    assert Dc.span_logp(Dc.SpanScores(spans.starts, spans.ends, spans.scores))[1] is None


def test_combine_drops_invalid_spans():
    s = np.zeros((40, 40))
    spans = Dc.combine_windows([window(s, 0, np.arange(40) + 3, 3, 42)], 30)
    assert ((spans.ends >= spans.starts) & (spans.ends - spans.starts < 30)).all()
    assert len(spans.scores) == sum(min(30, 40 - i) for i in range(40))
    assert spans.index((0, 29)) is not None and spans.index((0, 30)) is None and spans.index(None) is None


# Calibration.

def test_fit_temperature_recovers_the_true_temperature():
    rng = np.random.default_rng(1)
    items = []
    for _ in range(3000):
        z = rng.normal(0, 3, size=50)
        p = np.exp(Dc.log_softmax(z, 2.0))
        items.append((z, int(rng.choice(50, p=p))))
    t = Dc.fit_temperature(items)
    assert t == pytest.approx(2.0, rel=0.1)
    assert Dc.temperature_nll(items, t) <= Dc.temperature_nll(items, 1.0)


def test_fit_temperature_raises_at_the_edge():
    items = [(np.array([10.0, 0.0, 0.0]), 1)] * 10          # never right: T runs to the upper end
    with pytest.raises(RuntimeError, match="edge"):
        Dc.fit_temperature(items)


# Decision rules on hand-made distributions.

def cands(spans, probs):
    return Dc.Candidates(np.array([s for s, _ in spans]), np.array([e for _, e in spans]), np.log(np.array(probs)))


def test_map_groups_boundary_variants():
    text = "Visit the Eiffel Tower . Big Ben"
    f = flags_of(text)
    c = cands([(5, 6), (1, 3), (2, 3)], [0.4, 0.3, 0.3])     # Big Ben; the Eiffel Tower; Eiffel Tower
    assert Dc.decode(c, f, "map", "normalized") == (1, 3)    # group mass 0.6, its most probable span
    assert Dc.decode(c, f, "map", "positional") == (5, 6)


def test_same_text_elsewhere_is_not_grouped():
    f = flags_of("the cat and the cat")
    c = cands([(0, 0), (1, 1), (4, 4)], [0.4, 0.35, 0.25])   # "the" (no content), cat, cat elsewhere
    assert Dc.decode(c, f, "map", "normalized") == (0, 0)


def test_spans_without_content_words_never_pool():
    # From review of PR #30: "." and "the" are unrelated and must not add up to beat "dog".
    f = flags_of("the dog barked .")
    c = cands([(1, 1), (3, 3), (0, 0)], [0.45, 0.30, 0.25])
    assert Dc.decode(c, f, "map", "normalized") == (1, 1)
    assert Dc.decode(c, f, "mbr", "normalized") == (1, 1)


def test_mbr_prefers_the_span_that_overlaps_both():
    f = flags_of("red green blue")
    c = cands([(0, 0), (2, 2), (0, 2)], [0.35, 0.33, 0.32])
    assert Dc.decode(c, f, "map", "normalized") == (0, 0)
    # Expected F1: (0,0) .35 + .32*.5 = .51; (2,2) .33 + .32*.5 = .49; (0,2) .32 + .68*.5 = .66.
    assert Dc.decode(c, f, "mbr", "normalized", em_weight=0.0) == (0, 2)
    assert Dc.decode(c, f, "mbr", "normalized", em_weight=10.0) == (0, 0)    # large gamma -> MAP


def test_mbr_positional_counts_every_word():
    f = flags_of("the red car")
    c = cands([(1, 2), (0, 2)], [0.5, 0.5])
    assert Dc.decode(c, f, "mbr", "normalized") == (1, 2)       # one group: its most probable span
    assert Dc.decode(c, f, "mbr", "positional") == (1, 2)       # tie: the first candidate


# String grouping (mbr_utility: string): the scorer's normalization at any position, decoding only.

def texts_of(text, c):
    w = words_of(text)
    return [Dc.answer_text(text, w, int(i), int(j)) for i, j in zip(c.starts, c.ends)]


def test_string_groups_the_same_text_anywhere():
    text = "Paris beat Lyon and Paris won"
    f = flags_of(text)
    c = cands([(2, 2), (0, 0), (4, 4)], [0.4, 0.3, 0.3])       # Lyon; Paris; Paris elsewhere
    assert Dc.decode(c, f, "map", "normalized") == (2, 2)       # rule 4: the two Paris spans stay apart
    assert Dc.decode(c, f, "map", "string", texts=texts_of(text, c)) == (0, 0)   # one group, mass 0.6


def test_string_groups_boundary_variants_too():
    text = "Visit the Eiffel Tower . Big Ben"
    f = flags_of(text)
    c = cands([(5, 6), (1, 3), (2, 4)], [0.4, 0.3, 0.3])       # Big Ben; the Eiffel Tower; Eiffel Tower .
    assert Dc.decode(c, f, "map", "string", texts=texts_of(text, c)) == (1, 3)


def test_string_mbr_uses_the_scorers_bag_of_tokens():
    text = "red green blue . red blue"
    f = flags_of(text)
    c = cands([(0, 0), (2, 2), (4, 5)], [0.35, 0.33, 0.32])     # red; blue; "red blue" at other positions
    # Content-word F1 sees no overlap: expected F1 is each span's own mass, so "red" wins.
    assert Dc.decode(c, f, "mbr", "normalized", em_weight=0.0) == (0, 0)
    # Scorer F1("red blue", "red") = F1("red blue", "blue") = 2/3: .32 + .68 * 2/3 = .77 beats .35 + .32 * 2/3.
    assert Dc.decode(c, f, "mbr", "string", em_weight=0.0, texts=texts_of(text, c)) == (4, 5)
    f1, em = Dc.string_f1_em(["red blue", "red", "Red blue."])
    assert f1[0, 1] == pytest.approx(2 / 3) and em[0, 1] == 0.0 and f1[0, 2] == em[0, 2] == 1.0


def test_string_spans_that_normalize_to_nothing_never_pool():
    text = "the dog barked ."
    f = flags_of(text)
    c = cands([(1, 1), (3, 3), (0, 0)], [0.45, 0.30, 0.25])     # dog; "."; "the": both normalize to ""
    for rule in ("map", "mbr"):
        assert Dc.decode(c, f, rule, "string", texts=texts_of(text, c)) == (1, 1)
    f1, em = Dc.string_f1_em([".", "the", "dog"])
    assert f1.tolist() == em.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_string_needs_the_candidate_texts():
    c = cands([(0, 0), (1, 1)], [0.6, 0.4])
    with pytest.raises(ValueError):
        Dc.decode(c, np.ones(2, bool), "map", "string")
    with pytest.raises(ValueError):
        Dc.decode(c, np.ones(2, bool), "map", "string", texts=["a"])


def test_unknown_rule_or_utility_raises():
    c = cands([(0, 0)], [1.0])
    with pytest.raises(ValueError):
        Dc.decode(c, np.ones(1, bool), "greedy", "normalized")
    with pytest.raises(ValueError):
        Dc.decode(c, np.ones(1, bool), "map", "official")


def test_top_k_and_answer_text():
    spans = Dc.SpanScores(np.array([0, 1, 2]), np.array([0, 2, 2]), np.array([0.0, 2.0, 1.0], dtype=np.float32))
    c = Dc.top_k(spans, 2)
    assert c.starts.tolist() == [1, 2] and np.exp(Dc.log_softmax(spans.scores)).sum() == pytest.approx(1.0)
    text = "Visit the Eiffel Tower."
    assert Dc.answer_text(text, words_of(text), 1, 4) == "the Eiffel Tower."
