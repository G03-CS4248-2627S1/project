import numpy as np
import pytest

from squadqa import data as D
from squadqa import diagnostics as G
from squadqa.config import load_config
from squadqa.decode import SpanScores, content_flags
from squadqa.data import word_spans

CFG = load_config()
LOW = -30.0                                     # a word that is never a start or an end


def standard(start, end, max_len=30):
    """The spans of a standard head: s(i, j) = start_i + end_j over every valid span."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    n = len(start)
    i, j = np.nonzero((np.arange(n)[None, :] >= np.arange(n)[:, None])
                      & (np.arange(n)[None, :] - np.arange(n)[:, None] < max_len))
    return SpanScores(i.astype(np.int32), j.astype(np.int32), (start[i] + end[j]).astype(np.float32))


def flags_of(text):
    return content_flags(text, word_spans(text))


DATES = "March 1887 or May 1889"                # plan §5: words 0-1 and 3-4, the model split 50/50


def dates_head():
    start = [5.0, LOW, LOW, 5.0, LOW]
    end = [LOW, 5.0, LOW, LOW, 5.0]
    return start, end


def test_standard_head_split_between_two_answers_gives_a_third_to_the_cross_span():
    x = G.cross_span(standard(*dates_head()), flags_of(DATES))
    assert (x.a, x.b, x.c) == ((0, 1), (3, 4), (0, 4))
    assert x.p_a == pytest.approx(1 / 3) and x.p_b == pytest.approx(1 / 3) and x.p_c == pytest.approx(1 / 3)
    assert x.ratio == pytest.approx(1 / 3)
    assert x.p_top1 == pytest.approx(1 / 3)


def test_a_and_b_are_ordered_by_position():
    start, end = dates_head()
    start[3] = 6.0                                   # "May 1889" is now the most probable
    x = G.cross_span(standard(start, end), flags_of(DATES))
    assert (x.a, x.b, x.c) == ((0, 1), (3, 4), (0, 4))
    assert x.p_top1 == pytest.approx(x.p_b) and x.p_b > x.p_a


def test_a_joint_head_without_cross_mass_gives_zero():
    s = standard(*dates_head())
    s.scores[s.index((0, 4))] = -np.inf             # a joint head can score the cross span on its own
    x = G.cross_span(s, flags_of(DATES))
    assert x.c == (0, 4) and x.p_c == 0.0 and x.ratio == 0.0
    assert x.p_a == pytest.approx(0.5)


def test_temperature_spreads_the_mass():
    start, end = dates_head()
    start[3] = 6.0
    sharp = G.cross_span(standard(start, end), flags_of(DATES), temperature=0.5)
    flat = G.cross_span(standard(start, end), flags_of(DATES), temperature=2.0)
    assert sharp.p_top1 > flat.p_top1


def test_a_cross_span_over_the_length_limit_is_none():
    x = G.cross_span(standard(*dates_head(), max_len=4), flags_of(DATES))
    assert x.b == (3, 4) and x.c is None and x.p_c == 0.0 and x.ratio == 0.0


def test_b_needs_a_content_word():
    text = "Paris , France ."                      # words: Paris , France .
    start = [5.0, LOW, 1.0, 6.0]
    end = [5.0, LOW, 1.0, 0.0]
    x = G.cross_span(standard(start, end), flags_of(text))
    assert x.a == (0, 0) and x.b == (2, 2)         # not "." (3, 3), which scores higher
    assert x.c == (0, 2)


def test_no_second_span():
    x = G.cross_span(standard([2.0, LOW], [LOW, 2.0]), flags_of("the Beatles"))
    assert x.a == (0, 1) and x.b is None and x.c is None and x.ratio == 0.0   # every other span overlaps
    assert x.p_a == pytest.approx(x.p_top1)


def test_buckets():
    assert [G.bucket(p) for p in (0.0, 0.4999, 0.5, 0.9, 0.9001, 1.0)] == [
        "<0.5", "<0.5", "0.5-0.9", "0.5-0.9", ">0.9", ">0.9"]


def test_is_cross_counts_boundary_variants_only_when_normalized():
    text = "the March 1887 or May 1889 ."
    flags = flags_of(text)
    x = G.CrossSpan((1, 2), (4, 5), (1, 5), 0.4, 0.3, 0.4, 0.3)
    assert G.is_cross(x, (1, 5), flags, "normalized") and G.is_cross(x, (1, 5), flags, "positional")
    assert G.is_cross(x, (0, 6), flags, "normalized")          # "the March 1887 or May 1889 ."
    assert not G.is_cross(x, (0, 6), flags, "positional")
    assert not G.is_cross(x, (1, 2), flags, "normalized")      # a itself
    assert not G.is_cross(G.CrossSpan((1, 2), (4, 5), None, 0.4, 0.3, 0.4, 0.0), (1, 5), flags, "normalized")


def test_summarize():
    rows = {"q1": G.CrossSpan((0, 1), (3, 4), (0, 4), 1 / 3, 1 / 3, 1 / 3, 1 / 3),
            "q2": G.CrossSpan((0, 1), (3, 4), None, 0.95, 0.95, 0.01, 0.0),
            "q3": G.CrossSpan((0, 1), None, None, 0.7, 0.7, 0.0, 0.0)}
    picks = {"map": set(), "mbr": {"q1"}}
    scores = {"map": ({"q1": 1.0, "q2": 1.0, "q3": 0.0}, {"q1": 1.0, "q2": 1.0, "q3": 0.5}),
              "mbr": ({"q1": 0.0, "q2": 1.0, "q3": 0.0}, {"q1": 0.6, "q2": 1.0, "q3": 0.5})}
    out = G.summarize(rows, picks, scores)
    assert out["questions"] == 3 and out["no_second_span"] == 1 and out["no_cross_span"] == 2
    assert out["ratio"]["all"]["mean"] == pytest.approx(1 / 9)
    assert {k: out["ratio"][k]["questions"] for k in G.BUCKETS} == {"<0.5": 1, "0.5-0.9": 1, ">0.9": 1}
    assert out["ratio"]["<0.5"]["mean"] == pytest.approx(1 / 3)
    assert out["ratio"]["all"]["share_over_0.1"] == pytest.approx(1 / 3)
    assert out["scores"]["map"]["exact"] == pytest.approx(200 / 3)
    assert out["cross_answers"]["map"] == {"questions": 0}
    mbr = out["cross_answers"]["mbr"]
    assert mbr["questions"] == 1
    assert mbr["scores"]["mbr"]["f1"] == pytest.approx(60.0) and mbr["scores"]["map"]["exact"] == 100.0


# The whole path on validation questions, with a fake reader.

@pytest.fixture(scope="module")
def tok():
    try:
        return D.load_tokenizer(CFG["model"])
    except (OSError, RuntimeError) as e:
        pytest.skip(f"tokenizer not available: {e}")


def two_answers(gap=3):
    """A standard head split 50/50 between the gold span and the span of the same length that
    starts `gap` words after it, in every window that holds both."""
    import torch

    from squadqa.heads import valid_spans

    def score_fn(batch):
        W, n = batch["word_mask"].shape
        start = torch.full((W, n), LOW)
        end = torch.full((W, n), LOW)
        for r in range(W):
            i, j = int(batch["gold_start"][r]), int(batch["gold_end"][r])
            k = j + 1 + gap
            if i >= 0 and k + j - i < n and batch["word_mask"][r, k + j - i]:
                start[r, i] = start[r, k] = 20.0
                end[r, j] = end[r, k + j - i] = 20.0
        return valid_spans(start[:, :, None] + end[:, None, :], batch["word_mask"], CFG["max_answer_words"])
    return score_fn


def test_cross_spans_end_to_end(tok):
    pytest.importorskip("torch")
    from squadqa import predict as P

    feats = D.featurize(D.load_split("validation")[:300], tok, CFG, training=False)
    spans = P.score_questions(two_answers(), feats, tok.pad_token_id, CFG["max_answer_words"])
    rows, preds, picks = G.cross_spans(feats, spans, CFG, temperature=1.0)
    assert list(rows) == [ex.question.id for ex in feats.examples]
    gold = {ex.question.id: ex.gold for ex in feats.examples}
    split = [q for q, x in rows.items() if min(x.p_a, x.p_b) > 0.3 and x.c is not None]   # both got the bonus
    assert len(split) > 150
    for q in split:
        assert rows[q].a == gold[q] and rows[q].ratio == pytest.approx(1 / 3, rel=1e-3)
    assert set(preds) == set(CFG["decode_rules"]) and all(len(p) == len(rows) for p in preds.values())
    assert set(split) <= picks["mbr"]                   # expected F1 + EM favours the cross span
