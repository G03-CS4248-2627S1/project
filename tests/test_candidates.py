from types import SimpleNamespace

import numpy as np
import pytest

from squadqa import candidates as C
from squadqa.data import word_spans
from squadqa.decode import SpanScores, content_flags, log_softmax

CONTEXT = "The tower was built in 1889 for the World's Fair in Paris ."
WORDS = word_spans(CONTEXT)
FLAGS = content_flags(CONTEXT, WORDS)


def all_spans(n: int, max_len: int = 30, seed: int = 0) -> SpanScores:
    """Every valid span of an n-word passage, with random scores."""
    starts, ends = zip(*[(i, j) for i in range(n) for j in range(i, min(n, i + max_len))])
    scores = np.random.default_rng(seed).normal(size=len(starts)).astype(np.float32)
    return SpanScores(np.array(starts), np.array(ends), scores)


def word(w: str) -> int:
    return [CONTEXT[s:e] for s, e in WORDS].index(w)


# Sampling.

def test_sample_spans_are_distinct_and_capped_by_the_number_of_spans():
    scores = np.array([0.0, 1.0, 2.0])
    rng = np.random.default_rng(0)
    for k in (1, 2, 3, 10):
        got = C.sample_spans(scores, k, 1.0, rng)
        assert len(got) == min(k, 3) and len(set(got.tolist())) == len(got)


def test_sample_spans_draws_from_the_tempered_softmax():
    """Gumbel-top-k with k = 1 is a draw from softmax(scores / T)."""
    scores = np.array([0.0, 1.0, 2.0])
    for t in (0.5, 1.0, 3.0):
        rng = np.random.default_rng(1)
        draws = np.array([C.sample_spans(scores, 1, t, rng)[0] for _ in range(20000)])
        freq = np.bincount(draws, minlength=3) / len(draws)
        assert freq == pytest.approx(np.exp(log_softmax(scores, t)), abs=0.015)


def test_sample_spans_rejects_a_bad_temperature():
    with pytest.raises(ValueError):
        C.sample_spans(np.zeros(3), 1, 0.0, np.random.default_rng(0))


def test_question_rng_depends_only_on_the_seed_and_the_question():
    a = C.question_rng(0, "q1").random(4)
    assert np.array_equal(a, C.question_rng(0, "q1").random(4))
    assert not np.array_equal(a, C.question_rng(0, "q2").random(4))
    assert not np.array_equal(a, C.question_rng(1, "q1").random(4))


# Lists.

def test_boundary_variants_are_the_valid_one_word_changes():
    s = all_spans(len(WORDS))
    i, j = word("World"), word("Fair")
    got = {(int(s.starts[k]), int(s.ends[k])) for k in C.boundary_variants(s, (i, j))}
    assert got == {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
    # a one-word gold at the passage's start has only the longer variant on the right
    got = {(int(s.starts[k]), int(s.ends[k])) for k in C.boundary_variants(s, (0, 0))}
    assert got == {(0, 1)}


def test_training_list_holds_the_gold_its_variants_and_no_duplicates():
    s = all_spans(len(WORDS))
    gold = (word("1889"), word("1889"))
    for seed in range(20):
        c = C.training_list(s, gold, 8, 2.0, np.random.default_rng(seed), FLAGS, 1.0)
        spans = list(zip(c.starts.tolist(), c.ends.tolist()))
        assert len(spans) == len(set(spans))
        assert c.gold is not None and spans[c.gold] == gold
        assert {(gold[0] - 1, gold[1]), (gold[0], gold[1] + 1)} <= set(spans)
        assert 8 <= len(spans) <= 8 + 1 + 4
        assert c.f1[c.gold] == 1 and c.em[c.gold] == 1
        assert spans == sorted(spans)


def test_training_list_records_whether_the_sample_held_the_gold():
    s = all_spans(len(WORDS))
    gold = (word("1889"), word("1889"))
    g = s.index(gold)
    hot = SpanScores(s.starts, s.ends, np.where(np.arange(len(s.scores)) == g, 50.0, 0.0).astype(np.float32))
    cold = SpanScores(s.starts, s.ends, np.where(np.arange(len(s.scores)) == g, -50.0, 0.0).astype(np.float32))
    assert C.training_list(hot, gold, 3, 1.0, np.random.default_rng(0), FLAGS, 1.0).sampled_gold
    assert not C.training_list(cold, gold, 3, 1.0, np.random.default_rng(0), FLAGS, 1.0).sampled_gold


def test_training_list_refuses_an_invalid_gold():
    with pytest.raises(ValueError, match="not a valid span"):
        C.training_list(all_spans(5, max_len=2), (0, 4), 3, 1.0, np.random.default_rng(0), np.ones(5, bool), 1.0)


def test_a_boundary_variant_with_an_article_is_an_exact_match():
    """loss_utility normalized: "the World's Fair" covers the same content words as "World's Fair"."""
    s = all_spans(len(WORDS))
    gold = (word("World"), word("Fair"))
    c = C.training_list(s, gold, 8, 2.0, np.random.default_rng(0), FLAGS, 1.0)
    k = list(zip(c.starts.tolist(), c.ends.tolist())).index((word("World") - 1, word("Fair")))
    assert c.em[k] == 1 and c.f1[k] == 1


def test_inference_list_is_the_top_k_and_adds_nothing():
    s = all_spans(len(WORDS))
    logp = log_softmax(s.scores, 1.3)
    top = set(np.argsort(-logp)[:5].tolist())
    out = set(np.argsort(-logp)[5:].tolist())
    gold_in = (int(s.starts[min(top)]), int(s.ends[min(top)]))
    gold_out = (int(s.starts[min(out)]), int(s.ends[min(out)]))
    a = C.inference_list(s, 5, FLAGS, 1.3, gold_in)
    b = C.inference_list(s, 5, FLAGS, 1.3, gold_out)
    for c in (a, b):
        assert {s.index((int(i), int(j))) for i, j in zip(c.starts, c.ends)} == top
    assert a.gold is not None and a.sampled_gold
    assert b.gold is None and not b.sampled_gold
    assert a.logp == pytest.approx(logp[[s.index((int(i), int(j))) for i, j in zip(a.starts, a.ends)]])


def test_inference_list_without_a_gold_has_no_labels():
    c = C.inference_list(all_spans(6), 5, np.ones(6, bool), 1.0)
    assert c.gold is None and np.isnan(c.f1).all()


def test_build_lists_does_not_depend_on_the_batch():
    def example(qid, gold):
        q = SimpleNamespace(id=qid, context=CONTEXT)
        return SimpleNamespace(question=q, context=SimpleNamespace(words=WORDS), gold=gold)

    cfg = {"verifier": {"k_train": 6, "sample_temperature": 2.0, "k_infer": 5}, "loss_utility": "normalized"}
    exs = [example("a", (word("1889"),) * 2), example("b", (word("Paris"),) * 2)]
    spans = [all_spans(len(WORDS), seed=1), all_spans(len(WORDS), seed=2)]
    both = C.build_lists(exs, spans, "train", cfg, 1.0, seed=0)
    alone = C.build_lists(exs[1:], spans[1:], "train", cfg, 1.0, seed=0)
    assert np.array_equal(both["b"].starts, alone["b"].starts) and np.array_equal(both["b"].ends, alone["b"].ends)
    other_seed = C.build_lists(exs, spans, "train", cfg, 1.0, seed=1)
    assert any(not np.array_equal(both[q].starts, other_seed[q].starts) for q in "ab")


# Targets.

LIST_F1 = np.array([1.0, 1.0, 0.8, 0.4, 0.0])        # gold, a boundary variant, two partials, a miss
LIST_EM = np.array([1, 1, 0, 0, 0])


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("unit", ["span", "group"])
@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_targets_sum_to_one(alpha, unit, floor):
    t = C.targets(LIST_F1, LIST_EM, 0, alpha, 0.1, floor, unit)
    assert t.sum() == pytest.approx(1.0) and (t >= 0).all()


def test_one_hot_targets():
    assert C.targets(LIST_F1, LIST_EM, 0, 0.0, 0.1, 0.5, "span").tolist() == [1, 0, 0, 0, 0]
    assert C.targets(LIST_F1, LIST_EM, 0, 0.0, 0.1, 0.5, "group").tolist() == [0.5, 0.5, 0, 0, 0]


def test_f1_targets_follow_the_floor_and_count_the_gold_group_once():
    t = C.targets(LIST_F1, LIST_EM, 0, 1.0, 0.1, 0.5, "span")
    assert t[3] == 0 and t[4] == 0 and t[0] == t[1] > t[2] > 0
    g = C.targets(LIST_F1, LIST_EM, 0, 1.0, 0.1, 0.5, "group")
    # the group shares one weight: w(1) / 2 each, against w(0.8) = exp(-2) for the partial
    assert g[0] == g[1] and g[0] / g[2] == pytest.approx(0.5 / np.exp(-2.0))


def test_targets_reject_bad_settings():
    with pytest.raises(ValueError):
        C.targets(LIST_F1, LIST_EM, 0, 1.5, 0.1, 0.5)
    with pytest.raises(ValueError):
        C.targets(LIST_F1, LIST_EM, 0, 0.5, 0.1, 0.5, "word")


# Summary.

def _list(logp, f1, em, gold, sampled=True):
    n = len(logp)
    return C.CandidateList(np.arange(n), np.arange(n), np.asarray(logp, float), np.asarray(f1, float),
                           np.asarray(em), gold, sampled)


def test_summary_of_inference_lists():
    lists = [
        _list([-0.1, -2, -3], [1, 0.5, 0], [1, 0, 0], 0),           # gold top-1
        _list([-2, -0.1, -3], [1, 0.5, 0], [1, 0, 0], 0),           # gold second, top-1 partial
        _list([-0.1, -2, -3], [0, 0, 0], [0, 0, 0], None, False),   # gold missed, top-1 disjoint
    ]
    s = C.summary(lists, "validation")
    assert s["recall@1"] == pytest.approx(1 / 3) and s["recall@2"] == pytest.approx(2 / 3)
    assert s["recall@3"] == pytest.approx(2 / 3) and "recall@4" not in s
    assert s["top1"] == pytest.approx({"correct": 1 / 3, "partial": 1 / 3, "disjoint": 1 / 3})


def test_summary_of_training_lists():
    s = C.summary([_list([0.0], [1], [1], 0, True), _list([0.0, 0.0], [1, 0], [1, 0], 0, False)], "train")
    assert s["sampled_gold"] == 0.5 and s["mean_size"] == 1.5 and "recall@1" not in s


# The real data path (tokenizer, windows, one softmax per question) with a random scorer.

def test_lists_on_real_questions_with_a_random_reader():
    torch = pytest.importorskip("torch")
    from squadqa import data as D
    from squadqa.config import load_config
    from squadqa.predict import score_questions

    cfg = load_config()
    try:
        tok = D.load_tokenizer(cfg["model"])
    except (OSError, RuntimeError) as e:
        pytest.skip(f"tokenizer not available: {e}")
    gen = torch.Generator().manual_seed(0)

    def random_reader(batch):
        w, n = batch["word_mask"].shape
        return torch.randn(w, n, n, generator=gen)

    train = D.load_split("train")
    long_ones = [q for q in train if len(q.context) > 2000][:20]          # several windows each
    for split, questions in (("train", train[:200] + long_ones), ("validation", D.load_split("validation")[:200])):
        feats = D.featurize(questions, tok, cfg, training=split == "train")
        if split == "train":
            assert any(len(ex.windows) > 1 for ex in feats.examples)
        spans = score_questions(random_reader, feats, tok.pad_token_id, cfg["max_answer_words"])
        lists = C.build_lists(feats.examples, spans, split, cfg, 1.3, seed=0)
        assert len(lists) == len(feats.examples)
        for ex in feats.examples:
            c = lists[ex.question.id]
            pairs = list(zip(c.starts.tolist(), c.ends.tolist()))
            assert len(pairs) == len(set(pairs))
            assert all(0 <= i <= j < len(ex.context.words) and j - i < cfg["max_answer_words"] for i, j in pairs)
            if split == "train":
                assert pairs[c.gold] == tuple(ex.gold) and c.em[c.gold] == 1
            else:
                assert len(pairs) == cfg["verifier"]["k_infer"]
        assert C.summary(list(lists.values()), split)["questions"] == len(lists)
