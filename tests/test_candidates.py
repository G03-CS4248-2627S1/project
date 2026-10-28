from types import SimpleNamespace

import numpy as np
import pytest

from squadqa import candidates as C
from squadqa.data import word_spans
from squadqa.decode import SpanScores, content_flags, group_cores, log_softmax

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


def test_lists_save_as_strict_json_with_and_without_a_gold():
    """A validation question whose gold isn't word-aligned has no labels; they save as null."""
    import json
    s = all_spans(len(WORDS))
    for gold in (None, (word("1889"),) * 2):
        out = json.loads(json.dumps(C.inference_list(s, 5, FLAGS, 1.0, gold).to_json(), allow_nan=False))
        assert len(out["spans"]) == len(out["f1"]) == len(out["em"]) == 5
        if gold is None:
            assert out["f1"] == [None] * 5 and out["em"] == [None] * 5
    t = C.training_list(s, (word("1889"),) * 2, 8, 2.0, np.random.default_rng(0), FLAGS, 1.0).to_json()
    assert all(isinstance(x, int) for x in t["em"]) and t["gold"] is not None


def example(qid, first, last=None, occurrence=0):
    """An Example-like object whose gold (or plausible answer) is the passage's words first..last."""
    i = [k for k, (s, e) in enumerate(WORDS) if CONTEXT[s:e] == first][occurrence]
    j = i if last is None else [k for k, (s, e) in enumerate(WORDS) if CONTEXT[s:e] == last and k >= i][0]
    q = SimpleNamespace(id=qid, context=CONTEXT, answers=((CONTEXT[WORDS[i][0]:WORDS[j][1]], WORDS[i][0]),))
    return SimpleNamespace(question=q, context=SimpleNamespace(words=WORDS), gold=(i, j))


def list_cfg(unit="span"):
    return {"verifier": {"k_train": 6, "sample_temperature": 2.0, "k_infer": 5, "list_unit": unit},
            "loss_utility": "normalized"}


@pytest.mark.parametrize("unit", ["span", "group"])
def test_build_lists_does_not_depend_on_the_batch(unit):
    exs = [example("a", "1889"), example("b", "Paris")]
    spans = [all_spans(len(WORDS), seed=1), all_spans(len(WORDS), seed=2)]
    both = C.build_lists(exs, spans, "train", list_cfg(unit), 1.0, seed=0)
    alone = C.build_lists(exs[1:], spans[1:], "train", list_cfg(unit), 1.0, seed=0)
    assert np.array_equal(both["b"].starts, alone["b"].starts) and np.array_equal(both["b"].ends, alone["b"].ends)
    other_seed = C.build_lists(exs, spans, "train", list_cfg(unit), 1.0, seed=1)
    assert any(not np.array_equal(both[q].starts, other_seed[q].starts) for q in "ab")


# Targets.

LIST_F1 = np.array([1.0, 1.0, 0.8, 0.4, 0.0])        # gold, a boundary variant, two partials, a miss


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_targets_sum_to_one(alpha, floor):
    t = C.targets(LIST_F1, 0, alpha, 0.1, floor)
    assert t.sum() == pytest.approx(1.0) and (t >= 0).all()


def test_one_hot_targets():
    assert C.targets(LIST_F1, 0, 0.0, 0.1, 0.5).tolist() == [1, 0, 0, 0, 0]


def test_f1_targets_follow_the_floor_and_the_temperature():
    t = C.targets(LIST_F1, 0, 1.0, 0.1, 0.5)
    assert t[3] == 0 and t[4] == 0 and t[0] == t[1] > t[2] > 0
    assert t[0] / t[2] == pytest.approx(1 / np.exp(-2.0))          # w(1) = 1 against w(0.8) = exp(-0.2 / 0.1)


def test_targets_blend_one_hot_and_f1():
    soft = C.targets(LIST_F1, 0, 1.0, 0.1, 0.5)
    assert C.targets(LIST_F1, 0, 0.3, 0.1, 0.5) == pytest.approx(0.7 * np.eye(5)[0] + 0.3 * soft)


def test_targets_reject_bad_settings():
    with pytest.raises(ValueError):
        C.targets(LIST_F1, 0, 1.5, 0.1, 0.5)
    with pytest.raises(ValueError):
        C.targets(LIST_F1, 0, 0.5, 0.0, 0.5)


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



def test_summary_keeps_recall_at_5_when_one_list_is_shorter():
    """A passage with only 3 valid spans used to drop recall@4 and @5 for every question."""
    full = _list([-0.1, -1, -2, -3, -4], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], 4)    # gold fifth
    short = _list([-0.1, -2, -3], [1, 0, 0], [1, 0, 0], 0)
    s = C.summary([full, short], "validation")
    assert s["recall@1"] == 0.5 and s["recall@4"] == 0.5 and s["recall@5"] == 1.0


def test_summary_refuses_empty_input():
    with pytest.raises(ValueError):
        C.summary([], "validation")
    with pytest.raises(ValueError, match="word-aligned"):
        C.summary([_list([-0.1], [np.nan], [np.nan], None, False)], "validation")


def test_main_refuses_a_reader_trained_on_extra_data(monkeypatch):
    from squadqa import predict as P
    monkeypatch.setattr(P, "load_reader", lambda r, device: ({"extra_train_data": ["adversarialqa"]}, None, None, 0))
    with pytest.raises(NotImplementedError, match="extra_train_data"):
        C.main(["--run-id", "x", "--split", "train"])

def test_recall_em_counts_a_boundary_variant_and_recall_does_not():
    """A reader trained per group often ranks "the X" above the annotated "X": a hit for recall_em
    (EM 1 under normalized) but not for recall (the annotated span by position)."""
    lists = [_list([-0.1, -2.0, -3.0], [1, 1, 0], [1, 1, 0], 1)]     # top-1 a variant, the gold second
    s = C.summary(lists, "validation")
    assert s["recall@1"] == 0 and s["recall_em@1"] == 1
    assert s["recall@2"] == 1 and s["recall_em@2"] == 1


def test_sampled_em_sees_a_variant_the_sample_found():
    s = all_spans(len(WORDS))
    gold = (word("World"), word("Fair"))
    variant = s.index((word("World") - 1, word("Fair")))                # "the World's Fair"
    only_variant = SpanScores(s.starts, s.ends, np.where(np.arange(len(s.scores)) == variant, 50.0, -50.0).astype(np.float32))
    c = C.training_list(only_variant, gold, 1, 1.0, np.random.default_rng(0), FLAGS, 1.0)
    assert not c.sampled_gold and c.sampled_em


def test_summary_of_training_lists():
    s = C.summary([_list([0.0], [1], [1], 0, True), _list([0.0, 0.0], [1, 0], [1, 0], 0, False)], "train")
    assert s["sampled_gold"] == 0.5 and s["mean_size"] == 1.5 and "recall@1" not in s and "sampled_em" in s


# The entry point end to end, as a cluster job runs it, with a tiny reader in place of the checkpoint.

@pytest.mark.parametrize("split, source, unit", [("validation", "squad", "span"), ("train", "squad", "span"),
                                                ("train", "squad2-unanswerable", "group")])
def test_main_writes_the_lists_and_summary(split, source, unit, tmp_path, monkeypatch):
    """Two jobs (1008523, 1008579) failed after scoring, while saving: once on int(NaN), once on a
    missing output folder. This runs main() itself, chunks and all, into a fresh scratch folder."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import json
    from squadqa import config as Cf, data as D, heads as H, predict as P

    cfg = Cf.load_config("configs/experiments/row1-joint.yaml")
    try:
        tok = D.load_tokenizer(cfg["model"])
    except (OSError, RuntimeError) as e:
        pytest.skip(f"tokenizer not available: {e}")
    torch.manual_seed(0)
    enc = transformers.AutoModel.from_config(transformers.DebertaV2Config(
        vocab_size=128100, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=512, relative_attention=True, position_buckets=256, pos_att_type=["p2c", "c2p"],
        type_vocab_size=0, position_biased_input=False))
    reader = H.Reader(enc, cfg)
    run_id = "fake-reader-run"
    (tmp_path / "results" / run_id).mkdir(parents=True)
    (tmp_path / "results" / run_id / "calibration.json").write_text('{"temperature": 1.3}')
    real_split = D.load_split
    monkeypatch.setattr(Cf, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(P, "load_reader", lambda r, device: (cfg, tok, reader.to(device), 123))
    monkeypatch.setattr(D, "load_split", lambda name: real_split(name)[:40])
    if source == "squad2-unanswerable":
        if not D.UNANSWERABLE.exists():
            pytest.skip("run python -m squadqa.download_external")
        real_unanswerable = D.load_unanswerable
        monkeypatch.setattr(D, "load_unanswerable", lambda: real_unanswerable()[:40])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(C, "VERIFIER_DATA", tmp_path / "data" / "verifier")     # doesn't exist yet

    C.main(["--run-id", run_id, "--split", split, "--source", source, "--set", f"verifier.list_unit={unit}",
            "--chunk", "16", "--batch-size", "8"])

    name = split if source == "squad" else f"{source}_{split}"
    out = tmp_path / "data" / "verifier" / run_id
    lists = json.loads((out / f"candidates_{name}.json").read_text())
    summary = json.loads((out / f"summary_{name}.json").read_text())
    assert summary["questions"] == len(lists) > 30 and summary["checkpoint_step"] == 123
    assert summary["run_id"] == run_id and summary["temperature"] == 1.3 and summary["source"] == source
    unanswerable = source == "squad2-unanswerable"
    assert ("recall@5" in summary) == (split == "validation")
    assert ("sampled_gold" in summary) == (split == "train" and not unanswerable)
    assert ("sampled_plausible" in summary) == unanswerable
    first = next(iter(lists.values()))
    keys = {"spans", "logp", "f1", "em", "gold", "sampled_em", "top1", *C.LABELS, "official_f1", "official_em", "sibling"}
    if unanswerable:
        keys |= {"unanswerable", "plausible", "sampled_plausible", "size"}
    else:
        keys |= {"sampled_gold"}
    assert set(first) == keys
    assert ("recall_em@5" in summary) == (split == "validation") and ("sampled_em" in summary) == (split == "train")
    assert ("reader_top1" in summary) == (split == "train")
    assert set(summary) >= {"git_commit", "git_dirty", "slurm_job_id"}


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


@pytest.mark.parametrize("key, value", [("train_candidates", "crossfit"), ("sibling_negatives", True),
                                        ("unanswerable_negatives", True)])
def test_build_lists_refuses_list_options_that_are_not_built(key, value):
    cfg = list_cfg()
    cfg["verifier"][key] = value
    with pytest.raises(NotImplementedError, match=key):
        C.build_lists([], [], "train", cfg, 1.0, 0)


def test_build_lists_refuses_an_unknown_unit_and_unanswerable_inference_lists():
    with pytest.raises(ValueError, match="list_unit"):
        C.build_lists([], [], "train", list_cfg("word"), 1.0, 0)
    with pytest.raises(ValueError, match="training lists only"):
        C.build_lists([], [], "validation", list_cfg(), 1.0, 0, unanswerable=True)


def test_a_gold_longer_than_any_candidate_still_labels_the_list():
    """A word-aligned gold over max_answer_words is never a candidate, but F1 / EM need only its words."""
    s, gold = all_spans(40), (2, 37)                     # 36 words, candidates at most 30
    assert s.index(gold) is None
    c = C.inference_list(s, 5, np.ones(40, bool), 1.0, gold)
    overlap = np.maximum(0, np.minimum(c.ends, 37) - np.maximum(c.starts, 2) + 1)
    assert c.f1 == pytest.approx(2 * overlap / (c.ends - c.starts + 1 + 36))
    assert c.gold is None and not c.sampled_gold and (c.em == 0).all()
    summ = C.summary([c], "validation")
    assert summ["with_gold"] == 1 and summ["recall@1"] == summ["recall_em@1"] == 0


# Labels (spec §5.1): between content cores, so the same for every member of a group.

def only(spans, scores=None):
    """A SpanScores whose valid spans are exactly these, the first the most probable by default."""
    s, e = zip(*spans)
    sc = -np.arange(len(spans), dtype=np.float32) if scores is None else np.asarray(scores, np.float32)
    return SpanScores(np.array(s), np.array(e), sc)


def labelled(spans, gold):
    c = C.inference_list(only(spans), len(spans), FLAGS, 1.0, gold)
    return {(int(i), int(j)): {k: v[n] for k, v in c.labels.items()} | {"f1": c.f1[n], "em": c.em[n]}
            for n, (i, j) in enumerate(zip(c.starts, c.ends))}


def test_labels_against_the_reference():
    built, nineteen = word("built"), word("1889")                # "built in 1889" = words 3-5
    got = labelled([(4, 4), (5, 6), (2, 5), (13, 13), (3, 5)], (built, nineteen))
    assert got[(4, 4)]["relation"] == "inside"
    assert got[(4, 4)]["precision"] == 1 and got[(4, 4)]["recall"] == pytest.approx(1 / 3)
    assert (got[(4, 4)]["start_shift"], got[(4, 4)]["end_shift"], got[(4, 4)]["kind"]) == (1, -1, "boundary")
    assert got[(5, 6)]["relation"] == "crosses" and (got[(5, 6)]["start_shift"], got[(5, 6)]["end_shift"]) == (2, 1)
    assert got[(2, 5)]["relation"] == "contains" and got[(2, 5)]["recall"] == 1 and got[(2, 5)]["precision"] == 0.75
    assert got[(13, 13)]["relation"] == "disjoint" and got[(13, 13)]["kind"] == "wrong" and got[(13, 13)]["gap"] == 13 - 5 - 1
    assert got[(3, 5)]["relation"] == "same" and got[(3, 5)]["kind"] == "correct" and got[(3, 5)]["gap"] == 0
    # "The tower" has the core "tower" (word 1): its gap to "Paris" is counted from there
    assert labelled([(0, 1)], (word("Paris"),) * 2)[(0, 1)]["gap"] == 13 - 1 - 1


def test_a_boundary_variant_is_the_same_group_with_no_shift():
    got = labelled([(7, 11), (8, 11)], (word("World"), word("Fair")))
    for span in ((7, 11), (8, 11)):
        assert got[span]["relation"] == "same" and got[span]["kind"] == "correct"
        assert (got[span]["start_shift"], got[span]["end_shift"]) == (0, 0)


def test_labels_are_the_same_for_every_member_of_a_group():
    s = all_spans(len(WORDS))
    c = C.inference_list(s, len(s.scores), FLAGS, 1.0, (word("built"), word("1889")))
    cores = list(zip(*(x.tolist() for x in group_cores(FLAGS, c.starts, c.ends))))
    keys = [*C.LABELS, "official_f1"] if "official_f1" in c.labels else list(C.LABELS)
    by_core = {}
    for n, core in enumerate(cores):
        row = (c.f1[n], c.em[n], *(c.labels[k][n] for k in keys))
        assert by_core.setdefault(core, row) == row


def test_labels_without_a_reference_are_null():
    c = C.inference_list(all_spans(6), 3, np.ones(6, bool), 1.0)
    assert all(v == [None] * 3 for v in c.labels.values()) and c.top1["kind"] is None


# Group lists.

@pytest.mark.parametrize("gold", [(8, 11), (5, 5)])
def test_group_training_lists_hold_distinct_groups_and_the_gold_group(gold):
    s = all_spans(len(WORDS))
    logp = log_softmax(s.scores, 1.3)
    cs, ce = group_cores(FLAGS, s.starts, s.ends)
    for seed in range(20):
        c = C.group_list(s, 6, FLAGS, 1.3, gold, np.random.default_rng(seed), 2.0)
        cores = list(zip(*(x.tolist() for x in group_cores(FLAGS, c.starts, c.ends))))
        assert len(cores) == len(set(cores)) and 6 <= len(cores) <= 7
        assert c.gold is not None and c.em[c.gold] == 1 and c.labels["relation"][c.gold] == "same"
        for n, (a, b) in enumerate(cores):
            members = (cs == a) & (ce == b)
            assert c.logp[n] == pytest.approx(np.logaddexp.reduce(logp[members]))
            assert c.sizes[n] == members.sum()
            assert s.scores[s.index((int(c.starts[n]), int(c.ends[n])))] == s.scores[members].max()


def test_group_sampling_weights_are_the_groups_tempered_probabilities():
    s = all_spans(len(WORDS), seed=3)
    gid, rep, _, _, _ = C.span_groups(s, FLAGS)
    w = C.group_logsumexp(s.scores / 2.0, gid, len(rep))
    assert np.exp(log_softmax(w)) == pytest.approx(np.bincount(gid, weights=np.exp(log_softmax(s.scores, 2.0))))


def test_group_lists_sample_from_the_tempered_group_distribution():
    """k = 1 with no reference is one draw of a group. Every word counts here, so each span is its
    own group and the draw follows softmax(scores / sample_temperature)."""
    s = only([(0, 0), (1, 1), (2, 2)], [0.0, 1.0, 2.0])
    flags = np.ones(3, bool)
    for t in (1.0, 3.0):
        rng = np.random.default_rng(5)
        draws = [int(C.group_list(s, 1, flags, 1.0, None, rng, t).starts[0]) for _ in range(4000)]
        freq = np.bincount(draws, minlength=3) / len(draws)
        assert freq == pytest.approx(np.exp(log_softmax(s.scores, t)), abs=0.03)


def test_group_inference_lists_are_the_top_k_groups():
    s = all_spans(len(WORDS), seed=4)
    gid, rep, _, _, _ = C.span_groups(s, FLAGS)
    logp = C.group_logsumexp(log_softmax(s.scores, 1.3), gid, len(rep))
    c = C.group_list(s, 5, FLAGS, 1.3, (5, 5))
    assert sorted(c.logp.tolist(), reverse=True) == pytest.approx(np.sort(logp)[::-1][:5])
    assert c.top1["p"] == pytest.approx(np.exp(logp.max()))


# Unanswerable lists (SQuAD 2.0): the plausible answer in place of the gold.

@pytest.mark.parametrize("unit", ["span", "group"])
def test_unanswerable_lists_hold_the_plausible_answer_and_no_gold(unit):
    ex = example("u", "Paris")
    siblings = C.sibling_index([example("a", "Paris").question])      # an answerable question with that answer
    c = C.build_lists([ex], [all_spans(len(WORDS))], "train", list_cfg(unit), 1.0, 0, unanswerable=True,
                      siblings=siblings)["u"]
    pairs = list(zip(c.starts.tolist(), c.ends.tolist()))
    assert c.unanswerable and c.gold is None and c.em[c.plausible] == 1 and c.labels["relation"][c.plausible] == "same"
    if unit == "span":                                # a group list shows the group's most probable member
        assert pairs[c.plausible] == ex.gold
    j = c.to_json()
    assert j["gold"] is None and j["unanswerable"] and j["plausible"] == c.plausible
    assert "sampled_plausible" in j and "sampled_gold" not in j
    s = C.summary([c], "train")
    assert "sampled_plausible" in s and "sampled_gold" not in s and s["reader_top1"]["kind"]
    assert s["others"]["candidates"] == len(pairs) - 1 and s["reference_sibling"] == 1


def test_an_unanswerable_question_without_an_aligned_plausible_answer_has_no_labels():
    c = C.unanswerable_list(all_spans(len(WORDS)), None, 5, 2.0, np.random.default_rng(0), FLAGS, 1.0)
    assert c.plausible is None and len(c.starts) == 5 and np.isnan(c.f1).all()


# Labels that need the texts: the official scorer and siblings.

def test_official_labels_accept_the_same_text_elsewhere_and_em_does_not():
    ex = example("q", "in")                                       # the gold is the first "in" (word 4)
    s = only([(4, 4), (12, 12), (13, 13)])
    c = C.build_lists([ex], [s], "validation", list_cfg(), 1.0, 0)["q"]
    pairs = list(zip(c.starts.tolist(), c.ends.tolist()))
    k = pairs.index((12, 12))
    assert c.labels["official_em"][k] == 1 and c.em[k] == 0 and c.labels["relation"][k] == "disjoint"
    assert c.labels["official_em"][pairs.index((13, 13))] == 0 and c.top1["official_em"] == 1


def test_sibling_marks_another_questions_gold_on_the_passage():
    ex = example("q", "tower")
    siblings = C.sibling_index([SimpleNamespace(id="other", context=CONTEXT, answers=(("Paris", CONTEXT.index("Paris")),)),
                                ex.question])
    s = only([(1, 1), (13, 13), (0, 1)])
    c = C.build_lists([ex], [s], "validation", list_cfg(), 1.0, 0, siblings=siblings)["q"]
    pairs = list(zip(c.starts.tolist(), c.ends.tolist()))
    assert c.labels["sibling"][pairs.index((13, 13))]
    assert not c.labels["sibling"][pairs.index((1, 1))]           # its own gold is not a sibling
    assert not c.labels["sibling"][pairs.index((0, 1))] and c.top1["sibling"] is False
