import json

import numpy as np
import pytest

from squadqa import data as D
from squadqa.config import load_config

CFG = load_config()


# Words and answers: no tokenizer needed.

def test_word_rule():
    text = "U.S. $5,000 well-known Beyoncé's"
    words = [text[s:e] for s, e in D.word_spans(text)]
    assert words == ["U", ".", "S", ".", "$", "5", ",", "000", "well", "-", "known", "Beyoncé", "'", "s"]


def test_token_spans_strip_the_leading_space():
    assert D.token_spans("Who won", [(0, 3), (3, 7)]) == [(0, 3), (4, 7)]
    assert D.token_spans("a  b", [(1, 4)]) == [(3, 4)]


def test_answer_located_by_answer_start_not_by_search():
    text = "the cat and the cat"
    words = D.word_spans(text)
    assert D.answer_word_span(words, 16, "cat") == (4, 4)          # the second "cat", as annotated
    assert D.answer_word_span(words, 4, "cat") == (1, 1)
    assert D.answer_word_span(words, 0, "the cat and") == (0, 2)


@pytest.mark.parametrize("start, text", [(5, "at"), (4, "ca"), (3, " cat"), (4, "cat "), (19, "x")])
def test_misaligned_answers_are_none(start, text):
    words = D.word_spans("the cat and the cat")
    assert D.answer_word_span(words, start, text) is None


WINDOW_CASES = [(10, 20, 5), (20, 20, 5), (21, 20, 5), (24, 20, 5), (26, 20, 5), (100, 20, 5), (101, 20, 19),
                (367, 366, 128), (494, 366, 128), (495, 366, 128), (1000, 366, 128), (2000, 366, 128)]


@pytest.mark.parametrize("n, budget, stride", WINDOW_CASES)
def test_even_windows_are_full_spread_evenly_and_as_few_as_the_stride_allows(n, budget, stride):
    ranges = D.window_ranges(n, budget, stride, "even")
    assert ranges[0][0] == 0 and ranges[-1][1] == n
    assert all(b - a == min(budget, n) for a, b in ranges)          # every window full
    steps = [a1 - a0 for (a0, _), (a1, _) in zip(ranges, ranges[1:])]
    assert all(0 < s <= stride for s in steps)
    assert not steps or max(steps) - min(steps) <= 1                 # spread evenly
    assert len(ranges) == 1 or n - budget > stride * (len(ranges) - 2)   # one window fewer would need a longer step
    assert all(b0 - a1 >= budget - stride for (_, b0), (a1, _) in zip(ranges, ranges[1:]))


@pytest.mark.parametrize("n, budget, stride", WINDOW_CASES)
def test_left_windows_overlap_by_the_stride(n, budget, stride):
    ranges = D.window_ranges(n, budget, stride, "left")
    assert ranges[0][0] == 0 and ranges[-1][1] == n
    assert all(b - a <= budget for a, b in ranges)
    assert all(b0 - a1 == stride for (_, b0), (a1, _) in zip(ranges, ranges[1:]))
    assert len(ranges) == 1 or ranges[-2][1] < n                     # no window is redundant


@pytest.mark.parametrize("budget, stride, layout", [(5, 5, "even"), (5, 0, "even"), (5, 5, "left"), (20, 5, "right")])
def test_window_ranges_reject_bad_settings(budget, stride, layout):
    with pytest.raises(ValueError):
        D.window_ranges(10, budget, stride, layout)


# Splits.

def test_splits_are_disjoint_by_article():
    train, val = D.load_split("train"), D.load_split("validation")
    assert len(val) == 7708 and len(train) == 87599 - 7708
    assert not {q.title for q in train} & {q.title for q in val}
    with pytest.raises(KeyError):
        D.load_split("dev")


def test_adversarialqa_loader_drops_validation_articles():
    if not D.EXTRA_TRAIN["adversarialqa"].exists():
        pytest.skip("run python -m squadqa.download_external")
    train = D.load_split("train", ["adversarialqa"])
    val_titles = {q.title for q in D.load_split("validation")}
    assert not {q.title for q in train} & val_titles
    assert len(train) > 87599 - 7708


def test_dev_passage_raises_even_with_other_whitespace():
    dev = D.load_squad(D.DEV)[0]
    q = D.Question("x", "Not a dev title", "  " + dev.context.replace(" ", "\n", 3), "?", (("a", 0),))
    with pytest.raises(RuntimeError, match="dev-v1.1 passages"):
        D.check_no_dev_passages([q])


def test_skipped_ids_file():
    ids = D.SKIPPED_IDS.read_text(encoding="utf-8").split("\n")
    assert ids[-1] == "" and ids[:-1] == sorted(set(ids[:-1])) and len(ids) - 1 == 343


# Batching.

def test_bucket_batches():
    lengths = list(np.random.default_rng(0).integers(50, 384, size=1000))
    batches = D.bucket_batches(lengths, 32, seed=1, epoch=0, pool=5)
    assert sorted(i for b in batches for i in b) == list(range(1000))
    assert all(len(b) <= 32 for b in batches)
    assert batches == D.bucket_batches(lengths, 32, seed=1, epoch=0, pool=5)
    assert batches != D.bucket_batches(lengths, 32, seed=1, epoch=1, pool=5)
    spread = np.mean([max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches])
    assert spread < 100                                              # random batches spread about 300


# With the pinned tokenizer.

@pytest.fixture(scope="module")
def tok():
    try:
        return D.load_tokenizer(CFG["model"])
    except (OSError, RuntimeError) as e:
        pytest.skip(f"tokenizer not available: {e}")


@pytest.fixture(scope="module")
def val_feats(tok):
    return D.featurize(D.load_split("validation"), tok, CFG, training=False)


@pytest.fixture(scope="module")
def val_feats_left(tok):
    return D.featurize(D.load_split("validation"), tok, {**CFG, "window_layout": "left"}, training=False)


def test_pair_layout_is_cls_sep_sep(tok):
    assert D.pair_layout(tok) == ([tok.cls_token_id], [tok.sep_token_id], [tok.sep_token_id])


def test_round_trip_on_validation(val_feats):
    """chars -> words -> tokens -> chars gives the answer text back, through every window that
    holds it (the full check over train is the reference test, issue #19)."""
    checked = 0
    for ex in val_feats.examples:
        if ex.gold is None:
            continue
        text, start = ex.question.answers[0]
        ctx = ex.context
        i, j = ex.gold
        assert ex.question.context[ctx.spans[ctx.word_first[i]][0]:ctx.spans[ctx.word_last[j]][1]] == text
        wins = [val_feats.windows[w] for w in ex.windows if val_feats.windows[w].gold is not None]
        assert wins, ex.question.id
        for win in wins:                                          # the window's tokens are the passage's
            gi, gj = win.gold
            assert win.first_word + gi == i and win.first_word + gj == j
            ids = win.input_ids[win.first_tok[gi]:win.last_tok[gj] + 1].tolist()
            assert ids == ctx.ids[ctx.word_first[i]:ctx.word_last[j] + 1]
        checked += 1
    assert checked > 7600


def test_even_windows_on_validation(val_feats, val_feats_left):
    """Every window of a question has the same length (all full), the first starts at the passage's
    first word and the last ends at its last, and no question has fewer windows than under left."""
    multi = 0
    for ex, ex_left in zip(val_feats.examples, val_feats_left.examples):
        wins = [val_feats.windows[w] for w in ex.windows]
        assert len({len(w.input_ids) for w in wins}) == 1
        assert wins[0].first_word == 0 and wins[-1].first_word + wins[-1].n_words == len(ex.context.words)
        assert len(wins) >= len(ex_left.windows)
        multi += len(wins) > 1
    assert 50 < multi < 150


def test_left_windows_match_the_tokenizer_where_it_is_right(tok, val_feats_left):
    """With window_layout left, our windows equal return_overflowing_tokens' windows (its stride is
    the overlap) wherever tokenizers 0.23.2 covers the whole passage. Where it does not (at most two
    windows, the second sometimes cut short), each of its windows is the start of ours."""
    exs = val_feats_left.examples
    enc = tok([ex.question.question for ex in exs], [ex.question.context for ex in exs], truncation="only_second",
              max_length=CFG["max_seq_len"], stride=CFG["doc_stride"], return_overflowing_tokens=True)
    theirs: dict[int, list] = {}
    for k, e in enumerate(enc["overflow_to_sample_mapping"]):
        theirs.setdefault(e, []).append(enc["input_ids"][k])
    same = cut = 0
    for e, ex in enumerate(exs):
        ours = [val_feats_left.windows[w].input_ids.tolist() for w in ex.windows]
        if ours == theirs[e]:
            same += 1
            continue
        cut += 1
        assert len(theirs[e]) <= 2 and len(theirs[e]) <= len(ours)
        for t, o in zip(theirs[e], ours):
            assert o[:len(t) - 1] == t[:-1]                      # same start; theirs ends early
    assert same == len(exs) - cut and 0 < cut < 100
    print(f"{cut} of {len(exs)} validation questions lose passage text with return_overflowing_tokens")


def test_long_passage_windows(tok):
    context = " ".join(f"w{i}." for i in range(800))
    q = D.Question("q", "T", context, "Which word?", (("w700", context.index("w700.")),))
    f = D.featurize([q], tok, CFG, training=True)
    wins = f.windows
    assert len(wins) > 2
    assert all(len(w.input_ids) <= CFG["max_seq_len"] for w in wins)
    assert len({len(w.input_ids) for w in wins}) == 1                  # even layout: the last window is full too
    covered = set()
    for w in wins:
        covered |= set(range(w.first_word, w.first_word + w.n_words))
        ids = w.input_ids.tolist()
        assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
        assert ids[w.ctx_start - 1] == tok.sep_token_id
    assert covered == set(range(len(f.examples[0].context.words)))
    assert any(w.gold is not None for w in wins)


def test_training_skips_and_keeps(tok):
    context = "The quick brown fox jumps over the lazy dog."
    qs = [D.Question("ok", "T", context, "What?", (("brown fox", 10),)),
          D.Question("misaligned", "T", context, "What?", (("rown", 11),)),
          D.Question("long", "T", " ".join(["x"] * 40), "What?", ((" ".join(["x"] * 31), 0),))]
    f = D.featurize(qs, tok, CFG, training=True)
    assert [ex.question.id for ex in f.examples] == ["ok"]
    assert f.skipped == {"misaligned": "not word-aligned", "long": "longer than 30 words"}
    assert f.windows[0].gold == (2, 3)
    f = D.featurize(qs, tok, CFG, training=False)
    assert [ex.gold for ex in f.examples] == [(2, 3), None, (0, 30)]


def test_collate(tok, val_feats):
    torch = pytest.importorskip("torch")
    multi = next(e for e, ex in enumerate(val_feats.examples) if len(ex.windows) > 1)
    rows = val_feats.batch([0, multi])
    batch = D.collate(rows, tok.pad_token_id)
    W = len(rows)
    assert batch["input_ids"].shape[0] == W and batch["example"].tolist() == [0] + [1] * (W - 1)
    for r, (_, w) in enumerate(rows):
        n, T = w.n_words, len(w.input_ids)
        assert batch["attention_mask"][r].sum() == T
        assert (batch["input_ids"][r, T:] == tok.pad_token_id).all()
        assert batch["word_mask"][r].sum() == n
        assert batch["first_tok"][r, :n].tolist() == w.first_tok.tolist()
        assert (batch["gold_start"][r].item(), batch["gold_end"][r].item()) == (w.gold or (-1, -1))
    assert batch["first_tok"].dtype == torch.long
