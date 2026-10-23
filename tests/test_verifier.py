import numpy as np
import pytest

from squadqa import data as D
from squadqa import verifier as V
from squadqa.config import load_config

CFG = load_config()


# Marking: no tokenizer needed.

def test_mark_round_trip_and_marker_offsets():
    context = "The Eiffel Tower was built in 1889 for the World's Fair."
    words = D.word_spans(context)
    i = j = [context[s:e] for s, e in words].index("1889")
    m = V.mark(context, words, i, j, 0, len(context))
    assert m.text == "The Eiffel Tower was built in  <a> 1889 </a>  for the World's Fair."
    assert V.unmark(m.text) == context
    assert V.marked_span(m.text) == "1889"
    assert m.text[m.open_at:m.open_at + len(V.OPEN)] == V.OPEN
    assert m.text[m.close_at:m.close_at + len(V.CLOSE)] == V.CLOSE


def test_mark_at_the_edges_of_the_window():
    context = "Paris. Rome"
    words = D.word_spans(context)                       # Paris . Rome
    first = V.mark(context, words, 0, 0, 0, len(context))
    last = V.mark(context, words, 2, 2, 0, len(context))
    assert first.text.startswith(" <a> Paris </a> ") and last.text.endswith(" <a> Rome </a> ")
    for m, cand in ((first, "Paris"), (last, "Rome")):
        assert V.unmark(m.text) == context and V.marked_span(m.text) == cand


def test_mark_crops_to_the_window():
    context = "one two three four five"
    words = D.word_spans(context)
    lo, hi = words[1][0], words[3][1]                   # "two three four"
    m = V.mark(context, words, 2, 2, lo, hi)
    assert V.unmark(m.text) == "two three four" and V.marked_span(m.text) == "three"


def test_mark_refuses_a_passage_with_a_marker_and_a_candidate_outside_the_window():
    context = "a <a> b"
    with pytest.raises(ValueError, match="already contains"):
        V.mark(context, D.word_spans(context), 0, 0, 0, len(context))
    context = "one two three"
    words = D.word_spans(context)
    with pytest.raises(ValueError, match="not inside the window"):
        V.mark(context, words, 0, 0, words[1][0], len(context))


def test_unmark_needs_one_of_each_marker_in_order():
    with pytest.raises(ValueError):
        V.unmark("no markers")
    with pytest.raises(ValueError):
        V.unmark("x </a>  y  <a> z")


def _window(first_word, ctx_start, ctx_end, first_tok):
    """One token per word, so first_tok is also last_tok."""
    first_tok = np.asarray(first_tok)
    return D.Window(0, np.zeros(1, dtype=np.int32), first_word, first_tok, first_tok, ctx_start, ctx_end)


def test_candidate_window_takes_the_most_central_window():
    ctx = D.ContextTokens([(2 * k, 2 * k + 1) for k in range(10)], [], [], np.arange(10), np.arange(10))
    a = _window(0, 3, 8, np.arange(3, 9))               # words 0-5 at positions 3-8
    b = _window(3, 3, 9, np.arange(3, 10))              # words 3-9 at positions 3-9
    # word 4: left/right of 4 and 1 in a (min 1), 1 and 5 in b (min 1): a tie, so the earlier window
    assert V.candidate_window([a, b], ctx, 4, 4) == (ctx.words[0][0], ctx.words[5][1])
    # word 5: 5 and 0 in a (min 0), 2 and 4 in b (min 2)
    assert V.candidate_window([a, b], ctx, 5, 5) == (ctx.words[3][0], ctx.words[9][1])
    with pytest.raises(ValueError, match="no window"):
        V.candidate_window([a, b], ctx, 1, 8)            # in neither window whole


class _StubModel:
    def __init__(self, rows):
        self.rows = rows

    def get_input_embeddings(self):
        return type("E", (), {"weight": np.zeros((self.rows, 2))})()

    def resize_token_embeddings(self, n):
        self.rows = n


def test_resize_for_markers_grows_but_never_shrinks():
    spare = _StubModel(128100)
    assert V.resize_for_markers(spare, range(128003)) == 0 and spare.rows == 128100
    tight = _StubModel(50368)
    assert V.resize_for_markers(tight, range(50370)) == 2 and tight.rows == 50370


# V5 with each variant's tokenizer (spec §5.7), on validation questions only.

TOKENIZERS = [CFG["model"], CFG["verifier"]["separate_model"]]


@pytest.fixture(scope="module")
def reader_tok():
    try:
        return D.load_tokenizer(CFG["model"])
    except (OSError, RuntimeError) as e:
        pytest.skip(f"tokenizer not available: {e}")


@pytest.fixture(scope="module")
def cases(reader_tok):
    """(question, context, words, i, j, lo, hi) for the gold span and its boundary variants on the
    first 300 validation questions, plus every multi-window question, with the window chosen by the
    reader's tokenizer as both variants would get it."""
    layout = D.pair_layout(reader_tok)
    out, multi, ctx_cache = [], 0, {}
    for k, q in enumerate(D.load_split("validation")):
        if q.context not in ctx_cache:
            ctx_cache[q.context] = D.context_tokens(q.context, reader_tok)
        ctx = ctx_cache[q.context]
        q_ids = reader_tok(q.question, add_special_tokens=False)["input_ids"]
        windows = D.make_windows(q_ids, ctx, layout, CFG["max_seq_len"], CFG["doc_stride"],
                                 window_layout=CFG["window_layout"])
        if k >= 300 and len(windows) == 1:
            continue
        multi += len(windows) > 1
        text, start = q.answers[0]
        gold = D.answer_word_span(ctx.words, start, text)
        if gold is None:
            continue
        i, j = gold
        n = len(ctx.words)
        for a, b in {(i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}:
            if 0 <= a <= b < n:
                try:
                    lo, hi = V.candidate_window(windows, ctx, a, b)
                except ValueError:
                    continue                            # a variant reaching past every window
                out.append((q.question, q.context, ctx.words, a, b, lo, hi))
    assert multi > 0, "the sample has no multi-window question"
    return out


@pytest.mark.parametrize("name", TOKENIZERS)
def test_v5_marker_round_trip(name, cases):
    try:
        tok = D.load_tokenizer(name)
    except (OSError, RuntimeError) as e:
        pytest.skip(f"tokenizer not available: {e}")
    open_id, close_id = V.add_markers(tok)
    assert V.add_markers(tok) == (open_id, close_id)    # registering twice changes nothing
    for question, context, words, i, j, lo, hi in cases:
        cand = context[words[i][0]:words[j][1]]
        m = V.mark(context, words, i, j, lo, hi)
        assert V.unmark(m.text) == context[lo:hi]
        assert V.marked_span(m.text) == cand
        enc = V.encode(tok, question, m, open_id, close_id)
        assert 0 < enc.open_pos < enc.close_pos
        # the tokens between the markers cover exactly the candidate's characters
        offsets = tok(question, m.text, return_offsets_mapping=True)["offset_mapping"]
        inner = m.text[offsets[enc.open_pos][1]:offsets[enc.close_pos][0]]
        assert inner.strip() == cand
