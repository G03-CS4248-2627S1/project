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


# The shared_lora verifier (spec §5.4) on a tiny random reader, CPU only.

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from squadqa import heads as H                                          # noqa: E402

V_CFG = load_config(None, ["train.head_dropout=0.0", "verifier.lora.rank=4", "verifier.lora.alpha=8"])


def tiny_reader(seed=0):
    torch.manual_seed(seed)
    config = transformers.DebertaV2Config(
        vocab_size=128100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=512, relative_attention=True, position_buckets=256, pos_att_type=["p2c", "c2p"],
        type_vocab_size=0, position_biased_input=False, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    reader = H.Reader(transformers.AutoModel.from_config(config), V_CFG)
    with torch.no_grad():                                               # a "trained" pair term, not zero
        reader.head.W.normal_(0, 0.1)
    return reader.eval()


@pytest.fixture(scope="module")
def marked(reader_tok):
    """A reader batch for 4 validation questions, and a marked batch of each one's gold span and
    the span one word longer: 8 candidates, 2 per question, the gold first."""
    open_id, close_id = V.add_markers(reader_tok)
    qs = D.load_split("validation")[:4]
    feats = D.featurize(qs, reader_tok, V_CFG, training=False)
    reader_batch = D.collate(feats.batch(list(range(len(qs)))), reader_tok.pad_token_id)
    layout = D.pair_layout(reader_tok)
    encs, question = [], []
    for k, ex in enumerate(feats.examples):
        q_ids = reader_tok(ex.question.question, add_special_tokens=False)["input_ids"]
        windows = D.make_windows(q_ids, ex.context, layout, V_CFG["max_seq_len"], V_CFG["doc_stride"],
                                 window_layout=V_CFG["window_layout"])
        i, j = ex.gold
        for a, b in ((i, j), (i, j + 1)):
            lo, hi = V.candidate_window(windows, ex.context, a, b)
            m = V.mark(ex.question.context, ex.context.words, a, b, lo, hi)
            encs.append(V.encode(reader_tok, ex.question.question, m, open_id, close_id))
            question.append(k)
    batch = V.collate_marked(encs, reader_tok.pad_token_id)
    return open_id, close_id, reader_batch, batch, torch.tensor(question)


def _train(v, batch, question, steps=5):
    opt = torch.optim.AdamW(v.trainable_parameters(), lr=1e-2, weight_decay=0.01)
    targets = torch.tensor([1.0, 0.0] * (len(question) // 2))
    v.train()
    for _ in range(steps):
        opt.zero_grad()
        V.listwise_loss(v(batch), question, targets, int(question.max()) + 1).mean().backward()
        opt.step()
    v.eval()


def _base_params(reader):
    """The reader's own parameters by their pre-LoRA names (LoRA renames a wrapped layer's weight
    to <layer>.base_layer.weight) and without the adapters' own."""
    return {n.replace(".base_layer.", "."): p for n, p in reader.named_parameters() if "lora_" not in n}


def test_v4_reader_unchanged_by_verifier_training(marked):
    open_id, close_id, reader_batch, batch, question = marked
    reader = tiny_reader()
    with torch.no_grad():
        before_scores = reader.span_scores(reader_batch)[0].clone()
    before_params = {n: p.detach().clone() for n, p in reader.named_parameters()}
    v = V.SharedLoraVerifier(reader, open_id, close_id, V_CFG)
    with torch.no_grad():
        scores0 = v(batch).clone()
    _train(v, batch, question)
    with torch.no_grad():
        after_scores = v.reader_span_scores(reader_batch)[0]
        scores1 = v(batch)
    assert torch.equal(after_scores, before_scores)                     # V4: identical reader predictions
    after_params = _base_params(v.reader)
    assert set(after_params) == set(before_params)
    for n, p in before_params.items():                                  # every reader weight, the
        assert torch.equal(after_params[n], p), n                       # embedding table included
    assert not torch.allclose(scores1, scores0)                         # and training did change the verifier


def test_only_lora_markers_and_head_train(marked):
    open_id, close_id, *_ = marked
    v = V.SharedLoraVerifier(tiny_reader(), open_id, close_id, V_CFG)
    names = [n for n, p in v.named_parameters() if p.requires_grad]
    assert all("lora_" in n or n == "markers" or n.startswith("head.") for n in names)
    assert any("lora_" in n for n in names) and "markers" in names
    assert sum(p.numel() for p in v.trainable_parameters()) < 0.1 * sum(p.numel() for p in v.parameters())


def test_lora_starts_as_the_reader_and_the_markers_learn(marked):
    open_id, close_id, _, batch, question = marked
    v = V.SharedLoraVerifier(tiny_reader(), open_id, close_id, V_CFG)
    enc = v.reader.encoder
    with torch.no_grad():
        on = enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        with V.adapters(enc, False):
            off = enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    assert torch.equal(on, off)                                         # LoRA's B starts at zero
    V.listwise_loss(v(batch), question, torch.tensor([1.0, 0.0] * 4), 4).mean().backward()
    assert v.markers.grad is not None and v.markers.grad.abs().sum() > 0


def test_the_markers_replace_their_rows_of_the_embedding_table(marked):
    open_id, close_id, _, batch, _ = marked
    v = V.SharedLoraVerifier(tiny_reader(), open_id, close_id, V_CFG)
    with torch.no_grad():
        v.markers.copy_(torch.stack([torch.full((32,), 3.0), torch.full((32,), -3.0)]))
        out = v.reader.encoder.get_input_embeddings()(batch["input_ids"])
    b = torch.arange(len(out))
    assert torch.equal(out[b, batch["open_pos"]], torch.full((len(b), 32), 3.0))
    assert torch.equal(out[b, batch["close_pos"]], torch.full((len(b), 32), -3.0))


def test_adapters_come_back_on_after_an_error(marked):
    open_id, close_id, _, batch, question = marked
    v = V.SharedLoraVerifier(tiny_reader(), open_id, close_id, V_CFG)
    _train(v, batch, question, steps=2)
    with torch.no_grad():
        on = v(batch)
        with pytest.raises(KeyError), V.adapters(v.reader.encoder, False):
            raise KeyError
        assert torch.equal(v(batch), on)


def test_the_verifier_refuses_bad_setups(marked):
    open_id, close_id, *_ = marked
    bad = load_config(None, ["verifier.lora.targets=[no_such_module]"])
    with pytest.raises(Exception, match="not found in the base model"):
        V.SharedLoraVerifier(tiny_reader(), open_id, close_id, {**V_CFG, "verifier": bad["verifier"]})
    with pytest.raises(ValueError, match="outside the embedding table"):
        V.SharedLoraVerifier(tiny_reader(), 128100, 128101, V_CFG)


# The listwise loss and the combination.

def test_listwise_loss_matches_brute_force():
    scores = torch.tensor([2.0, 0.5, -1.0, 0.3, 1.2])
    question = torch.tensor([0, 0, 0, 1, 1])
    targets = torch.tensor([0.7, 0.3, 0.0, 0.0, 1.0])
    got = V.listwise_loss(scores, question, targets, 2)
    first = torch.log_softmax(scores[:3], 0)
    want = [-(0.7 * first[0] + 0.3 * first[1]), -torch.log_softmax(scores[3:], 0)[1]]
    assert got.tolist() == pytest.approx([float(w) for w in want])


def test_listwise_loss_ignores_masked_candidates_with_zero_target():
    scores = torch.tensor([1.0, float("-inf"), 0.0])
    got = V.listwise_loss(scores, torch.tensor([0, 0, 0]), torch.tensor([1.0, 0.0, 0.0]), 1)
    assert torch.isfinite(got).all()


def test_combine_and_fit_lambda():
    rng = np.random.default_rng(0)
    helpful, useless = [], []
    for _ in range(200):
        gold = int(rng.integers(5))
        reader = np.log(rng.dirichlet(np.ones(5)))
        helpful.append((reader, np.where(np.arange(5) == gold, 2.0, 0.0) + rng.normal(0, 0.3, 5), gold))
        confident = np.where(np.arange(5) == gold, np.log(0.9), np.log(0.025))
        useless.append((confident, rng.normal(0, 1, 5), gold))
    r, s, _ = helpful[0]
    assert V.combine(r, s, 0.0) == pytest.approx(r - np.logaddexp.reduce(r))   # λ = 0: the reader alone
    assert V.combine(r, s, 1.0, 2.0) == pytest.approx(np.log(np.exp((r + s) / 2) / np.exp((r + s) / 2).sum()))
    assert V.fit_lambda(helpful)[0] > 0.5                               # a verifier that knows the answer counts
    assert V.fit_lambda(useless)[0] <= 0.2                              # noise, against a sure reader, barely does
    with pytest.raises(ValueError):
        V.fit_lambda([])


def test_fit_lambda_cools_an_overconfident_sum_instead_of_dropping_the_verifier():
    """A reader and a verifier that are each calibrated and share most of their errors (as
    shared_lora does): their sum is too sure, so at T = 1 the fit nearly turns the verifier off (it
    did, λ = 0, on the 2026-10-28 base runs). Fit with T, λ grows and T settles inside its grid."""
    rng = np.random.default_rng(0)
    gold = np.arange(5) == 0
    shared = [rng.normal(0, 1, 5) for _ in range(400)]                 # what both get wrong
    reader = [2 * gold + c + rng.normal(0, 0.3, 5) for c in shared]
    verifier = [2 * gold + c + rng.normal(0, 0.3, 5) for c in shared]

    def calibrated(xs):                                                 # each alone at its best temperature
        nll = lambda t: np.mean([np.logaddexp.reduce(x / t) - x[0] / t for x in xs])
        t = min(np.arange(0.2, 5, 0.05), key=nll)
        return [x / t for x in xs]

    items = [(r - np.logaddexp.reduce(r), v, 0) for r, v in zip(calibrated(reader), calibrated(verifier))]
    alone, _ = V.fit_lambda(items, temperatures=(1.0,))
    lam, t = V.fit_lambda(items)
    assert alone <= 0.2 and lam >= 0.5 and 1.2 < t < 3                # seeds 0-4: 0.1-0.2; 0.6-1.6 at T 1.45-2.45
