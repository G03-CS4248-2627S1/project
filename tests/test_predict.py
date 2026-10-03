import json

import numpy as np
import pytest

from squadqa import data as D
from squadqa import evaluate as E
from squadqa import predict as P
from squadqa.config import load_config

torch = pytest.importorskip("torch")
CFG = load_config()


@pytest.fixture(scope="module")
def tok():
    try:
        return D.load_tokenizer(CFG["model"])
    except (OSError, RuntimeError) as e:
        pytest.skip(f"tokenizer not available: {e}")


@pytest.fixture(scope="module")
def val_feats(tok):
    return D.featurize(D.load_split("validation"), tok, CFG, training=False)


def oracle(noise=0.0, bonus=10.0, seed=0):
    """A fake reader: every valid span scores 0 (plus noise), the gold span gets a bonus."""
    gen = torch.Generator().manual_seed(seed)

    def score_fn(batch):
        W, n = batch["word_mask"].shape
        s = noise * torch.randn(W, n, n, generator=gen)
        for r in range(W):
            if batch["gold_start"][r] >= 0:
                s[r, batch["gold_start"][r], batch["gold_end"][r]] += bonus
        idx = torch.arange(n)
        length = idx[None, :] - idx[:, None]
        valid = (length >= 0) & (length < CFG["max_answer_words"])
        mask = valid[None] & batch["word_mask"][:, :, None] & batch["word_mask"][:, None, :]
        return s.masked_fill(~mask, float("-inf"))
    return score_fn


def test_gold_spans_score_100_with_the_official_scorer(tok, val_feats, tmp_path):
    spans = P.score_questions(oracle(), val_feats, tok.pad_token_id, CFG["max_answer_words"])
    preds, cands = P.decode_questions(val_feats, spans, CFG, temperature=1.0)
    P.write_outputs(tmp_path, "validation", preds, cands)
    data = E.load_eval_set("validation")
    aligned = {ex.question.id for ex in val_feats.examples                    # predictable gold spans
               if ex.gold is not None and ex.gold[1] - ex.gold[0] < CFG["max_answer_words"]}
    assert len(aligned) > 7650                       # 9 validation answers are over 30 words
    for rule in CFG["decode_rules"]:
        written = E.load_predictions(tmp_path / f"predictions_validation_{rule}.json")
        exact, f1 = E.per_question(data, written)                     # checks every id is present
        assert all(exact[q] == 1.0 and f1[q] == 1.0 for q in aligned), rule
    assert (tmp_path / "candidates_validation.json").read_bytes().isascii()
    top = json.loads((tmp_path / "candidates_validation.json").read_text())
    ex = next(ex for ex in val_feats.examples if ex.gold is not None)
    assert top[ex.question.id][0][:2] == list(ex.gold)
    assert len(top[ex.question.id]) == CFG["decode_topk"]


def test_multi_window_questions_get_one_softmax(tok, val_feats):
    multi = [e for e, ex in enumerate(val_feats.examples) if len(ex.windows) > 1]
    sub = D.Features([val_feats.examples[e] for e in multi], val_feats.windows, {})
    spans = P.score_questions(oracle(noise=1.0), sub, tok.pad_token_id, CFG["max_answer_words"])
    for ex, s in zip(sub.examples, spans):
        keys = set(zip(s.starts.tolist(), s.ends.tolist()))
        assert len(keys) == len(s.starts)                                # each span once
        assert max(s.ends) == len(ex.context.words) - 1                  # the passage end is reachable


def test_calibrate_on_a_noisy_reader(tok, val_feats):
    sub = D.Features(val_feats.examples[:800], val_feats.windows, {})
    spans = P.score_questions(oracle(noise=2.0, bonus=6.0), sub, tok.pad_token_id, CFG["max_answer_words"])
    cal = P.calibrate(sub, spans)
    assert cal["questions"] + cal["left_out"] == 800 and cal["left_out"] < 20
    assert cal["nll"] <= cal["nll_at_1"] and 0.05 < cal["temperature"] < 20


def test_empty_answers_are_refused(tmp_path):
    with pytest.raises(RuntimeError, match="empty"):
        P.write_outputs(tmp_path, "validation", {"map": {"q": " "}}, {})
