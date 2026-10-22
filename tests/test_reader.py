import copy

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

CFG = load_config(None, ["span_head=standard", "train.head_dropout=0.0"])


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
        return D.load_tokenizer(CFG["model"])
    except (OSError, RuntimeError) as e:
        pytest.skip(f"tokenizer not available: {e}")


@pytest.fixture(scope="module")
def small_train(tok):
    return D.featurize(D.load_split("train")[:24], tok, CFG, training=True)


# Loss with one softmax per question.

def test_question_nll_sums_every_gold_copy():
    inf = float("-inf")
    logits = torch.tensor([[1.0, 2.0, 0.5, inf],        # question 0, window 0: gold at 1
                           [0.0, 3.0, inf, inf],        # question 0, window 1: same gold word at 0
                           [0.3, 0.1, 0.2, 0.4]])       # question 1: gold at 3
    gold = torch.tensor([1, 0, 3])
    example = torch.tensor([0, 0, 1])
    nll = H.question_nll(logits, gold, example, 2)
    q0 = torch.tensor([1.0, 2.0, 0.5, 0.0, 3.0])
    expected0 = -torch.log((torch.exp(q0[1]) + torch.exp(q0[3])) / torch.exp(q0).sum())
    expected1 = -torch.log_softmax(logits[2], 0)[3]
    assert torch.allclose(nll, torch.stack([expected0, expected1]))


def test_question_nll_ignores_windows_without_the_answer():
    logits = torch.tensor([[1.0, 2.0], [0.5, 0.7]])
    nll = H.question_nll(logits, torch.tensor([0, -1]), torch.tensor([0, 0]), 1)
    assert torch.allclose(nll, -torch.log(torch.exp(torch.tensor(1.0)) / torch.exp(logits).sum()))
    with pytest.raises(RuntimeError, match="no window"):
        H.question_nll(logits, torch.tensor([-1, -1]), torch.tensor([0, 0]), 1)


def test_span_scores_are_start_plus_end_on_valid_spans():
    head = H.StandardSpanHead(8, 4, max_len=3, dropout=0.0)
    x = torch.randn(2, 5, 8)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    with torch.no_grad():
        out = head(x, x, mask)
        s = head.span_scores(out, mask)
    for w in range(2):
        for i in range(5):
            for j in range(5):
                ok = bool(mask[w, i] and mask[w, j]) and 0 <= j - i < 3
                expected = out["start"][w, i] + out["end"][w, j] if ok else float("-inf")
                assert s[w, i, j] == pytest.approx(float(expected))


def test_base_config_builds_the_row1_joint_head():
    model = H.Reader(tiny_encoder(), load_config())                 # base.yaml: ladder row 1, frozen at the 1b freeze
    assert model.kind == "joint" and isinstance(model.head, H.JointSpanHead)
    assert model.head.len_bias is None and model.head.form == "full" and model.head.pos == "none"
    assert not model.no_answer


# Training: fp32, loss falls, checkpoint and resume.

def test_check_fp32_raises_on_half_precision():
    model = H.Reader(tiny_encoder(), CFG)
    opt, _ = T.build_optimizer(model, CFG, 10)
    T.check_fp32(model, opt)
    model.head.u.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="fp32"):
        T.check_fp32(model, opt)


def run(feats, tok, path, cfg, **kw):
    model = H.Reader(tiny_encoder(), cfg)
    out = T.train(model, feats, cfg, 0, tok.pad_token_id, path, per_device=2, device="cpu", log=lambda m: None, **kw)
    return model, out


def test_loss_falls(small_train, tok, tmp_path):
    cfg = load_config(None, ["span_head=standard", "train.head_dropout=0.0", "train.lr=1e-3",
                             "train.effective_batch=4", "train.epochs=10", "train.warmup_frac=0.0"])
    _, out = run(small_train, tok, tmp_path / "ck.pt", cfg)
    assert out["steps"] == 60 and not out["stopped_early"]
    assert np.mean(out["losses"][-6:]) < 0.8 * np.mean(out["losses"][:6])


def test_resume_matches_an_uninterrupted_run(small_train, tok, tmp_path):
    cfg = load_config(None, ["span_head=standard", "train.head_dropout=0.0", "train.effective_batch=4",
                             "train.checkpoint_every=2", "train.lr=1e-3"])
    full, out = run(small_train, tok, tmp_path / "full.pt", cfg, max_steps=8)
    assert out["steps"] == 8
    _, out = run(small_train, tok, tmp_path / "cut.pt", cfg, max_steps=8, stop_after=3)
    assert out == {**out, "steps": 3, "stopped_early": True}
    assert torch.load(tmp_path / "cut.pt", weights_only=False)["step"] == 2      # the last checkpoint
    logs = []
    resumed = H.Reader(tiny_encoder(seed=5), cfg)                              # other init: the checkpoint wins
    out = T.train(resumed, small_train, cfg, 0, tok.pad_token_id, tmp_path / "cut.pt", 2, "cpu", max_steps=8,
                  log=logs.append)
    assert logs[0] == "resumed from step 2" and out["steps"] == 8
    for (name, a), (_, b) in zip(full.state_dict().items(), resumed.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-6), name


# Prediction from a reader.

def test_predict_split_writes_scored_outputs(tok, tmp_path):
    val = D.load_split("validation")
    title = val[0].title
    questions = [q for q in val if q.title == title]
    dataset = [a for a in E.load_eval_set("validation") if a["title"] == title]
    model = H.Reader(tiny_encoder(), CFG)
    out = P.predict_split(model, tok, CFG, tmp_path, "cpu", "validation", questions, dataset, temperature=1.0)
    assert set(out["validation"]) == set(CFG["decode_rules"])
    assert out["validation"]["map"]["total"] == len(questions)
    preds = E.load_predictions(tmp_path / "predictions_validation_mbr.json")
    assert set(preds) == {q.id for q in questions} and all(a.strip() for a in preds.values())

def test_epoch_end_hook_runs_after_each_epoch_but_the_last(small_train, tok, tmp_path):
    cfg = load_config(None, ["span_head=standard", "train.head_dropout=0.0", "train.effective_batch=4",
                             "train.epochs=3", "train.checkpoint_every=1000"])
    model = H.Reader(tiny_encoder(), cfg)
    seen = []

    def hook(k):
        seen.append((k, torch.load(tmp_path / "ck.pt", weights_only=False)["step"]))
        model.eval()                                      # what prediction does
    out = T.train(model, small_train, cfg, 0, tok.pad_token_id, tmp_path / "ck.pt", 2, "cpu", log=lambda m: None,
                  on_epoch_end=hook)
    assert seen == [(1, 6), (2, 12)] and out["steps"] == 18       # 24 questions / 4 per step = 6 steps an epoch
    assert model.training


def test_predict_split_can_skip_candidates(tok, tmp_path):
    val = D.load_split("validation")
    questions = [q for q in val if q.title == val[0].title]
    dataset = [a for a in E.load_eval_set("validation") if a["title"] == val[0].title]
    P.predict_split(H.Reader(tiny_encoder(), CFG), tok, CFG, tmp_path, "cpu", "validation", questions, dataset,
                    temperature=1.0, save_candidates=False)
    assert (tmp_path / "predictions_validation_map.json").exists()
    assert not (tmp_path / "candidates_validation.json").exists()

def test_micro_batches_cap_windows_and_keep_questions_whole():
    n_windows = [1, 3, 1, 1, 5, 2]
    passes = T.micro_batches(list(range(6)), 4, n_windows)
    assert [q for p in passes for q in p] == list(range(6))                  # order kept, nothing lost
    assert passes == [[0, 1], [2, 3], [4], [5]]
    assert all(sum(n_windows[q] for q in p) <= 4 or len(p) == 1 for p in passes)
    assert T.micro_batches([0, 1, 2], 2) == [[0, 1], [2]]                     # one window each by default
