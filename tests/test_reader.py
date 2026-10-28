import copy
import json

import numpy as np
import pytest
import yaml

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


def test_save_each_epoch_keeps_each_epochs_model_but_the_last(small_train, tok, tmp_path):
    cfg = load_config(None, ["span_head=standard", "train.head_dropout=0.0", "train.effective_batch=4",
                             "train.epochs=3", "train.checkpoint_every=1000"])
    model = H.Reader(tiny_encoder(), cfg)
    states = {}

    def hook(k):                                          # runs right after the epoch's model is saved
        states[k] = copy.deepcopy(model.state_dict())
    T.train(model, small_train, cfg, 0, tok.pad_token_id, tmp_path / "ck.pt", 2, "cpu", log=lambda m: None,
            on_epoch_end=hook, save_each_epoch=True)
    assert sorted(p.name for p in tmp_path.glob("model_epoch*")) == ["model_epoch1.pt", "model_epoch2.pt"]
    for k, step in ((1, 6), (2, 12)):
        saved = torch.load(tmp_path / f"model_epoch{k}.pt", weights_only=False)
        assert set(saved) == {"model", "step"} and saved["step"] == step
        loaded = H.Reader(tiny_encoder(seed=5), cfg)
        assert T.load_checkpoint(tmp_path / f"model_epoch{k}.pt", loaded) == step
        for name, value in loaded.state_dict().items():
            assert torch.equal(value, states[k][name]), (k, name)


def test_a_run_resumed_at_an_epoch_end_repeats_its_hooks(small_train, tok, tmp_path):
    cfg = load_config(None, ["span_head=standard", "train.head_dropout=0.0", "train.effective_batch=4",
                             "train.epochs=2", "train.checkpoint_every=1000"])
    seen = []
    model, out = run(small_train, tok, tmp_path / "ck.pt", cfg, stop_after=6, on_epoch_end=seen.append,
                     save_each_epoch=True)
    assert out["stopped_early"] and seen == [1]
    (tmp_path / "model_epoch1.pt").unlink()               # as if the job stopped before the epoch's hooks
    logs = []
    out = T.train(H.Reader(tiny_encoder(seed=5), cfg), small_train, cfg, 0, tok.pad_token_id, tmp_path / "ck.pt",
                  2, "cpu", log=logs.append, on_epoch_end=seen.append, save_each_epoch=True)
    assert logs[0] == "resumed from step 6" and seen == [1, 1] and out["steps"] == 12
    assert torch.load(tmp_path / "model_epoch1.pt", weights_only=False)["step"] == 6
    assert not (tmp_path / "model_epoch2.pt").exists()


# Continuing a finished run with a longer schedule (continue_from).

def epochs_cfg(n, *extra):
    return load_config(None, ["span_head=standard", "train.head_dropout=0.0", "train.effective_batch=4",
                              "train.lr=1e-3", f"train.epochs={n}", *extra])


def test_continue_from_takes_the_runs_state_and_this_schedule(small_train, tok, tmp_path):
    source, out = run(small_train, tok, tmp_path / "two.pt", epochs_cfg(2, "train.checkpoint_every=1000"))
    assert out["steps"] == 12                             # 6 steps an epoch
    cfg = epochs_cfg(3, "train.checkpoint_every=1")
    model, logs, seen = H.Reader(tiny_encoder(seed=5), cfg), [], []
    fresh_gap = max((a - b).abs().max().item() for a, b in zip(source.parameters(), model.parameters()))
    out = T.train(model, small_train, cfg, 0, tok.pad_token_id, tmp_path / "three.pt", 2, "cpu", stop_after=13,
                  log=logs.append, on_epoch_end=seen.append, continue_from=tmp_path / "two.pt")
    # 18 steps, 1 of warmup: step 12 is at (18 - 12) / 17 of the peak, where the 2-epoch schedule had 0
    assert logs[0] == f"continued from step 12 of {tmp_path.name} at lr {1e-3 * 6 / 17:.2e}"
    assert logs[1].startswith("step 13/18 ") and logs[1].split(" lr ")[1].startswith(f"{1e-3 * 5 / 17:.2e} ")
    assert out["stopped_early"] and len(out["losses"]) == 1 and seen == []   # epoch 2's model was the source's
    gap = max((a - b).abs().max().item() for a, b in zip(source.parameters(), model.parameters()))
    assert gap < 0.1 * fresh_gap                          # one small step from the source's weights
    logs = []
    out = T.train(H.Reader(tiny_encoder(seed=5), cfg), small_train, cfg, 0, tok.pad_token_id, tmp_path / "three.pt",
                  2, "cpu", log=logs.append, continue_from=tmp_path / "two.pt")
    assert logs[0] == "resumed from step 13" and out["steps"] == 18          # its own checkpoint wins
    with pytest.raises(ValueError, match="ends at 12"):
        run(small_train, tok, tmp_path / "again.pt", epochs_cfg(2), continue_from=tmp_path / "two.pt")


def test_continue_from_needs_the_same_config_and_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "RESULTS", tmp_path)
    monkeypatch.setenv("SQUADQA_SCRATCH", str(tmp_path / "scratch"))
    (tmp_path / "two").mkdir()
    (tmp_path / "two" / "config.yaml").write_text(yaml.safe_dump(load_config(None, ["train.epochs=2"])), encoding="utf-8")
    (tmp_path / "two" / "run.json").write_text(json.dumps({"seed": 0}), encoding="utf-8")
    assert T.continuation_source(load_config(None, []), 0) is None
    three = load_config(None, ["train.epochs=3", "continue_from=two"])
    assert T.continuation_source(three, 0) == tmp_path / "scratch" / "runs" / "two" / T.CHECKPOINT
    with pytest.raises(ValueError, match="seed 0"):
        T.continuation_source(three, 1)
    with pytest.raises(ValueError, match=r"\['span_head', 'train\.lr'\]"):
        T.continuation_source(load_config(None, ["train.epochs=3", "train.lr=1e-5", "span_head=standard",
                                                 "continue_from=two"]), 0)
    with pytest.raises(ValueError, match="must be more"):
        T.continuation_source(load_config(None, ["train.epochs=2", "continue_from=two"]), 0)


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
