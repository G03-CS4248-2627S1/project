import copy
import json

import numpy as np
import pytest

from squadqa import train_verifier as TV

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_question_targets_are_one_hot_on_the_gold_and_refuse_a_list_without_one():
    cfg = {"f1_target_weight": 0.0, "f1_target_temperature": 0.1, "f1_target_floor": 0.5}
    entry = {"spans": [[0, 0], [0, 1], [3, 4]], "f1": [0.67, 1.0, 0.0], "gold": 1}
    assert TV.question_targets(entry, cfg).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        TV.question_targets({**entry, "gold": None}, cfg)
    with pytest.raises(ValueError):
        TV.question_targets({**entry, "f1": [None, None, None]}, cfg)


def test_evaluate_picks_lambda_and_lets_the_verifier_overturn_the_reader():
    """Two lists where the reader's top-1 is wrong and the verifier prefers the right one."""
    from types import SimpleNamespace
    from squadqa.data import word_spans

    ctx = "The tower was built in 1889 in Paris ."
    words = word_spans(ctx)
    ex = [SimpleNamespace(question=SimpleNamespace(id=f"q{k}", context=ctx), context=SimpleNamespace(words=words))
          for k in range(2)]
    entry = {"spans": [[7, 7], [5, 5]], "logp": [-0.4, -1.1], "em": [0, 1]}   # reader: "Paris"; gold "1889"
    rcfg = {"decode_rules": ["map", "mbr"], "mbr_utility": "normalized", "mbr_em_weight": 1.0}
    fit, preds = TV.evaluate(ex, [entry, entry], [np.array([-3.0, 3.0])] * 2, rcfg)
    assert fit["lambda"] > 0 and fit["lambda_questions"] == 2
    assert preds["reader_map"]["q0"] == "Paris" and preds["combined_map"]["q0"] == "1889"
    assert preds["combined_mbr"]["q1"] == "1889"


def test_evaluate_groups_boundary_variants_like_the_readers_map():
    """"The tower" and "tower" are one group under normalized utility (an article is not a content
    word): together they outweigh "Paris", and the answer is the group's most probable span."""
    from types import SimpleNamespace
    from squadqa.data import word_spans

    ctx = "The tower was built in 1889 in Paris ."
    words = word_spans(ctx)
    ex = [SimpleNamespace(question=SimpleNamespace(id="q", context=ctx), context=SimpleNamespace(words=words))]
    entry = {"spans": [[7, 7], [0, 1], [1, 1]], "logp": [np.log(0.4), np.log(0.35), np.log(0.25)], "em": [0, 1, 1]}
    rcfg = {"decode_rules": ["map"], "mbr_utility": "normalized", "mbr_em_weight": 1.0}
    _, preds = TV.evaluate(ex, [entry], [np.zeros(3)], rcfg)
    assert preds["reader_map"]["q"] == "The tower"


def _trim(dataset, ids):
    """The evaluation set cut down to the given question ids."""
    out = []
    for a in dataset:
        paras = [{**p, "qas": [q for q in p["qas"] if q["id"] in ids]} for p in a["paragraphs"]]
        paras = [p for p in paras if p["qas"]]
        if paras:
            out.append({**a, "paragraphs": paras})
    return out


def test_main_trains_resumes_and_scores(tmp_path, monkeypatch, capsys):
    """candidates.main then train_verifier.main, as the cluster jobs run them, on a tiny reader and
    30 questions per split, into fresh folders: a run stopped early (--stop-after) resumes from its
    checkpoint, finishes and writes its results (jobs 1008523 and 1008579 failed while saving)."""
    from squadqa import candidates as Cand, config as Cf, data as D, evaluate as E, heads as H, predict as P

    rcfg = Cf.load_config("configs/experiments/row1-joint.yaml")
    try:
        tok = D.load_tokenizer(rcfg["model"])
    except (OSError, RuntimeError) as e:
        pytest.skip(f"tokenizer not available: {e}")
    torch.manual_seed(0)
    enc = transformers.AutoModel.from_config(transformers.DebertaV2Config(
        vocab_size=128100, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=512, relative_attention=True, position_buckets=256, pos_att_type=["p2c", "c2p"],
        type_vocab_size=0, position_biased_input=False))
    reader = H.Reader(enc, rcfg)
    reader_run = "fake-reader-run"
    (tmp_path / "results" / reader_run).mkdir(parents=True)
    (tmp_path / "results" / reader_run / "calibration.json").write_text('{"temperature": 1.3}')
    real_split, real_eval = D.load_split, E.load_eval_set
    monkeypatch.setattr(Cf, "RESULTS", tmp_path / "results")
    loaded = []                                                         # a fresh reader per job, as on the cluster
    monkeypatch.setattr(P, "load_reader", lambda r, device: (rcfg, tok, loaded.append(copy.deepcopy(reader)) or loaded[-1], 123))
    monkeypatch.setattr(D, "load_split", lambda name, extra=(): real_split(name)[:30])
    ids = {q.id for q in real_split("validation")[:30]}
    monkeypatch.setattr(E, "load_eval_set", lambda name, final=False: _trim(real_eval(name), ids))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SQUADQA_SCRATCH", str(tmp_path / "scratch"))   # nothing in it exists yet

    for split in ("train", "validation"):
        Cand.main(["--run-id", reader_run, "--split", split, "--chunk", "16", "--batch-size", "8"])

    run = "fake-verifier-run"
    common = ["--reader", reader_run, "--config", "configs/experiments/verifier-shared-lora.yaml", "--seed", "0",
              "--run-id", run, "--per-device-batch", "24", "--max-steps", "3",
              "--set", "verifier.lora.rank=4", "--set", "verifier.train.effective_batch=4"]
    TV.main(common + ["--stop-after", "2"])
    ckpt = tmp_path / "scratch" / "runs" / run / TV.CHECKPOINT
    assert torch.load(ckpt, weights_only=False)["step"] == 2
    assert not (tmp_path / "results" / run / "metrics.json").exists()
    capsys.readouterr()
    TV.main(common)
    assert "resumed from step 2" in capsys.readouterr().out             # not a fresh 3-step run
    assert torch.load(ckpt, weights_only=False)["step"] == 3

    out = tmp_path / "results" / run
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["steps"] == 3 and metrics["reader"] == reader_run and 0 <= metrics["lambda"] <= 5
    assert set(metrics["validation"]) == {f"{w}_{r}" for w in ("reader", "combined") for r in ("map", "mbr")}
    scores = json.loads((out / "verifier_scores_validation.json").read_text())
    lists = json.loads((tmp_path / "scratch" / "candidates" / reader_run / "candidates_validation.json").read_text())
    assert set(scores) == ids and all(len(scores[q]) == len(lists[q]["spans"]) for q in ids)
    for name in metrics["validation"]:
        assert set(json.loads((out / f"predictions_validation_{name}.json").read_text())) == ids
    trained = loaded[-1]                                                # V4 on the real loop
    after = {n.replace(".base_layer.", "."): p for n, p in trained.named_parameters() if "lora_" not in n}
    assert all(torch.equal(after[n], p) for n, p in reader.named_parameters())


def test_main_refuses_settings_that_are_not_built():
    with pytest.raises(NotImplementedError, match="shared_lora"):
        TV.main(["--reader", "x", "--seed", "0", "--set", "verifier.variant=separate"])
    with pytest.raises(NotImplementedError, match="sibling_negatives"):
        TV.main(["--reader", "x", "--seed", "0", "--set", "verifier.sibling_negatives=true"])
