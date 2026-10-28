import copy
import json
from types import SimpleNamespace

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


def test_fit_temperature_or_edge():
    from squadqa.decode import fit_temperature

    rng = np.random.default_rng(0)
    items = [(rng.normal(size=5), np.array([int(rng.integers(5))])) for _ in range(200)]
    noisy = [(s + 2.0 * (np.arange(5) == g[0]), g) for s, g in items]      # informative, not separating
    assert TV.fit_temperature_or_edge(noisy) == (pytest.approx(fit_temperature(noisy)), False)
    separating = [(0.1 * (np.arange(5) == g[0]), g) for _, g in items]     # the gold always ahead: NLL falls down to T = lo
    assert TV.fit_temperature_or_edge(separating) == (0.05, True)
    flat = [(np.zeros(5), g) for _, g in items]                             # no information: uniform at any T
    assert TV.fit_temperature_or_edge(flat)[1]


def _lists(n=80, seed=0):
    """n lists over the tower passage: random reader log-probabilities over 6 overlapping spans
    (so MBR's answer depends on the temperature), one of them the gold (EM 1), and verifier scores
    that favour the gold with noise."""
    from squadqa.data import word_spans

    ctx = "The tower was built in 1889 in Paris ."
    words = word_spans(ctx)
    rng = np.random.default_rng(seed)
    spans = [[2, 5], [3, 5], [4, 5], [5, 5], [5, 7], [7, 7]]
    exs, entries, scores = [], [], []
    for k in range(n):
        g = int(rng.integers(len(spans)))
        logp = np.log(rng.dirichlet(np.ones(len(spans))))
        exs.append(SimpleNamespace(question=SimpleNamespace(id=f"q{k}", context=ctx), context=SimpleNamespace(words=words)))
        entries.append({"spans": spans, "logp": logp.tolist(), "em": [int(i == g) for i in range(len(spans))]})
        scores.append(rng.normal(size=len(spans)) + 1.5 * (np.arange(len(spans)) == g))
    return exs, entries, scores


def test_evaluate_recalibrates_the_combination_and_scores_the_verifier_alone():
    """The combined and verifier-alone distributions are each refit to one temperature (spec §5.1,
    "Combining"), and every prediction is the decoded answer of that recalibrated distribution."""
    from squadqa.decode import Candidates, answer_text, content_flags, decode, fit_temperature, log_softmax

    exs, entries, scores = _lists()
    rcfg = {"decode_rules": ["map", "mbr"], "mbr_utility": "normalized", "mbr_em_weight": 1.0}
    fit, preds = TV.evaluate(exs, entries, scores, rcfg)
    gold = [np.flatnonzero(np.array(e["em"]) == 1) for e in entries]
    lam = fit["lambda"]
    combined = [(TV.V.combine(e["logp"], s, lam), g) for e, s, g in zip(entries, scores, gold)]
    t_comb, t_ver = fit_temperature(combined), fit_temperature([(s, g) for s, g in zip(scores, gold)])
    assert fit["temperature"] == {"combined": pytest.approx(t_comb), "verifier": pytest.approx(t_ver)}
    assert fit["temperature_at_edge"] == {"combined": False, "verifier": False}
    assert fit["nll"]["combined"] <= fit["nll"]["combined_at_1"] and fit["nll"]["combined"] < fit["nll"]["reader"]
    flips = {"combined": 0, "verifier": 0}             # the data must let the temperature matter, or this test proves nothing
    for ex, e, s in zip(exs, entries, scores):
        flags = content_flags(ex.question.context, ex.context.words)
        starts, ends = np.array([a for a, _ in e["spans"]]), np.array([b for _, b in e["spans"]])
        mbr = lambda lp: decode(Candidates(starts, ends, lp), flags, "mbr", "normalized", 1.0)
        flips["combined"] += mbr(TV.V.combine(e["logp"], s, lam)) != mbr(log_softmax(TV.V.combine(e["logp"], s, lam), t_comb))
        flips["verifier"] += mbr(log_softmax(s, t_comb)) != mbr(log_softmax(s, t_ver))
        for who, lp in (("verifier", log_softmax(s, t_ver)), ("combined", log_softmax(TV.V.combine(e["logp"], s, lam), t_comb))):
            for rule in ("map", "mbr"):
                i, j = decode(Candidates(starts, ends, lp), flags, rule, "normalized", 1.0)
                assert preds[f"{who}_{rule}"][ex.question.id] == answer_text(ex.question.context, ex.context.words, i, j)
    assert flips["combined"] > 0 and flips["verifier"] > 0


def test_lists_default_to_the_verifier_data_folder():
    from squadqa.candidates import VERIFIER_DATA

    assert TV.lists_dir("r", None) == VERIFIER_DATA / "r" and str(TV.lists_dir("r", "elsewhere")) == "elsewhere"


def _trim(dataset, ids):
    """The evaluation set cut down to the given question ids."""
    out = []
    for a in dataset:
        paras = [{**p, "qas": [q for q in p["qas"] if q["id"] in ids]} for p in a["paragraphs"]]
        paras = [p for p in paras if p["qas"]]
        if paras:
            out.append({**a, "paragraphs": paras})
    return out


@pytest.mark.parametrize("variant,unit", [("shared_lora", "span"), ("shared_lora", "group"), ("separate", "span")])
def test_main_trains_resumes_and_scores(variant, unit, tmp_path, monkeypatch, capsys):
    """candidates.main then train_verifier.main, as the cluster jobs run them, on a tiny reader and
    30 questions per split, into fresh folders: a run stopped early (--stop-after) resumes from its
    checkpoint, finishes and writes its results (jobs 1008523 and 1008579 failed while saving).
    Then --rescore rebuilds the same evaluation from the saved scores. Lists per span and per group
    (verifier.list_unit, #93); the separate variant on a tiny ModernBERT with its real tokenizer."""
    from squadqa import candidates as Cand, config as Cf, data as D, evaluate as E, heads as H, predict as P
    from squadqa import verifier as V

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
    monkeypatch.setattr(Cand, "VERIFIER_DATA", tmp_path / "data" / "verifier")
    monkeypatch.setattr(TV, "VERIFIER_DATA", tmp_path / "data" / "verifier")
    monkeypatch.setattr(P, "run_config", lambda r: rcfg)                # rescore and separate read the reader's config only
    encoders = []
    if variant == "separate":
        vtok_name = Cf.load_config("configs/experiments/verifier-separate.yaml")["verifier"]["separate_model"]
        try:
            D.load_tokenizer(vtok_name)
        except (OSError, RuntimeError) as e:
            pytest.skip(f"tokenizer not available: {e}")
        torch.manual_seed(1)
        tiny = transformers.AutoModel.from_config(transformers.ModernBertConfig(
            vocab_size=50368, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
            global_attn_every_n_layers=3, local_attention=8, pad_token_id=50283), attn_implementation=V.ATTENTION)
        monkeypatch.setattr(V, "load_separate_encoder",                 # a fresh encoder per job, as on the cluster
                            lambda name: encoders.append((name, copy.deepcopy(tiny))) or encoders[-1][1])

    unit_set = ["--set", f"verifier.list_unit={unit}"]
    for split in ("train", "validation"):
        Cand.main(["--run-id", reader_run, "--split", split, "--chunk", "16", "--batch-size", "8", *unit_set])
    readers_for_lists = len(loaded)

    run = "fake-verifier-run"
    config = f"configs/experiments/verifier-{variant.replace('_', '-')}.yaml"
    common = ["--reader", reader_run, "--config", config, "--seed", "0",
              "--run-id", run, "--per-device-batch", "24", "--max-steps", "3",
              "--set", "verifier.lora.rank=4", "--set", "verifier.train.effective_batch=4", *unit_set]
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
    assert set(metrics["validation"]) == {f"{w}_{r}" for w in ("reader", "verifier", "combined") for r in ("map", "mbr")}
    if metrics["lambda_questions"]:                                     # a random reader's top 5 rarely holds the gold
        assert set(metrics["temperature"]) == {"combined", "verifier"} and metrics["nll"]["combined"] <= metrics["nll"]["combined_at_1"]
    else:
        assert metrics["lambda"] == 0 and "temperature" not in metrics
    scores = json.loads((out / "verifier_scores_validation.json").read_text())
    lists = json.loads((tmp_path / "data" / "verifier" / reader_run / "candidates_validation.json").read_text())
    assert set(scores) == ids and all(len(scores[q]) == len(lists[q]["spans"]) for q in ids)
    assert ("size" in next(iter(lists.values()))) == (unit == "group")
    preds = {name: json.loads((out / f"predictions_validation_{name}.json").read_text()) for name in metrics["validation"]}
    assert all(set(p) == ids for p in preds.values())
    if variant == "shared_lora":
        trained = loaded[-1]                                            # V4 on the real loop
        after = {n.replace(".base_layer.", "."): p for n, p in trained.named_parameters() if "lora_" not in n}
        assert all(torch.equal(after[n], p) for n, p in reader.named_parameters())
    else:                                                               # no reader weights; the encoder trained and was saved
        assert len(loaded) == readers_for_lists                         # train_verifier loaded no reader
        assert [n for n, _ in encoders] == [vtok_name] * 2 and metrics["reader_step"] is None
        saved = torch.load(ckpt, weights_only=False)["trainable"]
        assert set(saved) == {f"encoder.{n}" for n, _ in tiny.named_parameters()} | {"head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias"}
        assert saved["encoder.embeddings.tok_embeddings.weight"].shape[0] == 50370
        assert not torch.equal(saved["encoder.layers.0.attn.Wqkv.weight"], tiny.layers[0].attn.Wqkv.weight)
        groups = torch.load(ckpt, weights_only=False)["opt"]["param_groups"]      # no decay on every norm (no_decay)
        no_wd = [g for g in groups if g["weight_decay"] == 0.0]
        assert len(no_wd) == 1 and len(no_wd[0]["params"]) == sum("norm" in n for n, _ in tiny.named_parameters()) + 2

    capsys.readouterr()
    TV.cli(["--rescore", run])                                          # from the saved scores, no model
    assert "RESCORED" in capsys.readouterr().out
    again = json.loads((out / "metrics.json").read_text())
    assert again["validation"] == metrics["validation"] and again["lambda"] == metrics["lambda"]
    assert again.get("temperature") == metrics.get("temperature")
    assert again["steps"] == 3 and again["gpu"] == metrics["gpu"] and set(again["rescored"]) >= {"git_commit", "lists"}
    assert all(json.loads((out / f"predictions_validation_{n}.json").read_text()) == p for n, p in preds.items())


def test_rescore_refuses_other_lists(tmp_path, monkeypatch):
    """Saved scores that don't fit the lists (other lists than the run scored) raise."""
    from squadqa import config as Cf, predict as P

    run = tmp_path / "results" / "v"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("verifier: {reader: r}\n")
    (run / "metrics.json").write_text("{}")
    (run / TV.SCORES).write_text(json.dumps({"q1": [0.1, 0.2]}))
    monkeypatch.setattr(Cf, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(P, "run_config", lambda r: {"model": "x"})
    monkeypatch.setattr(TV.D, "load_tokenizer", lambda name: None)
    monkeypatch.setattr(TV.D, "load_split", lambda name: [])
    monkeypatch.setattr(TV.D, "featurize", lambda qs, tok, cfg, training: SimpleNamespace(examples=[
        SimpleNamespace(question=SimpleNamespace(id="q1"))]))
    (tmp_path / "lists").mkdir()
    (tmp_path / "lists" / "candidates_validation.json").write_text(json.dumps({"q1": {"spans": [[0, 0]]}}))
    with pytest.raises(RuntimeError, match="other lists"):
        TV.rescore("v", str(tmp_path / "lists"))
    (run / TV.SCORES).write_text(json.dumps({"q2": [0.1]}))
    with pytest.raises(RuntimeError, match="different questions"):
        TV.rescore("v", str(tmp_path / "lists"))
    with pytest.raises(FileNotFoundError):
        TV.rescore("missing", str(tmp_path / "lists"))


def test_main_refuses_settings_that_are_not_built():
    with pytest.raises(NotImplementedError, match="in-place"):
        TV.main(["--reader", "x", "--seed", "0", "--set", "verifier.type=levitated"])
    with pytest.raises(NotImplementedError, match="in-place"):
        TV.main(["--reader", "x", "--seed", "0", "--set", "verifier.variant=both"])
    with pytest.raises(NotImplementedError, match="sibling_negatives"):
        TV.main(["--reader", "x", "--seed", "0", "--set", "verifier.sibling_negatives=true"])
