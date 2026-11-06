import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from squadqa import evaluate as E
from squadqa import predict as P
from squadqa import report as R

CONTEXT = "The cat sat on the mat."          # words: The cat sat on the mat .
LOG = math.log
CANDS = [[1, 1, LOG(0.5)], [0, 1, LOG(0.3)], [5, 5, LOG(0.15)], [4, 5, LOG(0.05)]]   # cat, The cat, mat, the mat


def test_ece_by_hand():
    # bins: 0.05 -> 0 (acc 0), 0.15 -> 1 (acc 0), 0.95 and 0.95 -> 9 (acc 0.5)
    got = R.ece([0.05, 0.95, 0.95, 0.15], [0, 1, 0, 0])
    assert got == pytest.approx(0.25 * 0.05 + 0.25 * 0.15 + 0.5 * 0.45)


def test_ece_edges():
    assert R.ece([1.0, 1.0, 0.0], [1, 1, 0]) == pytest.approx(0.0)
    assert R.ece([1.0, 0.95], [1, 0]) == pytest.approx(abs(0.5 - 0.975))     # 1.0 falls in the last bin
    for conf, ok in [([], []), ([0.5], [1, 0]), ([1.2], [1]), ([-0.1], [0])]:
        with pytest.raises(ValueError):
            R.ece(conf, ok)


def test_question_diagnostics_groups_boundary_variants_and_uses_the_official_em():
    text, prob, recall = R.question_diagnostics(CONTEXT, ["the cat"], CANDS, "normalized", (1, 2, 4))
    assert text == "cat" and prob == pytest.approx(0.8)          # "cat" and "The cat" are one answer
    assert recall == [True, True, True]                         # "cat" scores EM 1 against "the cat"
    text, prob, recall = R.question_diagnostics(CONTEXT, ["mat"], CANDS, "normalized", (1, 2, 3))
    assert (text, recall) == ("cat", [False, False, True])
    text, prob, _ = R.question_diagnostics(CONTEXT, ["mat"], CANDS, "positional")
    assert text == "cat" and prob == pytest.approx(0.5)          # positional: every span on its own
    _, _, recall = R.question_diagnostics(CONTEXT, ["dog", "mat"], CANDS, "normalized", (3,))
    assert recall == [True]                                      # the best match over references


def tiny_set(golds):
    qas = [{"id": f"q{n}", "question": "?", "answers": [{"text": g, "answer_start": CONTEXT.find(g)}]}
           for n, g in enumerate(golds)]
    return [{"title": "T", "paragraphs": [{"context": CONTEXT, "qas": qas}]}]


def write_run(root, run_id, preds, cands=None, temperature=1.2):
    d = root / run_id
    d.mkdir(parents=True)
    (d / "config.yaml").write_text("mbr_utility: normalized\ndecode_topk: 10\n", encoding="utf-8")
    (d / "calibration.json").write_text(json.dumps({"temperature": temperature, "nll": 0.9, "nll_at_1": 1.0,
                                                    "unit": "group"}), encoding="utf-8")
    for rule, p in preds.items():
        (d / f"predictions_validation_{rule}.json").write_text(json.dumps(p), encoding="utf-8")
    if cands is not None:
        (d / "candidates_validation.json").write_text(json.dumps(cands), encoding="utf-8")


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    monkeypatch.setattr(P, "RESULTS", tmp_path)
    monkeypatch.setattr(E, "load_eval_set", lambda name, final=False: tiny_set(["the cat", "mat"]))
    return tmp_path


def test_run_report(tiny):
    write_run(tiny, "r0", {"map": {"q0": "cat", "q1": "cat"}}, {"q0": CANDS, "q1": CANDS})
    rep = R.run_report("r0")
    assert rep["questions"] == 2 and rep["temperature"] == 1.2 and rep["map_rebuilt_differs"] == 0
    assert rep["ece_map"] == pytest.approx(0.3)                  # both at 0.8, one right
    assert rep["topk_recall"] == {"1": 0.5, "5": 1.0, "10": 1.0}


def test_run_report_needs_the_candidates(tiny):
    write_run(tiny, "r0", {"map": {"q0": "cat", "q1": "cat"}})
    with pytest.raises(FileNotFoundError):
        R.run_report("r0")


def test_compare_systems_and_a_contrast(tiny):
    # EM per question: A runs (1, 0) and (1, 0); B runs (1, 1) and (1, 0)
    write_run(tiny, "a1", {"map": {"q0": "cat", "q1": "cat"}, "mbr": {"q0": "cat", "q1": "mat"}})
    write_run(tiny, "a2", {"map": {"q0": "the cat", "q1": "sat"}, "mbr": {"q0": "cat", "q1": "on"}})
    systems = [R.parse_system("A=map:a1,a2"), R.parse_system("B=mbr:a1,a2")]
    out = R.compare(systems, ["B-A", "B+A-A-A"], resamples=200)
    assert out["systems"]["A"]["exact"]["mean"] == pytest.approx(50.0)
    assert out["systems"]["B"]["exact"]["mean"] == pytest.approx(75.0)
    assert out["systems"]["B"]["exact"]["std"] == pytest.approx(math.sqrt(2) * 25.0)
    assert out["contrasts"]["B-A"]["exact"]["diff"] == pytest.approx(25.0)
    assert out["contrasts"]["B+A-A-A"]["exact"]["diff"] == pytest.approx(25.0)   # repeated names add up
    assert "report" not in out["systems"]["A"]                                    # no run reports written
    assert "| B-A | +25.00" in R.markdown(out)


def test_parse_system_and_contrast():
    assert R.parse_system("B=mbr:r1,r2") == ("B", "mbr", ["r1", "r2"])
    assert R.parse_contrast("D+A-B-C", "ABCD") == {"D": 1, "A": 1, "B": -1, "C": -1}
    for bad in ["B", "B=greedy:r1", "B-1=map:r1", "B=map:"]:
        with pytest.raises(ValueError):
            R.parse_system(bad)
    for bad in ["B--A", "B-", "B-Z"]:
        with pytest.raises(ValueError):
            R.parse_contrast(bad, "AB")


def test_signed_sums():
    per = {"A": {"q": 0.2}, "B": {"q": 0.5}, "C": {"q": 0.1}}
    neg, pos = R.signed_sums(per, {"B": 1, "C": 1, "A": -2})
    assert pos["q"] - neg["q"] == pytest.approx(0.5 + 0.1 - 0.4)


def test_parse_system_takes_the_verifier_rules():
    for rule in ("combined_map", "combined_mbr", "reader_map", "verifier_mbr"):
        assert R.parse_system(f"B={rule}:r1,r2") == ("B", rule, ["r1", "r2"])
    for bad in ["B=combined:r1", "B=shared_map:r1", "B=combined_greedy:r1"]:
        with pytest.raises(ValueError):
            R.parse_system(bad)


def adversarial_set():
    """q0 has an original and two adversarial versions, q1 only its original."""
    context = CONTEXT
    qas = [{"id": i, "question": "?", "answers": [{"text": "cat", "answer_start": 4}]} for i in ("q0", "q0-a", "q0-b", "q1")]
    return [{"title": "T", "paragraphs": [{"context": context, "qas": qas}]}]


def write_preds(root, run_id, split, rule, preds):
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"predictions_{split}_{rule}.json").write_text(json.dumps(preds), encoding="utf-8")


def test_compare_on_adversarial_sets_uses_the_worst_case_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    monkeypatch.setattr(E, "load_eval_set", lambda name, final=False: adversarial_set())
    # Right on q0 and q1, wrong on one version of q0: the group takes its worst version, so EM is 1 of 2 groups.
    worst = {"q0": "cat", "q0-a": "cat", "q0-b": "mat", "q1": "cat"}
    right = {"q0": "cat", "q0-a": "cat", "q0-b": "cat", "q1": "cat"}
    write_preds(tmp_path, "r", "addsent", "map", worst)
    write_preds(tmp_path, "r", "addsent", "combined_map", right)
    out = R.compare([("A", "map", ["r"]), ("B", "combined_map", ["r"])], ["B-A"], "addsent", 100, final=True)
    assert out["split"] == "addsent"
    assert out["systems"]["A"]["exact"]["mean"] == pytest.approx(50.0)
    assert out["systems"]["A"]["exact"]["mean"] == pytest.approx(E.metrics("addsent", adversarial_set(), worst)["adversarial"]["exact"])
    assert out["contrasts"]["B-A"]["exact"]["diff"] == pytest.approx(50.0)
    # Beside the worst case, spec §6's averages over all 4 questions and over the 2 originals.
    assert out["systems"]["A"]["all_questions"]["exact"]["mean"] == pytest.approx(75.0)
    assert out["systems"]["A"]["original"]["exact"]["mean"] == pytest.approx(100.0)
    assert "| A | 75.00 ± 0.00 | 75.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |" in R.markdown(out)
    # Other sets score every question on its own.
    write_preds(tmp_path, "r", "dev", "map", worst)
    out = R.compare([("A", "map", ["r"])], [], "dev", 100, final=True)
    assert out["systems"]["A"]["exact"]["mean"] == pytest.approx(75.0)


def test_compare_refuses_final_sets_without_final():
    with pytest.raises(RuntimeError, match="final scoring"):
        R.compare([("A", "map", ["r"])], [], "dev")


def test_main_passes_split_and_final(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    seen = {}

    def load(name, final=False):
        seen.update(name=name, final=final)
        return adversarial_set()

    monkeypatch.setattr(E, "load_eval_set", load)
    write_preds(tmp_path, "r", "dev", "combined_mbr", {"q0": "cat", "q0-a": "cat", "q0-b": "cat", "q1": "cat"})
    R.main(["compare", "A=combined_mbr:r", "--split", "dev", "--final"])
    assert seen == {"name": "dev", "final": True}
    assert "| A | COMBINED_MBR | 1 | 100.00" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        R.main(["compare", "A=map:r", "--split", "not-a-set"])


def test_first_reference_scores_against_the_first_answer_alone(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    qas = [{"id": "q0", "question": "q?", "answers": [{"text": "cat", "answer_start": 0}, {"text": "mat", "answer_start": 0}]},
           {"id": "q1", "question": "q?", "answers": [{"text": "cat", "answer_start": 0}]}]
    monkeypatch.setattr(E, "load_eval_set", lambda name, final=False: [{"title": "T", "paragraphs": [{"context": "c", "qas": qas}]}])
    write_preds(tmp_path, "r", "dev", "map", {"q0": "mat", "q1": "cat"})
    assert R.compare([("A", "map", ["r"])], [], "dev", 100, final=True)["systems"]["A"]["exact"]["mean"] == pytest.approx(100.0)
    out = R.compare([("A", "map", ["r"])], [], "dev", 100, final=True, first_reference=True)
    assert out["first_reference"] is True
    assert out["systems"]["A"]["exact"]["mean"] == pytest.approx(50.0)
    R.main(["compare", "A=map:r", "--split", "dev", "--final", "--first-reference"])
    assert "| A | MAP | 1 | 50.00" in capsys.readouterr().out


def test_article_splits_use_each_half_both_ways():
    titles = [f"t{k}" for k in range(9)]
    got = R.article_splits(titles, 3)
    assert len(got) == 6
    for fit, test in got:
        assert fit | test == set(titles) and not fit & test
    assert got[0] == (got[1][1], got[1][0]) and got[2] == (got[3][1], got[3][0])
    assert got[0][0] != got[2][0]                                   # the seeds give different halves
    assert R.article_splits(reversed(titles), 3) == got             # independent of the input order


def heldout_lists(n=60, seed=0, signal=1.5):
    """n lists over one passage, in 6 articles: random reader log-probabilities over spans with distinct
    cores, one of them the gold, and verifier scores that favour it by `signal` (0: flat scores)."""
    from squadqa.data import word_spans

    context = "The tower was built in 1889 in Paris ."
    words = word_spans(context)
    rng = np.random.default_rng(seed)
    spans = [[1, 1], [3, 3], [5, 5], [7, 7], [5, 7]]
    examples, entries, scores = [], [], []
    for k in range(n):
        g = int(rng.integers(len(spans)))
        a, b = spans[g]
        text = context[words[a][0]:words[b][1]]
        q = SimpleNamespace(id=f"q{k}", title=f"t{k % 6}", context=context, answers=((text, words[a][0]),))
        examples.append(SimpleNamespace(question=q, context=SimpleNamespace(words=words)))
        entries.append({"spans": spans, "logp": np.log(rng.dirichlet(np.ones(len(spans)))).tolist(),
                        "em": [int(i == g) for i in range(len(spans))]})
        scores.append(signal * (np.arange(len(spans)) == g) + (rng.normal(size=len(spans)) if signal else 0.0))
    return examples, entries, scores


RCFG = {"decode_rules": ["map", "mbr"], "mbr_utility": "normalized", "mbr_em_weight": 1.0}


def test_heldout_gains_and_counts():
    pytest.importorskip("torch")
    out = R.heldout(*heldout_lists(), RCFG, splits=2)
    assert out["halves"] == 4 and len(out["fits"]) == 4
    for rule in ("map", "mbr"):
        for metric in ("exact", "f1"):
            m = out["rules"][rule][metric]
            assert len(m["gains"]) == 4 and m["mean"] == pytest.approx(np.mean(m["gains"]))
            assert m["positive"] == sum(x > 0 for x in m["gains"])
    assert out["rules"]["map"]["exact"]["mean"] > 5                 # the scores favour the gold, so the verifier helps
    assert all(f["lambda"] > 0 for f in out["fits"])


def test_heldout_with_flat_scores_gains_nothing_under_map():
    pytest.importorskip("torch")
    out = R.heldout(*heldout_lists(signal=0.0), RCFG, splits=1)
    assert out["rules"]["map"]["exact"]["gains"] == [0.0, 0.0]      # distinct cores: T can't move MAP
    assert out["rules"]["map"]["exact"]["positive"] == 0            # a zero gain doesn't count as positive


def test_heldout_fits_only_on_the_fit_half(monkeypatch):
    pytest.importorskip("torch")
    from squadqa import verifier as V

    sizes, fit = [], V.fit_lambda
    monkeypatch.setattr(V, "fit_lambda", lambda items: (sizes.append(len(items)), fit(items))[1])
    R.heldout(*heldout_lists(n=60), RCFG, splits=1)
    assert sizes == [30, 30]                                        # 3 of the 6 articles, 10 questions each


def test_heldout_markdown():
    m = {"exact": {"mean": 0.234, "positive": 9}, "f1": {"mean": -0.091, "positive": 5}}
    text = R.heldout_markdown({"r0": {"halves": 10, "rules": {"map": m}}})
    assert "| r0 | MAP | +0.23 | 9/10 | -0.09 | 5/10 |" in text
