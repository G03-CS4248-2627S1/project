import json
import math

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
