import json
import subprocess
import sys

import numpy as np
import pytest

from squadqa import evaluate as E


def make_dataset(questions):
    """questions: [(id, [gold answer texts])], all on one paragraph."""
    qas = [{"id": qid, "question": "q?", "answers": [{"text": t, "answer_start": 0} for t in golds]}
           for qid, golds in questions]
    return [{"title": "T", "paragraphs": [{"context": "c", "qas": qas}]}]


TINY = make_dataset([("q1", ["the Eiffel Tower"]), ("q2", ["1889", "in 1889"]), ("q3", ["Paris, France"])])
TINY_PREDS = {"q1": "Eiffel Tower", "q2": "1889.", "q3": "France"}


def test_summary_matches_the_scorer_command_line(tmp_path):
    data, pred = tmp_path / "data.json", tmp_path / "pred.json"
    data.write_text(json.dumps({"version": "1.1", "data": TINY}), encoding="ascii")
    pred.write_text(json.dumps(TINY_PREDS), encoding="ascii")
    out = subprocess.run([sys.executable, str(E.SCORER), str(data), str(pred)],
                         capture_output=True, text=True, check=True).stdout
    official = json.loads(out)
    ours = E.summarize(*E.per_question(TINY, TINY_PREDS))
    assert (ours["exact"], ours["f1"], ours["total"]) == (official["exact"], official["f1"], official["total"])
    assert ours["exact"] == pytest.approx(200 / 3)               # articles and punctuation are ignored


def test_per_question_takes_max_over_gold_answers():
    exact, f1 = E.per_question(TINY, TINY_PREDS)
    assert exact == {"q1": 1.0, "q2": 1.0, "q3": 0.0}
    assert f1["q3"] == pytest.approx(2 / 3)


# Unnormalized scoring: the official code without normalize_answer (a diagnostic).

@pytest.mark.parametrize("gold, pred, official, raw", [
    ("Paris", "Paris", (1, 1.0), (1, 1.0)),
    ("Paris", "paris", (1, 1.0), (0, 0.0)),                      # case
    ("the Eiffel Tower", "Eiffel Tower", (1, 1.0), (0, 0.8)),    # article: 2 of 3 whitespace tokens
    ("Paris", "Paris.", (1, 1.0), (0, 0.0)),                     # punctuation stays on the token
    ("the cat", "the dog", (0, 0.0), (0, 0.5)),                  # raw F1 can be higher: "the" now counts
])
def test_unnormalized_scoring_drops_only_the_normalization(gold, pred, official, raw):
    data = make_dataset([("q", [gold])])
    for normalize, (em, f1) in [(True, official), (False, raw)]:
        exact, f = E.per_question(data, {"q": pred}, normalize)
        assert exact["q"] == em and f["q"] == pytest.approx(f1)


def test_the_official_scorer_still_normalizes():
    E.per_question(TINY, TINY_PREDS, normalize=False)
    assert E.scorer() is not E.raw_scorer()
    assert E.scorer().normalize_answer("The Cat.") == "cat"
    assert E.raw_scorer().normalize_answer("The Cat.") == "The Cat."
    assert E.per_question(TINY, TINY_PREDS)[0] == {"q1": 1.0, "q2": 1.0, "q3": 0.0}


def test_unnormalized_metrics_say_so_and_take_max_over_golds():
    out = E.metrics("validation", TINY, TINY_PREDS, normalize=False)
    assert out["normalized"] is False and out["exact"] == 0.0
    assert out["f1"] == pytest.approx(100 * (0.8 + 0.0 + 2 / 3) / 3)   # "Paris," and "France" are 2 tokens
    assert "normalized" not in E.metrics("validation", TINY, TINY_PREDS)
    exact, _ = E.per_question(make_dataset([("q", ["Paris, France", "France"])]), {"q": "France"}, False)
    assert exact == {"q": 1.0}


@pytest.mark.parametrize("preds, problem", [
    ({"q1": "a", "q2": "b"}, "1 missing"),
    ({**TINY_PREDS, "q9": "x"}, "not in the dataset"),
    ({**TINY_PREDS, "q2": ""}, "empty"),
    ({**TINY_PREDS, "q2": "  "}, "empty"),
    ({**TINY_PREDS, "q2": None}, "not a string"),
])
def test_bad_predictions_raise(preds, problem):
    with pytest.raises(ValueError, match=problem):
        E.per_question(TINY, preds)


def test_robustness_sets_need_final():
    for name in E.FINAL_SETS:
        with pytest.raises(RuntimeError, match="final scoring"):
            E.load_eval_set(name)
    with pytest.raises(KeyError):
        E.load_eval_set("train")


def test_gold_predictions_score_100_on_validation():
    data = E.load_eval_set("validation")
    assert len(data) == 44
    gold = {qa["id"]: qa["answers"][0]["text"] for a in data for p in a["paragraphs"] for qa in p["qas"]}
    result = E.metrics("validation", data, gold)
    assert (result["exact"], result["f1"]) == (100.0, 100.0)
    assert result["total"] == 7708


# Adversarial SQuAD: the grouping must match Jia & Liang's eval_squad.py (see adversarial_groups).

ADV = make_dataset([
    ("a", ["red car"]), ("a-high-conf-turk0", ["red car"]), ("a-high-conf-turk1", ["red car"]),
    ("b", ["blue sky"]), ("b-high-conf-turk0", ["blue sky"]),
    ("c", ["green tea"]),
])


def test_adversarial_uses_versions_not_the_original():
    # b's original is wrong but its only version is right: the worst case is the version (EM 1).
    preds = {"a": "red car", "a-high-conf-turk0": "car", "a-high-conf-turk1": "nothing",
             "b": "nothing", "b-high-conf-turk0": "blue sky", "c": "green tea"}
    exact, f1 = E.adversarial_groups(ADV, *E.per_question(ADV, preds))
    assert exact == {"a": 0.0, "b": 1.0, "c": 1.0}
    assert f1 == {"a": 0.0, "b": 1.0, "c": 1.0}


def test_adversarial_em_comes_with_the_worst_f1_version():
    # Both versions of a have F1 1; the first (EM 1) is kept, although the second has EM 0.
    preds = {"a": "red car", "a-high-conf-turk0": "red car", "a-high-conf-turk1": "car red",
             "b": "blue sky", "b-high-conf-turk0": "sky", "c": "tea"}
    exact, f1 = E.adversarial_groups(ADV, *E.per_question(ADV, preds))
    assert exact == {"a": 1.0, "b": 0.0, "c": 0.0}
    assert f1["b"] == pytest.approx(2 / 3) and f1["c"] == pytest.approx(2 / 3)


def test_adversarial_metrics_report_three_averages():
    preds = {"a": "red car", "a-high-conf-turk0": "nothing", "a-high-conf-turk1": "red car",
             "b": "blue sky", "b-high-conf-turk0": "blue sky", "c": "green tea"}
    result = E.metrics("addsent", ADV, preds)
    assert result["total"] == 6 and result["exact"] == pytest.approx(500 / 6)       # every question
    assert result["original"]["exact"] == 100.0 and result["original"]["total"] == 3
    assert result["adversarial"]["exact"] == pytest.approx(200 / 3)                # a is broken


@pytest.mark.parametrize("name, questions", [("addsent", 3558), ("addonesent", 1787)])
def test_adversarial_files_have_1000_groups(name, questions):
    if not E.FINAL_SETS[name].exists():
        pytest.skip("run python -m squadqa.download_external")
    data = E.load_eval_set(name, final=True)
    ids = E.question_ids(data)
    assert len(ids) == questions
    assert len({i.split("-")[0] for i in ids}) == 1000
    assert {i.split("-")[0] for i in ids} <= set(ids)                           # every group has its original


# Paired bootstrap on synthetic scores.

def test_bootstrap_of_identical_systems_is_zero():
    a = {str(i): float(i % 2) for i in range(500)}
    assert E.paired_bootstrap(a, a, resamples=200) == {
        "diff": 0.0, "ci95": [0.0, 0.0], "n": 500, "resamples": 200, "seed": 0}


def test_bootstrap_of_a_constant_shift_is_that_shift():
    a = {str(i): 0.5 for i in range(300)}
    b = {k: 0.52 for k in a}
    result = E.paired_bootstrap(a, b, resamples=200)
    assert result["diff"] == pytest.approx(2.0)
    assert result["ci95"] == pytest.approx([2.0, 2.0])


def test_bootstrap_interval_matches_the_normal_approximation():
    rng = np.random.default_rng(1)
    n = 4000
    base = rng.random(n) < 0.75
    fixed = base | (rng.random(n) < 0.06)                     # b fixes about 1.5% of questions
    broken = fixed & ~(rng.random(n) < 0.01)                  # and breaks about 0.75%
    a = {str(i): float(x) for i, x in enumerate(base)}
    b = {str(i): float(x) for i, x in enumerate(broken)}
    result = E.paired_bootstrap(a, b, resamples=4000, seed=3)
    d = 100.0 * (broken.astype(float) - base.astype(float))
    half = 1.96 * d.std() / np.sqrt(n)
    lo, hi = result["ci95"]
    assert result["diff"] == pytest.approx(d.mean())
    assert lo < d.mean() < hi
    assert (hi - lo) / 2 == pytest.approx(half, rel=0.1)
    assert E.paired_bootstrap(a, b, resamples=4000, seed=3) == result              # seeded


def test_bootstrap_needs_the_same_questions():
    with pytest.raises(ValueError):
        E.paired_bootstrap({"q1": 1.0}, {"q2": 1.0})


def test_mean_over_seeds():
    assert E.mean_over_seeds([{"q": 1.0, "r": 0.0}, {"q": 0.0, "r": 0.0}]) == {"q": 0.5, "r": 0.0}
    with pytest.raises(ValueError):
        E.mean_over_seeds([{"q": 1.0}, {"r": 1.0}])


def test_compare_bootstraps_adversarial_groups():
    good = {"a": "red car", "a-high-conf-turk0": "red car", "a-high-conf-turk1": "red car",
            "b": "blue sky", "b-high-conf-turk0": "blue sky", "c": "green tea"}
    bad = {**good, "a-high-conf-turk1": "nothing"}
    result = E.compare("addsent", ADV, bad, good, resamples=100)
    assert result["exact"]["n"] == 3 and result["exact"]["diff"] == pytest.approx(100 / 3)


def test_command_line_writes_metrics(tmp_path):
    data = E.load_eval_set("validation")
    gold = {qa["id"]: qa["answers"][0]["text"] for a in data for p in a["paragraphs"] for qa in p["qas"]}
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps(gold), encoding="ascii")
    out = tmp_path / "metrics.json"
    E.main(["validation", str(pred), "-o", str(out)])
    assert json.loads(out.read_text())["f1"] == 100.0
    E.main(["validation", str(pred), "--compare", str(pred), "--resamples", "50", "-o", str(out)])
    assert json.loads(out.read_text())["f1"]["ci95"] == [0.0, 0.0]
    E.main(["validation", str(pred), "--no-normalize", "-o", str(out)])
    result = json.loads(out.read_text())
    assert result["normalized"] is False and result["exact"] == 100.0     # the gold strings themselves
