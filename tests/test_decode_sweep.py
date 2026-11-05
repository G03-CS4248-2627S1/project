import json
import math

import numpy as np
import pytest

from squadqa import decode_sweep as S
from squadqa import evaluate as E
from squadqa import predict as P
from squadqa import report as R
from squadqa.data import word_spans
from squadqa.decode import Candidates, content_flags

LOG = math.log
RGB = "red green blue"
RGB_WORDS = word_spans(RGB)
RGB_FLAGS = content_flags(RGB, RGB_WORDS)
RGB_TEXTS = ["red", "blue", RGB]


def rgb_cands(red, blue, cross):
    """Candidates "red", "blue" and "red green blue", most probable first."""
    return Candidates(np.array([0, 2, 0]), np.array([0, 2, 2]), np.array([LOG(red), LOG(blue), LOG(cross)]))


def answer(c, setting, temperature=1.0, texts=RGB_TEXTS):
    i, j = S.decode_question(c, RGB_FLAGS, texts, temperature, setting)
    return RGB[RGB_WORDS[i][0]:RGB_WORDS[j][1]]


def test_gamma_moves_mbr_between_expected_f1_and_map():
    # The plan's example (§7.2): expected F1 favours the cross span (0.625 against 0.525 for "red"),
    # and EM's weight brings MBR back to "red".
    c = rgb_cands(0.40, 0.35, 0.25)
    assert answer(c, "map") == "red"
    assert answer(c, "mbr-g0") == RGB
    assert answer(c, "mbr-g1") == "red"
    assert answer(c, "mbr-g2") == "red"


def test_t1_undoes_the_fitted_temperature():
    # At the fitted T = 4 the cross span wins at γ = 1 (0.95 against 0.87). At T = 1, p ∝ p_T^4, so
    # "red" has 0.44, and its 0.98 beats the cross span's 0.82.
    c = rgb_cands(0.36, 0.34, 0.30)
    assert answer(c, "mbr-g1", 4.0) == RGB
    assert answer(c, "mbr-g1-t1", 4.0) == "red"
    assert answer(c, "map-t1", 4.0) == "red"


def test_grouping_settings():
    context = "The cat sat on the mat."
    words = word_spans(context)
    flags = content_flags(context, words)
    # "mat" is the top span, but "The cat" and "cat" are one answer under the default grouping.
    c = Candidates(np.array([5, 0, 1]), np.array([5, 1, 1]), np.array([LOG(0.4), LOG(0.3), LOG(0.3)]))
    texts = ["mat", "The cat", "cat"]
    assert S.decode_question(c, flags, texts, 1.0, "map") == (0, 1)
    assert S.decode_question(c, flags, texts, 1.0, "map-positional") == (5, 5)
    # String grouping pools the same text at two positions; the other groupings keep them apart.
    context = "cat sat cat"
    words = word_spans(context)
    flags = content_flags(context, words)
    c = Candidates(np.array([1, 0, 2]), np.array([1, 0, 2]), np.array([LOG(0.45), LOG(0.3), LOG(0.25)]))
    texts = ["sat", "cat", "cat"]
    assert S.decode_question(c, flags, texts, 1.0, "map") == (1, 1)
    assert S.decode_question(c, flags, texts, 1.0, "map-string") == (0, 0)


def test_position_against_the_gold():
    context = "The cat sat on the mat."
    flags = content_flags(context, word_spans(context))
    assert S.position(flags, (1, 1), (1, 1)) == "identical"
    assert S.position(flags, (1, 1), (0, 1)) == "boundary_variant"      # "The cat" for "cat"
    assert S.position(flags, (1, 1), (5, 5)) == "elsewhere"
    assert S.position(flags, (1, 2), (1, 1)) == "elsewhere"             # too short is not a variant
    assert S.position(flags, None, (1, 1)) == "no_gold_span"


def test_gap_counts():
    got = S.gap_counts({"a": 1.0, "b": 1.0, "c": 0.0}, {"a": "identical", "b": "elsewhere", "c": "elsewhere"})
    assert got["identical_em1"] == 1 and got["elsewhere_em1"] == 1 and got["elsewhere_em0"] == 1
    assert got["boundary_variant_em1"] == 0


def test_parse_system():
    assert S.parse_system("row1=a,b") == ("row1", ["a", "b"])
    for bad in ["row1", "=a", "row1=", "row1=a,"]:
        with pytest.raises(ValueError):
            S.parse_system(bad)


def tiny_set():
    """Two questions on one passage: the gold is the second "cat" of the passage for q0, "sat" for q1."""
    context = "cat sat cat"
    qas = [{"id": "q0", "question": "?", "answers": [{"text": "cat", "answer_start": 8}]},
           {"id": "q1", "question": "?", "answers": [{"text": "sat", "answer_start": 4}]}]
    return [{"title": "T", "paragraphs": [{"context": context, "qas": qas}]}]


# Both questions: the first "cat" is the top span, and "sat" is close behind.
CANDS = [[0, 0, LOG(0.5)], [1, 1, LOG(0.4)], [2, 2, LOG(0.1)]]


def write_run(root, run_id, cands, preds, temperature=1.0):
    d = root / run_id
    d.mkdir(parents=True)
    (d / "calibration.json").write_text(json.dumps({"temperature": temperature}), encoding="utf-8")
    (d / "candidates_validation.json").write_text(json.dumps(cands), encoding="utf-8")
    for rule, p in preds.items():
        (d / f"predictions_validation_{rule}.json").write_text(json.dumps(p), encoding="utf-8")


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    for module in (S, R, P):
        monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(E, "load_eval_set", lambda name, final=False: tiny_set())
    return tmp_path


GOOD = {"map": {"q0": "cat", "q1": "cat"}, "mbr": {"q0": "cat", "q1": "cat"}}


def test_sweep_end_to_end(tiny):
    cands = {"q0": CANDS, "q1": CANDS}
    write_run(tiny, "r0", cands, GOOD)
    write_run(tiny, "r1", cands, GOOD)
    out = S.sweep([("A", ["r0", "r1"])], resamples=100)
    system = out["systems"]["A"]
    # q0 scores EM 1 officially although the answer is the other "cat"; q1 misses.
    assert system["settings"]["map"]["exact"]["mean"] == pytest.approx(50.0)
    assert system["settings"]["mbr-g1"]["vs_reference"]["diff"] == pytest.approx(0.0)
    assert system["settings"]["mbr-g1"]["wins_by_selection_rule"] is False
    assert system["gap"]["map"]["elsewhere_em1"] == 1 and system["gap"]["map"]["elsewhere_em0"] == 1
    assert system["gap"]["map"]["identical_em1"] == 0
    # String grouping pools the two "cat" spans, and the answer stays "cat".
    assert system["settings"]["map-string"]["exact"]["mean"] == pytest.approx(50.0)
    text = S.markdown(out)
    assert "| mbr-g1 | 50.00" in text and "| Elsewhere | 1.0 | 1.0 |" in text


def test_sweep_needs_the_candidates(tiny):
    (tiny / "r0").mkdir()
    with pytest.raises(FileNotFoundError):
        S.sweep([("A", ["r0"])], resamples=10)


def test_sweep_refuses_candidates_that_do_not_rebuild_the_predictions(tiny):
    write_run(tiny, "r0", {"q0": CANDS, "q1": CANDS}, {"map": {"q0": "sat", "q1": "sat"}, "mbr": GOOD["mbr"]})
    with pytest.raises(RuntimeError, match="differs from predictions_validation_map"):
        S.sweep([("A", ["r0"])], resamples=10)
