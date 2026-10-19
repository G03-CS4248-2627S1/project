"""Small checks for deterministic search decisions and official-metric extraction."""

import numpy as np

from squadqa.loss_search import ANCHORS, coordinates, parameters, score, suggest, trial_setup
from squadqa.loss_search_report import render


def test_suggest_uses_anchors_then_new_bounded_point():
    for phase in ("e", "c"):
        trials = []
        for anchor in ANCHORS[phase]:
            assert suggest(phase, trials) == anchor
            trials.append({"parameters": anchor, "objective": 160.0 + len(trials)})
        candidate = suggest(phase, trials)
        x = coordinates(phase, candidate)
        assert np.all((0 <= x) & (x <= 1))
        assert all(np.linalg.norm(x - coordinates(phase, t["parameters"])) >= 0.025
                   for t in trials)
        assert candidate == parameters(phase, x)
        assert candidate == suggest(phase, trials)


def test_score_optimizes_map_and_keeps_mbr_for_reporting():
    metrics = {"validation": {"map": {"exact": 74.0, "f1": 87.5},
                              "mbr": {"exact": 75.0, "f1": 88.0}},
               "calibration": {"temperature": 0.8}}
    result = score(metrics)
    assert result["map_em_plus_f1"] == 161.5
    assert result["mbr_f1"] == 88.0
    assert result["calibration_temperature"] == 0.8


def test_zero_risk_uses_onehot_config_and_positive_risk_uses_risk_config():
    assert trial_setup("e", {"beta": 0.0}) == ("configs/experiments/1a-onehot.yaml", [])
    assert trial_setup("e", {"beta": 1.0}) == ("configs/experiments/1a-risk.yaml", ["risk_weight=1.0"])
    config, overrides = trial_setup("c", {"alpha": 0.5, "tau": 0.3})
    assert config == "configs/experiments/1a-soft.yaml"
    assert overrides == ["f1_target_weight=0.5", "f1_target_temperature=0.3", "f1_target_floor=0.0"]


def test_report_compares_map_to_matched_mbr_baseline():
    summary = {"commit": "abc", "phase": "e", "trials": [
        {"trial": 1, "selection": "anchor", "parameters": {"beta": 0.0},
         "run_id": "baseline", "objective": 161.5,
         "scores": {"map_em": 74.0, "map_f1": 87.5, "mbr_em": 75.0, "mbr_f1": 88.0}},
        {"trial": 2, "selection": "expected_improvement", "parameters": {"beta": 1.2},
         "run_id": "risk", "objective": 164.0,
         "scores": {"map_em": 76.0, "map_f1": 88.0, "mbr_em": 76.0, "mbr_f1": 88.0}},
    ]}
    table = render([summary])
    assert "| E | 2 | GP expected improvement" in table
    assert "164.000 | 164.000 | +1.000 | `risk`" in table
