import json
import random
from datetime import datetime

import pytest
import yaml

from squadqa import config as C


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_base_only():
    cfg = C.load_config()
    assert cfg["model"] == "microsoft/deberta-v3-large"
    assert cfg["verifier"]["lora"]["rank"] == 16


def test_experiment_overrides_only_its_keys(tmp_path):
    exp = write_yaml(tmp_path / "row0-baseline.yaml", {"span_head": "standard", "verifier": {"k_infer": 3}})
    cfg = C.load_config(exp)
    assert cfg["span_head"] == "standard"
    assert cfg["verifier"]["k_infer"] == 3
    assert cfg["verifier"]["variant"] == "shared_lora"       # sibling keys survive
    assert cfg["max_seq_len"] == 384


def test_base_file_as_config_is_base():
    assert C.load_config(C.BASE) == C.load_config()


@pytest.mark.parametrize("bad", [{"spanhead": "joint"}, {"verifier": {"k_infr": 3}}])
def test_unknown_key_in_experiment_raises(tmp_path, bad):
    with pytest.raises(KeyError, match="unknown config key"):
        C.load_config(write_yaml(tmp_path / "typo.yaml", bad))


def test_section_replaced_by_value_raises(tmp_path):
    with pytest.raises(TypeError, match="section"):
        C.load_config(write_yaml(tmp_path / "x.yaml", {"verifier": 3}))


def test_experiment_must_be_a_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        C.load_config(p)


def test_overrides_parse_yaml_values_and_apply_last(tmp_path):
    exp = write_yaml(tmp_path / "e.yaml", {"decode_topk": 20})
    cfg = C.load_config(exp, ["decode_topk=5", "verifier.lora.rank=8", "decode_rules=[map]", "span_head=joint"])
    assert cfg["decode_topk"] == 5
    assert cfg["verifier"]["lora"]["rank"] == 8
    assert cfg["verifier"]["lora"]["alpha"] == 32
    assert cfg["decode_rules"] == ["map"]
    assert cfg["span_head"] == "joint"


@pytest.mark.parametrize("item", ["decode_topk", "=5", ""])
def test_override_needs_key_and_value(item):
    with pytest.raises(ValueError):
        C.parse_override(item)


def test_unknown_override_raises():
    with pytest.raises(KeyError):
        C.load_config(None, ["verifier.loraa.rank=8"])


def test_experiment_name():
    assert C.experiment_name("configs/experiments/row1-compound.yaml") == "row1-compound"
    assert C.experiment_name(None) == "base"


def test_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("SQUADQA_TEST_UNSET", raising=False)
    with pytest.raises(RuntimeError, match="SQUADQA_TEST_UNSET is not set"):
        C.env("SQUADQA_TEST_UNSET")
    monkeypatch.setenv("SQUADQA_TEST_UNSET", "")
    with pytest.raises(RuntimeError):
        C.env("SQUADQA_TEST_UNSET")
    monkeypatch.setenv("SQUADQA_TEST_UNSET", "/scratch")
    assert C.env("SQUADQA_TEST_UNSET") == "/scratch"


def test_run_id_uses_job_id_or_time(monkeypatch):
    now = datetime(2026, 10, 5, 9, 30, 1)
    monkeypatch.setenv("SLURM_JOB_ID", "887001")
    assert C.make_run_id("row0-baseline", 2, now) == "2026-10-05-row0-baseline-s2-j887001"
    monkeypatch.delenv("SLURM_JOB_ID")
    assert C.make_run_id("row0-baseline", 2, now) == "2026-10-05-row0-baseline-s2-t093001"


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(C, "git_state", lambda: ("a" * 40, False))
    return tmp_path / "results"


def test_start_run_writes_config_and_run_info(results):
    cfg = C.load_config()
    run = C.start_run(cfg, None, 1, "2026-10-05-base-s1-t000000")
    assert run == results / "2026-10-05-base-s1-t000000"
    assert yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) == cfg
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert (info["seed"], info["git_commit"], info["git_dirty"], info["experiment"]) == (1, "a" * 40, False, "base")


def test_resume_needs_same_config_seed_and_commit(results, monkeypatch):
    cfg = C.load_config()
    run = C.start_run(cfg, None, 1, "r1")
    assert C.start_run(cfg, None, 1, "r1") == run                       # same setup: resumes
    assert len(json.loads((run / "run.json").read_text())["resumed"]) == 1
    with pytest.raises(RuntimeError, match="different config"):
        C.start_run(C.load_config(None, ["decode_topk=5"]), None, 1, "r1")
    with pytest.raises(RuntimeError, match="seed"):
        C.start_run(cfg, None, 2, "r1")
    monkeypatch.setattr(C, "git_state", lambda: ("b" * 40, False))
    with pytest.raises(RuntimeError, match="commit"):
        C.start_run(cfg, None, 1, "r1")


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden"])
def test_invalid_run_ids_raise(results, bad):
    with pytest.raises(ValueError):
        C.start_run(C.load_config(), None, 0, bad or "/")


def test_git_state_reads_this_checkout():
    commit, dirty = C.git_state()
    assert len(commit) == 40 and isinstance(dirty, bool)


def test_write_json_is_ascii_and_rejects_nan(tmp_path):
    p = tmp_path / "m.json"
    C.write_json(p, {"answer": "café", "f1": 89.3})
    raw = p.read_bytes()
    assert raw.isascii() and raw.endswith(b"\n")
    assert json.loads(raw)["answer"] == "café"
    with pytest.raises(ValueError):
        C.write_json(p, {"f1": float("nan")})


def test_add_run_args():
    import argparse
    parser = argparse.ArgumentParser()
    C.add_run_args(parser)
    args = parser.parse_args(["--config", "c.yaml", "--set", "a=1", "--set", "b=2", "--seed", "3"])
    assert (args.config, args.set, args.seed, args.run_id) == ("c.yaml", ["a=1", "b=2"], 3, None)
    with pytest.raises(SystemExit):
        parser.parse_args([])                                   # --seed is required


def test_set_seed_is_reproducible():
    pytest.importorskip("torch")
    C.set_seed(7)
    a = random.random()
    C.set_seed(7)
    assert random.random() == a
