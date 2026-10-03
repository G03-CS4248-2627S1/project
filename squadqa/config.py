"""Configuration, machine profile and run folders (spec §2; AGENTS.md, "Profiles").

Every entry point builds its configuration the same way: configs/base.yaml, then one
experiment file from configs/experiments/, then --set overrides. An experiment file holds
only the keys it changes, and every key it or --set names must already exist in base.yaml,
so a typo raises instead of being ignored. Machine paths come from .env through env(),
which raises if a variable is missing; hyperparameters never go there.

    parser = argparse.ArgumentParser()
    add_run_args(parser)                                    # --config, --set, --seed, --run-id
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    run = start_run(cfg, args.config, args.seed, args.run_id)   # results/<run_id>/
    set_seed(args.seed)
    ...
    write_json(run / "metrics.json", {"exact": 76.5, "f1": 89.3})

Run from the repo root, like every entry point.
"""

import argparse
import json
import os
import random
import re
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

BASE = Path("configs/base.yaml")
RESULTS = Path("results")


def _merge(base: dict, override: dict, where: str = "") -> dict:
    """base updated with override, recursively. Every key in override must exist in base."""
    out = dict(base)
    for key, value in override.items():
        name = f"{where}{key}"
        if key not in base:
            raise KeyError(f"unknown config key {name!r}: it is not in {BASE}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"config key {name!r} is a section in {BASE}; set its keys, not a value")
            out[key] = _merge(base[key], value, f"{name}.")
        else:
            _check_type(name, base[key], value)
            out[key] = value
    return out


def _check_type(name: str, base, value) -> None:
    """A key keeps the type it has in base.yaml (an int may stand for a float; null takes anything)."""
    same_kind = isinstance(value, bool) == isinstance(base, bool)          # True is an int in Python
    if base is None or (same_kind and (isinstance(value, type(base)) or (isinstance(base, float) and isinstance(value, int)))):
        return
    hint = " (YAML reads 3e-5 as a string; write 3.0e-5)" if isinstance(base, float) and isinstance(value, str) else ""
    raise TypeError(f"config key {name!r} is {type(base).__name__} in {BASE}, got {value!r}{hint}")


_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def parse_override(item: str) -> dict:
    """'verifier.lora.rank=8' -> {'verifier': {'lora': {'rank': 8}}}. The value is parsed as YAML."""
    key, sep, value = item.partition("=")
    if not sep or not key:
        raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
    nested: dict = {}
    node = nested
    *parents, leaf = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str) and _NUMBER.fullmatch(parsed):
        parsed = float(parsed)                  # YAML 1.1 reads 3e-5 as a string
    node[leaf] = parsed
    return nested


def load_config(config: str | Path | None = None, overrides: list[str] | tuple[str, ...] = ()) -> dict:
    """base.yaml, then the experiment file (if any, and not base.yaml itself), then overrides."""
    with open(BASE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if config is not None and Path(config).resolve() != BASE.resolve():
        with open(config, encoding="utf-8") as f:
            experiment = yaml.safe_load(f) or {}
        if not isinstance(experiment, dict):
            raise TypeError(f"{config} must hold a mapping of config keys")
        cfg = _merge(cfg, experiment)
    for item in overrides:
        cfg = _merge(cfg, parse_override(item))
    return cfg


def experiment_name(config: str | Path | None) -> str:
    """The experiment's name in run ids: its config file name without .yaml, or 'base'."""
    return Path(config).stem if config is not None else "base"


_dotenv_loaded = False


def env(name: str) -> str:
    """A machine-specific setting from the environment or .env. Raises if it is missing or empty."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv()            # never overrides variables already set, e.g. by a Slurm job
        _dotenv_loaded = True
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; add it to .env (see .env.example)")
    return value


def make_run_id(experiment: str, seed: int, now: datetime | None = None) -> str:
    """<date>-<experiment>-s<seed>-j<Slurm job id>, or a time of day instead of the job id locally."""
    now = now or datetime.now()
    job = os.environ.get("SLURM_JOB_ID")
    suffix = f"j{job}" if job else f"t{now:%H%M%S}"
    return f"{now:%Y-%m-%d}-{experiment}-s{seed}-{suffix}"


def git_state() -> tuple[str, bool]:
    """The checked-out commit, and whether tracked files differ from it. Raises outside a git checkout."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, check=True).stdout.strip()
    return commit, bool(changes)


def start_run(cfg: dict, config: str | Path | None, seed: int, run_id: str | None = None) -> Path:
    """Create results/<run_id>/ with the resolved config and run.json, and return it.

    Resuming (the folder already exists) is allowed only with the same resolved config, seed and
    commit, so a resumed run can't silently mix two setups.
    """
    run_id = run_id or make_run_id(experiment_name(config), seed)
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        raise ValueError(f"invalid run id {run_id!r}")
    run_dir = RESULTS / run_id
    resolved = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False)
    commit, dirty = git_state()
    now = datetime.now().isoformat(timespec="seconds")

    if run_dir.exists():
        info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if (run_dir / "config.yaml").read_text(encoding="utf-8") != resolved:
            raise RuntimeError(f"{run_dir} exists with a different config; use a new run id")
        if info["seed"] != seed or info["git_commit"] != commit:
            raise RuntimeError(f"{run_dir} was started with seed {info['seed']} at commit "
                               f"{info['git_commit'][:7]}; this resume has seed {seed} at {commit[:7]}")
        info["resumed"] = info.get("resumed", []) + [now]
        write_json(run_dir / "run.json", info)
        return run_dir

    run_dir.mkdir(parents=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8", newline="\n") as f:
        f.write(resolved)
    write_json(run_dir / "run.json", {
        "run_id": run_id, "experiment": experiment_name(config),
        "config_file": str(config) if config is not None else None, "seed": seed,
        "git_commit": commit, "git_dirty": dirty, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(), "started": now,
    })
    return run_dir


def write_json(path: Path, data) -> None:
    """ASCII JSON with a trailing newline, the layout of qa/*.json. NaN or infinity raises."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every GPU)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """The arguments every entry point takes; slurm/template.sbatch passes them."""
    parser.add_argument("--config", help="experiment file in configs/experiments/ (default: configs/base.yaml only)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one config key, e.g. verifier.lora.rank=8; repeatable")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", help="default: <date>-<experiment>-s<seed>-j<Slurm job id>")
