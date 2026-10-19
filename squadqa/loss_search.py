"""Sequential validation-only Bayesian search of the E1 MAP training losses.

Run inside slurm/map_loss_search.sbatch. A search folder is a small, fetchable
manifest; each training trial keeps its own normal results/<run_id>/ record.
Reusing SEARCH_ID after interruption resumes the last trial from its checkpoint.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc

SEED = 0
TRIALS = 6
BOUNDS = {"e": {"beta": [0.0, 3.0]}, "c": {"alpha": [0.1, 0.9], "tau": [0.05, 1.0]}}
ANCHORS = {
    "e": [{"beta": 0.0}, {"beta": 1.0}, {"beta": 3.0}],
    "c": [{"alpha": 0.25, "tau": 0.1}, {"alpha": 0.5, "tau": 0.3}, {"alpha": 0.75, "tau": 1.0}],
}


def coordinates(phase: str, params: dict) -> np.ndarray:
    if phase == "e":
        return np.array([params["beta"] / 3.0])
    return np.array([(params["alpha"] - 0.1) / 0.8, math.log(params["tau"] / 0.05) / math.log(20.0)])


def parameters(phase: str, x: np.ndarray) -> dict:
    if phase == "e":
        return {"beta": round(float(3.0 * x[0]), 4)}
    return {"alpha": round(float(0.1 + 0.8 * x[0]), 4),
            "tau": round(float(0.05 * 20.0 ** x[1]), 4)}


def suggest(phase: str, trials: list[dict]) -> dict:
    """Maximize expected improvement under a fixed RBF Gaussian process surrogate."""
    if len(trials) < len(ANCHORS[phase]):
        return ANCHORS[phase][len(trials)]
    x = np.stack([coordinates(phase, t["parameters"]) for t in trials])
    y = np.array([t["objective"] for t in trials], dtype=float)
    center, scale = y.mean(), max(float(y.std()), 0.25)
    z = (y - center) / scale
    dim = x.shape[1]
    if dim == 1:
        pool = np.linspace(0.0, 1.0, 601)[:, None]
    else:
        pool = qmc.Sobol(d=dim, scramble=True, seed=2718 + len(trials)).random_base2(m=11)
        pool = np.vstack([pool, np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])])
    # Squared-exponential kernel, fixed length scale and a small numerical/noise nugget.
    length = 0.28 if dim == 1 else 0.35
    sq = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    k = np.exp(-0.5 * sq / length**2) + np.eye(len(x)) * 0.01
    cross = np.exp(-0.5 * ((pool[:, None, :] - x[None, :, :]) ** 2).sum(axis=2) / length**2)
    chol = np.linalg.cholesky(k)
    mean = cross @ np.linalg.solve(chol.T, np.linalg.solve(chol, z))
    projected = np.linalg.solve(chol, cross.T)
    sd = np.sqrt(np.maximum(1.0 - (projected**2).sum(axis=0), 1e-12))
    improvement = mean - z.max() - 0.01
    standardized = improvement / sd
    ei = improvement * ndtr(standardized) + sd * np.exp(-standardized**2 / 2) / math.sqrt(2 * math.pi)
    ei[np.min(np.linalg.norm(pool[:, None, :] - x[None, :, :], axis=2), axis=1) < 0.025] = -1.0
    return parameters(phase, pool[int(np.argmax(ei))])


def score(metrics: dict) -> dict:
    validation = metrics["validation"]
    return {"map_em": validation["map"]["exact"], "map_f1": validation["map"]["f1"],
            "mbr_em": validation["mbr"]["exact"], "mbr_f1": validation["mbr"]["f1"],
            "map_em_plus_f1": validation["map"]["exact"] + validation["map"]["f1"],
            "calibration_temperature": metrics["calibration"]["temperature"]}


def write_summary(folder: Path, summary: dict) -> None:
    tmp = folder / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(folder / "summary.json")


def main(argv=None) -> None:
    from squadqa import config as C

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["e", "c"], required=True)
    parser.add_argument("--search-id", required=True)
    parser.add_argument("--trials", type=int, default=TRIALS)
    args = parser.parse_args(argv)
    if args.trials < len(ANCHORS[args.phase]):
        parser.error("trials must include all three anchors")
    if args.search_id.startswith(".") or "/" in args.search_id or "\\" in args.search_id:
        parser.error("search-id must be a plain directory name")
    commit, dirty = C.git_state()
    if dirty:
        parser.error("tracked cluster code is dirty")
    folder = Path("results") / args.search_id
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "summary.json"
    if manifest.exists():
        summary = json.loads(manifest.read_text(encoding="utf-8"))
        if (summary["commit"], summary["phase"], summary["seed"], summary["trial_budget"]) != (commit, args.phase, SEED, args.trials):
            parser.error("search ID belongs to a different commit, phase, seed, or trial budget")
    else:
        summary = {"search_id": args.search_id, "commit": commit, "phase": args.phase,
                   "seed": SEED, "trial_budget": args.trials,
                   "method": "fixed-kernel GP expected improvement after three anchors; no pruning",
                   "objective": "official validation MAP EM + F1; one reference per question",
                   "bounds": BOUNDS[args.phase], "trials": []}
        write_summary(folder, summary)

    while len(summary["trials"]) < args.trials:
        index = len(summary["trials"])
        params = suggest(args.phase, summary["trials"])
        kind = "anchor" if index < len(ANCHORS[args.phase]) else "expected_improvement"
        run_id = f"{args.search_id}-t{index:02d}"
        result_dir = Path("results") / run_id
        metrics_path = result_dir / "metrics.json"
        if not metrics_path.exists():
            config = "configs/experiments/1a-risk.yaml" if args.phase == "e" else "configs/experiments/1a-soft.yaml"
            overrides = (["risk_weight=" + str(params["beta"])] if args.phase == "e" else
                         ["f1_target_weight=" + str(params["alpha"]),
                          "f1_target_temperature=" + str(params["tau"]), "f1_target_floor=0.0"])
            command = [sys.executable, "-m", "squadqa.train_reader", "--config", config,
                       "--seed", str(SEED), "--run-id", run_id]
            for override in overrides:
                command += ["--set", override]
            print(f"Trial {index + 1}/{args.trials}: {kind} {params}; run {run_id}", flush=True)
            subprocess.run(command, check=True)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        scores = score(metrics)
        row = {"trial": index + 1, "selection": kind, "parameters": params,
               "run_id": run_id, "scores": scores, "objective": scores["map_em_plus_f1"]}
        summary["trials"].append(row)
        summary["best"] = max(summary["trials"], key=lambda t: t["objective"])["run_id"]
        write_summary(folder, summary)
        # A completed trial has predictions and metrics. Its optimizer checkpoint is no longer
        # needed; interrupted trials retain theirs for a same-ID resume.
        checkpoint = Path(C.env("SQUADQA_SCRATCH")) / "runs" / run_id / "checkpoint.pt"
        checkpoint.unlink(missing_ok=True)
        print(f"MAP EM+F1 {row['objective']:.4f}; best {summary['best']}", flush=True)
    print(f"Search complete: {manifest}", flush=True)


if __name__ == "__main__":
    main()
