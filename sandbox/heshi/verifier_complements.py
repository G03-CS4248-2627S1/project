"""Plan or submit the verifier complement DAG; no GPU work occurs in the default plan mode."""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIALS = {"v0": "verifier-separate", "v1": "verifier-gold-group",
          "v2": "verifier-ce-margin", "v3": "verifier-deberta-independent"}
DEFAULT_RECORD = ROOT.parent / "cs4248 project/reports/2026-11-01-verifier-complement-jobs.json"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "submit"), nargs="?", default="plan")
    p.add_argument("--phase", choices=("setup", "smoke", "full"), default="smoke",
                   help="setup=CPU checks only; smoke=also lists and GPU checks; full=also all one-epoch trials")
    p.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = p.parse_args(argv)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    graph = {"commit": commit, "reader": "2026-10-23-row1-joint-s0-j998373",
             "setup": {"script": "slurm/verifier_complement_setup.sbatch", "partition": "normal", "time": "00:45:00"},
             "lists": {"script": "slurm/verifier_complement_candidates.sbatch", "gpu": "a100-40", "time": "00:45:00", "afterok": "setup"},
             "smoke": {"script": "slurm/verifier_complement_smoke.sbatch", "gpu": "a100-40", "time": "01:00:00", "afterok": "lists"},
             "trials": [{"id": k, "config": f"configs/experiments/{v}.yaml", "script": "slurm/verifier_complement_train.sbatch",
                         "gpu": "a100-40", "time": "03:00:00", "afterok": "smoke"} for k, v in TRIALS.items()],
             "v3_resume": "afterany of v3, another 3-hour allocation; skips if v3 already completed",
             "seed": 0, "verifier_epochs": 1, "per_device_batch": 16,
             "checkpoint_policy": "Cluster scratch only; no checkpoint downloads",
             "note": "Same frozen span lists for every trial. Full phase is 4 trainings including the control, plus a DeBERTa resume safety job."}
    if args.action == "plan":
        print(json.dumps(graph, indent=2))
        return
    record = json.loads(args.record.read_text()) if args.record.exists() else {
        "plan": graph, "commit": commit, "jobs": {}, "status": "submitting"}
    if record["commit"] != commit:
        raise ValueError("Recorded submissions use another commit; do not move their checkout")

    def save():
        args.record.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.record.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(args.record)

    def submit(role, script, options, after=None, dependency_type="afterok"):
        if role in record["jobs"]:
            return record["jobs"][role]["job_id"]
        command = ["bash", ".claude/skills/soc-cluster/soc.sh", "submit", *options]
        if after:
            command += [f"--dependency={dependency_type}:{after}", "--kill-on-invalid-dep=yes"]
        command.append(script)
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        print(result.stdout.strip(), flush=True)
        if result.returncode:
            record.update(status="submission_incomplete", error=result.stderr.strip())
            save()
            raise RuntimeError(result.stderr.strip())
        match = re.search(r"^Submitted batch job (\d+)$", result.stdout, re.M)
        if not match:
            record.update(status="submission_needs_review", error=result.stdout)
            save()
            raise RuntimeError("No job ID: inspect the queue before retrying")
        job = int(match[1])
        record["jobs"][role] = {"job_id": job, "script": script, "options": options,
                                  "dependency": f"{dependency_type}:{after}" if after else None}
        save()
        return job

    setup = submit("setup", graph["setup"]["script"], ["--time=00:45:00"])
    if args.phase != "setup":
        lists = submit("lists", graph["lists"]["script"], ["--gpus=a100-40", "--time=00:45:00"], setup)
        smoke = submit("smoke", graph["smoke"]["script"], ["--gpus=a100-40", "--time=01:00:00"], lists)
    if args.phase == "full":
        for trial, experiment in TRIALS.items():
            job = submit(trial, "slurm/verifier_complement_train.sbatch",
                         ["--gpus=a100-40", "--time=03:00:00", f"--export=ALL,EXPERIMENT={experiment}"], smoke)
            if trial == "v3":
                submit("v3_resume", "slurm/verifier_complement_train.sbatch",
                       ["--gpus=a100-40", "--time=03:00:00",
                        f"--export=ALL,EXPERIMENT={experiment},RESUME_JOB_ID={job}"], job, "afterany")
    record["status"] = f"{args.phase} phase submitted"
    save()
    print(args.record, flush=True)


if __name__ == "__main__":
    main()
