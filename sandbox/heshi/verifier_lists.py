"""Freeze/audit the shared candidate files for verifier complement experiments (standard library only)."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READER = "2026-10-23-row1-joint-s0-j998373"
FILES = [f"{kind}_{split}.json" for kind in ("candidates", "summary") for split in ("train", "validation")]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "audit"))
    parser.add_argument("--reader", default=READER)
    args = parser.parse_args(argv)
    if args.reader != READER:
        raise ValueError(f"these experiments freeze the stage-2 reader {READER}")
    folder = ROOT / "data/verifier" / args.reader
    path = folder / "complement_manifest.json"
    reader = ROOT / "results" / args.reader
    hashes = {name: sha(folder / name) for name in FILES}
    hashes.update({"reader_config": sha(reader / "config.yaml"), "reader_calibration": sha(reader / "calibration.json")})
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["hashes"] != hashes:
            raise RuntimeError("frozen reader/list hashes changed; refusing a mismatched comparison")
        print(f"AUDIT OK: {path}")
        return
    if args.action == "audit":
        raise FileNotFoundError("candidate manifest missing: run preparation/freeze first")
    for split, count in (("train", 79577), ("validation", 7708)):
        summary = json.loads((folder / f"summary_{split}.json").read_text())
        if (summary["run_id"] != args.reader or summary["split"] != split or summary["seed"] != 0
                or summary["checkpoint_step"] != 4974 or summary["loss_utility"] != "normalized"
                or summary["source"] != "squad"):
            raise ValueError(f"{split}: unexpected reader, schedule, labels or source")
        v = summary["verifier"]
        expected = {"list_unit": "span", "k_train": 8, "k_infer": 5, "sample_temperature": 2.0}
        if any(v[k] != value for k, value in expected.items()):
            raise ValueError(f"{split}: candidate settings differ from the fixed control")
        entries = json.loads((folder / f"candidates_{split}.json").read_text())
        if len(entries) != count or summary["questions"] != count:
            raise ValueError(f"{split}: expected {count} questions, got {len(entries)}")
        for qid, e in entries.items():
            n = len(e["spans"])
            if not n or len(e["em"]) != n or len(e["logp"]) != n:
                raise ValueError(f"{qid}: inconsistent candidate fields")
            if split == "train" and (e["gold"] is None or e["em"][e["gold"]] != 1
                                       or any(x not in (0, 1) for x in e["em"])):
                raise ValueError(f"{qid}: training gold/EM labels are invalid")
            if split == "validation" and n != 5:
                raise ValueError(f"{qid}: expected exactly five inference spans")
    state = {"reader": args.reader, "hashes": hashes, "train_questions": 79577, "validation_questions": 7708,
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    path.write_text(json.dumps(state, indent=2) + "\n")
    print(f"FROZEN: {path}")


if __name__ == "__main__":
    main()
