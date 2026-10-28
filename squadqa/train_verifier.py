"""Train the shared_lora verifier (spec §5.4; #67) on a reader's candidate lists, then score validation.

    python -m squadqa.train_verifier --reader <reader run_id> --config configs/experiments/verifier-shared-lora.yaml \
        --seed 0 [--run-id ID] [--set verifier.train.lr=2.0e-4] [--per-device-batch 16] [--max-steps N] [--stop-after N]

- The reader is the frozen final reader: its run's config sets the windows, its checkpoint the
  weights (predict.load_reader). The verifier adds LoRA, two marker embeddings and a head, and only
  those train (squadqa.verifier.SharedLoraVerifier); V4 is checked at the end of training.
- The lists are squadqa.candidates' training and validation lists of that reader, from
  data/verifier/<reader>/ (candidates.VERIFIER_DATA) unless --lists names another folder. Fields
  this script doesn't use are ignored, so lists with more labels (#93) load too, per span or per
  group (verifier.list_unit: with group lists the one-hot target is the gold's group).
- Each candidate is marked in place in its most central reader window (spec §5.3) and scored in its
  own pass. A question's candidates share one listwise softmax (verifier.listwise_loss) against
  candidates.targets: α, τ and φ are the reader's f1_target_weight, f1_target_temperature and
  f1_target_floor (spec §5.1, "same α, τ, φ form"); α = 0 is one-hot on the gold.
- verifier.train holds the optimizer settings, effective_batch in questions per step. bf16 autocast
  on the GPU, fp32 trainable parameters and optimizer state (spec §9).
- Checkpoints hold the trainable parameters, optimizer and schedule only (the reader is its own
  run's checkpoint), in $SQUADQA_SCRATCH/runs/<run_id>/checkpoint.pt every
  verifier.train.checkpoint_every steps and at the end. Re-running with the same run id resumes.

Then, on the validation top-k lists (evaluate; spec §5.1, "Combining"): λ is fit by
verifier.fit_lambda (the NLL of the gold's EM-1 candidates), then one temperature on the combined
scores log p_reader + λ · s, and one on the verifier's scores alone, both by the same NLL (plan
§4.3: "then calibrate and decode"). results/<run_id>/ gets the verifier scores of every candidate,
the predictions of the reader alone (λ = 0), the verifier alone (plan §9.2's ablation) and
combined, each decoded by the reader's own decode_rules (MAP and MBR, boundary variants grouped,
decode.decode), and metrics.json with all of them scored by the official scorer, plus λ, the
temperatures and the NLLs. λ and the temperatures are fit on the split they are scored on (plan
§4.3: tuned on validation), so the combined and verifier-alone scores are slightly optimistic.

--max-steps and --stop-after are for smoke runs (spec §9), as in train_reader: resume with the same
arguments minus --stop-after.

    python -m squadqa.train_verifier --rescore <verifier run_id> [--lists DIR]

re-runs that evaluation from a finished run's saved verifier_scores_validation.json, on the CPU
with no model (the reader's tokenizer only), and rewrites its predictions and the evaluation part
of metrics.json; a "rescored" entry records the commit. The lists must be the ones the run scored.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # pinned files from $HF_HOME only, never the network

import numpy as np
import torch

from squadqa import config as C
from squadqa import data as D
from squadqa import evaluate as E
from squadqa import verifier as V
from squadqa.candidates import NOT_BUILT, VERIFIER_DATA, targets

CHECKPOINT = "checkpoint.pt"
SCORES = "verifier_scores_validation.json"


def lists_dir(reader_run: str, override: str | None) -> Path:
    return Path(override) if override else VERIFIER_DATA / reader_run


def load_lists(folder: Path, split: str) -> dict[str, dict]:
    path = folder / f"candidates_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run python -m squadqa.candidates --split {split} first")
    return json.loads(path.read_text(encoding="ascii"))


class Marker:
    """Marks and encodes one question's candidates with the reader's windows (spec §5.3)."""

    def __init__(self, tok, rcfg: dict, open_id: int, close_id: int):
        self.tok, self.rcfg, self.ids = tok, rcfg, (open_id, close_id)
        self.layout = D.pair_layout(tok)

    def encode(self, ex: D.Example, spans: list[list[int]]) -> list[V.Encoded]:
        q_ids = self.tok(ex.question.question, add_special_tokens=False)["input_ids"]
        windows = D.make_windows(q_ids, ex.context, self.layout, self.rcfg["max_seq_len"], self.rcfg["doc_stride"],
                                 window_layout=self.rcfg["window_layout"])
        out = []
        for i, j in spans:
            lo, hi = V.candidate_window(windows, ex.context, i, j)
            m = V.mark(ex.question.context, ex.context.words, i, j, lo, hi)
            out.append(V.encode(self.tok, ex.question.question, m, *self.ids))
        return out


def question_targets(entry: dict, cfg: dict) -> np.ndarray:
    """The listwise target over one training list (candidates.targets)."""
    f1 = np.array([np.nan if x is None else x for x in entry["f1"]], dtype=np.float64)
    if entry["gold"] is None or np.isnan(f1).any():
        raise ValueError("a training list without its gold")
    return targets(f1, entry["gold"], cfg["f1_target_weight"], cfg["f1_target_temperature"], cfg["f1_target_floor"])


def batch_of(marker: Marker, examples: list[D.Example], entries: list[dict], device: str, pad_id: int):
    """One forward pass: every candidate of the given questions, each candidate's question index."""
    encs, question = [], []
    for k, (ex, e) in enumerate(zip(examples, entries)):
        got = marker.encode(ex, e["spans"])
        encs += got
        question += [k] * len(got)
    b = {k: v.to(device, non_blocking=True) for k, v in V.collate_marked(encs, pad_id).items()}
    return b, torch.tensor(question, device=device)


def micro_batches(order: list[int], sizes: list[int], per_device: int) -> list[list[int]]:
    """Questions per forward pass, at most per_device candidates (a question's candidates stay together)."""
    from squadqa.train_reader import micro_batches as split
    return split(order, per_device, sizes)


def reader_fingerprint(v: V.SharedLoraVerifier) -> float:
    """A float64 sum over every reader weight (base layers under LoRA included, the adapters not)."""
    with torch.no_grad():
        return float(sum(p.double().sum() for n, p in v.reader.named_parameters() if "lora_" not in n))


def save_checkpoint(path: Path, v, opt, sched, step: int) -> None:
    from squadqa.train_reader import check_fp32
    check_fp32(v, opt)
    trainable = {n: p.detach() for n, p in v.named_parameters() if p.requires_grad}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"trainable": trainable, "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, v, opt=None, sched=None) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)      # CPU first (train_reader.load_checkpoint)
    params = dict(v.named_parameters())
    if set(ck["trainable"]) != {n for n, p in params.items() if p.requires_grad}:
        raise RuntimeError(f"{path} holds other trainable parameters than this verifier")
    with torch.no_grad():
        for n, t in ck["trainable"].items():
            params[n].copy_(t)
    if opt is not None:
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
    return ck["step"]


def train(v, marker: Marker, examples: list[D.Example], entries: list[dict], cfg: dict, seed: int, pad_id: int,
          ckpt_path: Path, per_device: int, device: str, max_steps: int | None = None,
          stop_after: int | None = None, log=print) -> dict:
    """Train with resume from ckpt_path. Returns {"steps", "stopped_early", "losses", "minutes"}."""
    from squadqa.train_reader import build_optimizer, check_fp32

    t = cfg["verifier"]["train"]
    n = len(examples)
    n_batches = -(-n // t["effective_batch"])
    total_steps = max_steps or n_batches * t["epochs"]
    opt, sched = build_optimizer(v, {"train": t}, total_steps)
    step = load_checkpoint(ckpt_path, v, opt, sched) if ckpt_path.exists() else 0
    if step:
        log(f"resumed from step {step}")
    check_fp32(v, opt)
    sizes = [len(e["spans"]) for e in entries]
    tgts = [torch.tensor(question_targets(e, cfg), dtype=torch.float32) for e in entries]
    use_amp = device.startswith("cuda")
    v.train()
    losses, cand_sum, loss_sum, loss_n, t0 = [], 0, 0.0, 0, time.time()
    order, order_epoch = None, -1
    while step < total_steps:
        epoch, pos = divmod(step, n_batches)
        if epoch != order_epoch:                        # a fixed shuffle per (seed, epoch), so a resume continues it
            order, order_epoch = np.random.default_rng([seed, epoch]).permutation(n).tolist(), epoch
        batch_ids = order[pos * t["effective_batch"]:(pos + 1) * t["effective_batch"]]
        step_loss = 0.0
        for ids in micro_batches(batch_ids, sizes, per_device):
            b, question = batch_of(marker, [examples[i] for i in ids], [entries[i] for i in ids], device, pad_id)
            target = torch.cat([tgts[i] for i in ids]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                scores = v(b)
            per_q = V.listwise_loss(scores, question, target, len(ids))
            (per_q.sum() / len(batch_ids)).backward()
            step_loss += per_q.sum().item()
            cand_sum += len(question)
        torch.nn.utils.clip_grad_norm_(v.trainable_parameters(), t["grad_clip"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        loss_sum, loss_n = loss_sum + step_loss, loss_n + len(batch_ids)
        losses.append(step_loss / len(batch_ids))
        if step % 50 == 0 or step == total_steps or step == stop_after:
            el = time.time() - t0
            log(f"step {step}/{total_steps} loss {loss_sum / loss_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{cand_sum / max(el, 1e-9):.1f} candidates/s {el / 60:.1f} min")
            loss_sum, loss_n = 0.0, 0
        if step % t["checkpoint_every"] == 0 or step == total_steps or step == stop_after:
            save_checkpoint(ckpt_path, v, opt, sched, step)
        if stop_after is not None and step >= stop_after and step < total_steps:
            return {"steps": step, "stopped_early": True, "losses": losses, "minutes": (time.time() - t0) / 60}
    return {"steps": step, "stopped_early": False, "losses": losses, "minutes": (time.time() - t0) / 60}


def score_lists(v, marker: Marker, examples: list[D.Example], entries: list[dict], device: str, pad_id: int,
                per_device: int) -> list[np.ndarray]:
    """The verifier's score of every candidate of every list, adapters on, in eval mode."""
    v.eval()
    sizes = [len(e["spans"]) for e in entries]
    out: list[np.ndarray] = [None] * len(entries)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        for ids in micro_batches(list(range(len(entries))), sizes, per_device):
            b, question = batch_of(marker, [examples[i] for i in ids], [entries[i] for i in ids], device, pad_id)
            s = v(b).float().cpu().numpy()
            q = question.cpu().numpy()
            for k, i in enumerate(ids):
                out[i] = s[q == k]
    return out


def fit_temperature_or_edge(items: list[tuple[np.ndarray, np.ndarray]], lo: float = 0.05,
                            hi: float = 20.0) -> tuple[float, bool]:
    """decode.fit_temperature over [lo, hi], or the better end of the range when the optimum lies at
    an end (flat or perfectly separating scores, e.g. an untrained verifier or a tiny test list).
    The flag goes into metrics.json, so the fallback is recorded, and evaluation after hours of
    training doesn't crash on it."""
    from squadqa.decode import fit_temperature, temperature_nll

    try:
        return fit_temperature(items, lo, hi), False
    except RuntimeError:
        return (lo if temperature_nll(items, lo) <= temperature_nll(items, hi) else hi), True


def evaluate(examples: list[D.Example], entries: list[dict], scores: list[np.ndarray],
             rcfg: dict) -> tuple[dict, dict]:
    """The combination's settings and the predictions of the reader alone (λ = 0), the verifier
    alone and combined, under each of the reader's decode_rules (spec §5.1, "Combining").

    λ is fit first (verifier.fit_lambda), at the reader's own calibration; then one temperature on
    the combined scores log p_reader + λ · s, and one on the verifier's scores s alone, each by the
    NLL of the gold's EM-1 candidates, as the reader's calibration fits its temperature (spec
    §4.4). MBR weighs the candidates by these probabilities, so the temperature changes its
    answers; MAP's doesn't move. All go through decode.decode on the k candidates, so boundary
    variants are grouped as in the reader's own MAP and MBR; the reader alone over k = 5 can differ
    slightly from its run's predictions, which decode over decode_topk."""
    from squadqa.decode import Candidates, answer_text, content_flags, decode, log_softmax, temperature_nll

    items = []
    for e, s in zip(entries, scores):
        em = np.array([0 if x is None else x for x in e["em"]])
        if (em == 1).any():
            items.append((np.array(e["logp"]), s, np.flatnonzero(em == 1)))
    lam = V.fit_lambda(items) if items else 0.0         # no list holds its gold: the reader alone, not a crash after training
    fit = {"lambda": lam, "lambda_questions": len(items)}
    if items:
        combined = [(V.combine(r, s, lam), g) for r, s, g in items]
        verifier = [(np.asarray(s, dtype=np.float64), g) for _, s, g in items]
        t_comb, edge_comb = fit_temperature_or_edge(combined)
        t_ver, edge_ver = fit_temperature_or_edge(verifier)
        fit["temperature"] = {"combined": t_comb, "verifier": t_ver}
        fit["temperature_at_edge"] = {"combined": edge_comb, "verifier": edge_ver}
        fit["nll"] = {"reader": temperature_nll([(r, g) for r, _, g in items], 1.0),
                      "verifier": temperature_nll(verifier, t_ver),
                      "combined_at_1": temperature_nll(combined, 1.0), "combined": temperature_nll(combined, t_comb)}
    else:
        print("warning: no validation list holds an EM-1 candidate, so λ = 0 and no temperature is fit", flush=True)
        t_comb = t_ver = 1.0
    preds = {f"{who}_{rule}": {} for who in ("reader", "verifier", "combined") for rule in rcfg["decode_rules"]}
    for ex, e, s in zip(examples, entries, scores):
        q = ex.question
        flags = content_flags(q.context, ex.context.words)
        starts, ends = np.array([a for a, _ in e["spans"]]), np.array([b for _, b in e["spans"]])
        texts = [answer_text(q.context, ex.context.words, a, b) for a, b in e["spans"]]
        for who, lp in (("reader", np.array(e["logp"], dtype=np.float64)),
                        ("verifier", log_softmax(s, t_ver)),
                        ("combined", log_softmax(V.combine(e["logp"], s, lam), t_comb))):
            for rule in rcfg["decode_rules"]:
                i, j = decode(Candidates(starts, ends, lp), flags, rule, rcfg["mbr_utility"], rcfg["mbr_em_weight"], texts)
                preds[f"{who}_{rule}"][q.id] = answer_text(q.context, ex.context.words, i, j)
    return fit, preds


def by_id(feats: D.Features, lists: dict[str, dict]) -> tuple[list[D.Example], list[dict]]:
    """The examples that have a list, with their lists, in feature order. Raises if a list's question
    is missing, which would mean the lists come from other data."""
    have = {ex.question.id: ex for ex in feats.examples}
    missing = [q for q in lists if q not in have]
    if missing:
        raise RuntimeError(f"{len(missing)} listed questions are not in the split, e.g. {missing[:3]}")
    examples = [ex for ex in feats.examples if ex.question.id in lists]
    return examples, [lists[ex.question.id] for ex in examples]


def main(argv: list[str] | None = None) -> None:
    from squadqa import predict as P

    parser = argparse.ArgumentParser(description="Train the shared_lora verifier, then score validation.")
    C.add_run_args(parser)
    parser.add_argument("--reader", required=True, help="the frozen reader's run id (its lists and checkpoint)")
    parser.add_argument("--lists", help="folder of the candidate lists (default: data/verifier/<reader>)")
    parser.add_argument("--per-device-batch", type=int, default=16,
                        help="candidates per forward pass (a question's candidates stay together)")
    parser.add_argument("--max-steps", type=int, help="smoke runs: total optimizer steps instead of the epochs")
    parser.add_argument("--stop-after", type=int, help="smoke runs: exit after this step without scoring")
    args = parser.parse_args(argv)

    cfg = C.load_config(args.config, args.set)
    if cfg["verifier"]["variant"] != "shared_lora" or cfg["verifier"]["type"] != "in_place":
        raise NotImplementedError("only the in-place shared_lora verifier is built (#67)")
    for key, default in NOT_BUILT.items():
        if cfg["verifier"].get(key, default) != default:
            raise NotImplementedError(f"verifier.{key}: {cfg['verifier'][key]!r} is not built; only {default!r}")
    cfg["verifier"]["reader"] = args.reader             # part of the resolved config, so a resume checks it
    run_dir = C.start_run(cfg, args.config, args.seed, args.run_id)
    ckpt_dir = Path(C.env("SQUADQA_SCRATCH")) / "runs" / run_dir.name
    C.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    print(f"run {run_dir.name} on {gpu}; reader {args.reader}; checkpoints in {ckpt_dir}", flush=True)

    rcfg, tok, reader, reader_step = P.load_reader(args.reader, device)
    open_id, close_id = V.add_markers(tok)
    V.resize_for_markers(reader.encoder, tok)
    v = V.SharedLoraVerifier(reader, open_id, close_id, cfg).to(device)
    marker = Marker(tok, rcfg, open_id, close_id)
    folder = lists_dir(args.reader, args.lists)

    t0 = time.time()
    feats = D.featurize(D.load_split("train"), tok, rcfg, training=True)
    examples, entries = by_id(feats, load_lists(folder, "train"))
    print(f"{len(examples)} training lists, {sum(len(e['spans']) for e in entries)} candidates; "
          f"loaded in {time.time() - t0:.0f} s", flush=True)
    before = reader_fingerprint(v)
    result = train(v, marker, examples, entries, cfg, args.seed, tok.pad_token_id, ckpt_dir / CHECKPOINT,
                   args.per_device_batch, device, args.max_steps, args.stop_after, log=lambda m: print(m, flush=True))
    if reader_fingerprint(v) != before:
        raise RuntimeError("a reader weight changed during verifier training (V4)")
    if result["stopped_early"]:
        print(f"stopped after step {result['steps']} (--stop-after); rerun with --run-id {run_dir.name} to resume",
              flush=True)
        return

    peak_train = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    vfeats = D.featurize(D.load_split("validation"), tok, rcfg, training=False)
    vexamples, ventries = by_id(vfeats, load_lists(folder, "validation"))
    if len(vexamples) != len(vfeats.examples):
        raise RuntimeError(f"{len(vfeats.examples) - len(vexamples)} validation questions have no list")
    scores = score_lists(v, marker, vexamples, ventries, device, tok.pad_token_id, args.per_device_batch)
    # Evaluate the scores as saved (6 decimals), so a later --rescore reproduces this evaluation exactly.
    scores = [np.array([round(float(x), 6) for x in s]) for s in scores]
    C.write_json(run_dir / SCORES, {ex.question.id: s.tolist() for ex, s in zip(vexamples, scores)})
    metrics = write_evaluation(run_dir, vexamples, ventries, scores, rcfg, {
        "reader": args.reader, "reader_step": reader_step, "lists": str(folder),
        "steps": result["steps"], "train_minutes_last_job": round(result["minutes"], 1), "gpu": gpu,
        "per_device_batch": args.per_device_batch, "peak_gpu_gb_train": round(peak_train, 1)})
    print("RESULT", {k: metrics[k] for k in ("validation", "lambda", "steps")}, flush=True)


def write_evaluation(run_dir: Path, examples: list[D.Example], entries: list[dict], scores: list[np.ndarray],
                     rcfg: dict, info: dict) -> dict:
    """Evaluate (λ, temperatures, predictions), write the predictions and metrics.json (info's keys
    first, then the evaluation's), and return the metrics."""
    fit, preds = evaluate(examples, entries, scores, rcfg)
    dataset = E.load_eval_set("validation")
    for name, answers in preds.items():
        C.write_json(run_dir / f"predictions_validation_{name}.json", answers)
    metrics = {**info, "validation": {name: E.metrics("validation", dataset, p) for name, p in preds.items()}, **fit}
    C.write_json(run_dir / "metrics.json", metrics)
    return metrics


def rescore(run_id: str, lists: str | None = None) -> dict:
    """Re-run the validation evaluation of a finished verifier run from its saved scores: no model,
    no GPU, only the reader's tokenizer for the windows. The training entries of metrics.json are
    kept; the evaluation's are replaced, and "rescored" records the commit that did it."""
    import yaml
    from squadqa import predict as P

    run_dir = C.RESULTS / run_id
    for name in ("config.yaml", "metrics.json", SCORES):
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"{run_dir / name} is missing: not a finished verifier run")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    reader = cfg["verifier"]["reader"]
    rcfg = P.run_config(reader)
    tok = D.load_tokenizer(rcfg["model"])
    folder = lists_dir(reader, lists)
    vfeats = D.featurize(D.load_split("validation"), tok, rcfg, training=False)
    examples, entries = by_id(vfeats, load_lists(folder, "validation"))
    saved = json.loads((run_dir / SCORES).read_text(encoding="utf-8"))
    if set(saved) != {ex.question.id for ex in examples}:
        raise RuntimeError(f"{SCORES} and the lists in {folder} hold different questions")
    scores = [np.array(saved[ex.question.id], dtype=np.float64) for ex in examples]
    if any(len(s) != len(e["spans"]) for s, e in zip(scores, entries)):
        raise RuntimeError(f"{SCORES} doesn't match the lists in {folder}: other lists than the run scored")
    old = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    keep = {k: v for k, v in old.items() if k not in ("validation", "lambda", "lambda_questions", "temperature",
                                                      "temperature_at_edge", "nll", "rescored")}
    commit, dirty = C.git_state()
    keep["rescored"] = {"git_commit": commit, "git_dirty": dirty, "lists": str(folder)}
    metrics = write_evaluation(run_dir, examples, entries, scores, rcfg, keep)
    print("RESCORED", {k: metrics[k] for k in ("validation", "lambda")}, flush=True)
    return metrics


def cli(argv: list[str] | None = None) -> None:
    """main, or rescore with --rescore <run_id> [--lists DIR]."""
    argv = sys.argv[1:] if argv is None else argv
    if "--rescore" in argv:
        p = argparse.ArgumentParser(description="Re-run a finished verifier run's validation evaluation.")
        p.add_argument("--rescore", required=True, metavar="RUN_ID")
        p.add_argument("--lists", help="folder of the candidate lists (default: data/verifier/<reader>)")
        a = p.parse_args(argv)
        rescore(a.rescore, a.lists)
    else:
        main(argv)


if __name__ == "__main__":
    cli()
