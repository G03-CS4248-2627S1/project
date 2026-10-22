"""Train a reader from a config, then predict and score the validation split (spec §4, §9).

    python -m squadqa.train_reader --config configs/experiments/row0-baseline.yaml --seed 0 \
        [--run-id ID] [--set train.lr=3e-5] [--per-device-batch 16] [--max-steps N] [--stop-after N]

- fp32 master weights, gradients and AdamW state; bf16 autocast on the GPU. Raises if a trainable
  parameter or optimizer state is not fp32, after loading and at every checkpoint (spec §9).
- Effective batch cfg["train"]["effective_batch"] questions per optimizer step (all their
  windows); --per-device-batch questions per forward pass, with gradient accumulation between.
- Checkpoints to $SQUADQA_SCRATCH/runs/<run_id>/checkpoint.pt every cfg["train"]["checkpoint_every"]
  steps and at the end. Re-running with the same run id resumes from it (spec rule 8).
- Then writes validation predictions, candidates, calibration and metrics.json to
  results/<run_id>/ (squadqa.predict, squadqa.evaluate).

--eval-each-epoch also scores validation after every epoch but the last, into
results/<run_id>/epoch<k>/ (no candidates file), and lists all epochs under "per_epoch" in
metrics.json. Row 0 tunes the number of epochs this way (#23): one 3-epoch run per learning
rate. The epoch-k model of a longer run is an approximation of a k-epoch run, because its
learning rate has not decayed to zero yet; confirm the chosen setting with a real run.

--save-each-epoch keeps the model's parameters after every epoch but the last (the last is
checkpoint.pt) as $SQUADQA_SCRATCH/runs/<run_id>/model_epoch<k>.pt, without optimizer state, so
about a third of a checkpoint's size; load_checkpoint reads it.

--max-steps and --stop-after are for smoke runs (spec §9): --max-steps shortens the schedule,
--stop-after exits after that step without predicting, as if the job had been killed; resume
with the same arguments minus --stop-after (it is not part of the config, so the resume check passes).
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # pinned files from $HF_HOME only, never the network

import torch

from squadqa import config as C
from squadqa import data as D
from squadqa import evaluate as E

CHECKPOINT = "checkpoint.pt"
EPOCH_MODEL = "model_epoch{}.pt"


def load_encoder(name: str, dtype=torch.float32):
    """The pinned pretrained encoder, loaded with an explicit dtype (transformers 5 otherwise
    keeps the stored one, fp16 for deberta-v3-large)."""
    from transformers import AutoModel

    from squadqa.download_model import MODELS
    C.env("HF_HOME")
    if name not in MODELS:
        raise KeyError(f"{name} is not pinned; add it to squadqa/download_model.py first (spec §9)")
    return AutoModel.from_pretrained(name, revision=MODELS[name]["revision"], dtype=dtype)


def build_optimizer(model, cfg: dict, total_steps: int):
    from transformers import get_linear_schedule_with_warmup

    t = cfg["train"]
    no_decay = lambda n: n.endswith("bias") or "LayerNorm" in n or "layernorm" in n
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": [p for n, p in params if not no_decay(n)], "weight_decay": t["weight_decay"]},
        {"params": [p for n, p in params if no_decay(n)], "weight_decay": 0.0},
    ], lr=t["lr"], fused=torch.cuda.is_available())
    sched = get_linear_schedule_with_warmup(opt, int(t["warmup_frac"] * total_steps), total_steps)
    return opt, sched


def check_fp32(model, opt) -> None:
    """Raise unless every trainable parameter and every optimizer state tensor is fp32 (spec §9)."""
    dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    dtypes |= {v.dtype for s in opt.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0}
    if dtypes != {torch.float32}:
        raise RuntimeError(f"expected fp32 parameters and optimizer state, got {sorted(map(str, dtypes))}")


def save_checkpoint(path: Path, model, opt, sched, step: int) -> None:
    check_fp32(model, opt)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, tmp)
    tmp.replace(path)


def save_model(path: Path, model, step: int) -> None:
    """The model's parameters alone, without optimizer state, in the checkpoint's layout so
    load_checkpoint reads it."""
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "step": step}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model, opt=None, sched=None) -> int:
    """Load into CPU memory first; load_state_dict copies onto the model's device. Loading the
    5.2 GB checkpoint straight onto the GPU fragmented memory and caused out-of-memory retries
    on an A100-40 (smoke run, 2026-10-04)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if opt is not None:
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
    return ck["step"]


def micro_batches(example_ids: list[int], per_device: int, n_windows: list[int] | None = None) -> list[list[int]]:
    """Split one optimizer step's questions into forward passes of at most per_device windows.

    A question's windows always share a pass (one softmax, spec §3.4); a question with more
    windows than per_device gets a pass of its own. Counting questions instead let a pass grow
    past the cap with multi-window questions and ran deberta-v3-large out of memory on an
    A100-40 (jobs 909235-909236, 2026-10-04). n_windows: windows per question (default 1 each).
    """
    out, cur, used = [], [], 0
    for e in example_ids:
        w = n_windows[e] if n_windows is not None else 1
        if cur and used + w > per_device:
            out.append(cur)
            cur, used = [], 0
        cur.append(e)
        used += w
    if cur:
        out.append(cur)
    return out


def train(model, feats: D.Features, cfg: dict, seed: int, pad_id: int, ckpt_path: Path, per_device: int,
          device: str, max_steps: int | None = None, stop_after: int | None = None, log=print,
          on_epoch_end=None, save_each_epoch: bool = False) -> dict:
    """Train with resume from ckpt_path. Returns {"steps", "stopped_early", "losses", "minutes"}.
    After epoch k (not the last), once a checkpoint of that step is saved: with save_each_epoch the
    model's parameters go to model_epoch<k>.pt next to ckpt_path, then on_epoch_end(k) runs. A run
    resumed from such a checkpoint repeats both, since the job may have stopped before they finished."""
    t = cfg["train"]
    n_batches = len(D.bucket_batches(feats.question_lengths(), t["effective_batch"], seed, 0))
    total_steps = max_steps or n_batches * t["epochs"]
    opt, sched = build_optimizer(model, cfg, total_steps)
    hooks = save_each_epoch or on_epoch_end is not None

    def end_epoch(k: int) -> None:
        if save_each_epoch:
            save_model(ckpt_path.with_name(EPOCH_MODEL.format(k)), model, k * n_batches)
        if on_epoch_end is not None:
            on_epoch_end(k)
        model.train()

    step = 0
    if ckpt_path.exists():
        step = load_checkpoint(ckpt_path, model, opt, sched)
        log(f"resumed from step {step}")
    check_fp32(model, opt)
    if hooks and 0 < step < total_steps and step % n_batches == 0:
        end_epoch(step // n_batches)
    use_amp = device.startswith("cuda")
    lengths = feats.question_lengths()
    n_windows = [len(ex.windows) for ex in feats.examples]
    model.train()
    losses, window_sum, loss_sum, loss_n, t0 = [], 0, 0.0, 0, time.time()
    batches, batches_epoch = None, -1
    while step < total_steps:
        epoch, pos = divmod(step, n_batches)
        if epoch != batches_epoch:
            batches, batches_epoch = D.bucket_batches(lengths, t["effective_batch"], seed, epoch), epoch
        batch_ids = batches[pos]
        step_loss = 0.0
        for ids in micro_batches(batch_ids, per_device, n_windows):
            rows = feats.batch(ids)
            b = {k: v.to(device, non_blocking=True) for k, v in D.collate(rows, pad_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                per_q = model.loss(b, len(ids))
            (per_q.sum() / len(batch_ids)).backward()
            step_loss += per_q.sum().item()
            window_sum += len(rows)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        loss_sum, loss_n = loss_sum + step_loss, loss_n + len(batch_ids)
        losses.append(step_loss / len(batch_ids))
        if step % 50 == 0 or step == total_steps or step == stop_after:
            el = time.time() - t0
            check = getattr(model, "stats", {}).get("t_f1_expected_f1")     # spec §4.3's check on t_F1
            log(f"step {step}/{total_steps} loss {loss_sum / loss_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{window_sum / max(el, 1e-9):.1f} windows/s {el / 60:.1f} min"
                + (f" t_F1 expected F1 {check:.3f}" if check is not None else ""))
            loss_sum, loss_n = 0.0, 0
        epoch_end = hooks and step % n_batches == 0 and step < total_steps
        if step % t["checkpoint_every"] == 0 or step == total_steps or epoch_end:
            save_checkpoint(ckpt_path, model, opt, sched, step)
        if epoch_end:
            end_epoch(step // n_batches)
        if stop_after is not None and step >= stop_after and step < total_steps:
            return {"steps": step, "stopped_early": True, "losses": losses, "minutes": (time.time() - t0) / 60}
    return {"steps": step, "stopped_early": False, "losses": losses, "minutes": (time.time() - t0) / 60}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a reader, then predict and score the validation split.")
    C.add_run_args(parser)
    parser.add_argument("--per-device-batch", type=int, default=16,
                        help="windows per forward pass (a question's windows stay together); 8 for large on an A100-40")
    parser.add_argument("--max-steps", type=int, help="smoke runs: total optimizer steps instead of the epochs")
    parser.add_argument("--stop-after", type=int, help="smoke runs: exit after this step without predicting")
    parser.add_argument("--eval-each-epoch", action="store_true",
                        help="also score validation after every epoch, into results/<run_id>/epoch<k>/")
    parser.add_argument("--save-each-epoch", action="store_true",
                        help="also keep the model's parameters after every epoch but the last, as "
                             "model_epoch<k>.pt next to the checkpoint")
    args = parser.parse_args(argv)

    cfg = C.load_config(args.config, args.set)
    run_dir = C.start_run(cfg, args.config, args.seed, args.run_id)
    ckpt_dir = Path(C.env("SQUADQA_SCRATCH")) / "runs" / run_dir.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    C.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    print(f"run {run_dir.name} on {gpu}; checkpoints in {ckpt_dir}", flush=True)

    from squadqa.heads import Reader
    tok = D.load_tokenizer(cfg["model"])
    t0 = time.time()
    feats = D.featurize(D.load_split("train", cfg["extra_train_data"]), tok, cfg, training=True)
    print(f"{len(feats.examples)} training questions, {len(feats.windows)} windows, "
          f"{len(feats.skipped)} skipped; featurized in {time.time() - t0:.0f} s", flush=True)
    model = Reader(load_encoder(cfg["model"]), cfg).to(device)
    from squadqa.predict import predict_split, predict_validation

    def eval_epoch(k: int) -> None:
        epoch_dir = run_dir / f"epoch{k}"
        epoch_dir.mkdir(exist_ok=True)
        out = predict_split(model, tok, cfg, epoch_dir, device, "validation", D.load_split("validation"),
                            E.load_eval_set("validation"), save_candidates=False)
        C.write_json(epoch_dir / "metrics.json", out)
        print(f"epoch {k}:", {rule: (round(m["exact"], 2), round(m["f1"], 2)) for rule, m in out["validation"].items()},
              flush=True)

    result = train(model, feats, cfg, args.seed, tok.pad_token_id, ckpt_dir / CHECKPOINT, args.per_device_batch,
                   device, args.max_steps, args.stop_after, log=lambda m: print(m, flush=True),
                   on_epoch_end=eval_epoch if args.eval_each_epoch else None, save_each_epoch=args.save_each_epoch)
    if result["stopped_early"]:
        print(f"stopped after step {result['steps']} (--stop-after); rerun with --run-id {run_dir.name} to resume",
              flush=True)
        return

    peak_train = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    if device == "cuda":
        torch.cuda.empty_cache()                        # the optimizer is gone; release its cached blocks
        torch.cuda.reset_peak_memory_stats()
    metrics = predict_validation(model, tok, cfg, run_dir, device)
    peak_predict = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    metrics.update({"steps": result["steps"], "train_minutes_last_job": round(result["minutes"], 1), "gpu": gpu,
                    "per_device_batch": args.per_device_batch,
                    "peak_gpu_gb": {"train": round(peak_train, 1), "predict": round(peak_predict, 1)}})
    if args.eval_each_epoch:
        per_epoch = {p.name[5:]: json.loads((p / "metrics.json").read_text(encoding="utf-8"))["validation"]
                     for p in sorted(run_dir.glob("epoch*")) if (p / "metrics.json").exists()}
        per_epoch[str(cfg["train"]["epochs"])] = metrics["validation"]
        metrics["per_epoch"] = per_epoch
    C.write_json(run_dir / "metrics.json", metrics)
    print("RESULT", {k: metrics[k] for k in ("validation", "steps")}, flush=True)


if __name__ == "__main__":
    main()
