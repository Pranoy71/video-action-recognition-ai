"""Config-driven training entry point.

Usage:
    python -m src.train --config configs/framepool.yaml
    python -m src.train --config configs/r3d18.yaml --epochs 3   # quick run
    python -m src.train --config configs/r3d18.yaml --smoke      # CI/debug

Protocol notes (identical across models — this is what makes the
cross-model comparison meaningful):
  * same splits, same seed, same epochs budget, same cosine schedule shape
  * optimizer family differs where convention demands it (Adam for the
    transformer head, SGD+momentum for conv fine-tuning) — recorded in the
    config and in metrics.json, and discussed in the README
  * feature-input models (A/B) train on the same cached features
  * best-on-val checkpointing: the test split is touched ONLY by
    evaluate.py, once, at the very end
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common import (env_snapshot, find_repo_root, get_device, load_config,
                        save_json, set_global_seed)
from src.data.video_dataset import FeatureDataset, VideoDataset
from src.engine import count_params, evaluate, train_one_epoch
from src.models.classifiers import build_model, param_groups


def build_loaders(cfg: dict, data_root: Path) -> tuple:
    """Route each model to its dataset by input kind (feature vs video)."""
    kind = cfg["model"].get("input_kind") or _input_kind(cfg)
    dcfg = cfg["data"]
    if kind == "feature":
        feat_dir = data_root.parent / dcfg.get("features_dir", "features_resnet18")
        mk = lambda split: FeatureDataset(feat_dir, split)
        workers = 2
    else:
        def mk(split):
            return VideoDataset(
                data_root, split, num_frames=dcfg["num_frames"],
                spatial_size=dcfg.get("spatial_size", 112),
                norm=dcfg.get("norm", "kinetics"),
                train=(split == "train"),
                seed=cfg.get("seed", 42))
        workers = dcfg.get("workers", 4)
    tcfg = cfg["training"]
    train_loader = DataLoader(mk("train"), batch_size=tcfg["batch_size"],
                              shuffle=True, num_workers=workers,
                              pin_memory=torch.cuda.is_available(),
                              persistent_workers=False)
    val_loader = DataLoader(mk("val"), batch_size=tcfg["batch_size"] * 2,
                            shuffle=False, num_workers=workers,
                            pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, mk


def _input_kind(cfg: dict) -> str:
    name = cfg["model"]["name"]
    kind_map = {"framepool": "feature", "temporal_transformer": "feature",
                "r3d18": "video"}
    return kind_map[name]


def run(cfg: dict, run_name: str, out_dir: Path, smoke: bool = False,
        resume: bool = False, max_epochs_this_run: int | None = None) -> dict:
    device = get_device()
    set_global_seed(cfg.get("seed", 42))

    repo = find_repo_root()
    data_dir = cfg["data"].get("data_dir", "data")
    data_root = repo / data_dir / "ucf101"

    train_loader, val_loader, ds_factory = build_loaders(cfg, data_root)
    num_classes = ds_factory("val").num_classes

    model = build_model(cfg, num_classes).to(device)
    tcfg = cfg["training"]
    epochs = 1 if smoke else tcfg["epochs"]
    max_batches = 3 if smoke else None

    opt_name = tcfg.get("optimizer", "sgd")
    if opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups(model, cfg),
                                     weight_decay=tcfg.get("weight_decay", 1e-4))
    else:
        optimizer = torch.optim.SGD(param_groups(model, cfg),
                                    momentum=tcfg.get("momentum", 0.9),
                                    weight_decay=tcfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(len(train_loader), 1))

    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))
    amp = torch.cuda.is_available() and tcfg.get("amp", True)

    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: full training state (model/optimizer/scheduler/epoch/
    # best metric/history). Primarily for surviving Colab session drops; also
    # lets low-RAM machines train one epoch per process invocation.
    state_path = ckpt_dir / f"{run_name}_state.pt"
    history = []
    best_top1 = -1.0
    best_val_loss = float("inf")
    start_epoch = 0
    if resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        start_epoch = st["epoch"] + 1
        best_top1 = st["best_top1"]
        best_val_loss = st.get("best_val_loss", float("inf"))
        history = st["history"]
        print(f"[{run_name}] resumed from epoch {start_epoch} "
              f"(best val top1 so far: {best_top1:.3f})", flush=True)

    end_epoch = epochs if max_epochs_this_run is None else min(
        epochs, start_epoch + max_epochs_this_run)

    t_start = time.time()
    for epoch in range(start_epoch, end_epoch):
        tr = train_one_epoch(model, train_loader, device, criterion, optimizer,
                             epoch, amp=amp, max_batches=max_batches)
        for _ in range(len(train_loader) if max_batches is None else 1):
            scheduler.step()
        va = evaluate(model, val_loader, device, criterion,
                      max_batches=max_batches)
        history.append({"epoch": epoch + 1, "train_loss": round(tr["loss"], 5),
                        "train_top1": round(tr["top1"], 5),
                        "val_loss": round(va["loss"], 5),
                        "val_top1": round(va["top1"], 5),
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch_sec": round(tr["time_sec"], 1)})
        print(f"[{run_name}] epoch {epoch+1}/{epochs} "
              f"train_top1={tr['top1']:.3f} val_top1={va['top1']:.3f} "
              f"val_top5={va['top5']:.3f} ({tr['time_sec']:.0f}s)", flush=True)
        # Tie-aware model selection: on a small/saturated val split, plain
        # "strictly greater" freezes the checkpoint at the first 1.000 epoch.
        # Falling back to val loss breaks ties toward better-calibrated
        # later epochs.
        if va["top1"] > best_top1 or (
                va["top1"] == best_top1 and va["loss"] < best_val_loss):
            best_top1 = va["top1"]
            best_val_loss = va["loss"]
            torch.save({"model": model.state_dict(), "config": cfg,
                        "num_classes": num_classes, "val_top1": best_top1,
                        "classes": _classes_from(data_root)},
                       ckpt_dir / f"{run_name}_best.pt")
        # Training state saved every epoch: resume is always current.
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "best_top1": best_top1,
                    "best_val_loss": best_val_loss,
                    "history": history, "num_classes": num_classes},
                   state_path)

    total_min = (time.time() - t_start) / 60
    with open(log_dir / f"{run_name}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    summary = {"run_name": run_name, "best_val_top1": best_top1,
               "epochs": epochs, "train_minutes": round(total_min, 2),
               **count_params(model), "env": env_snapshot()}
    save_json(summary, log_dir / f"{run_name}_summary.json")
    print(f"[{run_name}] done in {total_min:.1f} min — best val top1 {best_top1:.3f}")
    return summary


def _classes_from(data_root: Path):
    f = data_root / "classes.txt"
    return f.read_text().split() if f.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <run>_state.pt (survives session drops)")
    ap.add_argument("--max-epochs-this-run", type=int, default=None,
                    help="run at most N more epochs this process (low-RAM mode)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
        cfg["data"]["workers"] = 0  # small-RAM machines: no worker processes
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    run_name = args.run_name or cfg.get("run_name") or \
        Path(args.config).stem
    out_dir = find_repo_root() / cfg.get("out_dir", "results")
    run(cfg, run_name, out_dir, smoke=args.smoke, resume=args.resume,
        max_epochs_this_run=args.max_epochs_this_run)


if __name__ == "__main__":
    main()
