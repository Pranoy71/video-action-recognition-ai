"""Evaluation: test-split metrics, plots, efficiency benchmark.

Deliberate protocol choices:
  * The test split is evaluated exactly once per model, on the best-on-val
    checkpoint. No test-set peeking during model selection.
  * Beyond Top-1/Top-5 we report macro-F1 and per-class recall: on a
    class-imbalanced subset, a single scalar accuracy hides per-class rot.
  * Efficiency is measured, not asserted: parameters, MACs per clip, weight
    size, and end-to-end CPU/GPU throughput (clips/s + latency). The
    accuracy-vs-throughput trade-off is the actual decision variable for
    deployment, which is the whole point of the quantization experiment.
  * For frame-level models (A/B) the benchmarked pipeline is
    frozen-ResNet18 backbone + trained head — i.e. the real inference cost,
    not just the cheap head on cached features.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common import env_snapshot, find_repo_root, get_device, load_config, save_json
from src.data.video_dataset import FeatureDataset, VideoDataset
from src.engine import count_params, evaluate
from src.models.classifiers import build_model


# --------------------------------------------------------------------------- #
# End-to-end pipeline wrapper (bench / inference only)
# --------------------------------------------------------------------------- #
class EndToEndFrameModel(nn.Module):
    """Frozen ResNet-18 backbone + trained temporal head, as one module.

    This is what actually runs in production for models A/B: the cached
    features are a training-speed optimisation, not a deployment artifact.
    """

    def __init__(self, head: nn.Module):
        super().__init__()
        from src.data.extract_features import ResNet18FeatureExtractor
        self.backbone = ResNet18FeatureExtractor()
        self.head = head

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(clips))


def load_trained(cfg: dict, run_name: str, out_dir: Path):
    ckpt = torch.load(out_dir / "checkpoints" / f"{run_name}_best.pt",
                      map_location="cpu", weights_only=False)
    num_classes = ckpt["num_classes"]
    model = build_model(cfg, num_classes)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


# --------------------------------------------------------------------------- #
# Efficiency benchmark
# --------------------------------------------------------------------------- #
@torch.no_grad()
def bench_throughput(model: nn.Module, device: torch.device, batch: int = 4,
                     num_frames: int = 16, warmup: int = 3, iters: int = 10,
                     size: int = 112) -> Dict[str, float]:
    """Clips/s and ms/clip on synthetic (batch, 3, T, H, W) input.

    Synthetic input is the right call for a throughput benchmark: we measure
    compute, not data loading (decode FPS is reported separately in the
    README discussion).
    """
    model = model.to(device).eval()
    x = torch.randn(batch, 3, num_frames, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return {"clips_per_sec": batch / dt, "ms_per_clip": 1000.0 * dt / batch,
            "batch": batch, "iters": iters, "device": str(device)}


def macs_per_clip(model: nn.Module, num_frames: int = 16, size: int = 112) -> float:
    """MACs (multiply-accumulates) for one clip via `thop`."""
    try:
        from thop import profile
        model = model.cpu().eval()
        x = torch.randn(1, 3, num_frames, size, size)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs)
    except Exception as e:  # thop is a soft dependency
        print(f"[bench] thop unavailable: {e}")
        return float("nan")


def weight_size_mb(model: nn.Module) -> float:
    tmp = Path(__file__).parent.parent / "results" / "_size_tmp.pt"
    torch.save(model.state_dict(), tmp)
    mb = tmp.stat().st_size / 1e6
    tmp.unlink(missing_ok=True)
    return mb


# --------------------------------------------------------------------------- #
# Metrics + plots
# --------------------------------------------------------------------------- #
def compute_metrics(preds: List[int], labels: List[int], n: int) -> Dict[str, float]:
    cm = np.zeros((n, n), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    top1 = float(np.trace(cm) / cm.sum())
    per_class = np.where(cm.sum(1) > 0, np.diag(cm) / np.maximum(cm.sum(1), 1), np.nan)
    macro_f1 = float(np.mean([2 * cm[i, i] / max(cm[i, :].sum() + cm[:, i].sum(), 1)
                              for i in range(n)]))
    return {"top1": top1, "macro_f1": macro_f1,
            "per_class_recall": per_class.tolist(), "confusion": cm.tolist()}


def top_confused_pairs(cm: np.ndarray, classes: List[str], k: int = 10):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for i, j in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape)):
        if off[i, j] == 0 or len(pairs) >= k:
            break
        pairs.append({"true": classes[i], "pred": classes[j],
                      "count": int(off[i, j])})
    return pairs


def plot_confusion(cm: np.ndarray, classes: List[str], out: Path,
                   run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(classes)
    fig, ax = plt.subplots(figsize=(1.9 + 0.42 * n, 1.6 + 0.38 * n),
                           constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{run_name}: test confusion matrix")
    thresh = cm.max() * 0.6 if cm.max() else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_per_class(recall: np.ndarray, classes: List[str], out: Path,
                   run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(recall)
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.barh([classes[i] for i in order], recall[order], color="#377eb8")
    ax.axvline(float(np.nanmean(recall)), color="#e41a1c", ls="--",
               label=f"mean {np.nanmean(recall):.3f}")
    ax.set_xlabel("recall (per-class accuracy)")
    ax.set_title(f"{run_name}: per-class recall on test")
    ax.legend()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_history(history_csv: Path, out: Path, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(history_csv)))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax1.plot(epochs, [float(r["train_top1"]) for r in rows], label="train")
    ax1.plot(epochs, [float(r["val_top1"]) for r in rows], label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("top-1 acc"); ax1.legend()
    ax1.set_title(f"{run_name}: accuracy")
    ax2.plot(epochs, [float(r["train_loss"]) for r in rows], label="train")
    ax2.plot(epochs, [float(r["val_loss"]) for r in rows], label="val")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("cross-entropy"); ax2.legend()
    ax2.set_title(f"{run_name}: loss")
    fig.savefig(out, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(cfg: dict, run_name: str, out_dir: Path) -> dict:
    device = get_device()
    repo = find_repo_root()
    data_dir = cfg["data"].get("data_dir", "data")
    data_root = repo / data_dir / "ucf101"

    model, ckpt = load_trained(cfg, run_name, out_dir)
    classes = ckpt.get("classes") or \
        (data_root / "classes.txt").read_text().split()
    n = len(classes)

    kind = cfg["model"].get("input_kind") or ("video" if cfg["model"]["name"] == "r3d18" else "feature")
    if kind == "feature":
        feat_dir = repo / data_dir / cfg["data"].get("features_dir", "features_resnet18")
        ds = FeatureDataset(feat_dir, "test")
    else:
        ds = VideoDataset(data_root, "test", num_frames=cfg["data"]["num_frames"],
                          spatial_size=cfg["data"].get("spatial_size", 112),
                          norm=cfg["data"].get("norm", "kinetics"), train=False,
                          seed=cfg.get("seed", 42))
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"] * 2,
                        num_workers=cfg["data"].get("workers", 2))

    model = model.to(device)
    res = evaluate(model, loader, device)
    met = compute_metrics(res["preds"], res["labels"], n)
    cm = np.array(met["confusion"])

    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, classes, plots / f"{run_name}_confusion.png", run_name)
    plot_per_class(np.array(met["per_class_recall"]), classes,
                   plots / f"{run_name}_perclass.png", run_name)
    hist = out_dir / "logs" / f"{run_name}_history.csv"
    if hist.exists():
        plot_history(hist, plots / f"{run_name}_curves.png", run_name)

    # Efficiency: benchmark the real end-to-end pipeline.
    if kind == "feature":
        bench_model = EndToEndFrameModel(model.cpu()).eval()
    else:
        bench_model = model.cpu().eval()
    cpu_fps = bench_throughput(bench_model, torch.device("cpu"),
                               num_frames=cfg["data"].get("num_frames", 16))
    gpu_fps = (bench_throughput(bench_model, torch.device("cuda"),
                                num_frames=cfg["data"].get("num_frames", 16))
               if torch.cuda.is_available() else None)

    summary = {
        "run_name": run_name, "model": cfg["model"]["name"],
        "num_classes": n, "classes": classes,
        "test": {"top1": met["top1"], "top5": res["top5"],
                 "macro_f1": met["macro_f1"], "n": res["n"]},
        "best_val_top1": ckpt.get("val_top1"),
        "efficiency": {
            # End-to-end pipeline accounting (backbone + head for A/B):
            # MACs, params and size must describe the same object that runs
            # in production, or the table quietly mixes apples and oranges.
            **count_params(bench_model),
            "params_trainable_head": count_params(model)["params_trainable"],
            "macs_per_clip": macs_per_clip(bench_model,
                                           cfg["data"].get("num_frames", 16)),
            "weights_mb": weight_size_mb(bench_model),
            "cpu": {k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in cpu_fps.items()},
            "gpu": ({k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in gpu_fps.items()} if gpu_fps else None),
        },
        "top_confused_pairs": top_confused_pairs(cm, classes),
        "decode_failures_test": getattr(ds, "decode_failures", [])[:20],
        "env": env_snapshot(),
    }
    save_json(summary, out_dir / "logs" / f"{run_name}_metrics.json")

    print(f"[{run_name}] TEST top1={met['top1']:.3f} top5={res['top5']:.3f} "
          f"macroF1={met['macro_f1']:.3f} | CPU {cpu_fps['clips_per_sec']:.2f} clips/s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    run_name = args.run_name or cfg.get("run_name") or Path(args.config).stem
    out_dir = find_repo_root() / cfg.get("out_dir", "results")
    run(cfg, run_name, out_dir)


if __name__ == "__main__":
    main()
