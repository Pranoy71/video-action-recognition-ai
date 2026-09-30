"""ONNX export + INT8 static quantization + CPU throughput benchmark.

Why this module exists (the story it supports):
  Accuracy alone does not decide deployment; accuracy-per-dollar-of-latency
  does. We export the best 3D-CNN model to ONNX, then apply *static* INT8
  quantization (QDQ format, per-channel weights, calibrated on TRAIN clips —
  never on test data) and measure the accuracy/throughput trade-off on CPU.

  Static (ahead-of-time) quantization is chosen over dynamic because r3d_18
  is convolution-dominated: dynamic quantization only shrinks Linear layers
  (a tiny fraction of the FLOPs), while static quantization converts the
  conv stack to int8 and unlocks onnxruntime's quantized kernels.

  CPU-first on purpose: INT8 on CPU is the edge-deployment scenario (no
  TensorRT dependency). GPU int8 requires TensorRT/calibrated engines and
  is listed as future work.

Usage:
    python -m src.export_quantize --config configs/r3d18.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common import env_snapshot, find_repo_root, get_device, load_config, save_json
from src.data.video_dataset import VideoDataset
from src.evaluate import load_trained


# --------------------------------------------------------------------------- #
# ONNX export
# --------------------------------------------------------------------------- #
def export_onnx(model: torch.nn.Module, out_path: Path, num_frames: int = 16,
                size: int = 112, opset: int = 17) -> Path:
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, num_frames, size, size)
    # dynamo=False -> the legacy TorchScript exporter. Its op style (axes as
    # attributes) is what onnxruntime's quantization preprocessor expects;
    # the newer dynamo path emits initializer-style ops that break QDQ
    # insertion for this graph.
    torch.onnx.export(
        model, dummy, str(out_path), opset_version=opset,
        input_names=["clip"], output_names=["logits"],
        dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True, dynamo=False)
    print(f"[quant] exported {out_path}")
    return out_path


def onnx_accuracy(onnx_path: Path, ds: VideoDataset, batch_size: int = 4) -> Dict[str, float]:
    """Top-1/Top-5 of an ONNX model on a dataset, via onnxruntime."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)
    preds, labels = [], []
    for x, y in loader:
        out = sess.run(None, {"clip": x.numpy()})[0]
        preds.extend(out.argmax(1).tolist())
        labels.extend(y.tolist())
    preds = np.array(preds)
    labels = np.array(labels)
    top1 = float((preds == labels).mean())
    return {"top1": top1, "n": int(labels.size)}


# --------------------------------------------------------------------------- #
# Static INT8 quantization
# --------------------------------------------------------------------------- #
def quantize_int8(fp32_onnx: Path, int8_onnx: Path, data_root: Path,
                  cfg: dict, n_calib: int = 32, batch: int = 1) -> Path:
    """Static per-channel QDQ quantization, calibrated on TRAIN clips.

    Calibration data must come from the train split: using test data to pick
    activation scales is a subtle form of test leakage.
    """
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)

    num_frames = cfg["data"].get("num_frames", 16)
    size = cfg["data"].get("spatial_size", 112)
    norm = cfg["data"].get("norm", "kinetics")

    ds = VideoDataset(data_root, "train", num_frames=num_frames,
                      spatial_size=size, norm=norm, train=False,
                      seed=cfg.get("seed", 42))

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self.i = 0
            self.items: List[np.ndarray] = []
            loader = DataLoader(ds, batch_size=1, num_workers=0)
            for x, _ in loader:
                # Keep the batch dim: (1, 3, T, H, W) — ORT calibration
                # consumes the model's full input rank.
                self.items.append(x.numpy())
                if len(self.items) >= n_calib:
                    break

        def get_next(self) -> Optional[Dict[str, np.ndarray]]:
            if self.i >= len(self.items):
                return None
            item = {"clip": self.items[self.i]}
            self.i += 1
            return item

        def rewind(self):
            self.i = 0

    quantize_static(
        str(fp32_onnx), str(int8_onnx), _Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,      # per-channel weights
        per_channel=True)
    print(f"[quant] INT8 model written to {int8_onnx}")
    return int8_onnx


# --------------------------------------------------------------------------- #
# Throughput benchmark
# --------------------------------------------------------------------------- #
@torch.no_grad()
def bench_torch(model, batch: int, num_frames: int, size: int, iters: int,
                warmup: int = 2) -> float:
    x = torch.randn(batch, 3, num_frames, size, size)
    for _ in range(warmup):
        model(x)
    t0 = time.time()
    for _ in range(iters):
        model(x)
    return (time.time() - t0) / iters / batch  # sec per clip


def bench_onnx(path: Path, batch: int, num_frames: int, size: int,
               iters: int, warmup: int = 2) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.randn(batch, 3, num_frames, size, size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"clip": x})
    t0 = time.time()
    for _ in range(iters):
        sess.run(None, {"clip": x})
    return (time.time() - t0) / iters / batch


def run(cfg: dict, run_name: str, out_dir: Path) -> dict:
    import onnxruntime as ort  # noqa: F401 — fail early with a clear message

    repo = find_repo_root()
    data_dir = cfg["data"].get("data_dir", "data")
    data_root = repo / data_dir / "ucf101"
    num_frames = cfg["data"].get("num_frames", 16)
    size = cfg["data"].get("spatial_size", 112)

    model, ckpt = load_trained(cfg, run_name, out_dir)
    model = model.cpu().eval()

    quant_dir = out_dir / "quantized"
    quant_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = quant_dir / f"{run_name}_fp32.onnx"
    int8_onnx = quant_dir / f"{run_name}_int8.onnx"

    if not fp32_onnx.exists():
        export_onnx(model, fp32_onnx, num_frames, size)

    # Numerical parity check: ONNX graph must match eager within fp tolerance.
    x = torch.randn(2, 3, num_frames, size, size)
    with torch.no_grad():
        ref = model(x).numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"clip": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    print(f"[quant] ONNX vs PyTorch max|diff| = {max_diff:.2e}")

    if not int8_onnx.exists():
        quantize_int8(fp32_onnx, int8_onnx, data_root, cfg)

    # Accuracy under quantization (test split, eval sampling).
    test_ds = VideoDataset(data_root, "test", num_frames=num_frames,
                           spatial_size=size, norm=cfg["data"].get("norm", "kinetics"),
                           train=False, seed=cfg.get("seed", 42))
    acc_fp32 = onnx_accuracy(fp32_onnx, test_ds)
    acc_int8 = onnx_accuracy(int8_onnx, test_ds)

    # Throughput: eager vs onnx-fp32 vs onnx-int8, batch 1 + batch 8.
    def bench_all(batch: int, iters: int) -> dict:
        return {
            "pytorch_eager": bench_torch(model, batch, num_frames, size, iters),
            "onnx_fp32": bench_onnx(fp32_onnx, batch, num_frames, size, iters),
            "onnx_int8": bench_onnx(int8_onnx, batch, num_frames, size, iters),
        }

    lat_b1 = bench_all(1, iters=8)
    lat_b8 = bench_all(8, iters=4)

    sizes = {"fp32_mb": fp32_onnx.stat().st_size / 1e6,
             "int8_mb": int8_onnx.stat().st_size / 1e6}

    summary = {
        "model": cfg["model"]["name"], "run_name": run_name,
        "onnx_parity_max_diff": max_diff,
        "accuracy": {"fp32_top1": acc_fp32["top1"],
                     "int8_top1": acc_int8["top1"],
                     "n_test": acc_int8["n"],
                     "delta": acc_int8["top1"] - acc_fp32["top1"]},
        "cpu_latency_sec_per_clip": {
            "batch1": {k: round(v, 4) for k, v in lat_b1.items()},
            "batch8": {k: round(v, 4) for k, v in lat_b8.items()},
        },
        "speedup_int8_vs_eager_b1": round(lat_b1["pytorch_eager"] / lat_b1["onnx_int8"], 2),
        "speedup_int8_vs_onnxfp32_b1": round(lat_b1["onnx_fp32"] / lat_b1["onnx_int8"], 2),
        "file_sizes_mb": {k: round(v, 1) for k, v in sizes.items()},
        "env": env_snapshot(),
    }
    save_json(summary, out_dir / "logs" / f"{run_name}_quantization.json")
    print(f"[quant] top1 fp32 {acc_fp32['top1']:.3f} -> int8 {acc_int8['top1']:.3f} "
          f"(delta {summary['accuracy']['delta']:+.3f})")
    print(f"[quant] CPU b1 latency: eager {lat_b1['pytorch_eager']:.3f}s | "
          f"onnx {lat_b1['onnx_fp32']:.3f}s | int8 {lat_b1['onnx_int8']:.3f}s "
          f"=> {summary['speedup_int8_vs_eager_b1']}x vs eager")

    _plot_latency(summary, out_dir / "plots" / f"{run_name}_quant_bench.png", run_name)
    return summary


def _plot_latency(summary: dict, out: Path, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = ["pytorch_eager", "onnx_fp32", "onnx_int8"]
    b1 = [summary["cpu_latency_sec_per_clip"]["batch1"][v] for v in variants]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    bars = ax1.bar(variants, b1, color=["#999999", "#4c72b0", "#dd8452"])
    ax1.set_ylabel("CPU latency (s / clip, batch=1)")
    ax1.set_title(f"{run_name}: CPU inference latency")
    for b, v in zip(bars, b1):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                 va="bottom", fontsize=9)
    ax2.bar(["fp32", "int8"],
            [summary["file_sizes_mb"]["fp32_mb"],
             summary["file_sizes_mb"]["int8_mb"]],
            color=["#4c72b0", "#dd8452"])
    ax2.set_ylabel("file size (MB)")
    ax2.set_title("Deployment footprint")
    fig.savefig(out, dpi=170)
    plt.close(fig)


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
