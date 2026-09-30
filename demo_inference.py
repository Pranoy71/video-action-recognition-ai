"""Zero-training demo: classify a single video with the trained r3d_18 models.

Run from the repository root.

=======================================================================
SETUP — where the trained model files go (NOT committed to git)
=======================================================================
The repo ships code + results, not weights (GitHub's 100 MB file limit;
weights are distributed via the GitHub Release). If you downloaded the
trained models from the Release page, place them exactly here — both
paths are already gitignored, so they will never be committed by accident:

    results/checkpoints/r3d18_best.pt      <- PyTorch weights (127 MB) (get from the Release)
    results/quantized/r3d18_int8.onnx      <- INT8 ONNX model (32 MB) (get from the Release)

Rename if needed, e.g.:
    ucf101_r3d18_best_trained.pt  ->  results/checkpoints/r3d18_best.pt
    ucf101_r3d18_int8.onnx        ->  results/quantized/r3d18_int8.onnx

Alternatively, point directly at any weights file with --weights (the
file may live anywhere — outside the repo is safest).

=======================================================================
USAGE — from the repo root, after `pip install -r requirements.txt`
=======================================================================
# 1) PyTorch backend (default path: results/checkpoints/r3d18_best.pt)
python demo_inference.py --video path/to/clip.avi

# 2) INT8 ONNX backend (default path: results/quantized/r3d18_int8.onnx)
python demo_inference.py --video path/to/clip.avi --backend onnx

# 3) Point at weights directly (backend auto-detected from the extension)
python demo_inference.py --video clip.avi --weights r3d18_best.pt
python demo_inference.py --video clip.avi --weights r3d18_int8.onnx

Any short action clip works (.avi / .mp4 / .mkv — decoded with PyAV).
The script prints top-5 classes with confidence and a decode-vs-model
latency split, using the exact training-time preprocessing (16 uniformly
sampled frames, 112x112, Kinetics normalization).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.infer import load_clip  # identical preprocessing to src/infer.py

# Fallback class list (alphabetical, as used in training). The torch
# checkpoint carries the authoritative list and is preferred when present.
UCF101_10_CLASSES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling",
    "BalanceBeam", "BandMarching", "BaseballPitch", "Basketball",
    "BasketballDunk", "BenchPress",
]

DEFAULT_TORCH_WEIGHTS = Path("results/checkpoints/r3d18_best.pt")
DEFAULT_ONNX_WEIGHTS = Path("results/quantized/r3d18_int8.onnx")


def resolve_classes(ckpt_classes, weights: Path) -> list[str]:
    """Priority: checkpoint metadata > classes.txt > built-in list."""
    if ckpt_classes:
        return list(ckpt_classes)
    txt = Path("data/ucf101/classes.txt")
    if txt.exists():
        return txt.read_text().split()
    print(f"[note] no class metadata found with {weights.name}; "
          f"using the training class list")
    return UCF101_10_CLASSES


def run_torch(weights: Path, clip: torch.Tensor):
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    classes = resolve_classes(ckpt.get("classes"), weights)

    from src.models.classifiers import build_model
    model = build_model(cfg, ckpt["num_classes"])
    model.load_state_dict(ckpt["model"])
    model = model.cpu().eval()

    with torch.no_grad():
        model(clip.unsqueeze(0))                       # warmup
        t0 = time.time()
        logits = model(clip.unsqueeze(0))
        infer_sec = time.time() - t0
    return logits.softmax(1)[0], classes, f"torch (CPU, fp32)", infer_sec


def run_onnx(weights: Path, clip: torch.Tensor):
    import onnxruntime as ort
    session = ort.InferenceSession(
        str(weights), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    x = clip.unsqueeze(0).numpy().astype(np.float32)

    session.run(None, {inp.name: x})                  # warmup
    t0 = time.time()
    (logits,) = session.run(None, {inp.name: x})
    infer_sec = time.time() - t0

    probs = torch.from_numpy(logits).softmax(1)[0]
    classes = resolve_classes(None, weights)
    return probs, classes, f"onnxruntime (CPU, INT8)", infer_sec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Zero-training demo: classify one video with trained weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path,
                    help="path to a video clip (.avi/.mp4/.mkv)")
    ap.add_argument("--weights", default=None, type=Path,
                    help="path to .pt or .onnx weights "
                         "(defaults: results/checkpoints/r3d18_best.pt | "
                         "results/quantized/r3d18_int8.onnx)")
    ap.add_argument("--backend", default=None, choices=["torch", "onnx"],
                    help="override backend (default: auto-detect from "
                         "the weights file extension)")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    # Resolve backend + weights (auto-detect from extension when unset).
    if args.backend is None:
        if args.weights and args.weights.suffix == ".onnx":
            args.backend = "onnx"
        else:
            args.backend = "torch"
    if args.weights is None:
        args.weights = (DEFAULT_ONNX_WEIGHTS if args.backend == "onnx"
                        else DEFAULT_TORCH_WEIGHTS)

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    if not args.weights.exists():
        raise SystemExit(
            f"weights not found: {args.weights}\n"
            f"Put the trained model at that path (see the header of this "
            f"script), or pass --weights directly.\n"
            f"Repo-root relative defaults checked: "
            f"{DEFAULT_TORCH_WEIGHTS} | {DEFAULT_ONNX_WEIGHTS}")

    # Preprocessing params from the training config the checkpoint carries.
    if args.backend == "torch":
        ckpt_cfg = torch.load(args.weights, map_location="cpu",
                              weights_only=False)["config"]
        data_cfg = ckpt_cfg["data"]
    else:  # onnx: r3d_18's export shape (1, 3, 16, 112, 112)
        data_cfg = {"num_frames": 16, "spatial_size": 112, "norm": "kinetics"}
    num_frames = data_cfg.get("num_frames", 16)
    size = data_cfg.get("spatial_size", 112)
    norm = data_cfg.get("norm", "kinetics")

    t0 = time.time()
    clip = load_clip(args.video, num_frames, size, norm)
    decode_sec = time.time() - t0

    run = run_torch if args.backend == "torch" else run_onnx
    probs, classes, backend_desc, infer_sec = run(args.weights, clip)

    top = probs.topk(min(args.top_k, len(classes)))
    print(f"video   : {args.video.name}")
    print(f"backend : {backend_desc}")
    print(f"weights : {args.weights}")
    print(f"decode  : {decode_sec * 1000:7.1f} ms "
          f"({num_frames} frames @ {size}px)")
    print(f"infer   : {infer_sec * 1000:7.1f} ms (CPU, after warmup)")
    print(f"top-{min(args.top_k, len(classes))}   :")
    for rank, (p, c) in enumerate(zip(top.values, top.indices), 1):
        print(f"  {rank}. {classes[c]:24s} {p.item() * 100:5.1f}%")


if __name__ == "__main__":
    main()
