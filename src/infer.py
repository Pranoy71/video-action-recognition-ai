"""Single-video inference CLI.

    python -m src.infer --video data/ucf101/videos/Biking/v_Biking_g01_c01.avi \
        --config configs/r3d18.yaml

Prints top-5 classes with softmax confidence and a latency split
(decode vs. model). The split is deliberate: on CPU the model dominates,
but once compute is accelerated (GPU / INT8) video decoding becomes the
residual floor — reporting both makes that trade-off visible instead of
hiding it inside one end-to-end number.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.common import find_repo_root, get_device, load_config
from src.data.extract_features import ResNet18FeatureExtractor
from src.data.video_dataset import NORM_STATS
from src.evaluate import load_trained
from src.models.classifiers import build_model


def load_clip(path: Path, num_frames: int, size: int, norm: str) -> torch.Tensor:
    from src.data.video_reader import read_video_frames
    vframes = read_video_frames(path)
    n = vframes.shape[0]
    step = max(n - 1, 1) / max(num_frames - 1, 1)
    idx = [min(int(round(i * step)), n - 1) for i in range(num_frames)]
    clip = vframes[idx].permute(3, 0, 1, 2).float() / 255.0

    import torchvision.transforms as T
    frames = clip.permute(1, 0, 2, 3)  # (T, C, H, W) float
    frames = torch.stack([
        T.Resize((size, size), antialias=True)(f) for f in frames])
    clip = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

    mean = torch.tensor(NORM_STATS[norm][0]).view(3, 1, 1, 1)
    std = torch.tensor(NORM_STATS[norm][1]).view(3, 1, 1, 1)
    return (clip - mean) / std


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    run_name = args.run_name or cfg.get("run_name") or Path(args.config).stem
    out_dir = find_repo_root() / cfg.get("out_dir", "results")
    device = get_device()

    model, ckpt = load_trained(cfg, run_name, out_dir)
    model = model.cpu().eval()
    classes = ckpt.get("classes") or \
        (find_repo_root() / cfg["data"].get("data_dir", "data")
         / "ucf101" / "classes.txt").read_text().split()  # ckpt carries them

    num_frames = cfg["data"].get("num_frames", 16)
    size = cfg["data"].get("spatial_size", 112)
    norm = cfg["data"].get("norm", "kinetics")

    t0 = time.time()
    clip = load_clip(args.video, num_frames, size, norm)
    decode_sec = time.time() - t0

    kind = cfg["model"].get("input_kind") or \
        ("video" if cfg["model"]["name"] == "r3d18" else "feature")
    if kind == "feature":
        # Compose the production pipeline: backbone + trained temporal head.
        class _Pipeline(torch.nn.Module):
            def __init__(self, head):
                super().__init__()
                self.backbone = ResNet18FeatureExtractor()
                self.head = head

            def forward(self, x):
                return self.head(self.backbone(x))

        model = _Pipeline(model).eval()

    t1 = time.time()
    with torch.no_grad():
        logits = model(clip.unsqueeze(0))
    infer_sec = time.time() - t1

    probs = logits.softmax(1)[0]
    top = probs.topk(min(args.top_k, len(classes)))
    print(f"video   : {args.video.name}")
    print(f"decode  : {decode_sec * 1000:7.1f} ms ({num_frames} frames @ {size}px)")
    print(f"infer   : {infer_sec * 1000:7.1f} ms (CPU)")
    print("top-5   :")
    for rank, (p, c) in enumerate(zip(top.values, top.indices), 1):
        print(f"  {rank}. {classes[c]:24s} {p.item() * 100:5.1f}%")


if __name__ == "__main__":
    main()
