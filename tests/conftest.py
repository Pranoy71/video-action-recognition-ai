"""Pytest fixtures: a tiny synthetic action dataset (real .avi files).

Motion is the class signal, so the dataset is genuinely learnable:
  move_right -> square drifts right
  move_up    -> square drifts up
  static     -> square stays centred

Writing actual .avi containers (via PyAV, mpeg4 codec) means the tests
exercise the REAL decode path (torchvision.io.read_video -> PyAV), not a
mocked one. CI therefore covers: container decode, clip sampling,
normalisation, splits, models, training loop, and export.
"""

import random
from pathlib import Path

import numpy as np
import pytest
import av

W, H, N_FRAMES = 64, 48, 24
CLASSES = ["move_right", "move_up", "static"]


def _frame(square_x: int, square_y: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # dark background
    s = 12
    x0, y0 = max(square_x, 0), max(square_y, 0)
    x1, y1 = min(square_x + s, W), min(square_y + s, H)
    img[y0:y1, x0:x1] = (220, 180, 60)
    return img


def _write_avi(path: Path, motion: str, phase: int) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=8)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    for t in range(N_FRAMES):
        if motion == "move_right":
            x, y = int(4 + (W - 24) * t / (N_FRAMES - 1)), H // 2 - 6
        elif motion == "move_up":
            x, y = W // 2 - 6, int(H - 20 - (H - 32) * t / (N_FRAMES - 1))
        else:
            x, y = W // 2 - 6, H // 2 - 6
        frame = av.VideoFrame.from_ndarray(_frame(x + phase, y), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def make_synthetic_dataset(root: Path, per_class: int = 3) -> Path:
    """3 classes x per_class videos, split 2/1/1 per class (train/val/test)."""
    rng = random.Random(0)
    ucf_root = root / "ucf101"
    (ucf_root / "videos").mkdir(parents=True, exist_ok=True)
    (ucf_root / "splits").mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    for ci, cls in enumerate(CLASSES):
        d = ucf_root / "videos" / cls
        d.mkdir(exist_ok=True)
        for k in range(per_class):
            name = f"v_{cls}_g{k:02d}_c01.avi"
            _write_avi(d / name, cls, phase=k * 3)
            bucket = "train" if k < per_class - 2 else (
                "val" if k == per_class - 2 else "test")
            splits[bucket].append((f"{cls}/{name}", ci))

    (ucf_root / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    for name, entries in splits.items():
        rng.shuffle(entries)
        with open(ucf_root / "splits" / f"{name}.txt", "w") as f:
            for rel, label in entries:
                f.write(f"{rel} {label}\n")
    return ucf_root


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    return make_synthetic_dataset(tmp_path_factory.mktemp("synth"))


@pytest.fixture(scope="session")
def synthetic_features(synthetic_root, tmp_path_factory):
    """Cached ResNet-18 features over the synthetic set (downloads ImageNet
    weights once per session)."""
    from src.data.extract_features import extract
    out = tmp_path_factory.mktemp("feats")
    extract(synthetic_root, out, num_frames=8, batch_size=4, workers=0)
    return out
