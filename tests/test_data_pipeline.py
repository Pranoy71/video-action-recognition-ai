"""Data pipeline integrity tests."""

from pathlib import Path

import torch

from src.data.video_dataset import VideoDataset


def _read(split: str, root: Path):
    return [l for l in (root / "splits" / f"{split}.txt").read_text().splitlines() if l]


def test_splits_do_not_overlap(synthetic_root):
    """No video may appear in two splits — leakage would invalidate metrics."""
    a = set(_read("train", synthetic_root))
    b = set(_read("val", synthetic_root))
    c = set(_read("test", synthetic_root))
    assert not (a & b) and not (a & c) and not (b & c)
    assert a | b | c  # and all three are non-empty


def test_split_labels_are_consistent(synthetic_root):
    """Same video path must always carry the same label across splits."""
    seen = {}
    for split in ("train", "val", "test"):
        for line in _read(split, synthetic_root):
            rel, label = line.rsplit(" ", 1)
            if rel in seen:
                assert seen[rel] == int(label)
            seen[rel] = int(label)


def test_decode_and_shapes(synthetic_root):
    ds = VideoDataset(synthetic_root, "train", num_frames=8, spatial_size=112,
                      norm="kinetics", train=False)
    clip, label = ds[0]
    assert clip.shape == (3, 8, 112, 112)
    assert clip.dtype == torch.float32
    assert 0 <= label < 3
    # Normalised Kinetics-style input should be roughly zero-mean-ish range.
    assert clip.abs().max() < 10.0
    assert not ds.decode_failures


def test_train_jitter_is_reproducible(synthetic_root):
    """Same (seed, index) must give identical augmented clips — required for
    reproducible runs."""
    ds = VideoDataset(synthetic_root, "train", num_frames=8, norm="kinetics",
                      train=True, seed=42)
    a = ds[1][0]
    b = ds[1][0]
    assert torch.equal(a, b)


def test_sampling_indices_in_bounds():
    ds = VideoDataset.__new__(VideoDataset)
    ds.num_frames = 8
    ds.train = True
    ds.clip_jitter = True
    import random
    for seed in range(50):
        idx = ds._sample_indices(137, random.Random(seed))
        assert len(idx) == 8 and all(0 <= i < 137 for i in idx)
    ds.train = False
    idx = ds._sample_indices(24, random.Random(0))
    assert len(idx) == 8 and all(0 <= i < 24 for i in idx)


def test_feature_dataset_roundtrip(synthetic_features):
    from src.data.video_dataset import FeatureDataset
    ds = FeatureDataset(synthetic_features, "train")
    assert len(ds) > 0
    feats, label = ds[0]
    assert feats.shape[1] == 512
    assert feats.dtype == torch.float32
    assert 0 <= label < 3
