"""Model contract tests: shapes, gradients, differential LRs, sanity."""

import torch
import torch.nn as nn

from src.models.classifiers import (FramePoolClassifier, R3D18Classifier,
                                    TemporalTransformerClassifier, build_model,
                                    param_groups)


def _cfg(name, lr=0.01):
    return {"model": {"name": name}, "training": {"lr": lr,
            "backbone_lr_scale": 0.1}}


def test_frame_pool_forward_backward():
    m = FramePoolClassifier(num_classes=7)
    x = torch.randn(4, 16, 512)
    out = m(x)
    assert out.shape == (4, 7)
    out.sum().backward()
    assert m.head.weight.grad is not None


def test_transformer_forward_shape_and_pos_embed():
    m = TemporalTransformerClassifier(num_classes=5)
    assert m.pos_embed.shape[1] >= 16  # positional embedding MUST exist
    out = m(torch.randn(3, 16, 512))
    assert out.shape == (3, 5)


def test_r3d18_forward():
    m = R3D18Classifier(num_classes=6, pretrained=False)  # random init: fast
    out = m(torch.randn(2, 3, 16, 112, 112))
    assert out.shape == (2, 6)


def test_differential_lr_only_for_r3d():
    """r3d_18 gets two param groups (backbone 0.1x, head 1x); the
    feature-input models get a single group."""
    cfg = _cfg("r3d18", lr=0.01)
    m = R3D18Classifier(num_classes=4, pretrained=False)
    groups = param_groups(m, cfg)
    assert len(groups) == 2
    lrs = sorted(g["lr"] for g in groups)
    assert lrs == [0.001, 0.01]

    groups = param_groups(FramePoolClassifier(4), _cfg("framepool"))
    assert len(groups) == 1 and groups[0]["lr"] == 0.01


def test_build_model_factory():
    assert isinstance(build_model(_cfg("framepool"), 3), FramePoolClassifier)
    assert isinstance(build_model(
        {"model": {"name": "temporal_transformer"}}, 3),
        TemporalTransformerClassifier)
    try:
        build_model({"model": {"name": "nope"}}, 3)
        assert False, "should raise"
    except ValueError:
        pass


def test_train_one_epoch_learns_something():
    """On a linearly separable feature problem, one epoch must beat chance."""
    from src.engine import train_one_epoch
    torch.manual_seed(0)
    n, d, c = 64, 32, 3
    centers = torch.randn(c, d) * 3
    y = torch.arange(n) % c
    x = centers[y].unsqueeze(1) + 0.1 * torch.randn(n, 4, d)  # (N, T, D)

    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=16)
    model = TemporalTransformerClassifier(num_classes=c, in_dim=d, d_model=64,
                                          nhead=4, num_layers=1,
                                          dim_feedforward=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    for _ in range(30):
        stats = train_one_epoch(model, loader, torch.device("cpu"), crit, opt, 0)
    assert stats["top1"] > 0.9
