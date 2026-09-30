"""Model zoo: three temporal modeling strategies, one unified contract.

The experiment design (each delta is attributable):

  A. FramePoolClassifier       frozen ResNet-18 feats -> mean pool -> linear
     "bag of frames": no temporal order, just appearance averaging.
     This is the floor. If A ~= B ~= C, temporal modeling does not matter
     for this subset (it does — see README).

  B. TemporalTransformerClassifier
     same frozen features -> pos. embedding + 2-layer Transformer encoder
     -> mean pool -> linear.
     A-vs-B isolates *learned temporal contextualization* on identical
     features (same split, same cache, same optimizer family).

  C. R3D18Classifier           Kinetics-400-pretrained r3d_18, fine-tuned
     end-to-end. 3D convolutions learn spatiotemporal features jointly
     instead of relying on frozen 2D appearance features.

Key implementation details that matter:
  * B MUST have positional embeddings. Without them, self-attention is
    permutation-invariant and the model degenerates into an expensive mean
    pool. Frame ORDER is exactly the signal we claim to be modeling.
  * A and B deliberately share the mean-pool aggregator so the A-vs-B delta
    measures contextualization, not the pooling choice.
  * C keeps torchvision's Kinetics normalisation (see video_dataset.NORM_STATS);
    mixing ImageNet stats into a Kinetics-pretrained 3D CNN is a classic
    silent accuracy leak.
"""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn


class FramePoolClassifier(nn.Module):
    """Model A — temporal mean pooling over frozen frame features."""

    input_kind = "feature"  # consumes (B, T, D) cached features

    def __init__(self, num_classes: int, in_dim: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, D). Mean over frames == mean pooling in 1D.
        pooled = feats.mean(dim=1)                     # (B, D)
        return self.head(pooled)                       # (B, num_classes)


class TemporalTransformerClassifier(nn.Module):
    """Model B — Transformer temporal encoder over frozen frame features.

    Learned positional embedding + nn.TransformerEncoder (2 layers).
    Aggregation is mean pooling — identical to Model A on purpose, so the
    A-vs-B comparison measures exactly the value of temporal attention.
    """

    input_kind = "feature"

    def __init__(self, num_classes: int, in_dim: int = 512, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_frames: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        b, t, _ = feats.shape
        x = self.proj(feats) + self.pos_embed[:, :t]   # (B, T, d)
        x = self.encoder(x)                            # temporal attention
        return self.head(self.norm(x).mean(dim=1))     # same pooling as Model A


class R3D18Classifier(nn.Module):
    """Model C — torchvision r3d_18 (Kinetics-400 pretrained), new head.

    3D convs factorize space and time jointly at every layer; fine-tuning
    adapts the spatiotemporal features to UCF101 instead of freezing
    appearance (A/B). Differential LRs (backbone 1e-4, head 1e-3) come from
    the standard transfer-learning recipe: pretrained features need small
    updates, the new head needs large ones.
    """

    input_kind = "video"  # consumes (B, 3, T, H, W) clips

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.net = r3d_18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.net(clips)


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """Factory: config['model']['name'] -> instantiated model."""
    name = cfg["model"]["name"].lower()
    mcfg = cfg["model"]
    if name == "framepool":
        return FramePoolClassifier(num_classes)
    if name == "temporal_transformer":
        return TemporalTransformerClassifier(
            num_classes, in_dim=mcfg.get("in_dim", 512),
            d_model=mcfg.get("d_model", 256), nhead=mcfg.get("nhead", 8),
            num_layers=mcfg.get("num_layers", 2),
            dim_feedforward=mcfg.get("dim_feedforward", 512),
            dropout=mcfg.get("dropout", 0.1))
    if name == "r3d18":
        return R3D18Classifier(num_classes, pretrained=mcfg.get("pretrained", True))
    raise ValueError(f"Unknown model: {name}")


def param_groups(model: nn.Module, cfg: dict) -> list:
    """Differential learning rates: pretrained backbone vs fresh head.

    For r3d_18: backbone at cfg lr * backbone_lr_scale (0.1), head at lr.
    Feature-input models have no pretrained part inside them (the frozen
    ResNet lives outside the training graph), so everything uses lr.
    """
    lr = float(cfg["training"]["lr"])
    if isinstance(model, R3D18Classifier):
        head_params = list(model.net.fc.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        scale = float(cfg["training"].get("backbone_lr_scale", 0.1))
        return [{"params": backbone_params, "lr": lr * scale},
                {"params": head_params, "lr": lr}]
    return [{"params": model.parameters(), "lr": lr}]
