"""Training/evaluation engine shared by train.py and evaluate.py.

Kept separate from CLI concerns so tests can exercise the loops directly.
Mixed precision is CUDA-only (CPU autocast for these ops is not profitable
and adds nondeterminism); the GradScaler dance is the standard torch.amp
recipe.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _accuracy(logits: torch.Tensor, labels: torch.Tensor,
              ks=(1, 5)) -> Dict[int, float]:
    """Top-k accuracy for one batch (eval mode, no grad)."""
    maxk = min(max(ks), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(labels.view(-1, 1))
    out = {}
    for k in ks:
        k_eff = min(k, logits.shape[1])
        out[k] = correct[:, :k_eff].any(dim=1).float().mean().item()
    return out


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                    criterion: nn.Module, optimizer, epoch: int,
                    amp: bool = False, max_batches: Optional[int] = None
                    ) -> Dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    total_loss, n = 0.0, 0
    top1 = 0.0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = y.size(0)
        total_loss += loss.item() * bs
        top1 += (logits.argmax(1) == y).float().sum().item()
        n += bs
    return {"loss": total_loss / max(n, 1), "top1": top1 / max(n, 1),
            "time_sec": time.time() - t0}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: Optional[nn.Module] = None, max_batches: Optional[int] = None
             ) -> Dict[str, object]:
    """Returns metrics + per-sample (pred, label, logits) for analysis."""
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels, all_logits = [], [], []
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_logits.append(logits.float().cpu())
    topk = _accuracy(torch.cat(all_logits), torch.tensor(all_labels))
    return {"loss": total_loss / max(n, 1), "top1": topk[1],
            "top5": topk.get(5, topk[1]), "preds": all_preds,
            "labels": all_labels, "time_sec": time.time() - t0, "n": n}


def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}
