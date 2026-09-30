"""End-to-end smoke: the full train -> evaluate -> export path on synthetic
data, using the same entry points the README documents.

This is the test that catches "works on my machine" rot: if any step of the
documented pipeline breaks, CI goes red.
"""

import json
from pathlib import Path

import torch


CFG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _tiny_cfg(name: str, features_dir: Path, epochs: int = 2) -> dict:
    return {
        "run_name": f"smoke_{name}",
        "seed": 42,
        "out_dir": "results",
        "model": {"name": name, "input_kind": "feature"},
        "data": {"data_dir": "data", "features_dir": str(features_dir)},
        "training": {"epochs": epochs, "batch_size": 4, "optimizer": "adam",
                     "lr": 0.003, "weight_decay": 1e-4, "label_smoothing": 0.0},
    }


def test_train_evaluate_end_to_end(synthetic_root, synthetic_features,
                                   tmp_path, monkeypatch):
    from src import train as train_mod
    from src import evaluate as eval_mod

    # Synthetic root is <tmp>/synth/ucf101; make the fake repo root <tmp>/synth
    # so that repo / data_dir(".") / "ucf101" resolves to the dataset.
    fake_repo = synthetic_root.parent
    monkeypatch.setattr(train_mod, "find_repo_root", lambda: fake_repo)
    monkeypatch.setattr(eval_mod, "find_repo_root", lambda: fake_repo)

    for name in ("framepool", "temporal_transformer"):
        cfg = _tiny_cfg(name, synthetic_features)
        cfg["data"]["data_dir"] = "."

        summary = train_mod.run(cfg, f"smoke_{name}", tmp_path, smoke=False)
        assert summary["best_val_top1"] >= 0.0
        assert (tmp_path / "checkpoints" / f"smoke_{name}_best.pt").exists()
        assert (tmp_path / "logs" / f"smoke_{name}_history.csv").exists()

        metrics = eval_mod.run(cfg, f"smoke_{name}", tmp_path)
        assert metrics["test"]["n"] > 0
        assert 0.0 <= metrics["test"]["top1"] <= 1.0
        assert len(metrics["classes"]) == 3
        assert metrics["efficiency"]["params_total"] > 0


def test_onnx_export_parity(synthetic_root, tmp_path, monkeypatch):
    """ONNX graph must reproduce eager outputs within fp tolerance."""
    from src.models.classifiers import R3D18Classifier
    from src.export_quantize import export_onnx
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(0)
    model = R3D18Classifier(num_classes=3, pretrained=False).eval()
    onnx_path = tmp_path / "m.onnx"
    export_onnx(model, onnx_path, num_frames=8, size=64)

    x = torch.randn(2, 3, 8, 64, 64)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"clip": x.numpy()})[0]
    assert np.abs(ref - got).max() < 1e-4
