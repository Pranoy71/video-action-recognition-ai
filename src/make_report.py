"""Aggregate all run metrics into README-ready tables and plots.

    python -m src.make_report

Reads results/logs/*_metrics.json (+ *_quantization.json) and writes:
  * results/results_table.md       — the headline comparison table
  * results/plots/accuracy_vs_fps.png — the accuracy/throughput Pareto view
  * results/quantization_table.md  — INT8 trade-off table

This is the mechanism that keeps the README honest: numbers in the write-up
are generated from run artifacts, never hand-typed.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.common import find_repo_root

DISPLAY = {
    "framepool": "A: ResNet-18 + mean pool",
    "temporal_transformer": "B: ResNet-18 + Transformer",
    "r3d18": "C: r3d_18 (3D CNN, fine-tuned)",
}


_ORDER = {"framepool": 0, "temporal_transformer": 1, "r3d18": 2}


def _load_all(out_dir: Path) -> list:
    rows = []
    for f in sorted((out_dir / "logs").glob("*_metrics.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    rows.sort(key=lambda r: _ORDER.get(r["model"], 99))
    return rows


def make_results_table(rows: list, out_md: Path) -> str:
    lines = [
        "| model | test top-1 | test top-5 | macro-F1 | pipeline params (M) | "
        "trainable (M) | MACs/clip (G) | weights (MB) | CPU clips/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        e = r["efficiency"]
        name = DISPLAY.get(r["model"], r["model"])
        lines.append(
            f"| {name} | {r['test']['top1']:.3f} | {r['test']['top5']:.3f} | "
            f"{r['test']['macro_f1']:.3f} | "
            f"{e['params_total'] / 1e6:.1f} | "
            f"{e.get('params_trainable_head', e['params_total']) / 1e6:.2f} | "
            f"{e['macs_per_clip'] / 1e9:.1f} | "
            f"{e['weights_mb']:.1f} | {e['cpu']['clips_per_sec']:.2f} |")
    table = "\n".join(lines)
    out_md.write_text(table + "\n")
    return table


def make_quant_table(out_dir: Path, out_md: Path) -> str:
    q = out_dir / "logs" / "r3d18_quantization.json"
    if not q.exists():
        return ""
    with open(q) as fh:
        d = json.load(fh)
    lat = d["cpu_latency_sec_per_clip"]["batch1"]
    lines = [
        "| variant | test top-1 | CPU latency b1 (s) | size (MB) |",
        "|---|---|---|---|",
        f"| PyTorch eager (fp32) | — | {lat['pytorch_eager']:.3f} | "
        f"{d['file_sizes_mb']['fp32_mb']:.0f} |",
        f"| ONNX fp32 | {d['accuracy']['fp32_top1']:.3f} | "
        f"{lat['onnx_fp32']:.3f} | {d['file_sizes_mb']['fp32_mb']:.0f} |",
        f"| ONNX INT8 (static) | {d['accuracy']['int8_top1']:.3f} | "
        f"{lat['onnx_int8']:.3f} | {d['file_sizes_mb']['int8_mb']:.0f} |",
        "",
        f"INT8 accuracy delta: {d['accuracy']['delta']:+.3f} top-1; "
        f"speedup vs eager: {d['speedup_int8_vs_eager_b1']}x; "
        f"vs ONNX fp32: {d['speedup_int8_vs_onnxfp32_b1']}x.",
    ]
    table = "\n".join(lines)
    out_md.write_text(table + "\n")
    return table


def plot_accuracy_vs_fps(rows: list, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for r in rows:
        name = DISPLAY.get(r["model"], r["model"])
        x = r["efficiency"]["cpu"]["clips_per_sec"]
        y = r["test"]["top1"]
        ax.scatter(x, y, s=90)
        ax.annotate(name, (x, y), xytext=(6, 4), textcoords="offset points",
                    fontsize=9)
    ax.set_xlabel("CPU throughput (clips/s, end-to-end model, batch=4)")
    ax.set_ylabel("test top-1 accuracy")
    ax.set_title("Accuracy vs. CPU inference cost")
    ax.grid(alpha=0.3)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    out_dir = find_repo_root() / "results"
    rows = _load_all(out_dir)
    if not rows:
        print("[report] no metrics found — run training + evaluation first")
        return
    table = make_results_table(rows, out_dir / "results_table.md")
    print(table)
    make_quant_table(out_dir, out_dir / "quantization_table.md")
    plot_accuracy_vs_fps(rows, out_dir / "plots" / "accuracy_vs_fps.png")
    print(f"[report] wrote {out_dir / 'results_table.md'} and plots/")


if __name__ == "__main__":
    main()
