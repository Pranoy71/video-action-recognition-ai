# UCF101 Action Recognition — A Controlled Bake-off of Temporal Modeling Strategies

[![CI](https://github.com/Pranoy71/video-action-recognition-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranoy71/video-action-recognition-ai/actions)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pranoy71/video-action-recognition-ai/blob/main/notebooks/walkthrough.ipynb)

Three architectures, one data pipeline, one evaluation protocol — so every
accuracy delta is attributable to a specific design decision, not noise:

| model | test top-1 | test top-5 | macro-F1 | pipeline params (M) | trainable (M) | MACs/clip (G) | weights (MB) | CPU clips/s |
|---|---|---|---|---|---|---|---|---|
| **A** ResNet-18 + mean pool *(bag of frames)* | 0.840 | 0.987 | 0.815 | 11.2 | 0.01 | 7.8 | 44.8 | 10.00 |
| **B** ResNet-18 + Transformer *(temporal attention)* | 0.853 | 1.000 | 0.811 | 12.4 | 1.21 | 7.8 | 49.6 | 9.57 |
| **C** r3d_18 fine-tuned *(3D CNN, end-to-end)* | **0.987** | 1.000 | **0.980** | 33.2 | 33.2 | **1.6** | 132.8 | 2.74 |
| **C-INT8** (static quantization, ONNX) | 0.987 | 1.000 | — | 33.2 | 33.2 | — | **33.3** | **5.10** |

**Headline findings:**

1. **Appearance alone gets you 84%.** Mean-pooled ImageNet features — no
   temporal order at all — already classify 8/10 test clips. On this subset,
   action recognition is mostly an appearance problem.
2. **Learned temporal aggregation adds only +1.3 points** (B vs A, identical
   frozen features). With 75 test clips, one clip = 1.3 points, so this delta
   is at the edge of noise; B's real edge is top-5 (1.000 vs 0.987). Temporal
   *ordering* of frozen appearance features is not where the value is.
3. **End-to-end spatiotemporal features are worth +13.4 points** (C vs A) —
   far outside noise. Learning *what* to look for across frames (3D convs,
   fine-tuned from Kinetics-400) beats post-hoc aggregation of frozen 2D
   features by an order of magnitude more than temporal attention does.
4. **INT8 quantization is free here**: 0.000 top-1 loss, 2.23x CPU speedup
   (0.438 → 0.196 s/clip, batch 1), 4x smaller file (132.7 → 33.3 MB).
5. **A systems trap worth knowing**: C has 4.9x *fewer* MACs than the A/B
   pipeline (1.6 vs 7.8 G/clip) yet is 3.7x *slower* on CPU — 2D conv
   kernels are far more CPU-optimized than 3D. MACs are a compute-cost
   proxy, not a wall-clock prediction; measure, don't assume.

All numbers are **generated from run artifacts** by `python -m src.make_report`
(`results/results_table.md`, `results/logs/*.json`) — nothing in this table is
hand-typed.

---

## 1. Task & dataset

**UCF101** (Soomro et al., 2012), 10-class subset: 300 train / 30 val / 75 test
clips, 25 fps, 320x240. Two data sources are scripted and idempotent:

- `--source crcv` — the **official release** + official `trainlist01`/`testlist01`
  split files (the paper protocol). Class subset = first-N alphabetical,
  deterministic and documented — no cherry-picking.
- `--source hf` — the HF mirror of the PyTorchVideo subset (used for the
  numbers in this README; the [Colab notebook](notebooks/walkthrough.ipynb)
  runs the full CRCV 20-class protocol on a T4).

**Leakage policy (four guardrails):** official train/test splits; validation
carved **by video group** (UCF101 clips `v_Class_gXX_cYY` sharing a group come
from the same recording session and never straddle splits); pytest asserts
split disjointness; INT8 calibration uses *train* clips only — calibrating
activation scales on test data is a subtle form of test leakage.

## 2. Architecture

```
                    ┌──────────────────────────────────────────────┐
 video ── decode ───►│ 16 frames, uniform, 112x112 (PyAV)           │
                    └──────────────┬───────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
   A: frozen ResNet-18    B: frozen ResNet-18    C: r3d_18 (Kinetics-400)
      16x(512-d) feats      16x(512-d) feats      3D convs, fine-tuned
      mean pool             + pos. embedding      (head lr 10x backbone)
      linear                + 2-layer Transformer
      (10K params)          mean pool + linear
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   ▼
                          class logits (10)
```

**Frame sampling.** 16 frames uniformly spaced over the whole clip — the
standard clip-level protocol and r3d_18's native Kinetics clip length.
Train-time **temporal jitter** (uniform sampling over a random 85-100% window)
is the only temporal augmentation; spatial augmentation (resize 128 → crop
112, hflip, ±0.2 brightness/contrast/saturation) is drawn **once per clip and
applied to all frames** — per-frame randomness would corrupt the motion
signal the models are supposed to learn. All randomness comes from a
per-item RNG `(seed, index)`, so runs are reproducible independent of
dataloader worker topology.

**Controlled comparison by construction.** A and B consume *byte-identical*
cached ResNet-18 features (one backbone pass per video, reused across
epochs — A/B train in seconds, and their difference is exactly the temporal
aggregation strategy). Both use **mean pooling** as the aggregator, so the
A-vs-B delta measures *temporal contextualization*, not the pooling choice.
B's Transformer (2 layers, d=256, 8 heads) **must** carry a positional
embedding — without it self-attention is permutation-invariant and collapses
into an expensive mean pool (asserted in the tests).

**Transformer vs 3D CNN trade-offs (measured, not asserted):** the
frame-level Transformer path is cheap to train (1.2M trainable params on
cached features), CPU-friendly at inference, and modular; but it is capped by
what frozen 2D features see. The 3D CNN learns *spatiotemporal* features
jointly — +13.4 points — at the cost of full-model fine-tuning, 3x the
weights, and (on CPU) 3.7x the wall-clock despite 4.9x fewer MACs. Frame
sampling rate and aggregation are config fields (`configs/*.yaml`), so the
trade-off space is explorable without touching code.

## 3. Evaluation

Protocol: single 16-frame center clip per test video, deterministic
transform; the test split is touched **once per model** by `src/evaluate.py`
on the best-on-val checkpoint (val-selection uses top-1 with a val-loss
tiebreak, because a 30-clip val split saturates quickly — see §4).
Metrics: top-1, top-5, macro-F1, per-class recall, full confusion matrix,
plus a measured efficiency bench (params, MACs/clip via `thop`, weight size,
end-to-end CPU throughput — A/B are benchmarked **as the full pipeline**
including the frozen backbone, not the head-on-features shortcut).

| plot | what it shows |
|---|---|
| `results/plots/r3d18_confusion.png` | C: a single test error (ApplyEyeMakeup → ApplyLipstick) |
| `results/plots/framepool_confusion.png` | A: errors concentrated in appearance-similar classes |
| `results/plots/accuracy_vs_fps.png` | the accuracy-vs-CPU-cost Pareto view |
| `results/plots/r3d18_quant_bench.png` | eager vs ONNX fp32 vs ONNX INT8 latency + footprint |

**Quantization** (`src/export_quantize.py`): r3d_18 → ONNX (opset 17,
parity 1e-6) → **static INT8** (QDQ, per-channel weights, 32 train-clip
calibration). Static rather than dynamic because a conv-dominated model
gains nothing from quantizing only Linear layers. Results (CPU, batch 1):
eager 0.438 s/clip → ONNX 0.564 → **INT8 0.196**; top-1 unchanged at 0.987;
file 132.7 → 33.3 MB.

## 4. Analysis & failure modes

- **The failure signature is appearance, not time.** A and B misclassify
  exactly where frame *appearance* is similar and motion is the discriminator:
  Basketball → BaseballPitch (both "person + ball + court"), Archery →
  BenchPress (both static equipment poses), ApplyEyeMakeup ↔ ApplyLipstick.
  C resolves nearly all of them by learning the motion — its one remaining
  error is the ApplyEyeMakeup/Lipstick pair, where even the motion is similar
  (close-up, mirror, hand-to-face) and the true discriminator is the
  *instrument*. That is a fine-grained appearance problem, which motivates
  the distillation/flow items below.
- **Val saturation & small-sample honesty.** r3d_18 hits 100% val top-1 in
  epoch 1 (30 val clips; Kinetics transfer is strong on UCF-style actions) —
  so val-based selection is coarse, and the reported test numbers carry
  75-clip granularity: one clip = 1.3 points. The C-vs-A/B gap (+13.4) is
  far outside that noise; the B-vs-A gap (+1.3) is not. I report both
  without inflating the latter.
- **Training stopped at 5/10 epochs** (train 0.58→0.97, val saturated;
  epoch 4 showed the first overfit dip). On a 300-clip subset, more epochs
  polish train loss, not generalization — documented rather than silently
  truncated.

## 5. Future improvements (concrete, prioritized)

1. **Two-stream RGB + optical flow (RAFT).** The confusion analysis says the
   residual errors are exactly motion-vs-appearance confusions — the case
   two-stream literature shows flow solves. Cost: flow extraction is
   expensive at inference; the practical variant is distilling the flow
   stream into the RGB model (one-stream distillation).
2. **Temporal-resolution & multi-clip TTA ablation.** 8/16/32 frames and
   k-clip logit averaging, benchmarked the same way I benchmarked
   quantization — accuracy-vs-latency curves, not vibes. Cheap to run: both
   are config fields.
3. **Quantization extensions + distillation.** QAT to recover the ~0-1
   point static quantization can cost at larger scale; TensorRT INT8 for
   GPU; distill C into B (a 27x-smaller trainable student) for a
   deployment-grade accuracy/efficiency point on the Pareto plot.

## 6. Reproduce

```bash
pip install -r requirements.txt
make data-hf                                   # or: make data (official CRCV, 20 classes)
make features && make train-a && make train-b  # ~3 min total (CPU)
make train-c                                   # ~7 min/epoch CPU; ~20-40 min on a T4
make eval-all && make quantize && make report  # tables + plots + INT8 bench
make test                                      # 14 tests, synthetic data, no download
```

The numbers above were produced on a **CPU-only box (2 cores / 4 GB RAM)** —
`configs/r3d18_cpu.yaml` is that exact profile; `configs/r3d18.yaml` is the
GPU profile. Training state checkpoints after every epoch
(`--resume` survives session drops). Single-video inference:
`python -m src.infer --video <path> --config configs/r3d18_cpu.yaml`
(reports decode and model latency separately — on CPU the model dominates
450 ms vs 41 ms decode, but once compute is accelerated — GPU or INT8 —
decode becomes the residual floor; splitting the two makes that visible).

<details><summary><b>Repo layout & engineering notes</b></summary>

```
configs/                 one YAML = one complete experiment spec (versioned)
src/data/                download (splits + leakage policy) -> PyAV decoder -> datasets
src/models/              the three architectures + factory + differential-LR groups
src/train.py             config-driven loop: AMP, cosine, resume, tie-aware checkpointing
src/evaluate.py          test metrics + plots + efficiency bench (pipeline-level)
src/export_quantize.py   ONNX + static INT8 + measured accuracy/latency trade-off
src/infer.py             single-video CLI
tests/                   synthetic .avi suite (real decode path) + e2e + ONNX parity
notebooks/walkthrough.ipynb   one-click Colab end-to-end
results/                 metrics.json, plots, generated tables (README numbers come from here)
```

- Decoding uses PyAV directly (`src/data/video_reader.py`): torchvision's
  video reader availability varies across builds; an explicit PyAV path is
  version-proof and debuggable. Decode failures degrade to tracked gray
  clips (counted into metrics.json) instead of crashing an epoch.
- Augmentation params are drawn from a per-item RNG — `(seed, index)`
  fully determines the augmented clip; reproducibility is independent of
  worker count and process hash seeds.
- CI (GitHub Actions) runs the full test suite plus a 1-epoch training smoke
  on synthetic data — the pipeline the README documents is the pipeline CI
  exercises.
- Environment snapshot (torch/onnxruntime versions, device) is written into
  every metrics.json, so any number in this README traces to a specific run.

</details>

<details><summary><b>Dataset & protocol details</b></summary>

- UCF101 classes used (10): ApplyEyeMakeup, ApplyLipstick, Archery,
  BabyCrawling, BalanceBeam, BandMarching, BaseballPitch, Basketball,
  BasketballDunk, BenchPress — includes two near-duplicate pairs
  (EyeMakeup/Lipstick, Basketball/Dunk), which makes the error analysis
  informative rather than decorative.
- Normalization: Kinetics stats for r3d_18 (its pretraining stats),
  ImageNet stats for the ResNet-18 feature extractor — per-model, in config.
  Mixing these silently costs accuracy points.
- Optimizers: Adam for the transformer head (convention), SGD+momentum with
  differential LRs (head 1e-2, backbone 1e-3) + cosine schedule for r3d_18;
  identical epoch budgets and schedule shapes across models.
- Label smoothing 0.1 on all three; single seed (42); train-time temporal
  jitter ON for C, unavailable to A/B by caching design (documented
  trade-off: the cache buys control and 100x cheaper head-training).

</details>
