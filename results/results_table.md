| model | test top-1 | test top-5 | macro-F1 | pipeline params (M) | trainable (M) | MACs/clip (G) | weights (MB) | CPU clips/s |
|---|---|---|---|---|---|---|---|---|
| A: ResNet-18 + mean pool | 0.840 | 0.987 | 0.815 | 11.2 | 0.01 | 7.8 | 44.8 | 10.00 |
| B: ResNet-18 + Transformer | 0.853 | 1.000 | 0.811 | 12.4 | 1.21 | 7.8 | 49.6 | 9.57 |
| C: r3d_18 (3D CNN, fine-tuned) | 0.987 | 1.000 | 0.980 | 33.2 | 33.17 | 1.6 | 132.8 | 2.74 |
