| variant | test top-1 | CPU latency b1 (s) | size (MB) |
|---|---|---|---|
| PyTorch eager (fp32) | — | 0.438 | 133 |
| ONNX fp32 | 0.987 | 0.564 | 133 |
| ONNX INT8 (static) | 0.987 | 0.196 | 33 |

INT8 accuracy delta: +0.000 top-1; speedup vs eager: 2.23x; vs ONNX fp32: 2.87x.
