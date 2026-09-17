# DPO (length-normalised) on synthesis preferences

| Pairs | Epochs | Best reward margin | Loss (first → last step) |
|---|---|---|---|
| 80 | 2 | 0.00034 | 0.6929 → 0.6919 |

Held-out record reward (policy-independent): mean 0.8080 ± 0.0151, 100/100 above 0.75.

The held-out statistics (mean 0.80799, std 0.01512) are reproduced exactly from `data/samples/synthesis/trajectories_literature.jsonl` records 400–499 by `tests/unit/test_benchmark_semantics.py`.
