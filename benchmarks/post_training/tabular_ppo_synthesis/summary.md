# Tabular actor-critic PPO on synthesis records

Every value below is a statistic of dataset records scored with a rule reward, reproduced exactly by `tests/unit/test_benchmark_semantics.py`. None depends on the trained actor-critic.

| Dataset (split in file order) | Reward | Train-record mean | Held-out record mean |
|---|---|---|---|
| labeled expanded (500) | current rule | 0.8577 (identical in all 5 epochs) | 0.8121 (100/100 above 0.75) |
| literature (500) | earlier efficiency term `0.1·(1 − steps/10)` | 0.8401 | 0.8011 (1/100 above the train mean) |
| improvable (500) | earlier efficiency term | 0.6033 | 0.5935 (30/100 above the train mean) |
| labeled small (100, first run) | current rule | 0.8559 | 0.8181 (20 records) |

Leave-one-molecule-out on the literature records (held-out molecule's record mean): Aspirin 0.897, Ibuprofen 0.852, Naproxen 0.761, Paracetamol 0.878, Ketoprofen 0.808 — also reproduced by `examples/05_synthesis_generalization.py`.

Status: cite only as dataset statistics; "+62.4%"-style improvements against a constant 0.5 are DO_NOT_USE.
