# Value-head PPO on synthesis conditions (Qwen2.5-7B-Instruct + LoRA r=8)

| Iterations | Batch | Logged iteration rewards (every 10) | Best iteration reward |
|---|---|---|---|
| 200 | 4 | 0.7811 – 0.8793 | 0.9007 |

**Interpretation.** The run shows the 7B LoRA PPO pipeline executing end to end on one GPU (generation, value head, clipped update, adapter-disabled reference). The rewards do not measure the policy: the rule reward reads outcomes stored in each record, which generated conditions do not change. Sampling four random training records per iteration reproduces the logged rewards (mean 0.848 vs 0.850; best-of-200 median 0.901).

Held-out record reward 0.808 (100/100 above 0.75) is the rule score of the 100 held-out Ketoprofen records, identical for every method.

Status: iteration count, logged rewards and 0.9007 CARRY_FORWARD_WITH_INTERPRETATION_NOTE (record statistics, not policy quality); "+61.6%" DO_NOT_USE.
