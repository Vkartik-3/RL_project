# Values without retained evidence

| Method | Documented value | Status | Why |
|---|---|---|---|
| GRPO (G=4), synthesis | 0.823 (peak 0.878) | NEEDS_EVIDENCE_RECOVERY | no results file, log or chart; the committed GRPO evaluation returns 0.8080 for any policy on that split |
| RLAIF self-judge → DPO | 0.814 (peak 0.867) | NEEDS_EVIDENCE_RECOVERY | no results file; the trainer reports DPO loss, not reward |
| STaR, 3 rounds | 0.791 (peak 0.843) | NEEDS_EVIDENCE_RECOVERY | no results file; the trainer reports accuracy, not reward |
| SFT warm-up | 0.412 | NEEDS_EVIDENCE_RECOVERY | the retained SFT file has losses only |

Not published as results. Rerun on GPU to replace them with verified numbers.
