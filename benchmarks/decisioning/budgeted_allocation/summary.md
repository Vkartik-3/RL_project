# Budgeted allocation benchmark

Environment: horizon 48, budget 24.0, 300 held-out episodes per policy, 5 training seeds for PPO. Oracle (hindsight upper bound) mean value 13.219.

| Policy | Value (mean ± seed std) | Utilisation | Pacing error | Early exhaustion | Violations | Regret vs oracle | Regret vs dual pacer |
|---|---|---|---|---|---|---|---|
| threshold_pacing | 7.292 ± 0.000 | 0.700 | 0.178 | 0.000 | 0.02 | 5.927 | +0.415 |
| dual_pacing | 7.707 ± 0.000 | 0.920 | 0.082 | 0.000 | 0.03 | 5.512 | +0.000 |
| ppo_mlp | 7.402 ± 0.098 | 0.873 | 0.153 | 0.045 | 0.60 | 5.818 | +0.306 |
| ppo_gru | 7.595 ± 0.115 | 0.948 | 0.091 | 0.198 | 1.82 | 5.624 | +0.112 |

## Offline policy evaluation (behaviour: ppo_mlp_eps0.2, 300 logged episodes)

| Target | Truth | IPS | SNIPS | PDIS | DR | DR clipped | ESS | max weight | warnings |
|---|---|---|---|---|---|---|---|---|---|
| ppo_mlp_best | 7.566 | 5.235 [0.99, 12.11] | 8.536 [5.44, 10.36] | 5.958 [3.76, 8.77] | 7.671 [6.49, 9.15] | 7.474 [6.62, 8.30] | 4.4 | 74.6 | 2 |
| ppo_gru_best | 7.742 | 0.257 [0.07, 0.47] | 5.535 [4.18, 8.13] | 2.374 [1.73, 3.08] | 6.817 [6.35, 7.20] | 6.928 [6.60, 7.25] | 5.2 | 4.6 | 2 |
| dual_pacing | 7.697 | 0.000 [0.00, 0.00] | — | 1.784 [0.62, 3.34] | 6.917 [6.27, 7.65] | 6.742 [6.37, 7.14] | 0.0 | 0.0 | 4 |
| threshold_pacing | 7.138 | 0.000 [0.00, 0.00] | — | 0.603 [0.37, 0.92] | 6.677 [6.36, 6.95] | 6.747 [6.53, 6.99] | 0.0 | 0.0 | 3 |
| behaviour_itself | 6.927 | 6.802 [6.49, 7.13] | 6.802 [6.49, 7.13] | 6.802 [6.49, 7.13] | 6.828 [6.51, 7.16] | 6.828 [6.51, 7.16] | 300.0 | 1.0 | 0 |

## Simulated A/B: dual_pacing (n=197) vs ppo_gru (n=203)

Δvalue +0.567 (95% bootstrap CI [-0.041, +1.157]), permutation p = 0.084; guardrail failures: ['primary metric CI lower bound -0.0405 < -0.0', 'violations increased by 1.029 > 0.5']; decision: reject.

## Shadow: divergence rate 0.398 (by horizon third [0.381, 0.378, 0.435]); incumbent value 7.656, candidate replay value 8.066

Throughput: evaluation 50886 env steps/s; OPE 297 trajectory-evaluations/s; A/B 518 episodes/s; PPO training ppo_mlp 8s/seed, ppo_gru 26s/seed. Total 179s.
