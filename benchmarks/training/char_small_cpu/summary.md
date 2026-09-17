# Character-level `small` model on CPU

| Metric | Value |
|---|---|
| Parameters | 10.6M |
| Best validation loss | 1.479 (step 1,500) |
| Training steps | 5,000 (batch 32 × 256 tokens) |
| Wall clock | ~45 min (Apple M-series CPU) |
| Generation speed (KV cache) | 49 tokens/sec |

Computation reproduced by `PretrainAlgorithm` (token cross-entropy, AdamW with matrix-only weight decay, cosine schedule, random-window memmap sampling). Status: applicable, no revalidation required.
