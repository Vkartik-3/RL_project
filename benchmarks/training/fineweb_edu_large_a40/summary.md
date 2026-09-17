# `large` model on FineWeb-Edu (single A40)

| Metric | Value |
|---|---|
| Best validation loss | **3.5834** (step 29,500) |
| Final train loss | 3.5478 |
| Steps / tokens | 30,000 / ~491M |
| Throughput | 20,700 tok/s average, 20,900 peak |
| torch.compile effect | ~20,800 vs ~15,000 tok/s |
| Generation (KV cache) | ~53 tok/s |

Loss: 12.12 (step 0) → 5.44 (1k) → 4.65 (3k) → 4.30 (5k) → 4.00 (10k) → 3.70 (20k) → 3.58 (30k).

Status: applicable — the forward pass, loss, optimizer grouping, schedule and bf16 autocast are unchanged. No revalidation required.
