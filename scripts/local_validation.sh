#!/usr/bin/env bash
# Full CPU validation: tests + an end-to-end pipeline on the tiny model. No GPU required.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/local-validation}
rm -rf "$OUT"

python -m pytest tests -q
forgeline data prepare --input data/samples/text/tiny_corpus.txt --output "$OUT/data" --tokenizer char
forgeline train configs/training/pretrain_tiny_cpu.yaml --max-steps 100 --output-dir "$OUT/pretrain" --set data.path="$OUT/data"
forgeline train configs/training/sft_tiny_cpu.yaml --max-steps 50 --output-dir "$OUT/sft" --set checkpoint_path="$OUT/pretrain/checkpoints/final"
forgeline train configs/post_training/dpo_tiny_cpu.yaml --max-steps 20 --output-dir "$OUT/dpo" --set checkpoint_path="$OUT/sft/checkpoints/final"
forgeline train configs/post_training/rlvr_math_tiny_cpu.yaml --max-steps 5 --output-dir "$OUT/rlvr"
forgeline evaluate --checkpoint "$OUT/dpo/checkpoints/final" --suites gsm8k mmlu --offline --max-tokens 8 --output "$OUT/eval.json"
forgeline export --checkpoint "$OUT/dpo/checkpoints/final" --output "$OUT/model-q8.gguf" --quantize q8_0
echo "local validation complete: $OUT"
