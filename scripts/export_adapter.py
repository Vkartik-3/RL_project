#!/usr/bin/env python3
"""Export the LoRA adapter of a HuggingFace-backend checkpoint as a PEFT adapter directory,
optionally publishing it to the Hugging Face Hub with a generated model card.

    python scripts/export_adapter.py --checkpoint runs/synthesis-dpo-qwen7b-lora/checkpoints/final \
        --base-model Qwen/Qwen2.5-7B-Instruct --output adapters/synthesis-dpo
    python scripts/export_adapter.py ... --push-to-hub <user>/<repo> [--private]   # needs HF_TOKEN
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

CARD = """---
base_model: {base}
library_name: peft
tags: [lora, forgeline, {algorithm}]
---

# {repo} — {algorithm} LoRA adapter

Adapter produced with Forgeline (`{algorithm}`), LoRA r={r}, alpha={alpha}, targets {targets}.

## Evaluation

{metrics}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{base}")
model = PeftModel.from_pretrained(base, "{repo}")
```
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--target-modules", nargs="*", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    p.add_argument("--metrics-file", default=None, help="evaluation results JSON to include in the model card")
    p.add_argument("--push-to-hub", default=None)
    p.add_argument("--private", action="store_true")
    a = p.parse_args()

    from forgeline.checkpoints.manager import CheckpointManager
    from forgeline.models.policy import HFPolicy

    state = CheckpointManager.load(a.checkpoint)
    if "adapter" not in state.model_state:
        raise SystemExit("checkpoint has no adapter state (was it trained with backend.type=huggingface and LoRA?)")
    policy = HFPolicy(a.base_model, lora_r=a.lora_r, lora_alpha=a.lora_alpha, target_modules=a.target_modules,
                      gradient_checkpointing=False, device_map=None)
    policy.load_checkpoint_state(state.model_state)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    policy.save_adapter(str(out))

    metrics = "No evaluation results attached."
    if a.metrics_file:
        rows = json.loads(Path(a.metrics_file).read_text())
        metrics = "| Suite | Metric | Value |\n|---|---|---|\n" + "\n".join(
            f"| {r['suite']} | {k} | {v:.4g} |" for r in rows for k, v in r["metrics"].items())
    repo = a.push_to_hub or out.name
    (out / "README.md").write_text(CARD.format(base=a.base_model, algorithm=state.algorithm or "post-training", repo=repo, r=a.lora_r,
                                               alpha=a.lora_alpha, targets=", ".join(a.target_modules), metrics=metrics))
    print(f"adapter written to {out}")

    if a.push_to_hub:
        from huggingface_hub import HfApi, create_repo

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("set HF_TOKEN to push to the Hub")
        create_repo(a.push_to_hub, token=token, private=a.private, exist_ok=True)
        HfApi(token=token).upload_folder(folder_path=str(out), repo_id=a.push_to_hub, repo_type="model",
                                         commit_message=f"Upload {state.algorithm} adapter ({date.today().isoformat()})")
        print(f"pushed to https://huggingface.co/{a.push_to_hub}")


if __name__ == "__main__":
    main()
