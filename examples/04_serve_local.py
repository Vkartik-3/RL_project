"""Serve a checkpoint over HTTP (OpenAI-compatible) on localhost.

    python examples/04_serve_local.py runs/sft-tiny-cpu/checkpoints/final
    curl -s localhost:8000/v1/completions -d '{"prompt": "Q: What is 2 + 2?\\nA:", "max_tokens": 4}'
"""

import sys

from forgeline.cli.main import main

sys.exit(main(["serve", "--checkpoint", sys.argv[1] if len(sys.argv) > 1 else "runs/sft-tiny-cpu/checkpoints/final", "--port", "8000"]))
