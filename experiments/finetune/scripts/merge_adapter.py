"""Merge a LoRA adapter into its base model and save as a standalone HF dir.

Used to produce a merged model directory suitable for `ollama create` (Ollama
0.5+ supports creating models from HF safetensors via Modelfile FROM) or for
llama.cpp GGUF conversion.

Example:
    uv run --project yaduha-ovp python yaduha-ovp/experiments/finetune/scripts/merge_adapter.py \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --adapter yaduha-ovp/experiments/finetune/adapters/qwen2.5-3b-instruct-forward \
        --out yaduha-ovp/experiments/finetune/merged/ft-qwen2.5-3b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True,
                   help="Output directory for the merged model + tokenizer.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading base {args.base_model} in bf16...", file=sys.stderr)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)

    print(f"attaching LoRA {args.adapter}...", file=sys.stderr)
    model = PeftModel.from_pretrained(base, args.adapter)

    print("merging adapter into base weights...", file=sys.stderr)
    merged = model.merge_and_unload()

    print(f"saving merged model to {out}...", file=sys.stderr)
    merged.save_pretrained(str(out))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(str(out))

    print(f"done. Contents:", file=sys.stderr)
    for f in sorted(out.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MiB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
