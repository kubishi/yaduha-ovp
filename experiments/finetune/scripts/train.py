"""LoRA fine-tune an instruct model on datagen output, one direction at a time.

Uses HuggingFace transformers + TRL + PEFT (no Unsloth/bitsandbytes) to avoid
version pinning conflicts with torch 2.11+cu130. At 3B scale this fits easily
in fp/bf16 on a single RTX 5000 Ada (32 GB).

Loss is masked to assistant tokens only (completion-only), so the large
get_prompt() system prompt does not consume training signal.

Example:
    uv run --project yaduha-ovp python yaduha-ovp/experiments/finetune/scripts/train.py \
        --direction forward \
        --model Qwen/Qwen2.5-3B-Instruct \
        --epochs 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
DATAGEN_OUT = HERE.parent.parent / "datagen" / "out"
ADAPTERS = HERE.parent / "adapters"
CHECKPOINTS = HERE.parent / "checkpoints"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_dataset(path: Path) -> Dataset:
    records = load_jsonl(path)
    # TRL SFTTrainer accepts datasets with a `messages` column directly.
    return Dataset.from_list([{"messages": r["messages"]} for r in records])


def resolve_target_modules(model) -> list[str]:
    """Return all nn.Linear module names (excluding the LM head)."""
    names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if leaf != "lm_head":
                names.add(leaf)
    return sorted(names)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--direction", choices=("forward", "backward"), required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--no-eval", action="store_true",
                   help="Skip periodic eval (some torch/CUDA combos hit kernel faults there).")
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--out-tag", default=None,
                   help="Adapter/checkpoint folder name; defaults to <model-slug>-<direction>")
    p.add_argument("--limit-train", type=int, default=None,
                   help="Cap training examples (smoke-test)")
    args = p.parse_args()

    train_path = DATAGEN_OUT / f"{args.direction}_train.jsonl"
    val_path = DATAGEN_OUT / f"{args.direction}_val.jsonl"
    if not train_path.exists():
        print(f"missing {train_path}", file=sys.stderr)
        return 2

    slug = args.model.split("/")[-1].lower()
    tag = args.out_tag or f"{slug}-{args.direction}"
    ckpt_dir = CHECKPOINTS / tag
    adapter_dir = ADAPTERS / tag

    print(f"--- train.py ---", file=sys.stderr)
    print(f"model          = {args.model}", file=sys.stderr)
    print(f"direction      = {args.direction}", file=sys.stderr)
    print(f"train_path     = {train_path}", file=sys.stderr)
    print(f"val_path       = {val_path}", file=sys.stderr)
    print(f"ckpt_dir       = {ckpt_dir}", file=sys.stderr)
    print(f"adapter_dir    = {adapter_dir}", file=sys.stderr)
    print(f"epochs         = {args.epochs}", file=sys.stderr)
    print(f"lr             = {args.lr}", file=sys.stderr)
    print(f"batch/grad     = {args.batch_size}/{args.grad_accum} "
          f"(effective {args.batch_size * args.grad_accum})", file=sys.stderr)
    print(f"max_seq_length = {args.max_seq_length}", file=sys.stderr)
    print(f"lora           = r={args.lora_r} alpha={args.lora_alpha} "
          f"dropout={args.lora_dropout}", file=sys.stderr)

    ds_train = to_dataset(train_path)
    ds_val = None if args.no_eval else (to_dataset(val_path) if val_path.exists() else None)
    if args.limit_train:
        ds_train = ds_train.select(range(min(args.limit_train, len(ds_train))))
    print(f"train size     = {len(ds_train)}", file=sys.stderr)
    print(f"val size       = {len(ds_val) if ds_val else 0}", file=sys.stderr)

    print("loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading base model (bf16)...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    target_modules = resolve_target_modules(model)
    print(f"lora targets   = {target_modules}", file=sys.stderr)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    sft_config = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if ds_val else "no",
        eval_steps=args.eval_steps if ds_val else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        max_length=args.max_seq_length,
        completion_only_loss=True,  # mask system/user tokens from loss
        dataset_text_field="messages",  # signal chat format to SFTTrainer
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=peft_config,
        args=sft_config,
    )

    print("begin training...", file=sys.stderr)
    trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"saved adapter to {adapter_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
