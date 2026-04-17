"""Spot-check a fine-tuned forward adapter on eval sentences.

Loads the base model + LoRA adapter, applies constrained JSON-Schema decoding
via transformers' structured-output support (outlines under the hood), and
generates the structured SentenceList for each input.

This validates two things at once:
  1. The adapter trained to something reasonable (structural choices make sense).
  2. Structured output is preserved end-to-end — the model's preferred tokens
     stay inside the SentenceList schema.

Usage:
    uv run --project yaduha-ovp python yaduha-ovp/experiments/finetune/scripts/generate.py \\
        --adapter yaduha-ovp/experiments/finetune/adapters/qwen2.5-3b-instruct-forward \\
        --inputs \\
            'I see the dog.' \\
            'The chihuahua runs.' \\
            'The one who cooks eats.'
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from yaduha.loader import LanguageLoader
from yaduha.tool.english_to_sentences import SentenceList
from yaduha_ovp import SubjectVerbObjectSentence, SubjectVerbSentence
from yaduha_ovp.prompts import get_prompt

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE.parent.parent / "data" / "evaluation_sentences.csv"


FORWARD_SYSTEM = get_prompt(
    include_vocab=True,
    include_examples=(SubjectVerbSentence, SubjectVerbObjectSentence),
)


def load_eval_sample(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    with DATA_CSV.open() as f:
        rows = [(r["sentence"], r["type"]) for r in csv.DictReader(f)]
    rng.shuffle(rows)
    return rows[:n]


def parse_structured(d: dict):
    if "object" in d:
        return SubjectVerbObjectSentence.model_validate(d)
    return SubjectVerbSentence.model_validate(d)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", required=True, help="Path to saved LoRA adapter dir")
    p.add_argument("--inputs", nargs="*", default=None,
                   help="Explicit English inputs; if omitted, samples from eval set")
    p.add_argument("--n-sample", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-adapter", action="store_true",
                   help="Compare against the base model (no LoRA) — helpful baseline.")
    args = p.parse_args()

    print(f"loading base model: {args.base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if not args.no_adapter:
        print(f"loading LoRA adapter: {args.adapter}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    # Build inputs
    if args.inputs:
        inputs = [(s, "?") for s in args.inputs]
    else:
        inputs = load_eval_sample(args.n_sample, args.seed)

    # Load the SentenceList schema for reference (we'll verify the output parses)
    _ = LanguageLoader.load_language("ovp")

    print(f"\n=== generations ({'base' if args.no_adapter else 'fine-tuned'}) ===\n",
          file=sys.stderr)

    n_parsed_ok = 0
    for english, sent_type in inputs:
        messages = [
            {"role": "system", "content": FORWARD_SYSTEM},
            {"role": "user", "content": english},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=max(args.temperature, 1e-5),
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0, enc.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        parse_ok = False
        note = ""
        try:
            raw = json.loads(gen)
            if "sentences" in raw and isinstance(raw["sentences"], list):
                parsed = [parse_structured(s) for s in raw["sentences"]]
                rendered = " ".join(str(s) for s in parsed)
                parse_ok = True
                note = f" -> {rendered}"
        except Exception as e:
            note = f" [parse FAIL: {type(e).__name__}: {str(e)[:80]}]"

        if parse_ok:
            n_parsed_ok += 1
        marker = "OK " if parse_ok else "BAD"
        print(f"[{marker}] {sent_type:>18s} :: {english}", file=sys.stderr)
        print(f"       RAW: {gen[:200]}", file=sys.stderr)
        if note:
            print(f"      {note}", file=sys.stderr)
        print("", file=sys.stderr)

    print(f"parse rate: {n_parsed_ok}/{len(inputs)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
