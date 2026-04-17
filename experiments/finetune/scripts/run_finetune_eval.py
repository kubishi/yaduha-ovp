"""Run the eval pipeline with a fine-tuned HF forward model.

Mirrors experiments/run_translations.py line-for-line except step 1 (forward):
the structured JSON comes from an HF model + LoRA adapter via unconstrained
generation, rather than from Ollama/OpenAI with structured-output mode.

Writes results to yaduha-ovp/experiments/results/<tag>.jsonl in the same
schema as run_translations.py, so run_metrics.py scores it directly.

Loads base+adapter once, generates sequentially (HF generate is single-GPU),
and parallelizes only the strong-model decoder calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from yaduha.agent.openai import OpenAIAgent
from yaduha.loader import LanguageLoader
from yaduha.tool.sentence_to_english import SentenceToEnglishTool
from yaduha_ovp import SubjectVerbObjectSentence, SubjectVerbSentence
from yaduha_ovp.prompts import get_prompt

HERE = Path(__file__).resolve().parent
DATAGEN = HERE.parent.parent / "datagen"
DATA = HERE.parent.parent / "data" / "evaluation_sentences.csv"
RESULTS = HERE.parent.parent / "results"

sys.path.insert(0, str(DATAGEN / "datagen"))
# Reuse the experiments' mask_oov to stay consistent with the eval pipeline.
# (Importing from experiments/ directly would require path hacks; just copy.)
from yaduha_ovp import (  # noqa: E402
    INTRANSITIVE_VERB_LOOKUP,
    NOUN_LOOKUP,
    TRANSITIVE_VERB_LOOKUP,
)


FORWARD_SYSTEM = get_prompt(
    include_vocab=True,
    include_examples=(SubjectVerbSentence, SubjectVerbObjectSentence),
)


def mask_oov(sentence: Any) -> tuple[Any, list[str]]:
    """Match experiments/run_translations.py::mask_oov exactly."""
    clone = sentence.model_copy(deep=True)
    oov: list[str] = []
    for field in ("subject", "object"):
        part = getattr(clone, field, None)
        if part is not None and hasattr(part, "head"):
            if part.head not in NOUN_LOOKUP:
                oov.append(part.head)
                part.head = "[NOUN]"
    verb = getattr(clone, "verb", None)
    if verb is not None and hasattr(verb, "lemma"):
        in_vocab = (
            verb.lemma in TRANSITIVE_VERB_LOOKUP
            or verb.lemma in INTRANSITIVE_VERB_LOOKUP
        )
        if not in_vocab:
            oov.append(verb.lemma)
            verb.lemma = "[VERB]"
    return clone, oov


def clean(s: str) -> str:
    s = s.strip()
    if s and s[-1] not in ".!?":
        s += "."
    if s:
        s = s[0].upper() + s[1:]
    return s


def parse_structured(d: dict):
    if "object" in d:
        return SubjectVerbObjectSentence.model_validate(d)
    return SubjectVerbSentence.model_validate(d)


def load_dataset() -> list[dict[str, str]]:
    with DATA.open() as f:
        return [{"sentence": r["sentence"], "type": r["type"]} for r in csv.DictReader(f)]


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("error") is None and rec.get("source"):
                    done.add(rec["source"])
            except Exception:
                pass
    return done


def generate_structured(
    model, tokenizer, english: str, max_new_tokens: int
) -> tuple[list[Any], dict[str, Any]]:
    """Returns (list of pydantic sentences, timing/token dict). Raises on parse fail."""
    messages = [
        {"role": "system", "content": FORWARD_SYSTEM},
        {"role": "user", "content": english},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    gen_ids = out[0, enc.input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    raw = json.loads(text)  # let JSONDecodeError propagate
    if "sentences" not in raw or not isinstance(raw["sentences"], list):
        raise ValueError(f"output missing sentences list: {text[:120]}")
    parsed = [parse_structured(s) for s in raw["sentences"]]
    return parsed, {
        "t_forward": dt,
        "forward_prompt_tokens": int(enc.input_ids.shape[1]),
        "forward_completion_tokens": int(gen_ids.shape[0]),
        "raw": text,
    }


def decode_one(
    strong: OpenAIAgent, sentence_types: tuple, structured: list[Any]
) -> dict[str, Any]:
    """Run the strong-model decoder for both backwards and comparator. Runs
    sequentially here (callers batch across source sentences in parallel)."""
    s2e = SentenceToEnglishTool(agent=strong, SentenceType=sentence_types)

    ovp_targets = [clean(str(s)) for s in structured]
    ovp_targets_masked = [clean(s.str_masked()) for s in structured]
    target = " ".join(ovp_targets)
    target_masked = " ".join(ovp_targets_masked)
    has_placeholders = target != target_masked

    bw_parts: list[str] = []
    bw_pt = bw_ct = 0
    t0 = time.time()
    for s in structured:
        r = s2e(s)
        bw_parts.append(clean(r.content))
        bw_pt += r.prompt_tokens
        bw_ct += r.completion_tokens
    t_bw = time.time() - t0
    backwards = " ".join(bw_parts)

    cmp_parts: list[str] = []
    cmp_pt = cmp_ct = 0
    oov_tokens: list[str] = []
    t0 = time.time()
    if has_placeholders:
        for s in structured:
            masked, oov = mask_oov(s)
            oov_tokens.extend(oov)
            r = s2e(masked)
            cmp_parts.append(clean(r.content))
            cmp_pt += r.prompt_tokens
            cmp_ct += r.completion_tokens
        comparator = " ".join(cmp_parts)
    else:
        comparator = backwards
    t_cmp = time.time() - t0

    return {
        "target": target,
        "target_masked": target_masked,
        "backwards": backwards,
        "comparator": comparator,
        "oov_tokens": oov_tokens,
        "has_placeholders": has_placeholders,
        "num_structured_sentences": len(structured),
        "structured_json": [s.model_dump() for s in structured],
        "bw_prompt_tokens": bw_pt,
        "bw_completion_tokens": bw_ct,
        "cmp_prompt_tokens": cmp_pt,
        "cmp_completion_tokens": cmp_ct,
        "t_backwards": t_bw,
        "t_comparator": t_cmp,
    }


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--strong-model", default="gpt-4o-mini")
    p.add_argument("--tag", required=True,
                   help="Output filename stem, e.g. 'ft-qwen2.5_3b__gpt-4o-mini'")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--decoder-parallel", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out_path = RESULTS / f"{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    done = load_done(out_path)
    todo = [r for r in rows if r["sentence"] not in done]
    print(f"total={len(rows)} done={len(done)} todo={len(todo)} out={out_path}", file=sys.stderr)
    if not todo:
        return 0

    print(f"loading base model {args.base_model}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto"
    )
    print(f"loading LoRA adapter {args.adapter}...", file=sys.stderr)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    language = LanguageLoader.load_language("ovp")
    sentence_types = language.sentence_types

    strong = OpenAIAgent(
        model=args.strong_model,
        api_key=__import__("os").environ["OPENAI_API_KEY"],
        temperature=0.0,
    )

    # Phase 1: generate all structured outputs sequentially (GPU-bound).
    # Phase 2: run decoder calls in parallel (network-bound).
    print("phase 1: generating structured outputs...", file=sys.stderr)
    forward_results: list[dict[str, Any]] = []
    t_start = time.time()
    for i, row in enumerate(todo):
        t0 = time.time()
        try:
            structured, info = generate_structured(
                model, tokenizer, row["sentence"], args.max_new_tokens,
            )
            forward_results.append({
                "row": row,
                "structured": structured,
                "info": info,
                "error": None,
            })
        except Exception as e:
            forward_results.append({
                "row": row,
                "structured": None,
                "info": {"t_forward": time.time() - t0, "raw": ""},
                "error": f"{type(e).__name__}: {e}",
            })
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(todo) - i - 1) / rate if rate > 0 else float("inf")
        status = "OK" if forward_results[-1]["error"] is None else "ERR"
        print(
            f"  [{i+1}/{len(todo)}] {status} {row['type']:>20s} "
            f"{elapsed:6.1f}s, {rate:4.2f}/s, ETA {eta:6.0f}s :: {row['sentence'][:60]}",
            file=sys.stderr,
        )

    # Phase 2
    print("\nphase 2: decoder calls...", file=sys.stderr)
    def finalize(item: dict[str, Any]) -> dict[str, Any]:
        row = item["row"]
        t0 = time.time()
        if item["error"] is not None:
            return {
                "source": row["sentence"],
                "type": row["type"],
                "target": None,
                "target_masked": None,
                "backwards": None,
                "comparator": None,
                "has_placeholders": None,
                "structured_json": None,
                "forward_prompt_tokens": item["info"].get("forward_prompt_tokens", 0),
                "forward_completion_tokens": item["info"].get("forward_completion_tokens", 0),
                "t_forward": item["info"].get("t_forward", 0.0),
                "t_backwards": 0.0,
                "t_comparator": 0.0,
                "wall_time": time.time() - t0,
                "raw_forward": item["info"].get("raw", ""),
                "error": item["error"],
            }
        try:
            decoded = decode_one(strong, sentence_types, item["structured"])
            rec = {
                "source": row["sentence"],
                "type": row["type"],
                **decoded,
                "forward_prompt_tokens": item["info"]["forward_prompt_tokens"],
                "forward_completion_tokens": item["info"]["forward_completion_tokens"],
                "t_forward": item["info"]["t_forward"],
                "wall_time": time.time() - t0 + item["info"]["t_forward"],
                "raw_forward": item["info"]["raw"],
                "error": None,
            }
            return rec
        except Exception as e:
            return {
                "source": row["sentence"],
                "type": row["type"],
                "error": f"decode failure: {type(e).__name__}: {e}",
                "wall_time": time.time() - t0,
            }

    completed = 0
    t2 = time.time()
    with out_path.open("a") as fout, ThreadPoolExecutor(max_workers=args.decoder_parallel) as ex:
        futures = {ex.submit(finalize, it): it for it in forward_results}
        for fut in as_completed(futures):
            rec = fut.result()
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            completed += 1
            elapsed = time.time() - t2
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(forward_results) - completed) / rate if rate > 0 else float("inf")
            status = "OK" if rec.get("error") is None else "ERR"
            print(
                f"  [{completed}/{len(forward_results)}] {status} {rec.get('type','?'):>20s} "
                f"{elapsed:5.1f}s {rate:4.2f}/s ETA {eta:5.0f}s :: {rec.get('source','')[:55]}",
                file=sys.stderr,
            )
    print(f"done in {time.time() - t_start:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
