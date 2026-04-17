"""Validate datagen output for fine-tuning compatibility.

Checks:
1. Every JSONL record parses; `messages` has the expected 3-turn chat shape.
2. Forward assistant content parses into pydantic SentenceList[SubjectVerbSentence | SubjectVerbObjectSentence].
3. Backward assistant content is non-empty; user content is valid JSON with `sentences` key.
4. `apply_chat_template` on Qwen2.5-3B-Instruct and (optionally) a Llama-3.2 mirror
   renders cleanly and the assistant turn is recoverable.

Exit code: 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from yaduha_ovp import SubjectVerbObjectSentence, SubjectVerbSentence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, jsonl_iter  # noqa: E402

REQUIRED_ROLES = ("system", "user", "assistant")

DEFAULT_TOKENIZERS = [
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("llama-3.2-3b", "unsloth/Llama-3.2-3B-Instruct"),
]


def parse_structured(d: dict[str, Any]) -> Any:
    if "object" in d:
        return SubjectVerbObjectSentence.model_validate(d)
    return SubjectVerbSentence.model_validate(d)


def check_message_shape(rec: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 3:
        errs.append(f"messages must be a 3-element list, got {type(msgs).__name__} len={len(msgs) if hasattr(msgs, '__len__') else '?'}")
        return errs
    for i, role in enumerate(REQUIRED_ROLES):
        m = msgs[i]
        if not isinstance(m, dict):
            errs.append(f"messages[{i}] not a dict")
            continue
        if m.get("role") != role:
            errs.append(f"messages[{i}] role={m.get('role')!r} expected {role!r}")
        if not isinstance(m.get("content"), str) or not m["content"]:
            errs.append(f"messages[{i}] content empty or non-string")
    return errs


def check_forward(rec: dict[str, Any]) -> list[str]:
    errs = check_message_shape(rec)
    if errs:
        return errs
    asst = rec["messages"][2]["content"]
    try:
        parsed = json.loads(asst)
    except json.JSONDecodeError as e:
        return [f"assistant not JSON: {e}"]
    if "sentences" not in parsed or not isinstance(parsed["sentences"], list):
        return ["assistant missing `sentences` list"]
    if not parsed["sentences"]:
        return ["assistant `sentences` list empty"]
    # Multi-clause records legitimately have len > 1; we just check each parses.
    for i, s in enumerate(parsed["sentences"]):
        try:
            parse_structured(s)
        except Exception as e:
            errs.append(f"sentences[{i}] pydantic validation failed: {type(e).__name__}: {str(e)[:120]}")
    return errs


def check_backward(rec: dict[str, Any]) -> list[str]:
    errs = check_message_shape(rec)
    if errs:
        return errs
    # Backward user follows SentenceToEnglishTool: double-encoded single sentence
    # json.dumps(sentence.model_dump_json()) — parse twice to reach the dict.
    user = rec["messages"][1]["content"]
    try:
        inner = json.loads(user)
    except json.JSONDecodeError as e:
        return [f"user outer not JSON: {e}"]
    if not isinstance(inner, str):
        return [f"user outer must decode to a JSON-string, got {type(inner).__name__}"]
    try:
        sent = json.loads(inner)
    except json.JSONDecodeError as e:
        return [f"user inner not JSON: {e}"]
    if not isinstance(sent, dict) or "subject" not in sent or "verb" not in sent:
        return ["user inner does not look like a Sentence dict (missing subject/verb)"]
    try:
        parse_structured(sent)
    except Exception as e:
        return [f"user pydantic validation failed: {type(e).__name__}: {str(e)[:120]}"]
    return errs


def validate_file(path: Path, check_fn, label: str) -> tuple[int, int, list[tuple[int, list[str]]]]:
    total = 0
    ok = 0
    bad: list[tuple[int, list[str]]] = []
    for i, rec in enumerate(jsonl_iter(path)):
        total += 1
        errs = check_fn(rec)
        if errs:
            bad.append((i, errs))
        else:
            ok += 1
    print(f"[{label:15s}] {path.name}  total={total} ok={ok} bad={len(bad)}", file=sys.stderr)
    for i, errs in bad[:5]:
        print(f"  record {i}: {errs[0]}", file=sys.stderr)
    if len(bad) > 5:
        print(f"  ... and {len(bad) - 5} more", file=sys.stderr)
    return total, ok, bad


def test_chat_templates(
    forward_path: Path, backward_path: Path, tokenizer_ids: list[tuple[str, str]], n: int
) -> bool:
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
    except ImportError:
        print("transformers not installed — skipping chat-template test", file=sys.stderr)
        return True

    f_samples = []
    for rec in jsonl_iter(forward_path):
        f_samples.append(rec)
        if len(f_samples) >= n:
            break
    b_samples = []
    for rec in jsonl_iter(backward_path):
        b_samples.append(rec)
        if len(b_samples) >= n:
            break

    all_ok = True
    for label, tid in tokenizer_ids:
        print(f"\n--- chat-template test: {label} ({tid}) ---", file=sys.stderr)
        try:
            tok = AutoTokenizer.from_pretrained(tid)
        except Exception as e:
            print(f"  skip: could not load tokenizer ({type(e).__name__}: {str(e)[:100]})", file=sys.stderr)
            continue
        for direction, samples in (("forward", f_samples), ("backward", b_samples)):
            lengths = []
            for rec in samples:
                try:
                    rendered = tok.apply_chat_template(
                        rec["messages"], tokenize=False, add_generation_prompt=False
                    )
                    ids = tok(rendered).input_ids
                    lengths.append(len(ids))
                    if rec["messages"][2]["content"][:20] not in rendered:
                        print(f"  [{direction}] warning: assistant content not found in rendered text", file=sys.stderr)
                        all_ok = False
                except Exception as e:
                    print(f"  [{direction}] FAIL apply_chat_template: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                    all_ok = False
                    break
            if lengths:
                print(
                    f"  [{direction}] n={len(lengths)}  token lengths: "
                    f"min={min(lengths)} med={sorted(lengths)[len(lengths)//2]} max={max(lengths)}",
                    file=sys.stderr,
                )
    return all_ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--sample-n", type=int, default=8,
                   help="How many records per direction to feed through each tokenizer")
    p.add_argument("--no-tokenizer", action="store_true", help="Skip chat-template check")
    args = p.parse_args()

    d = Path(args.out_dir)
    files = [
        ("forward_train", d / "forward_train.jsonl", check_forward),
        ("forward_val",   d / "forward_val.jsonl",   check_forward),
        ("backward_train", d / "backward_train.jsonl", check_backward),
        ("backward_val",   d / "backward_val.jsonl",   check_backward),
    ]

    print("=== schema validation ===", file=sys.stderr)
    overall_bad = 0
    for label, path, fn in files:
        if not path.exists():
            print(f"[{label:15s}] MISSING: {path}", file=sys.stderr)
            overall_bad += 1
            continue
        _, _, bad = validate_file(path, fn, label)
        overall_bad += len(bad)

    tmpl_ok = True
    if not args.no_tokenizer:
        print("\n=== chat-template test ===", file=sys.stderr)
        tmpl_ok = test_chat_templates(
            d / "forward_train.jsonl",
            d / "backward_train.jsonl",
            DEFAULT_TOKENIZERS,
            args.sample_n,
        )

    print("", file=sys.stderr)
    if overall_bad == 0 and tmpl_ok:
        print("ALL CHECKS PASSED", file=sys.stderr)
        return 0
    print(f"FAILURES: schema={overall_bad} template_ok={tmpl_ok}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
